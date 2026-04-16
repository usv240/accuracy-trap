#!/bin/bash
# Deploy main_zerve.py to AWS Lambda
# Run from repo root. Requires: aws CLI configured (aws configure)

set -e

FUNCTION_NAME="accuracy-trap-api"
REGION="us-east-1"
RUNTIME="python3.11"

echo "=== Step 1: Build package for Linux x86_64 ==="
rm -rf lambda_package && mkdir lambda_package

# Download Linux-compatible wheels (works even on Windows/Mac)
pip install \
  --platform manylinux2014_x86_64 \
  --target lambda_package \
  --implementation cp \
  --python-version 3.11 \
  --only-binary=:all: \
  fastapi uvicorn pydantic pydantic-core mangum requests pytrends numpy

echo "=== Step 2: Copy application code ==="
cp api/main_zerve.py lambda_package/main_zerve.py

echo "=== Step 3: Create zip ==="
cd lambda_package
zip -r ../lambda_deploy.zip . -q
cd ..
echo "Package size: $(du -sh lambda_deploy.zip | cut -f1)"

echo "=== Step 4: Create or update Lambda function ==="
# Check if function exists
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
  echo "Updating existing function..."
  aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --zip-file fileb://lambda_deploy.zip \
    --region $REGION
else
  echo "Creating new function..."
  # You need an IAM role ARN — create one at IAM console with AWSLambdaBasicExecutionRole
  read -p "Enter your Lambda IAM role ARN: " ROLE_ARN
  aws lambda create-function \
    --function-name $FUNCTION_NAME \
    --runtime $RUNTIME \
    --role $ROLE_ARN \
    --handler main_zerve.handler \
    --zip-file fileb://lambda_deploy.zip \
    --timeout 30 \
    --memory-size 512 \
    --region $REGION
fi

echo "=== Step 5: Enable Function URL (public HTTPS, no auth) ==="
aws lambda add-permission \
  --function-name $FUNCTION_NAME \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE \
  --region $REGION 2>/dev/null || true

URL_CONFIG=$(aws lambda create-function-url-config \
  --function-name $FUNCTION_NAME \
  --auth-type NONE \
  --region $REGION 2>/dev/null || \
  aws lambda get-function-url-config \
  --function-name $FUNCTION_NAME \
  --region $REGION)

FUNCTION_URL=$(echo $URL_CONFIG | python3 -c "import sys,json; print(json.load(sys.stdin)['FunctionUrl'])")

echo ""
echo "=== Done ==="
echo "Function URL: $FUNCTION_URL"
echo "Test it:      curl ${FUNCTION_URL}health"
echo "API docs:     ${FUNCTION_URL}docs"
