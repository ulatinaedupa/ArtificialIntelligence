import boto3
import botocore
import base64
import json

def save_to_s3_bucket(image_content, bucket, key):
    s3 = boto3.client('s3', region_name='us-east-1')
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=image_content)
        print(f'Code Saved to S3 Bucket {bucket}')
    except Exception as e:
        print(f'Error Saving the code to S3\n{e}')

def generate_image(model_id, prompt, accept, content_type, body):
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1", config=botocore.config.Config(read_timeout=300, retries=dict(max_attempts=3)))
    # Invoke the model with the request.
    try:
        # Invoke Bedrock API
        response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
    except Exception as e:
        print('Model Error')

    # Decode the response body.
    model_response = json.loads(response.get("body").read())

    # Extract the image data.
    base64_image_data = model_response["artifacts"][0].get("base64")
    image_content = base64.decodebytes(bytes(base64_image_data, "utf-8"))
    return image_content