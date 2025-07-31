import boto3
import botocore.config
import json


def invoke_bedrock_model(client, id, prompt, max_tokens=2000, temperature=0, top_p=0.9):
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}],}]
    }

    try:
        response = client.invoke_model(modelId=id, body=json.dumps(native_request))
    except Exception as e:
        print(e)
        return "Model invocation error"
    
    try:
        model_response = json.loads(response["body"].read())
        code = model_response["content"][0]["text"]
        return code
    except Exception as e:
        print(e)
        return "Output parsing error"



def generate_code(message: str, prog_lang: str) -> str:
    prompt = f"""Write {prog_lang} code for the following instructions:\n{message}\nDo not generate code examples or explanation, code only."""
    body = dict(prompt=prompt, temperature=0.2, max_tokens=2000, top_p=0.2)
    try:
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1", config=botocore.config.Config(read_timeout=300, retries=dict(max_attempts=3)))
        code = invoke_bedrock_model(client=bedrock, id="anthropic.claude-3-sonnet-20240229-v1:0", **body)
        print(code)
        return code
    except Exception as e:
        print(f"Error Generating the code.\n{e}")
        return ""



def save_to_s3_bucket(code, bucket, key):
    s3 = boto3.client('s3', region_name='us-east-1')
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=code)
        print(f'Code Saved to S3 Bucket {bucket}')
    except Exception as e:
        print(f'Error Saving the code to S3\n{e}')