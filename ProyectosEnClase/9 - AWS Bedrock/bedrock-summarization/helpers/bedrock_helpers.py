import boto3
import botocore.config
import json

from email import message_from_bytes


def save_s3_bucket(code, bucket, key):
    s3 = boto3.client('s3', region_name='us-east-1')
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=code)
        print(f'Code Saved to S3 Bucket {bucket}')
    except Exception as e:
        print(f'Error Saving the code to S3\n{e}')


def extract_text_from_multipart(data):
    msg = message_from_bytes(data)

    text_content = ''

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text_content += part.get_payload(decode=True).decode('utf-8') + "\n"

    if not msg.is_multipart():
        if msg.get_content_type() == "text/plain":
            text_content = msg.get_payload(decode=True).decode('utf-8')

    return text_content.strip() if text_content else None


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


def generate_summary(text_content: str) -> str:
    prompt = f"""Summarize the following meeting notes: {text_content}."""
    body = dict(prompt=prompt, temperature=0.2, max_tokens=2000, top_p=0.2)
    try:
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1", config=botocore.config.Config(read_timeout=300, retries=dict(max_attempts=3)))
        summary = invoke_bedrock_model(client=bedrock, id="anthropic.claude-3-sonnet-20240229-v1:0", **body)
        print(summary)
        return summary
    except Exception as e:
        print(f"Error Generating the summary.\n{e}")
        return ""