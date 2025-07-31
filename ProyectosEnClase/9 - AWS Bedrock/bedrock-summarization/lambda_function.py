import json
import base64
from datetime import datetime
from helpers.bedrock_helpers import (
    save_s3_bucket,
    extract_text_from_multipart,
    invoke_bedrock_model,
    generate_summary
)

def lambda_handler(event, context):
    print(event)

    document = event['body']
    decoded_document = base64.b64decode(document)
    text_content = extract_text_from_multipart(decoded_document)

    if not text_content:
        return dict(statusCode=400, body=json.dumps('Failed to extract content')) 

    summary = generate_summary(text_content)

    if summary:
        curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        key = f'summary-output/{curr_time}.txt'
        bucket = 'bedrock-summary-bucket'
        save_s3_bucket(summary, bucket, key)
        
    return {
        'statusCode': 200,
        'body': {
            'summary': json.dumps(summary), 
            'message': 'Summary generated successfully' if summary else 'Summary generation failed'
        }
    }
