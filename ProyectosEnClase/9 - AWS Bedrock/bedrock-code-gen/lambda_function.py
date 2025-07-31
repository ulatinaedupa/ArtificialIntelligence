
from datetime import datetime
from helpers.bedrock_helpers import (
    invoke_bedrock_model,
    generate_code,
    save_to_s3_bucket
) 


def lambda_handler(event, context):
    message = event['message']
    prog_lang = event['key']
    print(message, prog_lang)

    code = generate_code(message=message, prog_lang=prog_lang)
    if code:
        curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        key = f'code-output/{curr_time}.py'
        bucket = 'bedrock-output-code'
        save_to_s3_bucket(code, bucket, key)
        
    if not code:
        print('No code Saved')
        
    return {
        'statusCode': 200,
        'body': {
            'code': code, 
            'message': 'Code generated successfully' if code else 'Code generation failed'
        }
    }
