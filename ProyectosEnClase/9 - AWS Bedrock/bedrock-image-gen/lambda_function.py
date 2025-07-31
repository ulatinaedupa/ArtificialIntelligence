import json
from datetime import datetime
from helpers.bedrock_helpers import (
    save_to_s3_bucket,
    generate_image
)


def lambda_handler(event, context):
    prompt = event['message']
    print(prompt)
    
    # Stability AI Model Inputs
    model_id, accept, content_type = "stability.stable-diffusion-xl-v1", "application/json", "application/json"

    # Format the request payload using the model's native structure.
    body = json.dumps(
        {"text_prompts": [{"text":prompt, "weight":10}],
         "cfg_scale":10, "steps":50, "seed":0,
         "width":1024, "height":1024, "samples":1
    })
    
    try:
        image_content = generate_image(model_id=model_id, prompt=prompt, accept=accept, content_type=content_type, body=body)
    except Exception as e:
        return {
            'statusCode':400,
            'body': json.dumps(f'Could not generate image\n{e}')
        }        
    
    if image_content:
        curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        bucket, key = 'bedrock-img-gen', f'img-output/{curr_time}.png'
        try:
            save_to_s3_bucket(image_content, bucket, key)
        except Exception as e:
            return {
                'statusCode':400,
                'body': json.dumps('Error Saving image\n{e}')
            }
    
        return {
            'statusCode': 200,
            'body': json.dumps('Image Saved to S3!')
        }