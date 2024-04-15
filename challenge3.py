import boto3
import json
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Configure the Bedrock client with loaded credentials
bedrock = boto3.client(
    'bedrock-runtime',
    'us-east-1',
    endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN")
)

# Create a request body in JSON format with parameters for the Bedrock request
def get_completion(prompt, max_tokens_to_sample=1000):
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": max_tokens_to_sample,
        "temperature": 1,
        "top_k": 1,
        "top_p": 0.001,
        "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"],
        "anthropic_version": "bedrock-2023-05-31"
    })

    # Define necessary information for calling the Bedrock model
    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'

    # Call the Bedrock model with the request body
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType)

    # Decode the response body bytes to string and strip whitespace
    response_body = response.get('body').read().decode('utf-8').strip()

    # Check if the response body is not empty
    if response_body:
        completion = json.loads(response_body).get('completion')
        return completion
    else:
        return None

# Define the prompt for summarizing academic papers for different audiences
prompt = """
Human: Summarize this academic paper for different audiences.

Instructions:
- Please summarize the academic paper provided below in a way that suits different audiences.
- The content of the academic paper is as follows: "In this study, we investigate the impact of climate change on biodiversity in tropical rainforests. We analyze data collected over a 10-year period and identify key patterns and trends. Our findings highlight the urgent need for conservation efforts to mitigate the effects of climate change on vulnerable ecosystems."
- Tailor the summary based on the audience type: experts, general public, students.
- The summary length should be adjusted accordingly to meet the needs of each audience.
- Provide a brief and understandable overview of the paper's key points, avoiding technical jargon for non-expert audiences.

Assistant:
"""

# Process the prompt and print the response
print("Prompt:")
print(prompt)
print("----------------------")
response = get_completion(prompt, 6000)
print("Response:", response)
