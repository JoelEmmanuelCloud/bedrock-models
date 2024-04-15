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

# Define the prompt function, passing also the context.
def process_prompt(prompt):
    response = get_completion(prompt, 6000)
    return response

# Define the prompt for personalized recipe recommendations
prompt = """
Human: Generate personalized recipe recommendations based on my dietary preferences and available ingredients.

Instructions:
- Please provide recipes that match my dietary preferences and desired cuisine.
- I am interested in recipes suitable for the following dietary preferences: vegan, gluten-free.
- Here are the ingredients I have available: quinoa, spinach, chickpeas, tomatoes, avocado.
- I prefer recipes from the following cuisine: Mediterranean.

Assistant:
"""

# Process the prompt and print the response
print("Prompt:")
print(prompt)
print("----------------------")
response = process_prompt(prompt)
print("Response:", response)
