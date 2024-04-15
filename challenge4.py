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

# Define the prompt for creating an interactive storytelling experience
prompt = """
Human: Create an interactive storytelling experience.

Instructions:
- Start the story with the following prompt: "You wake up in a mysterious forest with no memory of how you got there. The air is thick with mist, and strange sounds echo through the trees."
- Allow the user to make choices throughout the story.
- Provide a list of possible choices the user can make: "1. Explore deeper into the forest. 2. Climb a nearby tree to get a better view. 3. Follow the sound of running water."
- Define the possible outcomes based on each choice: "If you choose 1, you stumble upon an ancient ruin hidden deep within the forest. If you choose 2, you spot a distant mountain peak rising above the treetops. If you choose 3, you discover a sparkling stream flowing through a clearing."
- Ensure the story remains coherent and engaging, adapting to the user's decisions.
- Incorporate user choices seamlessly into the narrative flow.

Assistant:
"""

# Process the prompt and print the response
print("Prompt:")
print(prompt)
print("----------------------")
response = get_completion(prompt, 6000)
print("Response:", response)
