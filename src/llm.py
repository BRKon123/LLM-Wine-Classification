from ai_clients.openai_helper import OpenAIHelper
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the AI helper with the API key from environment variables
ai_helper = OpenAIHelper(api_key=os.getenv("ARTIFICIAL_API_KEY"))

# Make a simple test call
messages = [
    {"role": "user", "content": "Say hello in a formal way"}
]

try:
    response = ai_helper.create_chat_completion(messages=messages)
    print("AI Response:", response)
except Exception as e:
    print(f"Error occurred: {e}")