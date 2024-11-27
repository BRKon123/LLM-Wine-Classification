import requests
import time
from typing import List, Dict

class OpenAIHelper:
    """Simple helper class to handle LLM API interactions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://candidate-llm.extraction.artificialos.com/v1/chat/completions"
        
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: int = 30,
        **kwargs
    ) -> str:
        """
        Create a chat completion with retry logic for rate limiting
        """
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        attempts = 0
        while attempts <= max_retries:
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                
                # If rate limited (429) and we have retries left
                if response.status_code == 429 and attempts < max_retries:
                    attempts += 1
                    print(f"Rate limited. Retrying in {retry_delay} seconds... (Attempt {attempts}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                
                # For all other cases, raise any HTTP errors
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
            except requests.exceptions.HTTPError as e:
                if attempts >= max_retries:
                    print(f"Failed after {max_retries} retries")
                    raise e
                attempts += 1
                print(f"HTTP error occurred. Retrying in {retry_delay} seconds... (Attempt {attempts}/{max_retries})")
                time.sleep(retry_delay)