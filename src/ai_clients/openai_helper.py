import requests
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
        **kwargs
    ) -> str:
        """
        Create a chat completion using synchronous HTTP requests
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to gpt-4-mini)
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Response text content
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
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()  # Raise exception for bad status codes
        
        return response.json()["choices"][0]["message"]["content"]