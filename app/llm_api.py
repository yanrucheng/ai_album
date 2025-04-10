from openai import OpenAI
import json

import base64
import requests
from pathlib import Path
from typing import Optional, Union, Dict, Any

import my_deco

class LLMClient:
    def __init__(self,
                 api_key='sk-ipaoxjdlloswleefiawvqxgfivfxdvkvlxfzktitiksvxtwu',
                 base_url="https://api.siliconflow.cn/v1",
                 default_model="Pro/deepseek-ai/DeepSeek-V3"):
        """
        Generic LLM API client for interacting with various models.
        
        Args:
            api_key (str): Your API key for the LLM provider
            base_url (str): Base URL of the API endpoint (default: siliconflow.cn)
            default_model (str): Default model to use if none specified
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.default_model = default_model
    
    @my_deco.retry_api(max_retries=3, delay=1.0)
    def query(self,
              user_prompt,
              system_prompt='You are a helpful assistant.',
              response_format=None,
              model=None):
        """
        Query the LLM with a prompt.
        
        Args:
            prompt (str): The input prompt/question
            model (str): Model to use (default: instance default_model)
            json_format (bool): Whether to request/parse JSON response
            **kwargs: Additional parameters for the API call
            
        Returns:
            dict or str: The response from the LLM
        """
        model = model or self.default_model

        if response_format is None:
            response_format = {'type': "text"}
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                stream=False,
                response_format=response_format,
            )
            
            content = response.choices[0].message.content
            
            if response_format['type'] == 'json_schema':
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"response": content, "code": 200}
            return content
            
        except Exception as e:
            return {"error": str(e), "code": 500}
    
    def __call__(self, prompt, **kwargs):
        """Shortcut for the query method"""
        return self.query(prompt, **kwargs)

class VLMClient:
    def __init__(self,
                 api_key: str = 'sk-ipaoxjdlloswleefiawvqxgfivfxdvkvlxfzktitiksvxtwu',
                 base_url: str = "https://api.siliconflow.cn/v1",
                 default_model: str = "deepseek-ai/deepseek-vl2"):
        """
        VLM (Vision-Language Model) API client for interacting with image+text models.
        
        Args:
            api_key (str): Your API key for the VLM provider
            base_url (str): Base URL of the API endpoint (default: siliconflow.cn)
            default_model (str): Default model to use if none specified
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @my_deco.retry_api(max_retries=3, delay=1.0)
    def query(self, 
              image_path: str, 
              user_prompt: str, 
              system_prompt: str = 'You are a helpful assistant',
              has_nude: bool = True,
              response_format: Optional[Dict[str, Any]] = None,
              model: Optional[str] = None,
              **kwargs) -> Union[str, Dict]:
        """
        Query the VLM with a prompt and image.
        
        Args:
            prompt (str): The input prompt/question
            image_path (str): Path to the image file
            model (str): Model to use (default: instance default_model)
            response_format (dict): Format for response (e.g., {'type': 'json_schema'})
            **kwargs: Additional parameters for the API call
            
        Returns:
            dict or str: The response from the VLM
        """

        assert not has_nude, 'You must ensure there is no nude content before sending it to cloud LLM.'
        model = model or self.default_model
        
        # Read and encode the image
        try:
            image_data = self._encode_image(image_path)
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}", "code": 400}
        
        # Prepare the payload
        payload = {
            "model": model,
            "stream": False,
            "max_tokens": 1024,          # Increased to allow longer descriptions
            "temperature": 0.3,          # Lowered for less randomness, more focus
            "top_p": 0.3,                # Narrower sampling for higher quality tokens
            "top_k": 30,                 # Focus on top 30 relevant tokens
            "frequency_penalty": 0.1,    # Allow slight repetition if needed for detail
            "n": 1,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "image_url": {
                                "detail": "high",
                                "url": f"data:image/TYPE;base64,{image_data}"
                            },
                            "type": "image_url"
                        },
                        {
                            "text": user_prompt,
                            "type": "text"
                        }
                    ]
                }
            ],
            **kwargs
        }
        
        # Add response format if specified
        if response_format:
            payload["response_format"] = response_format
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            
            return content
            
        except requests.exceptions.HTTPError as e:
            # Extract status code from HTTP errors
            status_code = e.response.status_code if hasattr(e, 'response') else 500
            return {"error": str(e), "code": status_code}
        except Exception as e:
            return {"error": str(e), "code": 500}
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def __call__(self, prompt: str, image_path: str, **kwargs) -> Union[str, Dict]:
        """Shortcut for the query method"""
        return self.query(prompt, image_path, **kwargs)

# Example usage:
if __name__ == "__main__":
    # Initialize with your API key and preferred defaults
    llm = LLMClient()
    
    # Basic query with JSON response
    response = llm("What are the benefits of using large language models?")
    print("JSON Response:", response)
    
    # Plain text response
    text_response = llm.query("Explain machine learning in simple terms", json_format=False)
    print("Text Response:", text_response)
    
    # Using different model
    custom_model_response = llm.query(
        "Generate a creative story about AI",
        model="Pro/deepseek-ai/DeepSeek-R1"
    )
    print("Custom Model Response:", custom_model_response)
