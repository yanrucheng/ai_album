from openai import OpenAI
import json

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
    
    def query(self, prompt, response_format=None, model=None):
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
                    {'role': 'user', 'content': prompt}
                ],
                stream=False,
                response_format=response_format,
            )
            
            content = response.choices[0].message.content
            
            if response_format['type'] == 'json_schema':
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"response": content}
            return content
            
        except Exception as e:
            return {"error": str(e)}
    
    def __call__(self, prompt, **kwargs):
        """Shortcut for the query method"""
        return self.query(prompt, **kwargs)


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
