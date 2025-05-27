from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict

def openai_chat(model: str, messages: List[Dict], api_key: str = None) -> Dict:
    """Invoke OpenAI model with retry logic."""
    from langchain_openai import ChatOpenAI
    client = ChatOpenAI(model_name=model, api_key=api_key, max_retries=3)
    prompt = ChatPromptTemplate.from_messages([(m["role"], m["content"]) for m in messages])
    response = client.invoke(prompt)
    return {"content": response.content, "metadata": response.response_metadata}
