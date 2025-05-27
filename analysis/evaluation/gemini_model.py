from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict

def gemini_chat(model: str, messages: List[Dict], api_key: str = None) -> Dict:
    """Invoke Google Gemini model with retry logic."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    client = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, max_retries=3)
    prompt = ChatPromptTemplate.from_messages([(m["role"], m["content"]) for m in messages])
    response = client.invoke(prompt)
    return {"content": response.content, "metadata": response.response_metadata}