
from typing import Any, List, Dict
from .openai_model import openai_chat
from .gemini_model import gemini_chat
import os


def get_llm_client(model: str, api_key: str = None, temperature: float = 0.0) -> Any:
    """Return the appropriate LLM client and chat function based on model name."""
    if model.startswith("gpt-"):
        from langchain_openai import ChatOpenAI
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    f"API key for {model} is missing. Provide it in model_config or set OPENAI_API_KEY environment variable."
                )
        client = ChatOpenAI(model_name=model, api_key=api_key, temperature=temperature, max_retries=3)
        return client, openai_chat
    elif model.startswith("gemini-"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    f"API key for {model} is missing. Provide it in model_config or set GOOGLE_API_KEY environment variable."
                )
        client = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, max_retries=3)
        return client, gemini_chat
    else:
        raise ValueError(f"Unsupported model: {model}")