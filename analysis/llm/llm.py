# llms.py

import os
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.deepinfra import ChatDeepInfra
from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(model_name: str) -> BaseChatModel:
    """
    Return a LangChain-compatible LLM based on the model name prefix.
    """
    if model_name.startswith("gpt-3.5") or model_name.startswith("gpt-4"):
        return ChatOpenAI(
            model=model_name,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif model_name.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
        )

    elif any(prefix in model_name for prefix in ["meta-llama", "mistralai", "Qwen", "deepinfra"]):
        return ChatDeepInfra(
            model=model_name,
            temperature=0.7,
            deepinfra_api_token=os.getenv("DEEPINFRA_API_KEY"),
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
