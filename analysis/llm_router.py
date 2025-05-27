import os

# Import provider-specific SDKs
import openai
import google.generativeai as genai

# Setup OpenAI API key (assumes env var is set)
openai.api_key = os.environ.get("OPENAI_API_KEY", "")
# Setup Gemini API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Import your DeepInfra chat utility (or use requests if custom)
from llm_settings.deepinfra_models import deepinfra_chat


def openai_chat(model_name, messages):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        raise RuntimeError(f"OpenAI error: {e}")


def gemini_chat(model_name, messages):
    try:
        chat = genai.GenerativeModel(model_name).start_chat()
        # Format only user content
        prompt = "\n".join([msg["content"] for msg in messages if msg["role"] in {"user", "system"}])
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")


def chat(model_name, messages):
    """
    Unified LLM chat interface. Dispatches to the correct provider.
    """
    if model_name.startswith(("gpt-3.5-turbo", "gpt-4")):
        return openai_chat(model_name, messages)

    elif model_name.startswith("gemini"):
        return gemini_chat(model_name, messages)

    elif model_name.startswith(("meta-llama", "mistralai", "Qwen")):
        return deepinfra_chat(model_name, messages)

    else:
        raise ValueError(f"Unsupported model: {model_name}")
