import logging

from groq import Groq
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

_nim_client = OpenAI(
    api_key=settings.nvidia_nim_api_key,
    base_url=settings.nvidia_nim_base_url,
    timeout=10.0,
    max_retries=1,
)
_groq_client = Groq(api_key=settings.groq_api_key, timeout=20.0, max_retries=1)


def _call(client, model: str, messages: list[dict], temperature: float, max_tokens: int | None) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("model returned empty content (likely hit max_tokens before emitting a response)")
    return content


def chat_completion(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        return _call(_nim_client, settings.llm_model_name, messages, temperature, max_tokens)
    except Exception as exc:
        logger.warning(f"NVIDIA NIM call failed ({exc}); falling back to Groq")

    try:
        return _call(_groq_client, settings.llm_model_name, messages, temperature, max_tokens)
    except Exception as exc:
        logger.error(f"Groq fallback also failed ({exc})")
        raise