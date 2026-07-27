import json

from src.llm.client import chat_completion

_SYSTEM_PROMPT = (
    "You are a PII redaction assistant. You receive a numbered list of ticket bodies "
    "that already had emails, phones, cards, SSNs and IPs removed by regex. Find any "
    "remaining PII (full names, physical addresses, dates of birth, account numbers, "
    "usernames) and return ONLY a JSON array where each element is the same ticket "
    "text with residual PII replaced by [PII_REDACTED]. Return exactly one string per "
    "input ticket, in the same order, with no extra commentary."
)


def batch_check_residual_pii(texts: list[str]) -> list[str]:
    if not texts:
        return []

    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    raw = chat_completion(_SYSTEM_PROMPT, numbered, temperature=0)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = next(iter(parsed.values()))
        if len(parsed) == len(texts):
            return parsed
    except (json.JSONDecodeError, StopIteration):
        pass

    # fall back to regex-only redaction if the LLM output doesn't parse cleanly
    return texts