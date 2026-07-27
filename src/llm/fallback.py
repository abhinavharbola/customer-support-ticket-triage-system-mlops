import json

from src.llm.client import chat_completion

_SYSTEM_PROMPT = (
    "You are a customer support ticket router. Given a ticket body, return ONLY a "
    "JSON object with keys: queue (string), priority (one of low/medium/high/urgent), "
    "and draft_reply (a short, professional draft response to the customer). Do not "
    "include any text outside the JSON object."
)


def fallback_classify(ticket_body: str) -> dict:
    raw = chat_completion(_SYSTEM_PROMPT, ticket_body, temperature=0.2)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    return {
        "queue": parsed.get("queue", "General"),
        "priority": parsed.get("priority", "medium"),
        "draft_reply": parsed.get("draft_reply", ""),
        "source": "llm_fallback",
    }