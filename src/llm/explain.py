import logging

from src.llm.client import chat_completion

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You write a one-sentence justification for a support ticket routing decision. "
    "Given the ticket body, predicted queue and predicted priority, explain briefly "
    "why that routing makes sense. One sentence, no preamble."
)

_FALLBACK_EXPLANATION = "Explanation unavailable."


def explain_prediction(ticket_body: str, queue: str, priority: str) -> str:
    user_content = f"Ticket: {ticket_body}\nRouted to: {queue} / {priority}"
    try:
        return chat_completion(_SYSTEM_PROMPT, user_content, temperature=0.3, max_tokens=200).strip()
    except Exception as exc:
        logger.warning(f"explain_prediction failed ({exc}); using fallback text")
        return _FALLBACK_EXPLANATION