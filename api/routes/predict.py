from fastapi import APIRouter

from api.schemas import PredictionResponse, TicketRequest
from src.config import settings
from src.data.pii_redaction import regex_redact
from src.db.models import Prediction, Ticket
from src.db.session import get_session
from src.llm.explain import explain_prediction
from src.llm.fallback import fallback_classify
from src.models.serving import predict as run_prediction

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
def create_prediction(request: TicketRequest) -> PredictionResponse:
    # live path only runs the always-on regex layer; the batched LLM PII pass
    # is an ingestion-time job, not part of the per-request serving path
    redacted_body, _ = regex_redact(request.body)

    result = run_prediction(redacted_body)

    draft_reply = None
    if result["needs_fallback"]:
        fallback = fallback_classify(redacted_body)
        queue, priority, source = fallback["queue"], fallback["priority"], fallback["source"]
        confidence = result["confidence"]
        draft_reply = fallback["draft_reply"]
    else:
        queue = result["queue"]
        priority = result["priority"]
        source = result["source"]
        confidence = result["confidence"]

    explanation = explain_prediction(redacted_body, queue, priority)

    with get_session() as session:
        ticket = Ticket(
            ingestion_batch_id="live",
            redacted_body=redacted_body,
            pii_redaction_method="regex",
        )
        session.add(ticket)
        session.flush()

        prediction = Prediction(
            ticket_id=ticket.id,
            predicted_queue=queue,
            predicted_priority=priority,
            confidence=confidence,
            source=source,
            explanation=explanation,
        )
        session.add(prediction)
        session.flush()

        ticket_id = ticket.id
        prediction_id = prediction.id

    return PredictionResponse(
        ticket_id=ticket_id,
        prediction_id=prediction_id,
        queue=queue,
        priority=priority,
        confidence=confidence,
        source=source,
        explanation=explanation,
        draft_reply=draft_reply,
    )