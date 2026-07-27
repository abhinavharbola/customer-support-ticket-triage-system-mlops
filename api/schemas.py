from pydantic import BaseModel


class TicketRequest(BaseModel):
    body: str


class PredictionResponse(BaseModel):
    ticket_id: int
    prediction_id: int
    queue: str
    priority: str
    confidence: float
    source: str
    explanation: str
    draft_reply: str | None = None


class FeedbackRequest(BaseModel):
    prediction_id: int
    corrected_queue: str
    corrected_priority: str
    corrected_by: str | None = None


class FeedbackResponse(BaseModel):
    correction_id: int
    prediction_id: int