from fastapi import APIRouter, HTTPException

from api.schemas import FeedbackRequest, FeedbackResponse
from src.db.models import Correction, Prediction
from src.db.session import get_session

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
def submit_correction(request: FeedbackRequest) -> FeedbackResponse:
    with get_session() as session:
        prediction = session.get(Prediction, request.prediction_id)
        if prediction is None:
            raise HTTPException(status_code=404, detail="prediction not found")

        correction = Correction(
            prediction_id=request.prediction_id,
            corrected_queue=request.corrected_queue,
            corrected_priority=request.corrected_priority,
            corrected_by=request.corrected_by,
        )
        session.add(correction)
        session.flush()
        correction_id = correction.id

    return FeedbackResponse(correction_id=correction_id, prediction_id=request.prediction_id)