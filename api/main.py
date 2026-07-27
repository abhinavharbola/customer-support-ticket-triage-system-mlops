from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import feedback, predict
from src.db.session import init_db
from src.models.model_registry import load_production_model
from src.monitoring.tracing import configure_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        load_production_model()
    except Exception:
        pass
    yield


app = FastAPI(title="Ticket Triage API", lifespan=lifespan)
configure_tracing(app)

app.include_router(predict.router)
app.include_router(feedback.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}