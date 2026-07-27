import logging

import logfire

from src.config import settings

logger = logging.getLogger(__name__)


def configure_tracing(app=None) -> None:
    try:
        logfire.configure(token=settings.logfire_token)
        if app is not None:
            logfire.instrument_fastapi(app)
    except Exception:
        logger.exception("Logfire tracing setup failed; continuing without it")