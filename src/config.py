import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    neon_database_url: str = os.getenv("NEON_DATABASE_URL", "")

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    dagshub_user: str = os.getenv("DAGSHUB_USER", "")
    dagshub_token: str = os.getenv("DAGSHUB_TOKEN", "")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    nvidia_nim_api_key: str = os.getenv("NVIDIA_NIM_API_KEY", "")
    nvidia_nim_base_url: str = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")

    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")

    logfire_token: str = os.getenv("LOGFIRE_TOKEN", "")

    kaggle_username: str = os.getenv("KAGGLE_USERNAME", "")
    kaggle_key: str = os.getenv("KAGGLE_KEY", "")

    fallback_confidence_threshold: float = float(os.getenv("FALLBACK_CONFIDENCE_THRESHOLD", "0.55"))
    onnx_model_path: str = os.getenv("ONNX_MODEL_PATH", "artifacts/baseline_model.onnx")
    pii_llm_length_threshold: int = int(os.getenv("PII_LLM_LENGTH_THRESHOLD", "800"))
    pii_llm_batch_size: int = int(os.getenv("PII_LLM_BATCH_SIZE", "10"))
    correction_alert_threshold: int = int(os.getenv("CORRECTION_ALERT_THRESHOLD", "50"))
    drift_alert_threshold: float = float(os.getenv("DRIFT_ALERT_THRESHOLD", "0.3"))


settings = Settings()

# MLflow's client reads these two env vars for basic auth; DagsHub's hosted
# MLflow needs them set or every mlflow.* call fails auth. The Kaggle notebook
# sets these itself since it runs in a separate process - this covers every
# local entry point (API, dashboard, Dagster) in one place instead of each
# call site remembering to do it.
if settings.dagshub_user and settings.dagshub_token:
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", settings.dagshub_user)
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", settings.dagshub_token)