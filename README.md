# Customer Support Ticket Triage — MLOps Lifecycle Demo

A production-shaped MLOps pipeline for routing customer support tickets to the right queue and priority. The classifier itself is deliberately simple — **the subject of this project is the lifecycle around it**: data validation, experiment tracking, CI/CD gating, deployment, monitoring, and a human-gated retraining loop, all built with free-tier tooling on a CPU-only local machine.

This is the third project in a three-part portfolio: (1) a RAG-based Terraform Q&A system, (2) retail demand forecasting with a self-verifying GenAI narrative layer, and (3) this project, which demonstrates MLOps engineering practice specifically rather than modeling sophistication.

## Why the model is not the point

Two models are trained and compared here — a TF-IDF + Logistic Regression baseline and a fine-tuned DistilBERT — but neither is the deliverable. What's being demonstrated is:

- a real promotion gate that only lets a new model become "Production" if it beats the current one on a fixed metric
- experiment tracking across multiple runs and both model families in one registry
- a model registry that actually reflects what gets served, not just what got logged
- monitoring and a human-in-the-loop retraining trigger, not full automation

LLM usage is intentionally minimal and scoped to three well-defined supporting roles (below) — never the core prediction mechanism.

## Tech stack

| Concern | Tool |
|---|---|
| Database | Neon (Postgres, free tier) |
| Experiment tracking / model registry | MLflow, hosted via DagsHub |
| Dataset/artifact versioning | DVC with a DagsHub remote |
| Orchestration | Dagster (jobs, sensors, schedules) |
| GPU training | Kaggle notebooks |
| LLM API (fallback, explanation, PII) | Groq / NVIDIA NIM (OpenAI-compatible) |
| Observability | Logfire |
| Drift detection | evidently |
| Data validation | pandera |
| Serving | FastAPI + ONNX Runtime |
| Dashboard | Streamlit |
| CI | GitHub Actions |
| Containerization | Docker |

All local work (orchestration, inference, dashboard) runs CPU-only on 16GB RAM. Fine-tuning runs on Kaggle's free GPU tier; everything else, including ONNX export and conversion, happens locally.

## LLM usage — exactly three places, never as the classifier

1. **Low-confidence fallback.** When the served model's confidence drops below a threshold, an LLM call produces a classification and a draft reply. This is the only place an LLM substitutes for the model, and only as a fallback.
2. **Prediction explanation.** A one-sentence natural-language justification for the routing decision, shown next to every prediction regardless of which path produced it.
3. **PII redaction on ingestion**, regex-first: a mandatory, always-on regex layer catches emails, phone numbers, card numbers, and similar patterns at zero API cost. An LLM pass only fires on a batched, conditionally-triggered subset of tickets where the regex layer found nothing but a heuristic (length, trigger phrases) suggests residual risk, batched 20-50 tickets per call with duplicate/template bodies deduplicated before sending. This keeps LLM-based redaction bounded and auditable rather than a per-ticket API call.

## The six MLOps phases, as implemented

**1. Data management** — Tickets ingested into Neon with a batch ID, validated against a pandera schema (fails loudly on drift or malformed rows), redacted via the hybrid regex/LLM pipeline above, and versioned with DVC against a DagsHub remote.

**2. Experimentation and training** — A Dagster job chains ingest to preprocess to train to evaluate to register. The baseline trains locally (CPU-trivial); the DistilBERT model trains on a Kaggle GPU notebook, triggered and polled from Dagster via the Kaggle API. Every run, baseline and transformer, is logged to MLflow with params, metrics, and artifacts.

**3. CI/CD for ML** — GitHub Actions runs lint and unit tests on every push. The real gate is in `evaluate_and_register_op`: a candidate model is only registered and promoted to "Production" if its `queue_f1_macro` beats the current Production version on the same held-out metric. Passing tests is necessary but not sufficient — the metric gate is what actually controls promotion.

**4. Deployment / serving** — FastAPI serves whatever model is currently tagged "Production" in the MLflow registry, resolved and downloaded once at process startup (cached locally, so restarts don't re-download). Each registered version carries a `model_type` tag (`sklearn_onnx` or `transformer_onnx`) that tells the serving layer which code path to use: TF-IDF-shaped input for the baseline, tokenizer plus ONNX transformer inference for DistilBERT. A Streamlit dashboard consumes the API: submit a ticket, see the prediction, confidence, explanation, and, if triggered, the LLM fallback draft reply. Documented as containerized via Docker; not deployed to a live free-tier host, per project scope.

**5. Monitoring and observability** — Logfire instruments the FastAPI service for request tracing, latency, and error rates. evidently compares live prediction/input distributions against the training distribution, run on an hourly schedule or on demand from a dashboard panel that reports drift share and lets you download the full report.

**6. Feedback loop and retraining, human-gated by design** — Agents can mark a prediction wrong and submit the correct label via the dashboard; corrections are written to Neon. When corrections cross a threshold, or drift crosses its own threshold, the system raises a retrain alert (visible on the dashboard, acknowledgeable by a human). Retraining is never auto-triggered; a person reviews the alert and manually kicks off the Kaggle notebook. The resulting model still has to pass the same promotion gate as everything else before it can become Production. This is a deliberate reflection of real-world practice, not a shortcut taken because of infrastructure limits, though those are real too.

## Results

| Run | Model | queue_f1_macro | priority_f1_macro |
|---|---|---|---|
| `baseline_maxfeat5000_C1.0` | TF-IDF + Logistic Regression | 0.306 | 0.470 |
| `distilbert_queue_priority` | DistilBERT (fine-tuned) | 0.353 | 0.520 |

DistilBERT beat the baseline on the gating metric and was promoted to Production automatically by the promotion gate.

## Project structure

```
support-ticket-mlops/
├── requirements.txt
├── requirements-kaggle.txt
├── .env.example
├── Dockerfile
├── README.md
├── src/
│   ├── config.py
│   ├── data/               # ingest, PII redaction, pandera schemas
│   ├── models/              # baseline, ONNX export, sklearn + transformer inference,
│   │                          model registry resolution, serving dispatcher
│   ├── llm/                 # fallback classifier, explanation, LLM-PII layer
│   ├── monitoring/           # evidently drift checks, logfire tracing setup
│   └── db/                   # SQLAlchemy models + Neon session
├── dagster_project/
│   ├── definitions.py
│   ├── assets.py             # ops and jobs for every phase
│   ├── kaggle_bridge.py       # push/poll/pull for the Kaggle training step
│   └── sensors.py             # kaggle completion sensor, retrain alert schedule
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/                # predict.py, feedback.py
├── dashboard/
│   └── app.py
├── kaggle_notebooks/
│   ├── train_distilbert.ipynb
│   └── kernel-metadata.json
└── tests/
```

## Setup

### Prerequisites

- Python 3.11
- A Neon Postgres database
- A DagsHub account (hosts MLflow tracking and the DVC remote)
- A Kaggle account with an API token
- A Groq and/or NVIDIA NIM API key
- (Windows only) see the note below before running anything

### Windows: set UTF-8 mode first

Several dependencies (Kaggle's API client, MLflow) write non-ASCII log output. Windows defaults Python's file I/O to `cp1252`, which crashes on that output. Set these once, permanently, before doing anything else:

Settings, Environment Variables, add for your account:
```
PYTHONUTF8=1
PYTHONLEGACYWINDOWSSTDIO=1
```
Open a fresh terminal afterward so they take effect.

### Install

```bash
python -m venv venv
venv\Scripts\activate        # or source venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```
Fill in every value in `.env.example`: Neon connection string, DagsHub-hosted MLflow tracking URI and credentials, Kaggle username/key, Groq/NVIDIA NIM keys, and Logfire token.

### Initialize the database

```python
from src.db.session import init_db
init_db()
```

### Set up DVC

```bash
dvc remote add -d origin https://dagshub.com/<your-username>/<your-repo>.dvc
dvc pull
```

## Running it

**Orchestration:**
```bash
dagster dev -f dagster_project/definitions.py
```
Open `http://localhost:3000`. Run `ingestion_job` first, then `baseline_training_job` to seed a Production baseline. For the Kaggle-trained model: edit `kaggle_notebooks/kernel-metadata.json` with your Kaggle username, launch `kaggle_training_job` from the Launchpad, add the four required secrets (`NEON_DATABASE_URL`, `MLFLOW_TRACKING_URI`, `DAGSHUB_USER`, `DAGSHUB_TOKEN`) to the notebook on kaggle.com, then start the `kaggle_completion_sensor`. It polls every 5 minutes and automatically fires `kaggle_promotion_job` once training finishes.

**API:**
```bash
uvicorn api.main:app --reload --port 8000
```
The first startup after a new model is promoted downloads it from the registry once (can take several minutes depending on model size and connection); every subsequent restart reuses the local cache.

**Dashboard:**
```bash
streamlit run dashboard/app.py
```

## API

- `POST /predict` — `{"body": "<ticket text>"}` returns queue, priority, confidence, explanation, and a fallback draft reply if confidence was below threshold.
- `POST /feedback` — submit a correction for a given `prediction_id`.
- `GET /health` — liveness check.

## Testing

```bash
pytest
ruff check .
```

## Docker

```bash
docker build -t ticket-triage-api .
docker run -p 8000:8000 --env-file .env ticket-triage-api
```
The image serves the API only; the dashboard and orchestration layer run outside the container in this setup.

## Known limitations, by design

- **Deployment is documented, not live.** The service is containerized and runnable, but no live free-tier hosting was stood up: an explicit, stated scope decision to prioritize finishing the full lifecycle over deployment polish.
- **Retraining is never automatic.** Alerts are raised on correction volume or drift; a human always decides whether and when to retrain.
- **Dataset choice was not optimized.** A suitable labeled Kaggle dataset was picked and used as-is; dataset curation was explicitly out of scope for this project.
- **Both models are intentionally simple.** Neither the TF-IDF baseline nor DistilBERT was tuned for maximum accuracy; the pipeline maturity around them is what this project demonstrates.
- **MLflow's stage-based registry API (`Production`/`Staging` via `transition_model_version_stage`) is used throughout** and is deprecated upstream in favor of aliases; kept as-is here since it's still functional and migrating is a mechanical follow-up, not a design change.

## Acknowledgments

Trained on a Kaggle multilingual customer support ticket dataset (queue/department and priority labels). Dataset selection was explicitly de-scoped per project constraints; any comparable labeled dataset would serve equally well here, since the dataset is not what this project evaluates.