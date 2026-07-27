import os

import dagster as dg
import mlflow
import pandas as pd

from dagster_project.kaggle_bridge import pull_kernel_output, push_training_notebook
from src.config import settings
from src.data.ingest import ingest
from src.db.models import Correction, RetrainAlert, Ticket
from src.db.session import get_session
from src.models.baseline import train_baseline
from src.models.onnx_export import export_sklearn_to_onnx, export_transformer_to_onnx


@dg.op(config_schema={"csv_path": str})
def ingest_tickets_op(context: dg.OpExecutionContext) -> dict:
    csv_path = context.op_config["csv_path"]
    result = ingest(csv_path)
    context.log.info(
        f"ingested batch {result['batch_id']}: {result['rows_written']} rows, "
        f"{result['llm_batch_calls']} LLM redaction calls"
    )
    return result


@dg.op
def load_training_data_op() -> pd.DataFrame:
    with get_session() as session:
        rows = session.query(Ticket).filter(Ticket.true_queue.isnot(None)).all()
        return pd.DataFrame(
            [
                {
                    "redacted_body": r.redacted_body,
                    "true_queue": r.true_queue,
                    "true_priority": r.true_priority,
                }
                for r in rows
            ]
        )


@dg.op
def train_baseline_op(context: dg.OpExecutionContext, df: pd.DataFrame) -> dict:
    result = train_baseline(df)
    context.log.info(
        f"baseline run {result['run_id']}: queue_f1={result['queue_f1']:.3f} "
        f"priority_f1={result['priority_f1']:.3f}"
    )
    return {
        "run_id": result["run_id"],
        "queue_f1": result["queue_f1"],
        "pipeline": result["pipeline"],
        "model_uri": result["model_uri"],
    }


@dg.op(config_schema={"output_path": dg.Field(str, default_value=settings.onnx_model_path)})
def export_baseline_onnx_op(context: dg.OpExecutionContext, train_result: dict) -> dict:
    import onnx

    output_path = context.op_config["output_path"]
    export_sklearn_to_onnx(train_result["pipeline"], output_path)
    context.log.info(f"exported baseline ONNX model to {output_path}")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    run_id = train_result["run_id"]
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    mlflow.set_experiment(experiment_id=run.info.experiment_id)

    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.onnx.log_model(onnx.load(output_path), name="model_onnx")

    return {
        "run_id": run_id,
        "queue_f1": train_result["queue_f1"],
        "model_uri": model_info.model_uri,
        "model_type": "sklearn_onnx",
    }


def _current_production_metric(client: mlflow.tracking.MlflowClient, model_name: str) -> float:
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        return 0.0

    if not versions:
        return 0.0
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get("queue_f1_macro", 0.0)


@dg.op
def evaluate_and_register_op(context: dg.OpExecutionContext, train_result: dict) -> bool:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    model_name = "ticket-triage-classifier"

    production_metric = _current_production_metric(client, model_name)
    candidate_metric = train_result["queue_f1"]

    if candidate_metric <= production_metric:
        context.log.info(
            f"candidate ({candidate_metric:.3f}) did not beat production "
            f"({production_metric:.3f}); not promoting"
        )
        return False

    registered = mlflow.register_model(train_result["model_uri"], model_name)
    client.set_model_version_tag(model_name, registered.version, "model_type", train_result["model_type"])
    client.transition_model_version_stage(
        model_name, registered.version, "Production", archive_existing_versions=True
    )
    context.log.info(
        f"promoted version {registered.version} "
        f"(queue_f1 {candidate_metric:.3f} > {production_metric:.3f})"
    )
    return True


@dg.op(config_schema={"notebook_dir": str})
def push_kaggle_training_op(context: dg.OpExecutionContext) -> str:
    notebook_dir = context.op_config["notebook_dir"]
    kernel_slug = push_training_notebook(notebook_dir)
    context.log.info(f"pushed kaggle kernel {kernel_slug}")
    return kernel_slug


@dg.op(config_schema={"kernel_slug": str, "output_dir": str})
def pull_kaggle_artifact_op(context: dg.OpExecutionContext) -> str:
    kernel_slug = context.op_config["kernel_slug"]
    output_dir = context.op_config["output_dir"]
    pull_kernel_output(kernel_slug, output_dir)
    context.log.info(f"pulled kaggle output for {kernel_slug} into {output_dir}")
    return output_dir


@dg.op
def read_kaggle_run_result_op(context: dg.OpExecutionContext, checkpoint_dir: str) -> dict:
    run_id_path = os.path.join(checkpoint_dir, "mlflow_run_id.txt")
    with open(run_id_path) as f:
        run_id = f.read().strip()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    queue_f1 = run.data.metrics.get("queue_f1_macro", 0.0)

    context.log.info(f"kaggle run {run_id}: queue_f1={queue_f1:.3f}")
    return {"run_id": run_id, "queue_f1": queue_f1, "checkpoint_dir": checkpoint_dir}


@dg.op
def export_kaggle_model_op(context: dg.OpExecutionContext, run_result: dict) -> dict:
    import onnx

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    run_id = run_result["run_id"]
    checkpoint_dir = run_result["checkpoint_dir"]

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    queue_onnx_dir = os.path.join(checkpoint_dir, "onnx_export", "queue")
    priority_onnx_dir = os.path.join(checkpoint_dir, "onnx_export", "priority")

    context.log.info("starting queue model export")
    export_transformer_to_onnx(
        os.path.join(checkpoint_dir, "model", "queue"), queue_onnx_dir, log=context.log.info
    )

    context.log.info("starting priority model export")
    export_transformer_to_onnx(
        os.path.join(checkpoint_dir, "model", "priority"), priority_onnx_dir, log=context.log.info
    )

    context.log.info("uploading ONNX artifacts to mlflow")
    mlflow.set_experiment(experiment_id=run.info.experiment_id)

    queue_model_path = os.path.join(queue_onnx_dir, "model.onnx")
    label_maps_path = os.path.join(checkpoint_dir, "label_maps.json")

    queue_extra_files = [
        os.path.join(queue_onnx_dir, f)
        for f in os.listdir(queue_onnx_dir)
        if f != "model.onnx" and os.path.isfile(os.path.join(queue_onnx_dir, f))
    ]
    if os.path.isfile(label_maps_path):
        queue_extra_files.append(label_maps_path)

    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.onnx.log_model(
            onnx.load(queue_model_path),
            artifact_path="model",
            extra_files=queue_extra_files,
        )
        mlflow.log_artifacts(priority_onnx_dir, artifact_path="priority_model")

    context.log.info(f"exported and logged ONNX artifacts for run {run_id}, model_uri={model_info.model_uri}")

    return {
        "run_id": run_id,
        "queue_f1": run_result["queue_f1"],
        "model_uri": model_info.model_uri,
        "model_type": "transformer_onnx",
    }


@dg.op
def check_retrain_alerts_op(context: dg.OpExecutionContext) -> None:
    with get_session() as session:
        pending = session.query(Correction).count()
        already_alerted = (
            session.query(RetrainAlert)
            .filter(RetrainAlert.reason == "correction_threshold", RetrainAlert.resolved.is_(False))
            .first()
        )

        if pending >= settings.correction_alert_threshold and not already_alerted:
            session.add(
                RetrainAlert(
                    reason="correction_threshold",
                    detail=f"{pending} total corrections logged",
                )
            )
            context.log.warning(
                f"retrain alert raised: {pending} corrections >= threshold "
                f"{settings.correction_alert_threshold}"
            )
        else:
            context.log.info(
                f"corrections={pending}, threshold={settings.correction_alert_threshold}, no new alert"
            )


@dg.op
def check_drift_alert_op(context: dg.OpExecutionContext) -> None:
    from src.monitoring.drift import check_drift

    result = check_drift()

    if result.get("insufficient_data"):
        context.log.info(
            f"skipping drift check: reference_rows={result['reference_rows']}, "
            f"current_rows={result['current_rows']}, need >= {result['min_rows_required']} each"
        )
        return

    context.log.info(f"drift_share={result['drift_share']:.3f}, detected={result['drift_detected']}")

    if not result["drift_detected"]:
        return

    with get_session() as session:
        already_alerted = (
            session.query(RetrainAlert)
            .filter(RetrainAlert.reason == "drift_threshold", RetrainAlert.resolved.is_(False))
            .first()
        )
        if already_alerted:
            context.log.info("drift alert already open, not duplicating")
            return

        session.add(
            RetrainAlert(
                reason="drift_threshold",
                detail=f"drift_share={result['drift_share']:.2f} >= threshold {settings.drift_alert_threshold}",
            )
        )
    context.log.warning(f"drift alert raised: drift_share={result['drift_share']:.2f}")


@dg.job
def ingestion_job():
    ingest_tickets_op()


@dg.job
def baseline_training_job():
    df = load_training_data_op()
    train_result = train_baseline_op(df)
    export_result = export_baseline_onnx_op(train_result)
    evaluate_and_register_op(export_result)


@dg.job
def kaggle_training_job():
    push_kaggle_training_op()


@dg.job
def kaggle_promotion_job():
    checkpoint_dir = pull_kaggle_artifact_op()
    run_result = read_kaggle_run_result_op(checkpoint_dir)
    export_result = export_kaggle_model_op(run_result)
    evaluate_and_register_op(export_result)


@dg.job
def alert_monitoring_job():
    check_retrain_alerts_op()
    check_drift_alert_op()