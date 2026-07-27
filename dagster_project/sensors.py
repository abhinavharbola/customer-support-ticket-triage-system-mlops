import json

import dagster as dg

from dagster_project.kaggle_bridge import get_kernel_status
from src.config import settings

KAGGLE_KERNEL_SLUG = f"{settings.kaggle_username}/ticket-triage-distilbert-train"


@dg.sensor(job_name="kaggle_promotion_job", minimum_interval_seconds=300)
def kaggle_completion_sensor(context: dg.SensorEvaluationContext):
    status = get_kernel_status(KAGGLE_KERNEL_SLUG)
    cursor_state = json.loads(context.cursor) if context.cursor else {"last_status": None, "cycle": 0}
    last_status = cursor_state["last_status"]

    if status == last_status:
        return dg.SkipReason(f"kernel status unchanged ({status})")

    if status == "complete" and last_status != "complete":
        cursor_state["cycle"] += 1

    cursor_state["last_status"] = status
    context.update_cursor(json.dumps(cursor_state))

    if status != "complete":
        return dg.SkipReason(f"kernel status: {status}")

    return dg.RunRequest(
        run_key=f"kaggle-complete-{KAGGLE_KERNEL_SLUG}-{cursor_state['cycle']}",
        run_config={
            "ops": {
                "pull_kaggle_artifact_op": {
                    "config": {
                        "kernel_slug": KAGGLE_KERNEL_SLUG,
                        "output_dir": "kaggle_output/latest",
                    }
                }
            }
        },
    )


@dg.schedule(job_name="alert_monitoring_job", cron_schedule="0 * * * *")
def retrain_alert_schedule(context: dg.ScheduleEvaluationContext) -> dg.RunRequest:
    return dg.RunRequest(run_key=f"alert-check-{context.scheduled_execution_time.isoformat()}")