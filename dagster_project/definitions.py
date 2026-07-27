import dagster as dg

from dagster_project.assets import (
    alert_monitoring_job,
    baseline_training_job,
    ingestion_job,
    kaggle_promotion_job,
    kaggle_training_job,
)
from dagster_project.sensors import kaggle_completion_sensor, retrain_alert_schedule

defs = dg.Definitions(
    jobs=[
        ingestion_job,
        baseline_training_job,
        kaggle_training_job,
        kaggle_promotion_job,
        alert_monitoring_job,
    ],
    sensors=[kaggle_completion_sensor],
    schedules=[retrain_alert_schedule],
)