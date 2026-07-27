import os

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

from src.config import settings
from src.db.models import Prediction, Ticket
from src.db.session import get_session

MIN_ROWS = 30


def build_drift_datasets(reference_limit: int = 2000, current_limit: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    with get_session() as session:
        reference_rows = (
            session.query(Ticket)
            .filter(Ticket.true_queue.isnot(None), Ticket.true_priority.isnot(None))
            .order_by(Ticket.id)
            .limit(reference_limit)
            .all()
        )
        reference_df = pd.DataFrame(
            [
                {
                    "redacted_body": r.redacted_body,
                    "predicted_queue": r.true_queue,
                    "predicted_priority": r.true_priority,
                }
                for r in reference_rows
            ]
        )

        current_rows = (
            session.query(Prediction)
            .join(Ticket, Prediction.ticket_id == Ticket.id)
            .order_by(Prediction.created_at.desc())
            .limit(current_limit)
            .all()
        )
        current_df = pd.DataFrame(
            [
                {
                    "redacted_body": p.ticket.redacted_body,
                    "predicted_queue": p.predicted_queue,
                    "predicted_priority": p.predicted_priority,
                }
                for p in current_rows
            ]
        )

    return reference_df, current_df


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_html: str) -> dict:
    column_mapping = ColumnMapping(
        text_features=["redacted_body"],
        categorical_features=["predicted_queue", "predicted_priority"],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    report.save_html(output_html)

    result = report.as_dict()
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

    return {
        "drift_share": drift_share,
        "drift_detected": drift_share >= settings.drift_alert_threshold,
        "report_path": output_html,
    }


def check_drift(output_html: str = "monitoring_reports/drift_report.html") -> dict:
    reference_df, current_df = build_drift_datasets()

    if len(reference_df) < MIN_ROWS or len(current_df) < MIN_ROWS:
        return {
            "insufficient_data": True,
            "reference_rows": len(reference_df),
            "current_rows": len(current_df),
            "min_rows_required": MIN_ROWS,
        }

    result = run_drift_report(reference_df, current_df, output_html)
    result["insufficient_data"] = False
    result["reference_rows"] = len(reference_df)
    result["current_rows"] = len(current_df)
    return result