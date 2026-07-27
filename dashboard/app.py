import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from src.config import settings
from src.db.models import RetrainAlert
from src.db.session import get_session
from src.monitoring.drift import check_drift

st.set_page_config(page_title="Ticket Triage Dashboard", page_icon=":material/inbox:", layout="centered")

PRIORITIES = ["low", "medium", "high", "urgent"]

PRIORITY_COLORS = {
    "low": ("#1e3a34", "#4ade80"),
    "medium": ("#1e2f4a", "#60a5fa"),
    "high": ("#4a3616", "#fb923c"),
    "urgent": ("#4a1e1e", "#f87171"),
}

SOURCE_LABELS = {
    "onnx": ("Model prediction", "#312e81", "#a5b4fc"),
    "llm_fallback": ("LLM fallback", "#3b1e4a", "#d8b4fe"),
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.app-header {
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    margin-bottom: 1.5rem;
}

.app-title {
    font-size: 1.9rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
}

.app-subtitle {
    font-size: 0.95rem;
    color: rgba(128, 128, 128, 0.9);
}

.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: rgba(128, 128, 128, 0.9);
    margin-bottom: 0.4rem;
}

.card-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.card-title {
    font-size: 1rem;
    font-weight: 600;
}

.stat-block {
    padding-left: 1rem;
}

.stat-block-first {
    padding-left: 0;
}

.stat-divider {
    border-left: 1px solid rgba(128, 128, 128, 0.25);
}

.queue-value {
    font-size: 1.5rem;
    font-weight: 700;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    white-space: nowrap;
}

.confidence-track {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: rgba(128, 128, 128, 0.2);
    overflow: hidden;
    margin-top: 0.4rem;
}

.confidence-fill {
    height: 100%;
    border-radius: 999px;
}

.confidence-value {
    font-size: 1.5rem;
    font-weight: 700;
}

.explanation-block {
    border-left: 3px solid rgba(165, 180, 252, 0.6);
    padding: 0.6rem 1rem;
    background: rgba(165, 180, 252, 0.08);
    border-radius: 0 8px 8px 0;
    font-size: 0.95rem;
    line-height: 1.5;
    margin-top: 1.25rem;
}

.reply-tag {
    font-size: 0.75rem;
    font-weight: 600;
    color: rgba(216, 180, 254, 0.9);
    background: rgba(216, 180, 254, 0.14);
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
}

.empty-state {
    text-align: center;
    padding: 2rem 1rem;
    color: rgba(128, 128, 128, 0.8);
    font-size: 0.9rem;
}

[data-testid="stTextArea"] textarea {
    border-radius: 10px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_badge(text: str, bg: str, fg: str) -> str:
    return f'<span class="badge" style="background:{bg};color:{fg};">{text}</span>'


def render_confidence(confidence: float, threshold: float) -> str:
    pct = max(0.0, min(1.0, confidence)) * 100
    color = "#4ade80" if confidence >= threshold else "#fbbf24"
    return f"""
    <div class="confidence-value">{confidence:.2f}</div>
    <div class="confidence-track">
        <div class="confidence-fill" style="width:{pct:.1f}%;background:{color};"></div>
    </div>
    """


def show_retrain_banner() -> None:
    with get_session() as session:
        alerts = session.query(RetrainAlert).filter(RetrainAlert.resolved.is_(False)).all()
        for alert in alerts:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.warning(f"Retraining recommended ({alert.reason}): {alert.detail}")
            with col2:
                if st.button("Acknowledge", key=f"resolve_{alert.id}"):
                    alert.resolved = True
                    st.rerun()


show_retrain_banner()

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Customer Support Ticket Triage</div>
        <div class="app-subtitle">Submit a ticket to see the routing decision, confidence, and explanation.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="section-label">New ticket</div>', unsafe_allow_html=True)
    with st.form("ticket_form"):
        body = st.text_area("Ticket body", height=150, label_visibility="collapsed")
        _, button_col = st.columns([3, 1])
        with button_col:
            submitted = st.form_submit_button("Classify", type="primary", use_container_width=True)

if submitted and body.strip():
    with st.spinner("Classifying ticket..."):
        response = requests.post(f"{settings.api_base_url}/predict", json={"body": body}, timeout=75)
    if response.ok:
        st.session_state["last_prediction"] = response.json()
    else:
        st.error(f"Prediction request failed: {response.status_code} {response.text}")

prediction = st.session_state.get("last_prediction")

st.write("")

if not prediction:
    with st.container(border=True):
        st.markdown(
            '<div class="empty-state">Submit a ticket above to see the routing decision.</div>',
            unsafe_allow_html=True,
        )
else:
    source_label, source_bg, source_fg = SOURCE_LABELS.get(
        prediction["source"], (prediction["source"], "#333", "#eee")
    )

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="card-header-row">
                <div class="card-title">Prediction</div>
                {render_badge(source_label, source_bg, source_fg)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="stat-block stat-block-first">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Queue</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="queue-value">{prediction["queue"]}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stat-block stat-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Priority</div>', unsafe_allow_html=True)
            bg, fg = PRIORITY_COLORS.get(prediction["priority"], ("#333", "#eee"))
            st.markdown(render_badge(prediction["priority"].capitalize(), bg, fg), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="stat-block stat-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Confidence</div>', unsafe_allow_html=True)
            st.markdown(
                render_confidence(prediction["confidence"], settings.fallback_confidence_threshold),
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="explanation-block">{prediction["explanation"]}</div>',
            unsafe_allow_html=True,
        )

    if prediction.get("draft_reply"):
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="card-header-row">
                    <div class="card-title">Suggested reply</div>
                    <div class="reply-tag">Fallback path</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(prediction["draft_reply"], language=None, wrap_lines=True)

    with st.expander("Correct this prediction"):
        with st.form("correction_form"):
            corrected_queue = st.text_input("Correct queue", value=prediction["queue"])
            default_priority_index = (
                PRIORITIES.index(prediction["priority"]) if prediction["priority"] in PRIORITIES else 1
            )
            corrected_priority = st.selectbox("Correct priority", PRIORITIES, index=default_priority_index)
            corrected_by = st.text_input("Your name (optional)")

            _, button_col = st.columns([3, 1])
            with button_col:
                correction_submitted = st.form_submit_button(
                    "Submit correction", type="primary", use_container_width=True
                )

        if correction_submitted:
            payload = {
                "prediction_id": prediction["prediction_id"],
                "corrected_queue": corrected_queue,
                "corrected_priority": corrected_priority,
                "corrected_by": corrected_by or None,
            }
            correction_response = requests.post(f"{settings.api_base_url}/feedback", json=payload, timeout=30)
            if correction_response.ok:
                st.success("Correction recorded.")
            else:
                st.error(f"Correction failed: {correction_response.status_code} {correction_response.text}")

st.write("")

with st.expander("Drift monitoring"):
    if st.button("Run drift check"):
        with st.spinner("Comparing live predictions against training distribution..."):
            drift_result = check_drift()
        st.session_state["drift_result"] = drift_result

    drift_result = st.session_state.get("drift_result")
    if drift_result:
        if drift_result.get("insufficient_data"):
            st.info(
                f"Not enough data yet: {drift_result['reference_rows']} reference rows, "
                f"{drift_result['current_rows']} current rows (need at least "
                f"{drift_result['min_rows_required']} of each)."
            )
        else:
            st.metric("Drift share", f"{drift_result['drift_share']:.2f}")
            if drift_result["drift_detected"]:
                st.error("Drift detected above threshold.")
            else:
                st.success("No significant drift detected.")
            st.caption(
                f"Compared {drift_result['reference_rows']} reference rows against "
                f"{drift_result['current_rows']} recent predictions."
            )
            with open(drift_result["report_path"], "rb") as f:
                st.download_button("Download full drift report (HTML)", f, file_name="drift_report.html")