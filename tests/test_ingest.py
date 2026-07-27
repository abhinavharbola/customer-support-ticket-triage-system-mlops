import pandas as pd
import pytest

from src.data import ingest as ingest_module


class _FakeSessionContext:
    def __init__(self):
        self.added = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add(self, obj):
        self.added.append(obj)


@pytest.fixture
def fake_session(monkeypatch):
    session = _FakeSessionContext()
    monkeypatch.setattr(ingest_module, "get_session", lambda: session)
    return session


def _write_csv(tmp_path, rows):
    path = tmp_path / "tickets.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_ingest_writes_one_ticket_per_row(tmp_path, fake_session, monkeypatch):
    monkeypatch.setattr(ingest_module, "batch_check_residual_pii", lambda texts: texts)
    csv_path = _write_csv(
        tmp_path,
        [
            {"body": "My package is late", "queue": "Shipping", "priority": "medium", "language": "en"},
            {"body": "I need a refund", "queue": "Billing", "priority": "high", "language": "en"},
        ],
    )

    result = ingest_module.ingest(csv_path)

    assert result["rows_written"] == 2
    assert len(fake_session.added) == 2


def test_ingest_dedupes_identical_flagged_tickets_before_llm_call(tmp_path, fake_session, monkeypatch):
    calls = []

    def fake_batch_check(texts):
        calls.append(list(texts))
        return [f"[LLM_CLEANED] {t}" for t in texts]

    monkeypatch.setattr(ingest_module, "batch_check_residual_pii", fake_batch_check)

    duplicate_body = "Hi, my name is Alex and I need help with my account settings today"
    csv_path = _write_csv(
        tmp_path,
        [
            {"body": duplicate_body, "queue": "Account", "priority": "low", "language": "en"},
            {"body": duplicate_body, "queue": "Account", "priority": "low", "language": "en"},
        ],
    )

    result = ingest_module.ingest(csv_path)

    assert result["rows_flagged_for_llm"] == 1
    assert len(calls) == 1
    assert len(calls[0]) == 1

    tickets = fake_session.added
    assert tickets[0].redacted_body == tickets[1].redacted_body
    assert tickets[0].pii_redaction_method == "regex+llm"


def test_ingest_reports_zero_llm_calls_when_nothing_flagged(tmp_path, fake_session, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ingest_module, "batch_check_residual_pii", lambda texts: (calls.append(texts), texts)[1]
    )

    csv_path = _write_csv(tmp_path, [{"body": "Where is my order", "queue": "Shipping", "priority": "low"}])

    result = ingest_module.ingest(csv_path)

    assert result["llm_batch_calls"] == 0
    assert result["rows_flagged_for_llm"] == 0
    assert calls == []