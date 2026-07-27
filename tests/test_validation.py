import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.data.validation import validate_raw_tickets


def test_validate_raw_tickets_accepts_valid_dataframe():
    df = pd.DataFrame(
        {
            "body": ["My order is late", "I need a refund"],
            "queue": ["Shipping", "Billing"],
            "priority": ["medium", "high"],
            "language": ["en", "en"],
        }
    )
    validated = validate_raw_tickets(df)
    assert len(validated) == 2


def test_validate_raw_tickets_allows_missing_optional_columns():
    df = pd.DataFrame({"body": ["Where is my package?"]})
    validated = validate_raw_tickets(df)
    assert len(validated) == 1


def test_validate_raw_tickets_rejects_empty_body():
    df = pd.DataFrame({"body": [""]})
    with pytest.raises(SchemaErrors):
        validate_raw_tickets(df)


def test_validate_raw_tickets_rejects_null_body():
    df = pd.DataFrame({"body": [None]})
    with pytest.raises(SchemaErrors):
        validate_raw_tickets(df)


def test_validate_raw_tickets_keeps_extra_columns_with_strict_false():
    df = pd.DataFrame({"body": ["Test ticket"], "ticket_id": [123]})
    validated = validate_raw_tickets(df)
    assert "ticket_id" in validated.columns