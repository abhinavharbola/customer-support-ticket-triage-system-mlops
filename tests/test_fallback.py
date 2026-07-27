from unittest.mock import patch

from src.llm.fallback import fallback_classify


def test_fallback_classify_parses_valid_json():
    raw = '{"queue": "Billing", "priority": "high", "draft_reply": "We are looking into it."}'
    with patch("src.llm.fallback.chat_completion", return_value=raw):
        result = fallback_classify("I was charged twice")

    assert result["queue"] == "Billing"
    assert result["priority"] == "high"
    assert result["draft_reply"] == "We are looking into it."
    assert result["source"] == "llm_fallback"


def test_fallback_classify_defaults_on_malformed_json():
    with patch("src.llm.fallback.chat_completion", return_value="not json"):
        result = fallback_classify("some ticket")

    assert result["queue"] == "General"
    assert result["priority"] == "medium"
    assert result["draft_reply"] == ""
    assert result["source"] == "llm_fallback"


def test_fallback_classify_fills_missing_keys():
    with patch("src.llm.fallback.chat_completion", return_value='{"queue": "Shipping"}'):
        result = fallback_classify("where is my order")

    assert result["queue"] == "Shipping"
    assert result["priority"] == "medium"
    assert result["draft_reply"] == ""