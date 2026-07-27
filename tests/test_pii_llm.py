from unittest.mock import patch

from src.llm.pii_llm import batch_check_residual_pii


def test_batch_check_residual_pii_empty_input_returns_empty():
    assert batch_check_residual_pii([]) == []


def test_batch_check_residual_pii_parses_json_array():
    texts = ["hi, my name is Alex", "please call me back"]
    raw = '["hi, my name is [PII_REDACTED]", "please call me back"]'
    with patch("src.llm.pii_llm.chat_completion", return_value=raw):
        result = batch_check_residual_pii(texts)

    assert result[0] == "hi, my name is [PII_REDACTED]"
    assert result[1] == "please call me back"


def test_batch_check_residual_pii_parses_json_object_wrapper():
    texts = ["ticket one", "ticket two"]
    raw = '{"redacted": ["ticket one", "ticket two"]}'
    with patch("src.llm.pii_llm.chat_completion", return_value=raw):
        result = batch_check_residual_pii(texts)

    assert result == ["ticket one", "ticket two"]


def test_batch_check_residual_pii_falls_back_on_malformed_json():
    texts = ["ticket one", "ticket two"]
    with patch("src.llm.pii_llm.chat_completion", return_value="not json"):
        result = batch_check_residual_pii(texts)

    assert result == texts


def test_batch_check_residual_pii_falls_back_on_length_mismatch():
    texts = ["ticket one", "ticket two"]
    raw = '["only one item"]'
    with patch("src.llm.pii_llm.chat_completion", return_value=raw):
        result = batch_check_residual_pii(texts)

    assert result == texts