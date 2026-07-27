from unittest.mock import patch

from src.llm.explain import explain_prediction


def test_explain_prediction_returns_stripped_text():
    with patch("src.llm.explain.chat_completion", return_value="  Routed due to urgent refund language.  \n"):
        result = explain_prediction("I need a refund now", "Billing", "high")

    assert result == "Routed due to urgent refund language."


def test_explain_prediction_passes_ticket_context():
    with patch("src.llm.explain.chat_completion", return_value="ok") as mock_call:
        explain_prediction("my card was charged twice", "Billing", "urgent")

    user_content = mock_call.call_args.args[1]
    assert "my card was charged twice" in user_content
    assert "Billing" in user_content
    assert "urgent" in user_content