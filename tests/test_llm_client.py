from unittest.mock import MagicMock

from src.llm import client as client_module


def _mock_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_chat_completion_uses_nim_when_it_succeeds(monkeypatch):
    nim_client = MagicMock()
    nim_client.chat.completions.create.return_value = _mock_response("from nim")
    groq_client = MagicMock()

    monkeypatch.setattr(client_module, "_nim_client", nim_client)
    monkeypatch.setattr(client_module, "_groq_client", groq_client)

    result = client_module.chat_completion("system", "user message")

    assert result == "from nim"
    nim_client.chat.completions.create.assert_called_once()
    groq_client.chat.completions.create.assert_not_called()


def test_chat_completion_falls_back_to_groq_on_nim_failure(monkeypatch):
    nim_client = MagicMock()
    nim_client.chat.completions.create.side_effect = RuntimeError("NIM unavailable")
    groq_client = MagicMock()
    groq_client.chat.completions.create.return_value = _mock_response("from groq")

    monkeypatch.setattr(client_module, "_nim_client", nim_client)
    monkeypatch.setattr(client_module, "_groq_client", groq_client)

    result = client_module.chat_completion("system", "user message")

    assert result == "from groq"
    groq_client.chat.completions.create.assert_called_once()


def test_chat_completion_uses_same_model_for_both_providers(monkeypatch):
    nim_client = MagicMock()
    nim_client.chat.completions.create.side_effect = RuntimeError("down")
    groq_client = MagicMock()
    groq_client.chat.completions.create.return_value = _mock_response("ok")

    monkeypatch.setattr(client_module, "_nim_client", nim_client)
    monkeypatch.setattr(client_module, "_groq_client", groq_client)

    client_module.chat_completion("system", "user message")

    nim_model = nim_client.chat.completions.create.call_args.kwargs["model"]
    groq_model = groq_client.chat.completions.create.call_args.kwargs["model"]
    assert nim_model == groq_model == client_module.settings.llm_model_name


def test_chat_completion_passes_temperature_and_max_tokens(monkeypatch):
    nim_client = MagicMock()
    nim_client.chat.completions.create.return_value = _mock_response("ok")
    monkeypatch.setattr(client_module, "_nim_client", nim_client)

    client_module.chat_completion("system", "user message", temperature=0.7, max_tokens=42)

    kwargs = nim_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 42