from unittest.mock import MagicMock, patch

import numpy as np

from src.models import inference


def _make_fake_session(queue_label, priority_label, queue_conf, priority_conf):
    session = MagicMock()
    input_mock = MagicMock()
    input_mock.name = "input"
    session.get_inputs.return_value = [input_mock]

    # verified real structure: outputs[0] is a single (1, 2) label array,
    # outputs[1] is a plain list of 2 per-target probability arrays (not
    # additional ndarray outputs at positions 2/3). Using dtype=None (float64)
    # here, not float32, so confidence == <exact value> assertions below don't
    # trip on float32->float64 rounding.
    labels = np.array([[queue_label, priority_label]], dtype=object)
    queue_probs = np.array([[queue_conf, 0.0, 0.0]])
    priority_probs = np.array([[priority_conf, 0.0, 0.0]])

    session.run.return_value = [labels, [queue_probs, priority_probs]]
    return session


def setup_function():
    inference._session = None


def test_predict_returns_onnx_source_when_confident():
    fake_session = _make_fake_session("Billing", "high", 0.95, 0.9)
    with patch("src.models.inference.load_onnx_session", return_value=fake_session):
        result = inference.predict("fake_model.onnx", "I need a refund")

    assert result["queue"] == "Billing"
    assert result["priority"] == "high"
    assert result["source"] == "onnx"
    assert result["needs_fallback"] is False


def test_predict_flags_fallback_when_confidence_is_low():
    fake_session = _make_fake_session("General", "medium", 0.3, 0.4)
    with patch("src.models.inference.load_onnx_session", return_value=fake_session):
        result = inference.predict("fake_model.onnx", "not sure what this is about")

    assert result["confidence"] == 0.3
    assert result["needs_fallback"] is True


def test_predict_confidence_is_min_of_both_heads():
    fake_session = _make_fake_session("Shipping", "low", 0.9, 0.4)
    with patch("src.models.inference.load_onnx_session", return_value=fake_session):
        result = inference.predict("fake_model.onnx", "where is my package")

    assert result["confidence"] == 0.4


def test_predict_reuses_cached_session():
    fake_session = _make_fake_session("Billing", "high", 0.95, 0.9)
    with patch("src.models.inference.load_onnx_session", return_value=fake_session) as mock_load:
        inference.predict("fake_model.onnx", "first call")
        inference.predict("fake_model.onnx", "second call")

    mock_load.assert_called_once()