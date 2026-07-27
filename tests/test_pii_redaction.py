from src.data.pii_redaction import hash_text, needs_llm_pass, redact_ticket, regex_redact


def test_regex_redact_email():
    text = "Contact me at john.doe@example.com please"
    redacted, hits = regex_redact(text)
    assert "[EMAIL_REDACTED]" in redacted
    assert "john.doe@example.com" not in redacted
    assert hits == 1


def test_regex_redact_phone():
    text = "Call me at 555-123-4567 today"
    redacted, hits = regex_redact(text)
    assert "[PHONE_REDACTED]" in redacted
    assert hits >= 1


def test_regex_redact_multiple_patterns():
    text = "Email test@example.com or call 555-123-4567"
    redacted, hits = regex_redact(text)
    assert hits == 2
    assert "[EMAIL_REDACTED]" in redacted
    assert "[PHONE_REDACTED]" in redacted


def test_regex_redact_no_pii():
    text = "My order hasn't arrived yet, can you help?"
    redacted, hits = regex_redact(text)
    assert redacted == text
    assert hits == 0


def test_needs_llm_pass_skipped_when_regex_hit():
    assert needs_llm_pass("some text", regex_hits=1, length_threshold=400) is False


def test_needs_llm_pass_triggered_by_length():
    long_text = "a" * 500
    assert needs_llm_pass(long_text, regex_hits=0, length_threshold=400) is True


def test_needs_llm_pass_triggered_by_trigger_phrase():
    text = "Hi, my name is Alex and I need help with my account"
    assert needs_llm_pass(text, regex_hits=0, length_threshold=400) is True


def test_needs_llm_pass_false_for_short_clean_text():
    text = "Please refund my order"
    assert needs_llm_pass(text, regex_hits=0, length_threshold=400) is False


def test_hash_text_is_case_and_whitespace_insensitive():
    assert hash_text("  Hello World  ") == hash_text("hello world")


def test_hash_text_differs_for_different_text():
    assert hash_text("hello") != hash_text("goodbye")


def test_redact_ticket_returns_all_fields():
    result = redact_ticket("Email me at test@example.com", length_threshold=400)
    assert result.regex_hits == 1
    assert result.needs_llm_check is False
    assert "[EMAIL_REDACTED]" in result.redacted_text
    assert isinstance(result.text_hash, str)