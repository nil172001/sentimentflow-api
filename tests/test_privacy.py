from app.privacy import apply_privacy_filter


def test_email_redaction():
    text = "Please contact me at daniel@example.com"
    result = apply_privacy_filter(text)
    assert "[EMAIL]" in result


def test_phone_redaction():
    text = "My phone number is 123456789"
    result = apply_privacy_filter(text)
    assert "[NUMBER]" in result or "[PHONE]" in result


def test_location_redaction():
    text = "I live in Barcelona"
    result = apply_privacy_filter(text)
    assert "[LOCATION]" in result


def test_name_redaction():
    text = "My name is Daniel"
    result = apply_privacy_filter(text)
    assert "[NAME]" in result