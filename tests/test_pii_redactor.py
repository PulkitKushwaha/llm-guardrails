import pytest
from src.output.pii_redactor import OutputPIIRedactor, ConditionalRedactor
from src.input.pii_detector import PIIType
 
 
def test_redacts_ssn_in_output():
    redactor = OutputPIIRedactor()
    result = redactor.redact("The customer SSN is 123-45-6789 as per records.")
    assert "123-45-6789" not in result.redacted_text
    assert result.pii_found is True
    assert result.redacted_count >= 1
 
 
def test_clean_output_unchanged():
    redactor = OutputPIIRedactor()
    text = "Your return policy allows 30 days for all purchases."
    result = redactor.redact(text)
    assert result.redacted_text == text
    assert result.pii_found is False
 
 
def test_hard_redaction_replaces_fully():
    redactor = OutputPIIRedactor(redaction_mode="hard")
    result = redactor.redact("Email: test@example.com for details.")
    assert "test@example.com" not in result.redacted_text
    assert "REDACTED" in result.redacted_text
 
 
def test_soft_redaction_shows_partial():
    redactor = OutputPIIRedactor(redaction_mode="soft")
    result = redactor.redact("Call 555-123-4567 for help.")
    assert result.pii_found is True
    assert "555-123-4567" not in result.redacted_text
 
 
def test_multiple_pii_all_redacted():
    redactor = OutputPIIRedactor()
    text = "SSN: 123-45-6789. Card: 4532-1234-5678-9012. Email: a@b.com"
    result = redactor.redact(text)
    assert "123-45-6789" not in result.redacted_text
    assert result.redacted_count >= 3
 
 
def test_audit_log_populated():
    redactor = OutputPIIRedactor()
    result = redactor.redact("SSN: 987-65-4321")
    assert result.audit_log["pii_found"] is True
    assert result.audit_log["redacted_count"] >= 1
 
 
def test_batch_redaction():
    redactor = OutputPIIRedactor()
    texts = [
        "SSN: 123-45-6789",
        "Clean text here",
        "Email: test@test.com"
    ]
    results = redactor.redact_batch(texts)
    assert len(results) == 3
    assert results[0].pii_found is True
    assert results[1].pii_found is False
    assert results[2].pii_found is True
