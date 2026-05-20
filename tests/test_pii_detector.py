import pytest
from src.input.pii_detector import (
    RegexPIIDetector,
    CompositePIIDetector,
    PIIType
)
 
 
def test_ssn_detection():
    """Should detect formatted SSN"""
    detector = RegexPIIDetector()
    result = detector.detect("My SSN is 123-45-6789 and I need help.")
    assert result.contains_pii
    assert PIIType.SSN in result.pii_types_found
 
 
def test_email_detection():
    """Should detect email addresses"""
    detector = RegexPIIDetector()
    result = detector.detect("Contact me at john.doe@example.com for details.")
    assert result.contains_pii
    assert PIIType.EMAIL in result.pii_types_found
 
 
def test_phone_detection():
    """Should detect phone numbers"""
    detector = RegexPIIDetector()
    result = detector.detect("Call me at 555-123-4567 anytime.")
    assert result.contains_pii
    assert PIIType.PHONE in result.pii_types_found
 
 
def test_credit_card_detection():
    """Should detect credit card numbers"""
    detector = RegexPIIDetector()
    result = detector.detect("My card number is 4532-1234-5678-9012.")
    assert result.contains_pii
    assert PIIType.CREDIT_CARD in result.pii_types_found
 
 
def test_no_pii_clean_text():
    """Clean text should not trigger detection"""
    detector = RegexPIIDetector()
    result = detector.detect("What is the return policy for online orders?")
    assert not result.contains_pii
    assert result.redacted_text == "What is the return policy for online orders?"
 
 
def test_redacted_text_replaces_pii():
    """Redacted text should replace PII with placeholder"""
    detector = RegexPIIDetector()
    result = detector.detect("Email me at test@example.com please.")
    assert result.contains_pii
    assert "test@example.com" not in result.redacted_text
    assert "REDACTED" in result.redacted_text
 
 
def test_multiple_pii_types():
    """Should detect multiple PII types in one text"""
    detector = RegexPIIDetector()
    result = detector.detect(
        "Name: John. SSN: 123-45-6789. Email: john@test.com. Phone: 555-123-4567."
    )
    assert result.contains_pii
    assert len(result.pii_types_found) >= 3
 
 
def test_custom_patterns():
    """Custom patterns should be detected"""
    detector = RegexPIIDetector(custom_patterns={"employee_id": r"EMP-\d{6}"})
    result = detector.detect("Employee EMP-123456 filed a request.")
    assert result.contains_pii
 
 
def test_composite_detector_without_presidio():
    """Composite detector should work without Presidio"""
    detector = CompositePIIDetector(use_presidio=False)
    result = detector.detect("SSN: 987-65-4321")
    assert result.contains_pii
