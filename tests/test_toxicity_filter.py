import pytest
from src.output.toxicity_filter import (
    KeywordToxicityFilter,
    CompositeToxicityFilter,
    ToxicitySeverity,
    ToxicityCategory
)
 
 
def test_clean_output_not_toxic():
    f = KeywordToxicityFilter()
    result = f.filter("Our return policy allows returns within 30 days.")
    assert not result.is_toxic
    assert result.overall_severity == ToxicitySeverity.NONE
 
 
def test_threat_detected():
    f = KeywordToxicityFilter()
    result = f.filter("You deserve to suffer for this mistake.")
    assert result.is_toxic
    assert result.overall_severity in [ToxicitySeverity.HIGH, ToxicitySeverity.CRITICAL]
 
 
def test_harassment_detected():
    f = KeywordToxicityFilter()
    result = f.filter("You are so stupid and worthless.")
    assert result.is_toxic
 
 
def test_custom_policy_pattern():
    f = KeywordToxicityFilter(
        custom_patterns={"competitor_mention": [r"\bcompetitorX\b"]}
    )
    result = f.filter("You should use competitorX instead.")
    assert result.is_toxic
    assert any(m.category == ToxicityCategory.POLICY_VIOLATION for m in result.matches)
 
 
def test_severity_is_valid_enum():
    f = KeywordToxicityFilter()
    result = f.filter("Normal helpful response here.")
    assert isinstance(result.overall_severity, ToxicitySeverity)
 
 
def test_recommendation_populated():
    f = KeywordToxicityFilter()
    result = f.filter("Normal response.")
    assert result.recommendation != ""
 
 
def test_composite_filter_no_ml():
    f = CompositeToxicityFilter(use_ml=False)
    result = f.filter("Helpful customer support response.")
    assert not result.is_toxic
 
 
def test_should_block_high_severity():
    f = CompositeToxicityFilter(block_severity=ToxicitySeverity.HIGH)
    from src.output.toxicity_filter import ToxicityResult, ToxicityMatch
    mock_result = ToxicityResult(
        text="test",
        is_toxic=True,
        overall_severity=ToxicitySeverity.CRITICAL,
        matches=[]
    )
    assert f.should_block(mock_result) is True
 
 
def test_should_not_block_low_severity():
    f = CompositeToxicityFilter(block_severity=ToxicitySeverity.HIGH)
    from src.output.toxicity_filter import ToxicityResult
    mock_result = ToxicityResult(
        text="test",
        is_toxic=True,
        overall_severity=ToxicitySeverity.LOW,
        matches=[]
    )
    assert f.should_block(mock_result) is False
