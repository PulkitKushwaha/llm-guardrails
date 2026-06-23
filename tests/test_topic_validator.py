import pytest
from src.validators.topic_validator import (
    KeywordTopicValidator,
    CompositeTopicValidator,
    TopicValidationResult,
    CUSTOMER_SUPPORT_TOPIC
)
 
 
def test_in_scope_query_passes():
    validator = KeywordTopicValidator(CUSTOMER_SUPPORT_TOPIC)
    result = validator.validate("What is your return policy?")
    assert result.result == TopicValidationResult.IN_SCOPE
 
 
def test_blocked_keyword_rejected():
    validator = KeywordTopicValidator(CUSTOMER_SUPPORT_TOPIC)
    result = validator.validate("Give me medical advice for my condition")
    assert result.result == TopicValidationResult.OUT_OF_SCOPE
 
 
def test_off_topic_query_uncertain():
    validator = KeywordTopicValidator(CUSTOMER_SUPPORT_TOPIC)
    result = validator.validate("What is the capital of France?")
    assert result.result == TopicValidationResult.UNCERTAIN
 
 
def test_composite_blocks_uncertain_when_configured():
    validator = CompositeTopicValidator(
        CUSTOMER_SUPPORT_TOPIC,
        block_uncertain=True
    )
    result = validator.validate("What is the capital of France?")
    assert result.result == TopicValidationResult.OUT_OF_SCOPE
 
 
def test_composite_allows_uncertain_by_default():
    validator = CompositeTopicValidator(
        CUSTOMER_SUPPORT_TOPIC,
        block_uncertain=False
    )
    result = validator.validate("What is the capital of France?")
    # Without embedder, uncertain queries pass through by default
    assert result.result in [
        TopicValidationResult.IN_SCOPE,
        TopicValidationResult.UNCERTAIN
    ]
 
 
def test_confidence_is_between_0_and_1():
    validator = KeywordTopicValidator(CUSTOMER_SUPPORT_TOPIC)
    for query in ["return policy", "my order", "capital of France"]:
        result = validator.validate(query)
        assert 0.0 <= result.confidence <= 1.0
 
 
def test_result_has_reason():
    validator = KeywordTopicValidator(CUSTOMER_SUPPORT_TOPIC)
    result = validator.validate("What is your return policy?")
    assert result.reason != ""
