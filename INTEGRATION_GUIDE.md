# llm-guardrails Integration Guide
 
A practical guide to integrating llm-guardrails into your
LLM-powered application: from minimal setup to full
compliance-grade deployment.
 
---
 
## TL;DR
 
```python
from src.pipeline.guardrails_wrapper import GuardrailsWrapper, GuardrailConfig
from src.validators.topic_validator import CUSTOMER_SUPPORT_TOPIC
 
# Wrap your existing pipeline (zero changes to pipeline code)
wrapped = GuardrailsWrapper(
    pipeline=your_existing_rag_pipeline,
    config=GuardrailConfig(
        topic_config=CUSTOMER_SUPPORT_TOPIC,
        redact_output_pii=True,
        filter_toxicity=True
    )
)
 
result = wrapped.run(user_query)
print(result.response)          # Safe response
print(result.blocked)           # True if query was blocked
print(result.summary())         # Full audit trail
```
 
---
 
## Components
 
### Input Guardrails
 
**PII Detection** (`src/input/pii_detector.py`)
 
Detects sensitive data in user queries before they reach the LLM
or get logged. Three tiers:
 
| Detector | Speed | Accuracy | Use case |
|---|---|---|---|
| `RegexPIIDetector` | Instant | Structured PII only | Production default |
| `PresidioPIIDetector` | ~100ms | Natural language PII | High-compliance |
| `CompositePIIDetector` | ~100ms | Both | Recommended |
 
```python
from src.input.pii_detector import CompositePIIDetector
 
detector = CompositePIIDetector(use_presidio=False)
result = detector.detect("My SSN is 123-45-6789")
print(result.contains_pii)      # True
print(result.redacted_text)     # "My SSN is [SSN_REDACTED]"
```
 
**Topic Scope Validation** (`src/validators/topic_validator.py`)
 
Enforces topic boundaries: blocks or warns on off-topic queries.
 
```python
from src.validators.topic_validator import (
    CompositeTopicValidator,
    CUSTOMER_SUPPORT_TOPIC,
    HR_ASSISTANT_TOPIC
)
 
validator = CompositeTopicValidator(
    topic_config=CUSTOMER_SUPPORT_TOPIC,
    block_uncertain=True
)
result = validator.validate("What is the capital of France?")
print(result.result)  # TopicValidationResult.OUT_OF_SCOPE
```
 
Custom topic configuration:
 
```python
from src.validators.topic_validator import TopicConfig
 
my_topic = TopicConfig(
    name="legal_assistant",
    description="Legal document analysis and contract review",
    allowed_keywords=["contract", "clause", "agreement", "liability", "indemnity"],
    blocked_keywords=["medical advice", "financial advice"],
    examples=[
        "Summarize this contract clause",
        "What are the termination conditions?",
        "Is this indemnity clause standard?"
    ]
)
```
 
---
 
### Output Guardrails
 
**PII Redaction** (`src/output/pii_redactor.py`)
 
Redacts PII from LLM responses before they reach users or logs.
 
```python
from src.output.pii_redactor import OutputPIIRedactor
 
# Hard redaction (default)
redactor = OutputPIIRedactor(redaction_mode="hard")
result = redactor.redact("Customer John Smith (SSN: 123-45-6789) called.")
print(result.redacted_text)
# "Customer John Smith (SSN: [SSN_REDACTED]) called."
 
# Role-based — show emails to admins, redact for users
from src.output.pii_redactor import ConditionalRedactor
from src.input.pii_detector import PIIType
 
redactor = ConditionalRedactor(
    redactor=OutputPIIRedactor(),
    always_redact=[PIIType.SSN, PIIType.CREDIT_CARD],
    user_role_limits={"admin": [PIIType.EMAIL, PIIType.PHONE]}
)
result = redactor.redact_for_user(text, user_role="analyst")
```
 
**Toxicity Filtering** (`src/output/toxicity_filter.py`)
 
Detects and blocks harmful content in LLM outputs.
 
```python
from src.output.toxicity_filter import CompositeToxicityFilter, ToxicitySeverity
 
filter = CompositeToxicityFilter(
    use_ml=False,
    block_severity=ToxicitySeverity.HIGH,
    custom_patterns={"competitor": [r"\bCompetitorX\b"]}
)
result = filter.filter(llm_response)
if filter.should_block(result):
    return safe_fallback_response
```
 
---
 
## Integration Patterns
 
### Pattern 1: Minimal (PII + toxicity only)
 
```python
config = GuardrailConfig(
    topic_config=None,
    detect_input_pii=True,
    redact_output_pii=True,
    filter_toxicity=True
)
```
 
Best for: internal tools, low-risk applications.
 
### Pattern 2: Standard (adds topic scope)
 
```python
config = GuardrailConfig(
    topic_config=CUSTOMER_SUPPORT_TOPIC,
    block_off_topic=True,
    detect_input_pii=True,
    redact_output_pii=True,
    filter_toxicity=True
)
```
 
Best for: customer-facing chatbots with defined scope.
 
### Pattern 3: Full compliance
 
```python
config = GuardrailConfig(
    topic_config=your_topic_config,
    block_off_topic=True,
    detect_input_pii=True,
    block_input_pii=False,       # Redact, don't block
    redact_output_pii=True,
    filter_toxicity=True,
    block_toxic_severity=ToxicitySeverity.MEDIUM,
    system_prompt_fragments=["secret_keyword", "internal@company.com"],
    log_all_checks=True
)
```
 
Best for: regulated industries — healthcare, finance, legal.
 
---
 
## Reading the GuardrailReport
 
Every `wrapped.run()` call returns a `GuardrailReport`:
 
```python
result = wrapped.run(query)
 
result.response          # Safe response to return to user
result.blocked           # True if request was blocked
result.input_modified    # True if query had PII redacted
result.output_modified   # True if response had PII redacted
result.overall_action    # PASSED / BLOCKED / REDACTED / WARNED
result.checks            # List of individual guardrail results
result.summary()         # Human-readable audit summary
```
 
Each check in `result.checks`:
 
```python
check.name      # "topic_validator", "input_pii_detector", etc.
check.action    # PASSED / BLOCKED / REDACTED / WARNED
check.reason    # Human-readable explanation
check.details   # Dict with additional context
```
 
---
 
## Compliance logging
 
For regulated industries, log the full audit trail:
 
```python
import json
import logging
 
def process_query_with_audit(query: str, user_id: str):
    result = wrapped.run(query)
 
    # Log to your compliance system
    logging.info(json.dumps({
        "event": "llm_query",
        "user_id": user_id,
        "action": result.overall_action.value,
        "blocked": result.blocked,
        "pii_in_input": result.input_modified,
        "pii_in_output": result.output_modified,
        "checks": [
            {"name": c.name, "action": c.action.value}
            for c in result.checks
        ]
    }))
 
    return result.response
```
 
---
 
## Related repos
 
- [rag-pipeline](https://github.com/pulkitkushwaha/rag-pipeline) — RAG pipeline this wraps
- [llm-security-playbook](https://github.com/pulkitkushwaha/llm-security-playbook) — threat modeling + attack demos
- [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework) — evaluation framework
- [multi-agent-system](https://github.com/pulkitkushwaha/multi-agent-system) — agentic AI with LangGraph
---
 
*Built by [Pulkit Kushwaha](https://linkedin.com/in/pulkit-kushwaha-514764197)
· Part of [ai-engineering-portfolio](https://github.com/pulkitkushwaha/ai-engineering-portfolio)*