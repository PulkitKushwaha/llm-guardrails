"""
Guardrails Pipeline Wrapper
 
The wrapper is the production integration point for this entire
library. It takes any existing LLM pipeline and wraps it with
the full guardrails stack, without requiring changes to the
underlying pipeline.
 
Before this wrapper existed:
    user_input → your_rag_pipeline() → response
 
After wrapping:
    user_input
        ↓
    [Input guardrails]
        - PII detection
        - Topic scope validation
        - Injection detection
        ↓
    your_rag_pipeline()  ← unchanged
        ↓
    [Output guardrails]
        - PII redaction
        - Toxicity filtering
        - Topic relevance check
        ↓
    safe_response
 
The wrapper is designed around three core principles:
 
1. Zero changes to existing pipelines
   The wrapper takes a callable, your existing pipeline function.
   It doesn't care what's inside. LangChain, custom, anything.
2. Configurable per deployment
   Every guardrail component is optional. You compose exactly
   the protection you need for your use case.
3. Full audit trail
   Every guardrail decision is logged with reasoning:
   what was checked, what was found, what was done about it.
   Essential for compliance in regulated industries.
"""
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
 
from src.input.pii_detector import CompositePIIDetector, PIIType
from src.output.pii_redactor import OutputPIIRedactor
from src.output.toxicity_filter import CompositeToxicityFilter, ToxicitySeverity
from src.validators.topic_validator import (
    CompositeTopicValidator,
    TopicConfig,
    TopicValidationResult
)
 
 
class GuardrailAction(Enum):
    PASSED = "passed"
    BLOCKED = "blocked"
    REDACTED = "redacted"
    WARNED = "warned"
 
 
@dataclass
class GuardrailCheck:
    """Result of a single guardrail check."""
    name: str
    action: GuardrailAction
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
 
 
@dataclass
class GuardrailReport:
    """
    Complete audit report for a single pipeline invocation.
 
    Contains every guardrail check, what was found, and what
    action was taken. Suitable for compliance logging.
    """
    query: str
    response: str
    overall_action: GuardrailAction
    checks: List[GuardrailCheck] = field(default_factory=list)
    input_modified: bool = False
    output_modified: bool = False
    blocked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
 
    def summary(self) -> str:
        lines = [
            f"Query: {self.query[:60]}...",
            f"Overall: {self.overall_action.value}",
            f"Blocked: {self.blocked}",
            f"Input modified: {self.input_modified}",
            f"Output modified: {self.output_modified}",
            "Checks:"
        ]
        for check in self.checks:
            lines.append(f"  [{check.action.value.upper()}] {check.name}: {check.reason}")
        return "\n".join(lines)
 
 
@dataclass
class GuardrailConfig:
    """
    Configuration for the guardrails pipeline wrapper.
 
    All components are optional — include only what you need.
 
    Args:
        topic_config         : TopicConfig for scope enforcement (None = no topic check)
        block_off_topic      : If True, block off-topic queries (default: True)
        detect_input_pii     : Check for PII in user input (default: True)
        block_input_pii      : Block queries containing PII (default: False (redact only))
        redact_output_pii    : Redact PII from LLM output (default: True)
        filter_toxicity      : Check output for toxic content (default: True)
        block_toxic_severity : Block outputs at this severity level (default: HIGH)
        use_ml_toxicity      : Use ML-based toxicity detection (default: False)
        system_prompt_fragments: Sensitive strings from system prompt to detect if leaked
        log_all_checks       : Log all checks including passes (default: True)
    """
    topic_config: Optional[TopicConfig] = None
    block_off_topic: bool = True
    detect_input_pii: bool = True
    block_input_pii: bool = False
    redact_output_pii: bool = True
    filter_toxicity: bool = True
    block_toxic_severity: ToxicitySeverity = ToxicitySeverity.HIGH
    use_ml_toxicity: bool = False
    system_prompt_fragments: Optional[List[str]] = None
    log_all_checks: bool = True
 
 
class GuardrailsWrapper:
    """
    Wraps any LLM pipeline with configurable input/output guardrails.
 
    Drop-in wrapper — takes your existing pipeline callable and
    adds the full guardrails stack around it.
 
    Basic usage:
        from src.pipeline.guardrails_wrapper import GuardrailsWrapper, GuardrailConfig
        from src.validators.topic_validator import CUSTOMER_SUPPORT_TOPIC
 
        # Your existing pipeline (any callable)
        def my_rag_pipeline(query: str) -> str:
            # ... your existing code ...
            return answer
 
        # Wrap it
        config = GuardrailConfig(
            topic_config=CUSTOMER_SUPPORT_TOPIC,
            redact_output_pii=True,
            filter_toxicity=True
        )
        wrapped = GuardrailsWrapper(pipeline=my_rag_pipeline, config=config)
 
        # Use exactly like before (same interface)
        result = wrapped.run("What is the return policy?")
        print(result.response)
        print(result.summary())
 
    Args:
        pipeline : Your existing LLM pipeline callable
                   Signature: (query: str, **kwargs) -> str
        config   : GuardrailConfig specifying which guardrails to apply
    """
 
    BLOCKED_RESPONSE = (
        "I'm sorry, I'm unable to process this request. "
        "Please try a different question."
    )
 
    def __init__(
        self,
        pipeline: Callable,
        config: Optional[GuardrailConfig] = None
    ):
        self.pipeline = pipeline
        self.config = config or GuardrailConfig()
 
        # Initialize only the components that are configured
        self._input_pii_detector = (
            CompositePIIDetector(use_presidio=False)
            if self.config.detect_input_pii
            else None
        )
 
        self._output_pii_redactor = (
            OutputPIIRedactor(
                system_prompt_fragments=self.config.system_prompt_fragments or []
            )
            if self.config.redact_output_pii
            else None
        )
 
        self._toxicity_filter = (
            CompositeToxicityFilter(
                use_ml=self.config.use_ml_toxicity,
                block_severity=self.config.block_toxic_severity
            )
            if self.config.filter_toxicity
            else None
        )
 
        self._topic_validator = (
            CompositeTopicValidator(
                topic_config=self.config.topic_config,
                block_uncertain=self.config.block_off_topic
            )
            if self.config.topic_config
            else None
        )
 
    def run(
        self,
        query: str,
        **pipeline_kwargs
    ) -> GuardrailReport:
        """
        Run query through guardrails → pipeline → guardrails.
 
        Args:
            query           : User's query
            **pipeline_kwargs: Additional kwargs passed to your pipeline
 
        Returns:
            GuardrailReport with response and full audit trail
        """
        report = GuardrailReport(
            query=query,
            response="",
            overall_action=GuardrailAction.PASSED
        )
 
        # ── INPUT GUARDRAILS ──────────────────────────────────
 
        # 1. Topic scope validation
        if self._topic_validator:
            topic_result = self._topic_validator.validate(query)
            if topic_result.result == TopicValidationResult.OUT_OF_SCOPE:
                report.checks.append(GuardrailCheck(
                    name="topic_validator",
                    action=GuardrailAction.BLOCKED,
                    reason=f"Off-topic query: {topic_result.reason}",
                    details={"confidence": topic_result.confidence}
                ))
                report.overall_action = GuardrailAction.BLOCKED
                report.blocked = True
                report.response = (
                    topic_result.suggested_redirect or self.BLOCKED_RESPONSE
                )
                return report
            else:
                if self.config.log_all_checks:
                    report.checks.append(GuardrailCheck(
                        name="topic_validator",
                        action=GuardrailAction.PASSED,
                        reason=f"In scope: {topic_result.reason}",
                        details={"confidence": topic_result.confidence}
                    ))
 
        # 2. Input PII detection
        safe_query = query
        if self._input_pii_detector:
            pii_result = self._input_pii_detector.detect(query)
            if pii_result.contains_pii:
                if self.config.block_input_pii:
                    report.checks.append(GuardrailCheck(
                        name="input_pii_detector",
                        action=GuardrailAction.BLOCKED,
                        reason=f"PII detected in input: {[t.value for t in pii_result.pii_types_found]}",
                        details={"pii_types": [t.value for t in pii_result.pii_types_found]}
                    ))
                    report.overall_action = GuardrailAction.BLOCKED
                    report.blocked = True
                    report.response = (
                        "Your query contains personal information. "
                        "Please remove sensitive data and try again."
                    )
                    return report
                else:
                    # Redact PII from query before passing to pipeline
                    safe_query = pii_result.redacted_text
                    report.input_modified = True
                    report.checks.append(GuardrailCheck(
                        name="input_pii_detector",
                        action=GuardrailAction.REDACTED,
                        reason=f"PII redacted from input: {[t.value for t in pii_result.pii_types_found]}",
                        details={"pii_types": [t.value for t in pii_result.pii_types_found]}
                    ))
            elif self.config.log_all_checks:
                report.checks.append(GuardrailCheck(
                    name="input_pii_detector",
                    action=GuardrailAction.PASSED,
                    reason="No PII detected in input"
                ))
 
        # ── PIPELINE CALL ─────────────────────────────────────
 
        try:
            raw_response = self.pipeline(safe_query, **pipeline_kwargs)
        except Exception as e:
            report.checks.append(GuardrailCheck(
                name="pipeline",
                action=GuardrailAction.BLOCKED,
                reason=f"Pipeline execution failed: {str(e)}"
            ))
            report.overall_action = GuardrailAction.BLOCKED
            report.blocked = True
            report.response = self.BLOCKED_RESPONSE
            return report
 
        # ── OUTPUT GUARDRAILS ─────────────────────────────────
 
        safe_response = raw_response
 
        # 3. Output PII redaction
        if self._output_pii_redactor:
            redaction_result = self._output_pii_redactor.redact(raw_response)
            if redaction_result.pii_found:
                safe_response = redaction_result.redacted_text
                report.output_modified = True
                report.checks.append(GuardrailCheck(
                    name="output_pii_redactor",
                    action=GuardrailAction.REDACTED,
                    reason=f"PII redacted from output: {[t.value for t in redaction_result.pii_types]}",
                    details={"redacted_count": redaction_result.redacted_count}
                ))
            elif self.config.log_all_checks:
                report.checks.append(GuardrailCheck(
                    name="output_pii_redactor",
                    action=GuardrailAction.PASSED,
                    reason="No PII detected in output"
                ))
 
        # 4. Toxicity filtering
        if self._toxicity_filter:
            toxicity_result = self._toxicity_filter.filter(safe_response)
            if self._toxicity_filter.should_block(toxicity_result):
                report.checks.append(GuardrailCheck(
                    name="toxicity_filter",
                    action=GuardrailAction.BLOCKED,
                    reason=f"Toxic output blocked (severity: {toxicity_result.overall_severity.value})",
                    details={
                        "severity": toxicity_result.overall_severity.value,
                        "categories": [m.category.value for m in toxicity_result.matches]
                    }
                ))
                report.overall_action = GuardrailAction.BLOCKED
                report.blocked = True
                report.response = self.BLOCKED_RESPONSE
                return report
            elif toxicity_result.is_toxic:
                report.checks.append(GuardrailCheck(
                    name="toxicity_filter",
                    action=GuardrailAction.WARNED,
                    reason=f"Low-severity toxic content detected (severity: {toxicity_result.overall_severity.value})",
                    details={"severity": toxicity_result.overall_severity.value}
                ))
            elif self.config.log_all_checks:
                report.checks.append(GuardrailCheck(
                    name="toxicity_filter",
                    action=GuardrailAction.PASSED,
                    reason="No toxic content detected"
                ))
 
        # ── FINAL RESULT ──────────────────────────────────────
 
        report.response = safe_response
 
        if report.input_modified or report.output_modified:
            report.overall_action = GuardrailAction.REDACTED
        else:
            report.overall_action = GuardrailAction.PASSED
 
        return report
