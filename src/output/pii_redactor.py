"""
Output PII Redaction
 
Why redact PII from LLM outputs?
 
Input PII detection (src/input/pii_detector.py) catches PII
that users send IN. But LLM outputs can contain PII from a
different source, that is, the retrieved context.
 
When a RAG system retrieves documents containing PII and passes
them to the LLM, the LLM may include that PII in its response:
 
    User: "Tell me about John Smith's account"
    Retrieved chunk: "John Smith (SSN: 123-45-6789) opened account #8472..."
    LLM response: "John Smith's SSN is 123-45-6789 and account is 8472"
 
Even if the retrieval was authorized (the user has access to
John Smith's record), logging this response or sending it to
a third-party analytics system creates compliance risk.
 
Output redaction is the last line of defense. It ensures that
PII never leaves the system boundary in plain text, regardless
of what the LLM generated.
 
Design decision: Redact AFTER generation, not before
    We could strip PII from retrieved chunks before passing to the LLM.
    But that degrades answer quality as the LLM needs the full context
    to answer correctly. Instead we let the LLM see the full context,
    then redact the output before it reaches the user or gets logged.
    The user gets a useful answer; the logs stay clean.
"""
 
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from src.input.pii_detector import (
    RegexPIIDetector,
    CompositePIIDetector,
    PIIDetectionResult,
    PIIType
)
 
 
@dataclass
class RedactionResult:
    """
    Result of output PII redaction.
 
    Attributes:
        original_text   : The LLM's raw output
        redacted_text   : Output with PII replaced by placeholders
        pii_found       : Whether any PII was detected
        redacted_count  : Number of PII instances redacted
        pii_types       : Types of PII that were redacted
        audit_log       : Log entry for compliance tracking
    """
    original_text: str
    redacted_text: str
    pii_found: bool
    redacted_count: int
    pii_types: List[PIIType]
    audit_log: Dict[str, Any] = field(default_factory=dict)
 
    def __post_init__(self):
        self.audit_log = {
            "pii_found": self.pii_found,
            "redacted_count": self.redacted_count,
            "pii_types": [t.value for t in self.pii_types],
            "text_length_original": len(self.original_text),
            "text_length_redacted": len(self.redacted_text),
        }
 
 
class OutputPIIRedactor:
    """
    Redacts PII from LLM output before it reaches the user or logs.
 
    Uses the same detection logic as InputPIIDetector but applied
    to outputs. Supports both hard redaction (replace with placeholder)
    and soft redaction (partial masking — show first/last characters).
 
    Args:
        use_presidio    : Enable ML-based PII detection (slower, more thorough)
        redaction_mode  : 'hard' replaces fully, 'soft' masks partially
        custom_patterns : Domain-specific PII patterns
        log_redactions  : If True, includes redaction details in audit log
 
    Usage:
        redactor = OutputPIIRedactor()
        result = redactor.redact(llm_response)
        safe_response = result.redacted_text
        log_compliance_event(result.audit_log)
    """
 
    def __init__(
        self,
        use_presidio: bool = False,
        redaction_mode: str = "hard",
        custom_patterns: Optional[Dict[str, str]] = None,
        log_redactions: bool = True
    ):
        if redaction_mode not in ("hard", "soft"):
            raise ValueError(f"redaction_mode must be 'hard' or 'soft', got '{redaction_mode}'")
 
        self.detector = CompositePIIDetector(
            use_presidio=use_presidio,
            custom_patterns=custom_patterns
        )
        self.redaction_mode = redaction_mode
        self.log_redactions = log_redactions
 
    def redact(self, text: str) -> RedactionResult:
        """
        Detect and redact PII from LLM output text.
 
        Args:
            text: Raw LLM output to scan and redact
 
        Returns:
            RedactionResult with redacted text and audit information
        """
        detection = self.detector.detect(text)
 
        if not detection.contains_pii:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                pii_found=False,
                redacted_count=0,
                pii_types=[]
            )
 
        if self.redaction_mode == "hard":
            redacted = detection.redacted_text
        else:
            redacted = self._soft_redact(text, detection)
 
        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            pii_found=True,
            redacted_count=len(detection.matches),
            pii_types=detection.pii_types_found
        )
 
    def redact_batch(self, texts: List[str]) -> List[RedactionResult]:
        """
        Redact PII from a list of texts.
 
        Useful for redacting conversation history or
        multiple retrieved chunks before logging.
 
        Args:
            texts: List of strings to redact
 
        Returns:
            List of RedactionResult objects
        """
        return [self.redact(text) for text in texts]
 
    def _soft_redact(
        self,
        text: str,
        detection: PIIDetectionResult
    ) -> str:
        """
        Partially mask PII — show first and last characters.
 
        More user-friendly than hard redaction when the user
        needs to verify they got the right record without
        seeing the full sensitive value.
 
        Example:
            SSN 123-45-6789 → 1**-**-**89
            Email john@example.com → jo**@ex*****.com
        """
        result = text
        sorted_matches = sorted(
            detection.matches,
            key=lambda m: m.start,
            reverse=True
        )
 
        for match in sorted_matches:
            value = match.value
            if len(value) <= 4:
                masked = "*" * len(value)
            else:
                # Show first 2 and last 2 characters
                masked = value[:2] + "*" * (len(value) - 4) + value[-2:]
 
            result = result[:match.start] + masked + result[match.end:]
 
        return result
 
 
class ConditionalRedactor:
    """
    Redacts PII only when certain conditions are met.
 
    Useful for systems where PII in outputs is acceptable for
    authorized users but should be redacted for others
    or where some PII types are more sensitive than others.
 
    Args:
        redactor         : Base OutputPIIRedactor instance
        always_redact    : PII types to always redact regardless of conditions
        never_redact     : PII types to never redact (e.g. emails in a CRM)
        user_role_limits : Dict of {role: [allowed_pii_types]}
 
    Usage:
        redactor = ConditionalRedactor(
            redactor=OutputPIIRedactor(),
            always_redact=[PIIType.SSN, PIIType.CREDIT_CARD],
            never_redact=[PIIType.EMAIL],
        )
        result = redactor.redact_for_user(text, user_role="analyst")
    """
 
    def __init__(
        self,
        redactor: OutputPIIRedactor,
        always_redact: Optional[List[PIIType]] = None,
        never_redact: Optional[List[PIIType]] = None,
        user_role_limits: Optional[Dict[str, List[PIIType]]] = None
    ):
        self.redactor = redactor
        self.always_redact = set(always_redact or [PIIType.SSN, PIIType.CREDIT_CARD])
        self.never_redact = set(never_redact or [])
        self.user_role_limits = user_role_limits or {}
 
    def redact_for_user(
        self,
        text: str,
        user_role: str = "default"
    ) -> RedactionResult:
        """
        Apply role-based redaction rules.
 
        Args:
            text      : Text to potentially redact
            user_role : User's role — determines which PII types are redacted
 
        Returns:
            RedactionResult with role-appropriate redaction applied
        """
        # Get allowed PII types for this role
        allowed_types = set(self.user_role_limits.get(user_role, []))
 
        # Detect all PII first
        base_result = self.redactor.redact(text)
 
        if not base_result.pii_found:
            return base_result
 
        # Re-redact based on role rules
        # Always redact high-risk types regardless of role
        # Never redact types explicitly allowed
        # Redact everything else not explicitly allowed
        from src.input.pii_detector import RegexPIIDetector
        detector = RegexPIIDetector()
        detection = detector.detect(text)
 
        result_text = text
        redacted_count = 0
        redacted_types = []
 
        sorted_matches = sorted(
            detection.matches,
            key=lambda m: m.start,
            reverse=True
        )
 
        for match in sorted_matches:
            pii_type = match.pii_type
            should_redact = (
                pii_type in self.always_redact or
                (pii_type not in self.never_redact and pii_type not in allowed_types)
            )
 
            if should_redact:
                placeholder = f"[{pii_type.value.upper()}_REDACTED]"
                result_text = result_text[:match.start] + placeholder + result_text[match.end:]
                redacted_count += 1
                redacted_types.append(pii_type)
 
        return RedactionResult(
            original_text=text,
            redacted_text=result_text,
            pii_found=redacted_count > 0,
            redacted_count=redacted_count,
            pii_types=redacted_types
        )
