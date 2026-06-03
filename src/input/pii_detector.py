"""
PII Detection for LLM Input Validation
 
Why detect PII in inputs?
 
When users send queries to an LLM system, they sometimes include
personal information, either their own or someone else's. This
creates two risks:
 
1. PII logging — the query (with PII) gets logged to your observability
   stack, creating a compliance problem. GDPR, HIPAA, and SOC2 all have
   strict requirements about logging personal data.
2. PII in context — if the query is used to retrieve documents, the
   retrieved documents may contain the same PII, which then appears
   in the LLM response visible to other users in multi-tenant systems.
Detecting PII at input time lets you:
    - Redact before logging
    - Block queries that are asking to process PII inappropriately
    - Alert on sensitive data patterns for compliance monitoring
 
Detection approaches used here:
    1. Regex patterns — fast, deterministic, good for structured PII
       (SSNs, credit cards, phone numbers, emails)
    2. Microsoft Presidio — ML-based NER for unstructured PII
       (names, addresses, organizations in free text)
    3. Custom patterns — domain-specific PII (employee IDs, account numbers)
"""
 
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
 
 
class PIIType(Enum):
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    MEDICAL_RECORD = "medical_record"
    CUSTOM = "custom"
 
 
@dataclass
class PIIMatch:
    """A single PII detection result."""
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float
    detector: str  # "regex" or "presidio"
    redacted_value: str = "[REDACTED]"
 
 
@dataclass
class PIIDetectionResult:
    """Aggregated result of PII detection on a text."""
    text: str
    matches: List[PIIMatch] = field(default_factory=list)
    contains_pii: bool = False
    redacted_text: str = ""
    pii_types_found: List[PIIType] = field(default_factory=list)
 
    def __post_init__(self):
        self.contains_pii = len(self.matches) > 0
        self.pii_types_found = list({m.pii_type for m in self.matches})
        if self.matches:
            self.redacted_text = self._build_redacted_text()
        else:
            self.redacted_text = self.text
 
    def _build_redacted_text(self) -> str:
        """Replace all PII matches with redacted placeholders."""
        result = self.text
        # Sort by start position descending to replace from end
        sorted_matches = sorted(self.matches, key=lambda m: m.start, reverse=True)
        for match in sorted_matches:
            placeholder = f"[{match.pii_type.value.upper()}_REDACTED]"
            result = result[:match.start] + placeholder + result[match.end:]
        return result
 
 
class RegexPIIDetector:
    """
    Fast, deterministic PII detection using regex patterns.
 
    Best for structured PII that follows a predictable format:
    SSNs, credit cards, phone numbers, email addresses.
 
    Limitations:
        - Cannot detect unstructured PII (names, addresses in free text)
        - Regex patterns have false positive rates
        - Does not understand context — flags all pattern matches
 
    Cost: Zero — pure string matching, extremely fast.
 
    Usage:
        detector = RegexPIIDetector()
        result = detector.detect("My SSN is 123-45-6789")
        print(result.contains_pii)  # True
        print(result.redacted_text)  # "My SSN is [SSN_REDACTED]"
    """
 
    PATTERNS = {
        PIIType.SSN: [
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b\d{9}\b(?!\s*[-/]\s*\d)"
        ],
        PIIType.CREDIT_CARD: [
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b"
        ],
        PIIType.EMAIL: [
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ],
        PIIType.PHONE: [
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            r"\b\+\d{1,3}[-.\s]?\d{3,14}\b"
        ],
        PIIType.IP_ADDRESS: [
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        ],
        PIIType.DATE_OF_BIRTH: [
            r"\b(?:born|dob|date of birth)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b"
        ],
        PIIType.BANK_ACCOUNT: [
            r"\b(?:account\s*(?:number|#|no\.?)[:\s]+)\d{8,17}\b",
            r"\b\d{8,17}\s*(?:account|acct)\b"
        ],
    }
 
    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        """
        Args:
            custom_patterns: Dict of {label: regex_pattern} for domain-specific PII.
                             E.g. {"employee_id": r"EMP-\d{6}"}
        """
        self.custom_patterns = custom_patterns or {}
 
    def detect(self, text: str) -> PIIDetectionResult:
        """
        Detect PII in text using regex patterns.
 
        Args:
            text: Input text to scan for PII
 
        Returns:
            PIIDetectionResult with all matches and redacted text
        """
        matches = []
 
        # Built-in patterns
        for pii_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        detector="regex"
                    ))
 
        # Custom patterns
        for label, pattern in self.custom_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(PIIMatch(
                    pii_type=PIIType.CUSTOM,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    detector=f"custom_{label}"
                ))
 
        # Deduplicate overlapping matches
        matches = self._deduplicate(matches)
 
        return PIIDetectionResult(text=text, matches=matches)
 
    def _deduplicate(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping higher confidence ones."""
        if not matches:
            return matches
 
        sorted_matches = sorted(matches, key=lambda m: (m.start, -m.confidence))
        deduplicated = [sorted_matches[0]]
 
        for match in sorted_matches[1:]:
            last = deduplicated[-1]
            if match.start >= last.end:
                deduplicated.append(match)
 
        return deduplicated
 
 
class PresidioPIIDetector:
    """
    ML-based PII detection using Microsoft Presidio.
 
    Better than regex for unstructured PII — person names,
    addresses, and organizations in free text. Uses NLP-based
    Named Entity Recognition (NER) to identify PII in context.
 
    Requires: presidio-analyzer, presidio-anonymizer, spacy model
 
    Installation:
        pip install presidio-analyzer presidio-anonymizer spacy
        python -m spacy download en_core_web_lg
 
    Cost: Moderate — NLP model inference per request. Slower than
    regex but handles natural language PII that regex misses.
 
    Usage:
        detector = PresidioPIIDetector()
        result = detector.detect("My name is John Smith and I live in NYC")
    """
 
    def __init__(self, language: str = "en", score_threshold: float = 0.6):
        self.language = language
        self.score_threshold = score_threshold
        self._analyzer = None
        self._anonymizer = None
 
    def _load(self):
        """Lazy loading of Presidio — only load when first used."""
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            except ImportError:
                raise ImportError(
                    "presidio-analyzer and presidio-anonymizer are required. "
                    "Install with: pip install presidio-analyzer presidio-anonymizer\n"
                    "Then run: python -m spacy download en_core_web_lg"
                )
 
    def detect(self, text: str) -> PIIDetectionResult:
        """
        Detect PII using Presidio NLP analysis.
 
        Args:
            text: Input text to analyze
 
        Returns:
            PIIDetectionResult with all matches
        """
        self._load()
 
        results = self._analyzer.analyze(
            text=text,
            language=self.language,
            score_threshold=self.score_threshold
        )
 
        matches = []
        for result in results:
            pii_type = self._map_entity_type(result.entity_type)
            matches.append(PIIMatch(
                pii_type=pii_type,
                value=text[result.start:result.end],
                start=result.start,
                end=result.end,
                confidence=result.score,
                detector="presidio"
            ))
 
        return PIIDetectionResult(text=text, matches=matches)
 
    def _map_entity_type(self, presidio_entity: str) -> PIIType:
        """Map Presidio entity types to our PIIType enum."""
        mapping = {
            "PERSON": PIIType.PERSON_NAME,
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "PHONE_NUMBER": PIIType.PHONE,
            "US_SSN": PIIType.SSN,
            "CREDIT_CARD": PIIType.CREDIT_CARD,
            "IP_ADDRESS": PIIType.IP_ADDRESS,
            "US_BANK_NUMBER": PIIType.BANK_ACCOUNT,
            "LOCATION": PIIType.ADDRESS,
            "MEDICAL_LICENSE": PIIType.MEDICAL_RECORD,
        }
        return mapping.get(presidio_entity, PIIType.CUSTOM)
 
 
class CompositePIIDetector:
    """
    Combines regex and Presidio detection for comprehensive coverage.
 
    Uses regex for structured PII (fast, no false negatives on
    formatted patterns) and Presidio for unstructured PII
    (names, addresses in natural language).
 
    Usage:
        detector = CompositePIIDetector(use_presidio=True)
        result = detector.detect(user_input)
        if result.contains_pii:
            log_pii_alert(result.pii_types_found)
            safe_input = result.redacted_text
    """
 
    def __init__(
        self,
        use_presidio: bool = False,
        custom_patterns: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            use_presidio    : Enable Presidio ML detection (slower but more thorough)
            custom_patterns : Domain-specific regex patterns
        """
        self.regex_detector = RegexPIIDetector(custom_patterns=custom_patterns)
        self.presidio_detector = PresidioPIIDetector() if use_presidio else None
 
    def detect(self, text: str) -> PIIDetectionResult:
        """
        Run all enabled detectors and merge results.
 
        Args:
            text: Input text to scan
 
        Returns:
            Merged PIIDetectionResult from all detectors
        """
        regex_result = self.regex_detector.detect(text)
        all_matches = list(regex_result.matches)
 
        if self.presidio_detector:
            try:
                presidio_result = self.presidio_detector.detect(text)
                # Only add Presidio matches that don't overlap with regex matches
                for presidio_match in presidio_result.matches:
                    overlaps = any(
                        not (presidio_match.end <= m.start or presidio_match.start >= m.end)
                        for m in all_matches
                    )
                    if not overlaps:
                        all_matches.append(presidio_match)
            except Exception as e:
                print(f"[CompositePIIDetector] Presidio failed: {e}. Using regex only.")
 
        return PIIDetectionResult(text=text, matches=all_matches)
