"""
Toxicity Filter for LLM Output Validation
 
Why filter toxicity in LLM outputs?
 
Modern LLMs have built-in safety training, but they can still
produce problematic outputs in edge cases:
 
    - Jailbreak attempts that partially succeed
    - Adversarial prompts that slip through safety filters
    - Domain-specific content that generic safety training misses
    - Outputs that are technically safe but violate company policy
 
The toxicity filter is a post-generation safety layer that
catches these cases before they reach users.
 
Categories covered:
    - Explicit harmful content
    - Hate speech and discriminatory language
    - Threats and incitement to violence
    - Self-harm content
    - Personal attacks and harassment
    - Policy violations (custom per deployment)
 
Detection approaches:
 
1. Keyword/pattern matching (fast, deterministic)
   Zero cost. Catches obvious cases. High false positive rate
   for context-dependent language.
2. ML classifier (Detoxify / HuggingFace)
   More nuanced. Understands context. Adds latency.
   Best for production deployments handling sensitive topics.
3. LLM-based classification
   Most accurate. Understands full context and nuance.
   Highest cost and latency. Use for highest-stakes outputs.
Design decision: Non-blocking by default
    The filter returns a ToxicityResult with a severity score
    and recommendation. The calling code decides whether to block,
    warn, or log. This makes the filter usable in different contexts
    (strict enterprise chat vs internal research tool).
"""
 
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
 
 
class ToxicitySeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
 
 
class ToxicityCategory(Enum):
    HATE_SPEECH = "hate_speech"
    THREATS = "threats"
    HARASSMENT = "harassment"
    EXPLICIT = "explicit"
    SELF_HARM = "self_harm"
    MISINFORMATION = "misinformation"
    POLICY_VIOLATION = "policy_violation"
 
 
@dataclass
class ToxicityMatch:
    """A single toxicity detection result."""
    category: ToxicityCategory
    severity: ToxicitySeverity
    matched_text: str
    reason: str
    detector: str
 
 
@dataclass
class ToxicityResult:
    """Aggregated toxicity detection result for an output."""
    text: str
    is_toxic: bool
    overall_severity: ToxicitySeverity
    matches: List[ToxicityMatch] = field(default_factory=list)
    recommendation: str = ""
    safe_alternative: Optional[str] = None
 
    def __post_init__(self):
        if not self.recommendation:
            self.recommendation = self._get_recommendation()
 
    def _get_recommendation(self) -> str:
        if self.overall_severity == ToxicitySeverity.NONE:
            return "Output is safe to display"
        elif self.overall_severity == ToxicitySeverity.LOW:
            return "Minor concern — log and monitor"
        elif self.overall_severity == ToxicitySeverity.MEDIUM:
            return "Moderate concern — consider filtering or warning"
        elif self.overall_severity == ToxicitySeverity.HIGH:
            return "High severity — block and log for review"
        else:
            return "Critical — block immediately and escalate"
 
 
class KeywordToxicityFilter:
    """
    Fast pattern-based toxicity detection.
 
    Catches explicit harmful content using regex patterns.
    Zero latency, zero cost. Not context-aware — may produce
    false positives for legitimate discussion of sensitive topics.
 
    Best used as a first-pass filter before more sophisticated checks.
 
    Args:
        custom_patterns : Dict of {category_name: [patterns]} for policy violations
        case_sensitive  : If True, patterns are case-sensitive (default: False)
    """
 
    # Core toxicity patterns by category
    # Note: These are illustrative. Production systems use more sophisticated pattern libraries
    PATTERNS: Dict[ToxicityCategory, List[Tuple[str, ToxicitySeverity]]] = {
        ToxicityCategory.THREATS: [
            (r"\b(?:i will|i'm going to|i am going to)\s+(?:kill|hurt|harm|attack)\s+(?:you|them|him|her)\b", ToxicitySeverity.CRITICAL),
            (r"\b(?:you|they)\s+(?:should|deserve to|will)\s+(?:die|suffer|be hurt)\b", ToxicitySeverity.HIGH),
            (r"\b(?:threatening|threat|threaten)\b", ToxicitySeverity.LOW),
        ],
        ToxicityCategory.HARASSMENT: [
            (r"\b(?:you are|you're)\s+(?:stupid|idiot|moron|worthless|pathetic)\b", ToxicitySeverity.MEDIUM),
            (r"\b(?:shut up|go away|nobody likes you|you're useless)\b", ToxicitySeverity.MEDIUM),
        ],
        ToxicityCategory.SELF_HARM: [
            (r"\b(?:how to|ways to|methods to)\s+(?:hurt yourself|self.harm|end your life)\b", ToxicitySeverity.CRITICAL),
            (r"\b(?:suicide|self-harm)\s+(?:methods|ways|instructions|guide)\b", ToxicitySeverity.CRITICAL),
        ],
        ToxicityCategory.MISINFORMATION: [
            (r"\b(?:vaccines? (?:cause|causes) autism)\b", ToxicitySeverity.HIGH),
            (r"\b(?:the earth is flat|flat earth is real)\b", ToxicitySeverity.MEDIUM),
        ],
    }
 
    def __init__(
        self,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        case_sensitive: bool = False
    ):
        self.case_sensitive = case_sensitive
        self.flags = 0 if case_sensitive else re.IGNORECASE
 
        # Compile built-in patterns
        self._compiled_patterns: Dict[ToxicityCategory, List[Tuple[re.Pattern, ToxicitySeverity]]] = {}
        for category, pattern_list in self.PATTERNS.items():
            self._compiled_patterns[category] = [
                (re.compile(pattern, self.flags), severity)
                for pattern, severity in pattern_list
            ]
 
        # Add custom policy violation patterns
        self._custom_patterns = []
        if custom_patterns:
            for policy_name, patterns in custom_patterns.items():
                for pattern in patterns:
                    self._custom_patterns.append((
                        re.compile(pattern, self.flags),
                        ToxicitySeverity.MEDIUM,
                        policy_name
                    ))
 
    def filter(self, text: str) -> ToxicityResult:
        """
        Check text for toxic content using pattern matching.
 
        Args:
            text: LLM output to check
 
        Returns:
            ToxicityResult with matches and severity assessment
        """
        matches = []
 
        # Check built-in patterns
        for category, pattern_list in self._compiled_patterns.items():
            for compiled_pattern, severity in pattern_list:
                match = compiled_pattern.search(text)
                if match:
                    matches.append(ToxicityMatch(
                        category=category,
                        severity=severity,
                        matched_text=match.group(),
                        reason=f"Matched {category.value} pattern",
                        detector="keyword"
                    ))
 
        # Check custom policy patterns
        for compiled_pattern, severity, policy_name in self._custom_patterns:
            match = compiled_pattern.search(text)
            if match:
                matches.append(ToxicityMatch(
                    category=ToxicityCategory.POLICY_VIOLATION,
                    severity=severity,
                    matched_text=match.group(),
                    reason=f"Policy violation: {policy_name}",
                    detector=f"custom_{policy_name}"
                ))
 
        # Determine overall severity
        overall_severity = self._compute_severity(matches)
 
        return ToxicityResult(
            text=text,
            is_toxic=len(matches) > 0,
            overall_severity=overall_severity,
            matches=matches
        )
 
    def _compute_severity(self, matches: List[ToxicityMatch]) -> ToxicitySeverity:
        """Return the highest severity across all matches."""
        if not matches:
            return ToxicitySeverity.NONE
 
        severity_order = {
            ToxicitySeverity.NONE: 0,
            ToxicitySeverity.LOW: 1,
            ToxicitySeverity.MEDIUM: 2,
            ToxicitySeverity.HIGH: 3,
            ToxicitySeverity.CRITICAL: 4
        }
 
        return max(matches, key=lambda m: severity_order[m.severity]).severity
 
 
class MLToxicityFilter:
    """
    ML-based toxicity detection using the Detoxify library.
 
    More context-aware than keyword matching — understands that
    "I want to kill this project" is not a threat, while
    keyword matching would flag it.
 
    Requires: pip install detoxify
 
    Models available:
        'original'       — Trained on Wikipedia comments
        'unbiased'       — Reduced demographic bias
        'multilingual'   — Supports 7 languages
 
    Args:
        model_name  : Detoxify model to use (default: 'original')
        thresholds  : Dict mapping toxicity type to score threshold
    """
 
    DEFAULT_THRESHOLDS = {
        "toxicity": 0.7,
        "severe_toxicity": 0.5,
        "obscene": 0.7,
        "threat": 0.6,
        "insult": 0.7,
        "identity_attack": 0.6,
    }
 
    def __init__(
        self,
        model_name: str = "original",
        thresholds: Optional[Dict[str, float]] = None
    ):
        self.model_name = model_name
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self._model = None
 
    def _load_model(self):
        """Lazy model loading."""
        if self._model is None:
            try:
                from detoxify import Detoxify
                self._model = Detoxify(self.model_name)
                print(f"[MLToxicityFilter] Loaded Detoxify model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "detoxify is required for ML toxicity detection. "
                    "Install with: pip install detoxify"
                )
 
    def filter(self, text: str) -> ToxicityResult:
        """
        Check text using Detoxify ML classifier.
 
        Args:
            text: LLM output to check
 
        Returns:
            ToxicityResult with ML-based toxicity scores
        """
        self._load_model()
 
        scores = self._model.predict(text)
        matches = []
 
        category_map = {
            "toxicity": ToxicityCategory.HARASSMENT,
            "severe_toxicity": ToxicityCategory.HARASSMENT,
            "threat": ToxicityCategory.THREATS,
            "insult": ToxicityCategory.HARASSMENT,
            "identity_attack": ToxicityCategory.HATE_SPEECH,
            "obscene": ToxicityCategory.EXPLICIT,
        }
 
        for score_type, score_value in scores.items():
            threshold = self.thresholds.get(score_type, 0.7)
            if score_value >= threshold:
                severity = self._score_to_severity(score_value)
                matches.append(ToxicityMatch(
                    category=category_map.get(score_type, ToxicityCategory.POLICY_VIOLATION),
                    severity=severity,
                    matched_text=text[:100],
                    reason=f"Detoxify {score_type} score: {score_value:.3f}",
                    detector=f"detoxify_{score_type}"
                ))
 
        severity_order = {
            ToxicitySeverity.NONE: 0, ToxicitySeverity.LOW: 1,
            ToxicitySeverity.MEDIUM: 2, ToxicitySeverity.HIGH: 3,
            ToxicitySeverity.CRITICAL: 4
        }
        overall = max(matches, key=lambda m: severity_order[m.severity]).severity \
                  if matches else ToxicitySeverity.NONE
 
        return ToxicityResult(
            text=text,
            is_toxic=len(matches) > 0,
            overall_severity=overall,
            matches=matches
        )
 
    def _score_to_severity(self, score: float) -> ToxicitySeverity:
        if score >= 0.9:
            return ToxicitySeverity.CRITICAL
        elif score >= 0.8:
            return ToxicitySeverity.HIGH
        elif score >= 0.65:
            return ToxicitySeverity.MEDIUM
        else:
            return ToxicitySeverity.LOW
 
 
class CompositeToxicityFilter:
    """
    Combines keyword and ML detection for comprehensive coverage.
 
    Strategy:
        1. Keyword filter first (fast — catches obvious cases)
        2. If CRITICAL detected by keywords → block immediately
        3. ML filter for medium/low/no keyword matches (slower, more accurate)
        4. Combine results — take highest severity
 
    Args:
        use_ml           : Enable ML-based detection (default: False)
        block_severity   : Minimum severity level to block (default: HIGH)
        custom_patterns  : Additional policy violation patterns
    """
 
    SEVERITY_ORDER = {
        ToxicitySeverity.NONE: 0,
        ToxicitySeverity.LOW: 1,
        ToxicitySeverity.MEDIUM: 2,
        ToxicitySeverity.HIGH: 3,
        ToxicitySeverity.CRITICAL: 4
    }
 
    def __init__(
        self,
        use_ml: bool = False,
        block_severity: ToxicitySeverity = ToxicitySeverity.HIGH,
        custom_patterns: Optional[Dict[str, List[str]]] = None
    ):
        self.keyword_filter = KeywordToxicityFilter(custom_patterns=custom_patterns)
        self.ml_filter = MLToxicityFilter() if use_ml else None
        self.block_severity = block_severity
 
    def filter(self, text: str) -> ToxicityResult:
        """Run composite toxicity detection."""
        keyword_result = self.keyword_filter.filter(text)
 
        # Critical keyword match — return immediately
        if keyword_result.overall_severity == ToxicitySeverity.CRITICAL:
            return keyword_result
 
        # Run ML filter if enabled
        if self.ml_filter:
            try:
                ml_result = self.ml_filter.filter(text)
                # Merge matches, take highest severity
                all_matches = keyword_result.matches + ml_result.matches
                overall = max(
                    [keyword_result.overall_severity, ml_result.overall_severity],
                    key=lambda s: self.SEVERITY_ORDER[s]
                )
                return ToxicityResult(
                    text=text,
                    is_toxic=len(all_matches) > 0,
                    overall_severity=overall,
                    matches=all_matches
                )
            except Exception as e:
                print(f"[CompositeToxicityFilter] ML filter failed: {e}. Using keyword only.")
 
        return keyword_result
 
    def should_block(self, result: ToxicityResult) -> bool:
        """Determine if output should be blocked based on severity."""
        return self.SEVERITY_ORDER[result.overall_severity] >= \
               self.SEVERITY_ORDER[self.block_severity]
