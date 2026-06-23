"""
Topic Relevance Validator
 
Enforces topic scope for LLM-powered applications.
 
The problem without topic validation:
    A customer support bot is deployed to answer questions about
    products and return policies. Without topic validation:
    - Users ask it for medical advice → it answers
    - Users ask it to write code → it complies
    - Users ask about competitors → it responds
    - Users ask controversial political questions → it engages
 
    This creates liability, degrades user experience, and wastes
    compute on off-topic queries.
 
The solution:
    Define the allowed topics for your application.
    Validate every query against those topics before processing.
    Block off-topic queries with a helpful redirect.
 
Two validation approaches:
 
1. Keyword-based (fast, zero cost)
   Checks if query contains topic-relevant keywords.
   High false negative rate, misses paraphrased queries.
   Best for: simple scope enforcement, low-latency requirements.
2. Embedding-based (slower, more accurate)
   Embeds the query and compares to embeddings of topic descriptions.
   Catches paraphrased off-topic queries.
   Best for: production systems where accuracy matters.
3. LLM-based (slowest, most accurate)
   Asks an LLM to classify the query against allowed topics.
   Near-human accuracy. Adds latency and cost.
   Best for: high-stakes applications where off-topic responses
   have real consequences.
"""
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
 
 
class TopicValidationResult(Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    UNCERTAIN = "uncertain"
 
 
@dataclass
class TopicConfig:
    """
    Configuration for a topic scope definition.
 
    Args:
        name            : Topic name (e.g. "customer_support")
        description     : Natural language description of what's in scope
        allowed_keywords: Keywords that indicate an in-scope query
        blocked_keywords: Keywords that always indicate out-of-scope
        examples        : Example in-scope queries (used for embedding comparison)
    """
    name: str
    description: str
    allowed_keywords: List[str]
    blocked_keywords: List[str] = None
    examples: List[str] = None
 
    def __post_init__(self):
        self.blocked_keywords = self.blocked_keywords or []
        self.examples = self.examples or []
 
 
@dataclass
class ValidationResult:
    """Result of topic relevance validation."""
    result: TopicValidationResult
    confidence: float
    reason: str
    matched_topic: Optional[str] = None
    suggested_redirect: Optional[str] = None
 
 
class KeywordTopicValidator:
    """
    Fast keyword-based topic scope enforcement.
 
    Zero latency, zero cost. Good as a first-pass filter
    to catch obvious off-topic queries before more expensive
    validation methods.
 
    Args:
        topic_config    : TopicConfig defining allowed scope
        redirect_message: Message to return for out-of-scope queries
    """
 
    def __init__(
        self,
        topic_config: TopicConfig,
        redirect_message: Optional[str] = None
    ):
        self.config = topic_config
        self.redirect_message = redirect_message or (
            f"I can only help with {topic_config.name}-related questions. "
            f"Could you ask something about {topic_config.description}?"
        )
 
        # Pre-compile patterns for efficiency
        self._allowed_patterns = [
            re.compile(kw, re.IGNORECASE)
            for kw in topic_config.allowed_keywords
        ]
        self._blocked_patterns = [
            re.compile(kw, re.IGNORECASE)
            for kw in topic_config.blocked_keywords
        ]
 
    def validate(self, query: str) -> ValidationResult:
        """
        Validate query against topic scope using keyword matching.
 
        Args:
            query: User's query to validate
 
        Returns:
            ValidationResult with scope determination
        """
        # Check blocked keywords first, immediate rejection
        for pattern in self._blocked_patterns:
            if pattern.search(query):
                return ValidationResult(
                    result=TopicValidationResult.OUT_OF_SCOPE,
                    confidence=0.95,
                    reason=f"Query contains blocked keyword: '{pattern.pattern}'",
                    suggested_redirect=self.redirect_message
                )
 
        # Check allowed keywords
        matched_keywords = [
            pattern.pattern
            for pattern in self._allowed_patterns
            if pattern.search(query)
        ]
 
        if matched_keywords:
            confidence = min(0.5 + len(matched_keywords) * 0.1, 0.95)
            return ValidationResult(
                result=TopicValidationResult.IN_SCOPE,
                confidence=confidence,
                reason=f"Matched topic keywords: {matched_keywords[:3]}",
                matched_topic=self.config.name
            )
 
        # No keywords matched — uncertain
        return ValidationResult(
            result=TopicValidationResult.UNCERTAIN,
            confidence=0.4,
            reason="No topic keywords matched — query may be off-topic",
            suggested_redirect=self.redirect_message
        )
 
 
class EmbeddingTopicValidator:
    """
    Embedding-based topic scope validation.
 
    More accurate than keyword matching as it catches paraphrased
    off-topic queries that use different vocabulary.
 
    Compares query embedding to embeddings of:
    - Topic description
    - Example in-scope queries
    - Example out-of-scope queries (if provided)
 
    Args:
        topic_config        : TopicConfig defining allowed scope
        embedder            : Embedding model with embed_query() method
        similarity_threshold: Minimum cosine similarity for in-scope (default 0.6)
        out_of_scope_examples: Example out-of-scope queries for better boundary
    """
 
    def __init__(
        self,
        topic_config: TopicConfig,
        embedder=None,
        similarity_threshold: float = 0.6,
        out_of_scope_examples: Optional[List[str]] = None
    ):
        self.config = topic_config
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.out_of_scope_examples = out_of_scope_examples or []
        self._topic_embeddings = None
 
    def _load_embeddings(self):
        """Lazily compute topic embeddings on first use."""
        if self._topic_embeddings is not None or self.embedder is None:
            return
 
        reference_texts = [self.config.description] + self.config.examples
        self._topic_embeddings = self.embedder.embed_documents(reference_texts)
 
        if self.out_of_scope_examples:
            self._oos_embeddings = self.embedder.embed_documents(
                self.out_of_scope_examples
            )
        else:
            self._oos_embeddings = []
 
    def validate(self, query: str) -> ValidationResult:
        """
        Validate query using embedding similarity.
 
        Args:
            query: User's query to validate
 
        Returns:
            ValidationResult with similarity-based scope determination
        """
        if self.embedder is None:
            return ValidationResult(
                result=TopicValidationResult.UNCERTAIN,
                confidence=0.0,
                reason="No embedder configured, cannot perform embedding validation"
            )
 
        self._load_embeddings()
 
        import numpy as np
 
        query_embedding = np.array(self.embedder.embed_query(query))
        topic_embeddings = np.array(self._topic_embeddings)
 
        # Compute similarity to topic embeddings
        similarities = []
        for topic_emb in topic_embeddings:
            sim = np.dot(query_embedding, topic_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(topic_emb) + 1e-8
            )
            similarities.append(float(sim))
 
        max_topic_sim = max(similarities) if similarities else 0.0
 
        # Check similarity to out-of-scope examples
        oos_sim = 0.0
        if self._oos_embeddings:
            oos_similarities = []
            for oos_emb in np.array(self._oos_embeddings):
                sim = np.dot(query_embedding, oos_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(oos_emb) + 1e-8
                )
                oos_similarities.append(float(sim))
            oos_sim = max(oos_similarities)
 
        # Decision logic
        if oos_sim > max_topic_sim and oos_sim > 0.7:
            return ValidationResult(
                result=TopicValidationResult.OUT_OF_SCOPE,
                confidence=oos_sim,
                reason=f"Query more similar to out-of-scope examples (sim={oos_sim:.3f})"
            )
 
        if max_topic_sim >= self.threshold:
            return ValidationResult(
                result=TopicValidationResult.IN_SCOPE,
                confidence=max_topic_sim,
                reason=f"Query similar to topic scope (sim={max_topic_sim:.3f})",
                matched_topic=self.config.name
            )
 
        return ValidationResult(
            result=TopicValidationResult.UNCERTAIN,
            confidence=max_topic_sim,
            reason=f"Query similarity below threshold ({max_topic_sim:.3f} < {self.threshold})"
        )
 
 
class CompositeTopicValidator:
    """
    Combines keyword and embedding validation for production use.
 
    Strategy:
        1. Keyword check first (fast — catches obvious cases)
        2. If uncertain, embedding check (slower — resolves ambiguous cases)
        3. Blocked keywords always override — no appeal
 
    This gives keyword speed for clear cases and embedding accuracy
    for ambiguous ones.
 
    Args:
        topic_config         : TopicConfig defining allowed scope
        embedder             : Embedding model (optional)
        similarity_threshold : Embedding similarity threshold
        block_uncertain      : If True, treat UNCERTAIN as OUT_OF_SCOPE
    """
 
    def __init__(
        self,
        topic_config: TopicConfig,
        embedder=None,
        similarity_threshold: float = 0.6,
        block_uncertain: bool = False
    ):
        self.keyword_validator = KeywordTopicValidator(topic_config)
        self.embedding_validator = EmbeddingTopicValidator(
            topic_config,
            embedder=embedder,
            similarity_threshold=similarity_threshold
        )
        self.block_uncertain = block_uncertain
 
    def validate(self, query: str) -> ValidationResult:
        """
        Run composite validation.
 
        Args:
            query: Query to validate
 
        Returns:
            ValidationResult from the most appropriate validator
        """
        # Stage 1 — keyword check
        keyword_result = self.keyword_validator.validate(query)
 
        # Definitive results from keyword check, return immediately
        if keyword_result.result == TopicValidationResult.OUT_OF_SCOPE:
            return keyword_result
        if keyword_result.result == TopicValidationResult.IN_SCOPE and keyword_result.confidence >= 0.8:
            return keyword_result
 
        # Stage 2 — embedding check for uncertain/low-confidence cases
        embedding_result = self.embedding_validator.validate(query)
 
        if embedding_result.result != TopicValidationResult.UNCERTAIN:
            return embedding_result
 
        # Both uncertain — apply block_uncertain policy
        if self.block_uncertain:
            return ValidationResult(
                result=TopicValidationResult.OUT_OF_SCOPE,
                confidence=0.5,
                reason="Query could not be confirmed as in-scope",
                suggested_redirect=self.keyword_validator.redirect_message
            )
 
        return ValidationResult(
            result=TopicValidationResult.IN_SCOPE,
            confidence=0.4,
            reason="Query could not be confirmed as out-of-scope — allowing through"
        )
 
 
# ── Pre-built topic configs ───────────────────────────────────
 
CUSTOMER_SUPPORT_TOPIC = TopicConfig(
    name="customer_support",
    description="Product support, orders, shipping, returns, refunds, and account management",
    allowed_keywords=[
        "order", "return", "refund", "shipping", "delivery", "track",
        "product", "account", "payment", "invoice", "warranty", "cancel",
        "exchange", "discount", "price", "stock", "available"
    ],
    blocked_keywords=[
        "medical advice", "legal advice", "investment advice",
        "competitor", "hack", "exploit", "bypass"
    ],
    examples=[
        "What is your return policy?",
        "Where is my order?",
        "How do I get a refund?",
        "Is this product still in stock?",
        "Can I cancel my order?"
    ]
)
 
HR_ASSISTANT_TOPIC = TopicConfig(
    name="hr_assistant",
    description="HR policies, benefits, leave, payroll, and employee procedures",
    allowed_keywords=[
        "leave", "vacation", "sick", "benefit", "payroll", "salary",
        "policy", "procedure", "onboarding", "offboarding", "performance",
        "training", "holiday", "expense", "reimbursement"
    ],
    blocked_keywords=[
        "personal relationship", "dating", "outside work"
    ],
    examples=[
        "How many vacation days do I get?",
        "What is the parental leave policy?",
        "How do I submit an expense report?",
        "When is the performance review cycle?"
    ]
)
