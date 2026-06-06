"""
Example: Wrapping a RAG pipeline with llm-guardrails
 
This example shows the complete integration pattern —
taking an existing RAG pipeline and adding the full
guardrails stack with minimal code changes.
 
Three integration patterns shown:
    1. Minimal: just PII redaction and toxicity filtering
    2. Standard: adds topic scope enforcement
    3. Full: all guardrails + audit logging + compliance report
 
Run:
    python examples/rag_with_guardrails.py
"""
 
from src.pipeline.guardrails_wrapper import GuardrailsWrapper, GuardrailConfig
from src.output.toxicity_filter import ToxicitySeverity
from src.validators.topic_validator import CUSTOMER_SUPPORT_TOPIC
 
 
# ── Your existing RAG pipeline (unchanged) ───────────────────
 
def my_rag_pipeline(query: str) -> tuple:
    """
    Simulated RAG pipeline — replace with your actual pipeline.
 
    In production this would:
        1. Embed the query
        2. Retrieve relevant chunks from FAISS
        3. Build a prompt with retrieved context
        4. Call Azure OpenAI / OpenAI
        5. Return (answer, contexts)
    """
    # Mock response for demonstration
    mock_answers = {
        "return": (
            "Our return policy allows returns within 30 days of purchase. "
            "Items must be unused and in original packaging.",
            ["Our return policy: returns accepted within 30 days."]
        ),
        "shipping": (
            "Standard shipping takes 5-7 business days. "
            "Express shipping takes 2-3 business days.",
            ["Shipping options: standard (5-7 days), express (2-3 days)."]
        )
    }
 
    query_lower = query.lower()
    for keyword, response in mock_answers.items():
        if keyword in query_lower:
            return response
 
    return (
        "I can help you with questions about our products, shipping, and returns.",
        ["General customer support information."]
    )
 
 
# ── Pattern 1: Minimal guardrails ────────────────────────────
 
def demo_minimal():
    """
    Minimal integration: PII redaction + toxicity filtering only.
    Best for: internal tools where topic scope isn't a concern.
    """
    print("\n" + "="*55)
    print("PATTERN 1: Minimal guardrails")
    print("="*55)
 
    config = GuardrailConfig(
        topic_config=None,       # No topic restriction
        detect_input_pii=True,
        block_input_pii=False,   # Redact PII, don't block
        redact_output_pii=True,
        filter_toxicity=True,
        block_toxic_severity=ToxicitySeverity.HIGH
    )
 
    wrapped = GuardrailsWrapper(pipeline=my_rag_pipeline, config=config)
 
    queries = [
        "What is the return policy?",
        "My SSN is 123-45-6789, what is my order status?",
    ]
 
    for query in queries:
        result = wrapped.run(query)
        print(f"\nQuery:    {query}")
        print(f"Response: {result.response[:80]}...")
        print(f"Action:   {result.overall_action.value}")
        print(f"Input modified:  {result.input_modified}")
        print(f"Output modified: {result.output_modified}")
 
 
# ── Pattern 2: Standard guardrails ───────────────────────────
 
def demo_standard():
    """
    Standard integration: adds topic scope enforcement.
    Best for: customer-facing chatbots with defined scope.
    """
    print("\n" + "="*55)
    print("PATTERN 2: Standard guardrails (with topic scope)")
    print("="*55)
 
    config = GuardrailConfig(
        topic_config=CUSTOMER_SUPPORT_TOPIC,
        block_off_topic=True,
        detect_input_pii=True,
        redact_output_pii=True,
        filter_toxicity=True
    )
 
    wrapped = GuardrailsWrapper(pipeline=my_rag_pipeline, config=config)
 
    queries = [
        "What is the return policy?",           # In scope
        "What is the capital of France?",        # Out of scope
        "Can you write my essay for me?",        # Out of scope
        "How long does shipping take?",          # In scope
    ]
 
    for query in queries:
        result = wrapped.run(query)
        status = "✅ PASSED" if not result.blocked else "🚫 BLOCKED"
        print(f"\n{status} | {query}")
        print(f"  Response: {result.response[:70]}...")
 
 
# ── Pattern 3: Full guardrails with audit logging ─────────────
 
def demo_full():
    """
    Full integration: all guardrails + complete audit trail.
    Best for: regulated industries (healthcare, finance, legal).
    """
    print("\n" + "="*55)
    print("PATTERN 3: Full guardrails with audit logging")
    print("="*55)
 
    config = GuardrailConfig(
        topic_config=CUSTOMER_SUPPORT_TOPIC,
        block_off_topic=True,
        detect_input_pii=True,
        block_input_pii=False,
        redact_output_pii=True,
        filter_toxicity=True,
        block_toxic_severity=ToxicitySeverity.MEDIUM,
        system_prompt_fragments=[
            "internal policy",
            "10% discount",
        ],
        log_all_checks=True
    )
 
    wrapped = GuardrailsWrapper(pipeline=my_rag_pipeline, config=config)
 
    query = "What is your return policy for email john@test.com?"
    result = wrapped.run(query)
 
    print(f"\nQuery: {query}")
    print(f"\nResponse: {result.response}")
    print(f"\n--- Audit Trail ---")
    print(result.summary())
 
    # In production: send result.checks to your compliance logging system
    print(f"\n--- Compliance Log (JSON) ---")
    import json
    audit_data = {
        "query": result.query,
        "action": result.overall_action.value,
        "blocked": result.blocked,
        "input_modified": result.input_modified,
        "output_modified": result.output_modified,
        "checks": [
            {
                "name": c.name,
                "action": c.action.value,
                "reason": c.reason
            }
            for c in result.checks
        ]
    }
    print(json.dumps(audit_data, indent=2))
 
 
if __name__ == "__main__":
    print("llm-guardrails Integration Examples")
    print("github.com/pulkitkushwaha/llm-guardrails")
 
    demo_minimal()
    demo_standard()
    demo_full()
 
    print("\n" + "="*55)
    print("See INTEGRATION_GUIDE.md for full documentation.")
    print("="*55)