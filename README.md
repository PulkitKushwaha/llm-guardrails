# llm-guardrails
 
Production LLM guardrails: input validation, output validation,
PII detection, toxicity filtering, and topic relevance checks
for enterprise AI pipelines.
 
> Most LLM systems are built with a single question in mind:
> "Does it give good answers?" Production systems need a second
> question: "Does it give safe answers?" Guardrails are the
> answer to that second question.
 
---
 
## Why guardrails matter
 
Enterprise LLM systems handle sensitive data, serve regulated
industries, and operate at scale. Without guardrails:
 
- **PII leaks**: user data in prompts surfaces in responses
  visible to other users or logged in plain text
- **Prompt injection**: malicious inputs manipulate the LLM
  into ignoring its intended behavior
- **Off-topic responses**: the LLM answers questions outside
  its intended scope, creating liability
- **Hallucinated outputs**: confident wrong answers fed into
  downstream automated systems cause compounding errors
- **Toxic outputs**: unfiltered generation produces content
  that violates policy or harms users
Guardrails address all of these, not by making the LLM smarter,
but by adding a validation layer around it.
 
---
 
## Architecture
 
```
User Input
    ↓
┌─────────────────────────────┐
│     INPUT GUARDRAILS        │
│  • PII detection            │
│  • Injection detection      │
│  • Topic scope check        │
│  • Input length limits      │
└──────────────┬──────────────┘
               ↓
         LLM Pipeline
               ↓
┌─────────────────────────────┐
│     OUTPUT GUARDRAILS       │
│  • PII redaction            │
│  • Toxicity filtering       │
│  • Topic relevance check    │
│  • Hallucination detection  │
│  • Schema validation        │
└──────────────┬──────────────┘
               ↓
         User Response
```
 
Input and output guardrails are independent layers. Each can
be configured, extended, or disabled independently. This means
you can add output validation to a pipeline that already exists
without touching the input layer, and vice versa.
 
---
 
## What this covers
 
| Component | Description |
|---|---|
| Input PII detection | Detect SSNs, emails, phone numbers, credit cards before they reach the LLM |
| Prompt injection detection | Pattern and ML-based detection of injection attempts |
| Topic scope filter | Block questions outside the intended domain |
| Output PII redaction | Redact PII from LLM responses before returning to user |
| Toxicity filter | Flag or block toxic, harmful, or policy-violating outputs |
| Topic relevance check | Verify response addresses the actual question |
| Hallucination detection | Grounding check: is the response supported by context? |
| Schema validation | Enforce structured output schemas using Pydantic |
| Pipeline wrapper | Drop-in wrapper to add guardrails to any existing LLM pipeline |
 
---
 
## Design principles
 
**Non-blocking by default**
Guardrails flag issues and return structured results, they don't
silently modify behavior. The calling code decides what to do with
a flagged result. This makes guardrails auditable and testable.
 
**Layered, not monolithic**
Input and output guardrails are independent. Each validator within
a layer is independent. You compose only what you need and you don't
pay the cost of running toxicity detection on an internal admin tool.
 
**Fail safe, not fail open**
When a guardrail check fails due to an error (network issue, model
timeout), it defaults to blocking rather than passing. A false
positive is better than a missed violation in production.
 
**Production observable**
Every guardrail decision is logged with reasoning, not just pass/fail.
This is essential for debugging, auditing, and improving guardrails
over time.
 
---
 
## Structure
 
```
llm-guardrails/
├── src/
│   ├── input/           # Input validation layer
│   ├── output/          # Output validation layer
│   ├── validators/      # Reusable validation components
│   └── pipeline/        # Pipeline integration wrappers
├── examples/            # Integration examples
├── notebooks/           # Evaluation notebooks
└── tests/               # Unit tests
```
 
---
 
## Status
 
| Component | Status |
|---|---|
| Input PII detection | 🟡 In progress |
| Prompt injection detection | ⬜ Coming soon |
| Topic scope filter | ⬜ Coming soon |
| Output PII redaction | ⬜ Coming soon |
| Toxicity filter | ⬜ Coming soon |
| Topic relevance check | ⬜ Coming soon |
| Hallucination detection | ⬜ Coming soon |
| Schema validation | ⬜ Coming soon |
| Pipeline wrapper | ⬜ Coming soon |
 
---
 
*Part of the [ai-engineering-portfolio](https://github.com/pulkitkushwaha/ai-engineering-portfolio)
— built by [Pulkit Kushwaha](https://linkedin.com/in/pulkit-kushwaha)*
