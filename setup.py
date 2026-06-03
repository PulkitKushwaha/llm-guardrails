from setuptools import setup, find_packages
 
setup(
    name="llm-guardrails",
    version="0.1.0",
    author="Pulkit Kushwaha",
    author_email="pulkitkushwahadev@gmail.com",
    description="Production LLM guardrails — input/output validation, PII detection, and topic filtering for enterprise AI pipelines",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.14.0",
        "pydantic>=2.6.4",
        "presidio-analyzer>=2.2.354",
        "presidio-anonymizer>=2.2.354",
        "transformers>=4.39.3",
        "regex>=2024.4.16",
    ],
)
