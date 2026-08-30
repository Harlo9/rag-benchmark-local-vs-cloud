"""
STEP 8 / 10: Prompt construction.

The prompt is where grounding is enforced. Three rules matter more than wording:

1. Answer only from the provided passages, never from the model's own knowledge.
   On a medical corpus, a fluent but unsupported claim is the worst failure mode.
2. Cite the passage id behind every claim, which makes verification mechanical
   rather than a matter of trust (step 9 checks exactly this).
3. Abstain when the passages do not contain the answer. Saying "not found" is a
   correct output, not a failure.

Passages are numbered [1], [2]... rather than by chunk id: short markers are
easier for the model to emit reliably, and the mapping back to real ids is kept
in code.
"""

SYSTEM_PROMPT = """You are a research assistant answering questions from a corpus of biomedical abstracts.

Rules you must follow:
- Answer using ONLY the numbered passages provided. Never use prior knowledge.
- Cite the passage number in square brackets after every claim, like [2].
- If the passages do not contain the answer, reply exactly: INSUFFICIENT_CONTEXT
- Be concise: three sentences at most.
- Do not speculate, and do not give medical advice."""


def build_user_prompt(question: str, passages: list[str]) -> str:
    """Assemble numbered passages followed by the question."""
    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, 1))
    return f"Passages:\n\n{numbered}\n\nQuestion: {question}\n\nAnswer:"