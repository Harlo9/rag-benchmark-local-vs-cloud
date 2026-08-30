"""
STEP 9 / 10: Evaluation question sets.

Two sets, testing opposite behaviours:

IN_DOMAIN     questions the corpus can answer. The system should answer and cite.
OUT_OF_DOMAIN questions the corpus cannot answer. The system should abstain.

The second set is the interesting one. Retrieval always returns something, even
for a question about football, because nearest-neighbour search has no notion of
"nothing is relevant". Whether the system notices is a property of the prompt and
the guardrails, and it is exactly what separates a demo from a usable system.
"""

IN_DOMAIN = [
    "does vitamin d reduce mortality",
    "is red meat linked to colorectal cancer",
    "what are the effects of dietary fiber on cholesterol",
    "does green tea help with weight loss",
    "are artificial sweeteners linked to diabetes",
    "what is the effect of soy on breast cancer risk",
    "does coffee consumption affect liver disease",
    "is there a link between dairy and prostate cancer",
    "what are the cardiovascular effects of omega 3",
    "does fasting improve insulin sensitivity",
]

OUT_OF_DOMAIN = [
    "how do I renew my french passport",
    "who won the 2014 football world cup",
    "what is the capital of mongolia",
    "how do I configure a kubernetes ingress",
    "what is the population of Lisbon",
]