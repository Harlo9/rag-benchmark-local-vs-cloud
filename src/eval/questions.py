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

# Robustness set: the same 10 answerable questions, deliberately mis-phrased.
#
# The out-of-domain set above tests whether the system refuses what it cannot
# answer. This one tests the opposite failure: refusing something it CAN answer
# because the wording degraded retrieval. In production that is the failure that
# kills adoption, since users reformulate twice and then stop asking.
#
# Three styles, each stressing a different part of the pipeline:
#   vague       underspecified wording, weak lexical signal for BM25
#   colloquial  spoken register and typos, unseen surface forms for the embedder
#   verbose     the real question buried in irrelevant context

PARAPHRASED = {
    "does vitamin d reduce mortality": {
        "vague": "is vitamin d actually useful",
        "colloquial": "does vit D realy help peopl live longer",
        "verbose": "my grandmother's doctor mentioned supplements last week and I was wondering whether taking vitamin D has any effect on how long people live",
    },
    "is red meat linked to colorectal cancer": {
        "vague": "is meat bad for you",
        "colloquial": "does eating red meat gives u colon cancer",
        "verbose": "we had a family argument about barbecues and someone said red meat causes bowel cancer, is there evidence for that",
    },
    "what are the effects of dietary fiber on cholesterol": {
        "vague": "does fiber do anything",
        "colloquial": "whats fibre do to cholestrol",
        "verbose": "my nutritionist keeps telling me to eat more fibre and I want to understand what it actually changes about cholesterol levels",
    },
    "are artificial sweeteners linked to diabetes": {
        "vague": "are sweeteners risky",
        "colloquial": "is aspartame gonna give me diabetes",
        "verbose": "I switched from sugar to sweeteners in my coffee two years ago and now I read they might be connected to diabetes, what does the research say",
    },
    "what is the effect of soy on breast cancer risk": {
        "vague": "is soy safe",
        "colloquial": "does soya effect breast cancer",
        "verbose": "a friend recovering from breast cancer was told to avoid tofu entirely and I would like to know whether soy consumption changes the risk",
    },
    "does coffee consumption affect liver disease": {
        "vague": "is coffee good or bad",
        "colloquial": "coffe and liver, good or no",
        "verbose": "I drink four cups a day and my last blood test showed elevated liver enzymes, is there a documented link between coffee and liver disease",
    },
    "is there a link between dairy and prostate cancer": {
        "vague": "is milk a problem",
        "colloquial": "does drinkin milk cause prostrate cancer",
        "verbose": "my father was recently diagnosed and someone in the waiting room claimed dairy was involved, what do the studies actually show",
    },
    "what are the cardiovascular effects of omega 3": {
        "vague": "are fish oils worth it",
        "colloquial": "omega3 good for the heart or nah",
        "verbose": "the pharmacy sells omega 3 capsules at the counter with a heart symbol on the box and I want to know what effect they actually have on the cardiovascular system",
    },
    "does green tea help with weight loss": {
        "vague": "does tea burn fat",
        "colloquial": "green tea for loosing weight, works?",
        "verbose": "I keep seeing green tea extract in every fat burner on the shelf and I want to know whether it does anything for weight loss",
    },
    "does fasting improve insulin sensitivity": {
        "vague": "is fasting healthy",
        "colloquial": "does skipin meals help with insuline",
        "verbose": "I have been doing 16 8 intermittent fasting for three months and I want to understand its effect on insulin sensitivity",
    },
}