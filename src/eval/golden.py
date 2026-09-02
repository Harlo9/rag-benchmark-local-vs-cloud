"""
Golden dataset: expected answer points per question.

Goal: measure completeness, which faithfulness cannot see. A faithful answer can
still report one of the four things its sources say. Faithfulness judges what was
said; coverage judges what was left out.

Every point is stated by a passage retrieved under the SAME configuration used
for evaluation (hybrid + rerank + MMR). That matters: an earlier draft was
written from a truncated dump under a different configuration, and scored
correct answers as misses because the reference had never seen the full passage.

`absences` are claims a correct answer should NOT make: the corpus supports the
surrounding context but not the conclusion. Stating an absence is a correct
answer, not a missing point.
"""

GOLDEN: dict[str, dict] = {
    "does vitamin d reduce mortality": {
        "points": [
            "overall vitamin D decreased mortality, RR 0.97 (95% CI 0.94 to 1.00)",
            "only vitamin D3 decreased mortality significantly, RR 0.94 (95% CI 0.91 to 0.98)",
            "vitamin D2, alfacalcidol and calcitriol had no statistically significant effect",
            "the trials mostly involved elderly women, largely in institutions or dependent care",
            "vitamin D3 combined with calcium increased the risk of nephrolithiasis",
        ],
        "absences": [],
    },
    "is red meat linked to colorectal cancer": {
        "points": [
            "high intake of red and processed meat increases colorectal cancer risk",
            "summary RR 1.22 for highest versus lowest intake, and 1.14 per 100 g per day",
            "risk increases roughly linearly up to about 140 g per day, then plateaus",
            "fresh red meat RR 1.17 per 100 g per day, processed meat RR 1.18 per 50 g per day",
            "proposed mechanisms include heterocyclic amines, polycyclic aromatic hydrocarbons, N-nitroso compounds and heme iron",
        ],
        "absences": [],
    },
    "what is the effect of soy on breast cancer risk": {
        # The corpus contains two findings that disagree on menopausal status.
        # A good answer reports the inconsistency rather than picking a side, so
        # both are expected points and neither is required to exclude the other.
        "points": [
            "soy intake is inversely associated with breast cancer risk, OR 0.36 for highest versus lowest quartile",
            "one study found the protective effect only among postmenopausal women, OR 0.08",
            "another study found reduced risk among premenopausal women, RR 0.57, and no significant association postmenopause",
            "studies report no adverse effect of soy on breast cancer prognosis in survivors",
        ],
        "absences": [],
    },
    "does coffee consumption affect liver disease": {
        "points": [
            "coffee intake is inversely associated with incident liver cancer, RR 0.82 per cup per day",
            "chronic liver disease mortality was lower among coffee drinkers, RR 0.55",
            "hepatocellular carcinoma risk is reduced by about 40 percent for any coffee consumption",
            "the inverse association held regardless of sex, alcohol intake or hepatitis history",
            "similar associations were found for boiled and filtered coffee",
        ],
        "absences": [],
    },
    "is there a link between dairy and prostate cancer": {
        "points": [
            "cohort meta-analysis found a positive association, summary RR 1.13",
            "case-control meta-analysis found a stronger association, combined OR 1.68",
            "a dose-response relationship was identified",
            "proposed mechanism involves IGF-1 and steroid hormones present in milk",
            "cow's milk stimulated growth of LNCaP prostate cancer cells in vitro by over 30 percent",
        ],
        "absences": [],
    },
    "what are the cardiovascular effects of omega 3": {
        "points": [
            "no statistically significant association was observed with all-cause mortality",
            "a small reduction in cardiovascular death was observed, RR 0.91",
            "that reduction disappeared once a methodologically flawed study was excluded",
            "the meta-analysis found insufficient evidence of a secondary preventive effect on cardiovascular events",
        ],
        "absences": [],
    },
    "does green tea help with weight loss": {
        "points": [
            "laboratory studies have shown potential efficacy of green or black tea for obesity prevention",
            "results of human intervention studies are mixed",
            "the underlying mechanisms remain unclear and the role of caffeine is not established",
            "high doses of tea polyphenols may have adverse side effects",
        ],
        "absences": [],
    },
    "are artificial sweeteners linked to diabetes": {
        "points": [
            "artificial sweeteners provide sweetness without calories and may suit people who cannot tolerate sugar, such as diabetics",
            "these substances have received increased attention for their effects on glucose regulation",
            "scientists disagree about the safety of artificial sweeteners",
        ],
        "absences": [
            "that artificial sweeteners cause or increase the risk of developing diabetes",
        ],
    },
    "what are the effects of dietary fiber on cholesterol": {
        "points": [
            "water-soluble fibres showed higher in vitro binding capacity for cholesterol and bile acids than insoluble ones",
            "a dietary portfolio including viscous fibre reduced LDL cholesterol by about 13 percent",
        ],
        "absences": [
            "that wheat fibre's protective effect in cardiovascular disease is explained by lowering serum cholesterol",
        ],
    },
}

# No retrieved passage discusses fasting as a practice: the term appears only in
# "fasting glucose" and "fasting insulin", which name a measurement taken in the
# fasted state. BM25 matched the surface form, not the meaning. Declining is the
# correct behaviour, so this question has no expected points.
UNANSWERABLE = ["does fasting improve insulin sensitivity"]