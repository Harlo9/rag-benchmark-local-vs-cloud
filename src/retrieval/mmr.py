"""
STEP 8 / 10: Maximal Marginal Relevance.

Goal: pick a diverse set of passages for the LLM context, instead of the top-k
most relevant ones which often say the same thing.

Trade-off, made explicit: MMR usually *lowers* ranking metrics like nDCG,
because relevance is all those metrics measure. It is applied here at generation
time only, where the objective is different: five passages from one document
give the model one source, while five passages from five documents give it five.

lambda_ balances the two forces: 1.0 is pure relevance (no diversification),
0.0 is pure novelty. 0.7 keeps relevance dominant while breaking up duplicates.
"""

import numpy as np


def mmr_select(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int,
               lambda_: float = 0.7) -> list[int]:
    """Return indices of k candidates, greedily balancing relevance and novelty."""
    relevance = cand_vecs @ query_vec          # vectors are already normalised
    similarity = cand_vecs @ cand_vecs.T       # pairwise similarity between candidates

    selected: list[int] = [int(np.argmax(relevance))]

    while len(selected) < min(k, len(cand_vecs)):
        # Penalty = how close a candidate is to whatever is already selected.
        redundancy = similarity[:, selected].max(axis=1)
        score = lambda_ * relevance - (1 - lambda_) * redundancy
        score[selected] = -np.inf              # never pick the same one twice
        selected.append(int(np.argmax(score)))

    return selected