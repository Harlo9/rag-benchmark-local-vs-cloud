"""
Figure generation for the README.

Goal: build every chart from the JSON files in results/, never by hand. When a
new configuration is evaluated, rerunning this script updates the figures, and
nobody has to trust that the numbers in the README match the ones measured.

Outputs, in docs/:
    retrieval_ablation.png    nDCG and recall per retrieval configuration (local)
    quality_vs_cost.png       nDCG against total latency, log scale (local)
    backend_comparison.png    same strategies, local embedder against hosted one
    answer_comparison.png     answer-level metrics, local model against hosted
    prompt_effect.png         answer metrics before and after the prompt fix

Note on latency: all retrieval runs must be measured back to back with the
models already loaded, otherwise a cold start inflates whichever ran first and
the configurations stop being comparable.

Usage: python -m scripts.plot_results
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed: we only write files
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

# Muted palette, readable in both GitHub themes and when printed greyscale.
INK, ACCENT, WARM, GREY = "#1b2a33", "#2b6c7f", "#c08a2e", "#9aa5a3"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GREY,
    "figure.dpi": 160,
})


def load_retrieval() -> list[dict]:
    """All retrieval runs, tagged by backend so local and cloud can be compared."""
    runs = []
    for path in RESULTS.glob("*.json"):
        if path.name.startswith("answers-"):
            continue
        data = json.load(path.open())
        cfg = data["config"]
        backend = "azure" if cfg["embedder"].startswith("azure") else "local"
        label = cfg["strategy"] + (" + rerank" if cfg.get("rerank") else "")
        runs.append({
            "backend": backend,
            "label": label,
            "ndcg": data["metrics"]["ndcg@10"],
            "r10": data["metrics"]["recall@10"],
            "r50": data["metrics"]["recall@50"],
            "elapsed": data["elapsed_s"],
        })
    return runs


def plot_ablation(runs: list[dict]) -> None:
    """Grouped bars: what each configuration scores on each metric."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = range(len(runs))
    width = 0.26

    for offset, key, name, color in [
        (-width, "ndcg", "nDCG@10", ACCENT),
        (0.0, "r10", "Recall@10", INK),
        (width, "r50", "Recall@50", GREY),
    ]:
        values = [r[key] for r in runs]
        bars = ax.bar([i + offset for i in x], values, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=2, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in runs])
    ax.set_ylabel("score")
    ax.set_ylim(0, 0.42)
    ax.set_title("Retrieval quality by configuration (local, NFCorpus)", loc="left")
    ax.legend(frameon=False, ncols=3, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "retrieval_ablation.png")
    plt.close(fig)


def plot_quality_vs_cost(runs: list[dict]) -> None:
    """The trade-off chart: quality against what it costs to get it."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for run in runs:
        ax.scatter(run["elapsed"], run["ndcg"], s=70, color=ACCENT, zorder=3)
        ax.annotate(f"{run['label']}\n{run['elapsed']:.1f} s",
                    (run["elapsed"], run["ndcg"]),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=7.5, color=INK)

    # Log scale: latencies span two orders of magnitude.
    ax.set_xscale("log")
    ax.set_xlabel("total time for 323 queries (s, log scale)")
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(min(r["ndcg"] for r in runs) - 0.03, max(r["ndcg"] for r in runs) + 0.05)
    ax.set_title("Quality against cost: BM25 sits on the efficient frontier", loc="left")
    ax.grid(axis="y", color=GREY, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(DOCS / "quality_vs_cost.png")
    plt.close(fig)


def plot_backend_comparison(runs: list[dict]) -> None:
    """Same four strategies, two embedding backends, side by side.

    The point this figure makes: the embedding model matters more than the
    retrieval strategy. Stacking hybrid and reranking compensates for a weak
    embedder, and stops paying off once the embedder is strong.
    """
    order = ["dense", "bm25", "hybrid", "hybrid + rerank"]
    by_backend = {
        b: {r["label"]: r["ndcg"] for r in runs if r["backend"] == b}
        for b in ("local", "azure")
    }

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = range(len(order))
    width = 0.36

    for offset, backend, name, color in [
        (-width / 2, "local", "local (qwen3-embedding:0.6b)", GREY),
        (width / 2, "azure", "azure (text-embedding-3-small)", ACCENT),
    ]:
        values = [by_backend[backend].get(label, 0) for label in order]
        bars = ax.bar([i + offset for i in x], values, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=2, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 0.45)
    ax.set_title("Embedding backend matters more than retrieval strategy", loc="left")
    ax.legend(frameon=False, ncols=2, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(DOCS / "backend_comparison.png")
    plt.close(fig)


def plot_answer_comparison() -> None:
    """Answer-level metrics, local model against hosted model.

    Both were measured after the citation-parsing fix, so the numbers are
    comparable: an earlier version of the guardrail reported valid grouped
    citations as uncited, which unfairly penalised the hosted model.
    """
    local = json.load((RESULTS / "answers-local-strict-v2.json").open())["summary"]
    azure = json.load((RESULTS / "answers-azure-strict-v2.json").open())["summary"]

    metrics = [
        ("hallucination_rate", "Uncited claims\n(lower is better)"),
        ("guardrail_pass", "Guardrails\npassed"),
        ("faithfulness", "Faithfulness"),
        ("abstention_rate_ood", "Declines\nout of domain"),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    x = range(len(metrics))
    width = 0.36

    for offset, data, name, color in [
        (-width / 2, local, "local (qwen2.5:3b)", GREY),
        (width / 2, azure, "azure (gpt-5-mini)", ACCENT),
    ]:
        values = [data[key] or 0 for key, _ in metrics]
        bars = ax.bar([i + offset for i in x], values, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=2, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics], fontsize=7.5)
    ax.set_ylim(0, 1.2)
    ax.set_title("Answer quality: local model against hosted model", loc="left")
    ax.legend(frameon=False, ncols=2, loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(DOCS / "answer_comparison.png")
    plt.close(fig)


def plot_prompt_effect() -> None:
    """Before and after the citation rule was tightened.

    Both runs were re-measured after the citation-parsing fix, so the comparison
    isolates the prompt change rather than mixing it with a harness change.
    """
    before = json.load((RESULTS / "answers-local-strict-v1.json").open())["summary"]
    after = json.load((RESULTS / "answers-local-strict-v2.json").open())["summary"]

    # Labels spell out the direction of improvement: two of these metrics are
    # good when they rise, one is good when it falls, and one is only correct
    # because the corpus genuinely lacks those answers.
    metrics = [
        ("hallucination_rate", "Uncited claims\n(lower is better)"),
        ("guardrail_pass", "Guardrails\npassed"),
        ("faithfulness", "Faithfulness"),
        ("abstention_rate_ood", "Declines\nout of domain"),
        ("abstention_rate_in_domain", "Declines when corpus\nlacks the answer"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = range(len(metrics))
    width = 0.36

    for offset, data, name, color in [
        (-width / 2, before, "before", GREY),
        (width / 2, after, "after", ACCENT),
    ]:
        values = [data[key] or 0 for key, _ in metrics]
        bars = ax.bar([i + offset for i in x], values, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=2, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics], fontsize=7.5)
    ax.set_ylim(0, 1.15)
    ax.set_title("Effect of enforcing a citation on every factual claim", loc="left")
    ax.legend(frameon=False, ncols=2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "prompt_effect.png")
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    runs = load_retrieval()

    # The ablation figures stay local-only: mixing backends would confound two
    # variables at once. The comparison gets its own figure.
    local = sorted([r for r in runs if r["backend"] == "local"], key=lambda r: r["ndcg"])
    plot_ablation(local)
    plot_quality_vs_cost(local)

    plot_backend_comparison(runs)
    plot_answer_comparison()
    plot_prompt_effect()
    print(f"wrote 5 figures to {DOCS}")


if __name__ == "__main__":
    main()