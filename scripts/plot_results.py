"""
Figure generation for the README.

Goal: build every chart from the JSON files in results/, never by hand. When a
new configuration is evaluated, rerunning this script updates the figures, and
nobody has to trust that the numbers in the README match the ones measured.

Outputs, in docs/:
    retrieval_ablation.png   nDCG and recall per retrieval configuration
    quality_vs_cost.png      nDCG against total latency, log scale
    prompt_effect.png        answer metrics before and after the prompt fix

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
    """Retrieval runs, ordered by score so the progression reads left to right."""
    runs = []
    for path in RESULTS.glob("ollama*.json"):
        data = json.load(path.open())
        cfg = data["config"]
        label = cfg["strategy"] + (" + rerank" if cfg.get("rerank") else "")
        runs.append({
            "label": label,
            "ndcg": data["metrics"]["ndcg@10"],
            "r10": data["metrics"]["recall@10"],
            "r50": data["metrics"]["recall@50"],
            "elapsed": data["elapsed_s"],
        })
    return sorted(runs, key=lambda r: r["ndcg"])


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
    ax.set_title("Retrieval quality by configuration (NFCorpus, 323 queries)", loc="left")
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


def plot_prompt_effect() -> None:
    """Before and after the citation rule was tightened."""
    before = json.load((RESULTS / "answers-strict-v1.json").open())["summary"]
    after = json.load((RESULTS / "answers-strict-v2.json").open())["summary"]

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
    plot_ablation(runs)
    plot_quality_vs_cost(runs)
    plot_prompt_effect()
    print(f"wrote 3 figures to {DOCS}")


if __name__ == "__main__":
    main()