#!/usr/bin/env python3
"""Generate figures for the full-sequence Social Reasoning baseline.

This script reads the saved baseline result JSON and writes a PNG and PDF for
its model-mean loss and paired-effect displays. It performs no model inference.
The clustered primary and Copeland sensitivity analyses have a separate script.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "social_reasoning" / "full_sequence_baseline" / "baseline_results.json"
OUTPUT_DIR = ROOT / "figures" / "social_reasoning_baseline"

# Display order for the full-sequence baseline figures.
DISPLAY_ORDER = ["LlamaBase", "LlamaInstruct", "QwenInstruct", "QwenBase"]
DISPLAY = {
    "LlamaBase": "Llama\nBase",
    "LlamaInstruct": "Llama\nInstruct",
    "QwenInstruct": "Qwen\nInstruct",
    "QwenBase": "Qwen\nBase",
}
_CANON_BY_NORMALIZED = {
    "llama-3.1-8b-base": "LlamaBase",
    "llama-3.1-8b-instruct": "LlamaInstruct",
    "qwen3-8b": "QwenInstruct",
    "qwen3-8b-base": "QwenBase",
}


def short_name(tag):
    t = tag[2:] if tag.startswith("./") else tag
    return _CANON_BY_NORMALIZED[t.lower()]


def figure3_loss_by_model(model_results):
    """Reproduce Figure 3: full-sequence loss by model, with 95% CI."""
    by_short = {short_name(tag): v for tag, v in model_results.items()}

    means = [by_short[s]["mean_loss"] for s in DISPLAY_ORDER]
    lo = [by_short[s]["ci_95_lower"] for s in DISPLAY_ORDER]
    hi = [by_short[s]["ci_95_upper"] for s in DISPLAY_ORDER]
    err_lo = [m - l for m, l in zip(means, lo)]
    err_hi = [h - m for m, h in zip(means, hi)]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.arange(len(DISPLAY_ORDER))
    bars = ax.bar(x, means, yerr=[err_lo, err_hi], capsize=4, width=0.55,
                   color="#7A9CC6", edgecolor="black", linewidth=0.8,
                   error_kw=dict(elinewidth=1.1, ecolor="black"))

    for xi, m, l, h in zip(x, means, lo, hi):
        ax.text(xi + 0.14, h + 0.05, f"{m:.3f} [{h:.3f}, {l:.3f}]",
                 rotation=35, ha="left", va="bottom", fontsize=8)

    ax.set_title("Social Reasoning loss, ProlificAI", fontsize=13, fontweight="bold")
    ax.set_ylabel("loss [95% CI]")
    ax.set_ylim(0, 3.0)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[s] for s in DISPLAY_ORDER])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def figure4_effect_size_heatmap(model_results, pairwise_statistics):
    """Reproduce Figure 4: lower-triangular Cohen's d heatmap.

    Uses the manuscript's verified-correct sign convention: cell(row, col)
    = d(row) - d(col) on loss, so a positive value means the row model has
    the higher loss (worse Social Reasoning) than the column model -- this
    matches the LaTeX caption of the manuscript's Figure 4. The title text here follows the manuscript caption and sign convention.
    """
    # Raw tags, in the same order and spelling as the manuscript's Figure 4.
    row_tags = ["Qwen3-8B-Base", "llama-3.1-8b-base", "llama-3.1-8B-instruct"]
    col_tags = ["Qwen3-8B", "Qwen3-8B-Base", "llama-3.1-8b-base"]

    # Look up each pairwise comparison regardless of which of the two tags
    # the source file happened to store it under, and orient the sign to
    # (row - col).
    def signed_d(row_tag, col_tag):
        for key, v in pairwise_statistics.items():
            m1, m2 = v["model1"], v["model2"]
            if {m1, m2} == {row_tag, col_tag}:
                d = v["cohens_d"]  # stored as model1 - model2
                return d if m1 == row_tag else -d
        return None

    # row_tags[i] and col_tags[j] are offset by one position in the shared
    # canonical order [Qwen Instruct, Qwen Base, Llama Base, Llama Instruct]
    # (row_tags = positions 1..3, col_tags = positions 0..2), so the valid,
    # single-appearance-per-pair lower-triangular cells are exactly j <= i.
    grid = np.full((len(row_tags), len(col_tags)), np.nan)
    for i, rt in enumerate(row_tags):
        for j, ct in enumerate(col_tags):
            if j <= i:
                grid[i, j] = signed_d(rt, ct)

    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    vmax = float(np.nanmax(np.abs(grid))) * 1.05
    im = ax.imshow(np.ma.masked_invalid(np.abs(grid)), cmap=cmap, vmin=0, vmax=vmax, aspect="equal")

    for i in range(len(row_tags)):
        for j in range(len(col_tags)):
            if not np.isnan(grid[i, j]):
                mag = abs(grid[i, j])
                txt_color = "white" if mag > vmax * 0.55 else "black"
                ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center",
                         color=txt_color, fontsize=11)

    ax.set_xticks(range(len(col_tags)))
    ax.set_xticklabels(col_tags, rotation=0)
    ax.set_yticks(range(len(row_tags)))
    ax.set_yticklabels(row_tags)
    ax.set_title("Effect Sizes of Pairwise Comparisons", fontsize=13, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's $d$ (magnitude)")

    fig.tight_layout()
    return fig


def main():
    with open(RESULTS) as f:
        data = json.load(f)

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig3 = figure3_loss_by_model(data["model_results"])
    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"figure3_social_reasoning_loss.{ext}"
        fig3.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path.relative_to(ROOT).as_posix()}")
    plt.close(fig3)

    fig4 = figure4_effect_size_heatmap(data["model_results"], data["pairwise_statistics"])
    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"figure4_effect_size_heatmap.{ext}"
        fig4.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path.relative_to(ROOT).as_posix()}")
    plt.close(fig4)

    print("\nDone. Generated figures for the full-sequence Social Reasoning baseline.")
    print("For the primary clustered and Copeland sensitivity designs, run")
    print("generate_social_reasoning_figures.py.")


if __name__ == "__main__":
    main()
