#!/usr/bin/env python3
"""Summarize saved behavioral results and full-sequence Social Reasoning results.

The script recomputes descriptive and pairwise statistics from the result JSON
files included in this repository. It does not load model weights or perform
model inference. The clustered primary and Copeland sensitivity analyses are
summarized by scripts/social_reasoning/recompute_saved_results.py.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Paths (relative to this script's location in the repository)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
REFUSAL_STATS = ROOT / "results" / "behavioral_assessment" / "refusal" / "statistical_analysis.json"
STAGE2_STATS = ROOT / "results" / "behavioral_assessment" / "ai_self_identification_assistive_behavior" / "statistical_analysis.json"
SOCIAL_REASONING_STATS = ROOT / "results" / "social_reasoning" / "full_sequence_baseline" / "baseline_results.json"

# ---------------------------------------------------------------------------
# Tag -> manuscript label (mirrors manuscript Table 2's Hugging Face ID ->
# label mapping). Case-insensitive by construction -- see the module
# docstring for why that matters here.
# ---------------------------------------------------------------------------
_CANON_BY_NORMALIZED = {
    "llama-3.1-8b-base": "LlamaBase",
    "llama-3.1-8b-instruct": "LlamaInstruct",
    "qwen3-8b": "QwenInstruct",
    "qwen3-8b-base": "QwenBase",
}
DISPLAY = {
    "LlamaBase": "Llama Base",
    "LlamaInstruct": "Llama Instruct",
    "QwenInstruct": "Qwen Instruct",
    "QwenBase": "Qwen Base",
}
CANON_PAIRS = [
    ("LlamaBase", "LlamaInstruct"),
    ("LlamaBase", "QwenInstruct"),
    ("LlamaBase", "QwenBase"),
    ("LlamaInstruct", "QwenInstruct"),
    ("LlamaInstruct", "QwenBase"),
    ("QwenInstruct", "QwenBase"),
]


def short_name(tag):
    """Map any tag (any casing, with or without leading './') to its
    canonical short name, e.g. 'LlamaBase'."""
    t = tag[2:] if tag.startswith("./") else tag
    return _CANON_BY_NORMALIZED[t.lower()]


def disp(tag):
    """Map any tag to its manuscript display label, e.g. 'Llama Base'."""
    return DISPLAY[short_name(tag)]


# ---------------------------------------------------------------------------
# Shared statistical helpers
# ---------------------------------------------------------------------------
def two_prop_cohens_d(p1, p2):
    """Pooled-Bernoulli-variance Cohen's d for two proportions.

    d = (p1 - p2) / sqrt[(p1(1-p1) + p2(1-p2)) / 2]

    Sign matches (p1 - p2) by construction: report rate_diff = p1 - p2
    alongside this d and they are guaranteed to agree in sign.
    """
    s1sq, s2sq = p1 * (1 - p1), p2 * (1 - p2)
    pooled_sd = np.sqrt((s1sq + s2sq) / 2)
    return 0.0 if pooled_sd == 0 else (p1 - p2) / pooled_sd


def pairwise_rate_table(counts, title):
    """Build and print a Holm-corrected pairwise comparison table.

    counts: dict of {short_name: (successes, n)}, short_name in
    {'LlamaBase','LlamaInstruct','QwenInstruct','QwenBase'}.
    """
    rows = []
    for a, b in CANON_PAIRS:
        c1, n1 = counts[a]
        c2, n2 = counts[b]
        p1, p2 = c1 / n1, c2 / n2
        table = np.array([[c1, n1 - c1], [c2, n2 - c2]])
        chi2, p, dof, _ = chi2_contingency(table, correction=False)
        d = two_prop_cohens_d(p1, p2)
        rows.append(dict(a=a, b=b, delta=p1 - p2, chi2=chi2, p=p, d=d))
    pvals = [r["p"] for r in rows]
    reject, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    for r, ph, rj in zip(rows, p_holm, reject):
        r["p_holm"] = ph
        r["sig_holm"] = bool(rj)

    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"{'Comparison':38s} {'Delta':>9s} {'chi2':>9s} {'p_raw':>12s} {'p_holm':>12s} {'d':>8s}  sig")
    for r in rows:
        print(f"{DISPLAY[r['a']]:>15s} vs {DISPLAY[r['b']]:<15s} "
              f"{r['delta'] * 100:>+7.1f}pp {r['chi2']:>9.2f} {r['p']:>12.4g} "
              f"{r['p_holm']:>12.4g} {r['d']:>+8.3f}  {r['sig_holm']}")
    return rows


def check_ai_awareness_sign_bug(model_comparisons):
    """Informational only: documents the upstream sign-convention bug
    this script sidesteps by never reading `cohens_d` from the upstream
    JSON (see module docstring). Does not affect any computed value."""
    print(f"\n{'=' * 100}\nSanity check: upstream sign-convention bug (informational only)\n{'=' * 100}")
    mismatches = sum(
        1 for v in model_comparisons.values()
        if (v["ai_self_reference_rate_diff"] >= 0) != (v["cohens_d"] >= 0)
    )
    print(f"{mismatches}/{len(model_comparisons)} rows in statistical_analysis_stage2.json's "
          f"model_comparisons have rate_diff and cohens_d in opposite sign conventions.")
    print("This script never reads that field; Supplementary Table S7 below is computed fresh (two_prop_cohens_d).")


# ---------------------------------------------------------------------------
# Refusal (manuscript Section 6.1, Table 3; Supplementary Table S6)
# ---------------------------------------------------------------------------
def refusal_tables(stats):
    print(f"\n{'#' * 100}\n# REFUSAL (manuscript Table 3; Supplementary Table S6)\n{'#' * 100}")

    overall = {m: v["overall_harmful"] for m, v in stats["refusal_rates"].items()}

    print(f"\n{'=' * 100}\nManuscript Table 3 (rate columns). Refusal rate, all four models.\n{'=' * 100}")
    for m, v in overall.items():
        print(f"  {disp(m):18s}: {v['n_refused']:4d}/{v['n']:4d} = {v['n_refused'] / v['n'] * 100:5.2f}%")

    print(f"\n{'=' * 100}\nManuscript Table 3 (format-dependence columns), pooled across 8 categories.\n{'=' * 100}")
    pooled_fd = {}
    for m, cats in stats["format_dependence"].items():
        pooled = {}
        for catdata in cats.values():
            for fmt, fmtdata in catdata["by_format"].items():
                n = fmtdata["n"]
                refused = round(fmtdata["refusal_rate"] * n)
                pooled.setdefault(fmt, {"refused": 0, "total": 0})
                pooled[fmt]["refused"] += refused
                pooled[fmt]["total"] += n
        formats = list(pooled.keys())
        table = np.array([[pooled[f]["refused"], pooled[f]["total"] - pooled[f]["refused"]] for f in formats])
        chi2, p, dof, _ = chi2_contingency(table)
        n_dep_categories = sum(1 for cat in cats.values() if cat["format_dependent"])
        pooled_fd[short_name(m)] = (chi2, p)
        print(f"  {disp(m):18s}: pooled chi2={chi2:7.2f}  df={dof}  p={p:.4g}  "
              f"-> {'DEPENDENT' if p < 0.05 else 'not significant'}   "
              f"({n_dep_categories}/8 categories individually significant)")

    counts = {short_name(m): (v["n_refused"], v["n"]) for m, v in overall.items()}
    pairwise_rate_table(counts, "Supplementary Table S6. Pairwise comparisons, Refusal rate.")
    return pooled_fd


# ---------------------------------------------------------------------------
# AI Self-Awareness and Assistive Behavior (Section 6.1, manuscript Table 4;
# Supplementary Tables S5, S7-S9)
# ---------------------------------------------------------------------------
def stage2_tables(stats, refusal_fd):
    print(f"\n{'#' * 100}\n# AI SELF-AWARENESS AND ASSISTIVE BEHAVIOR (manuscript Table 4; Supplementary Tables S5, S7-S9)\n{'#' * 100}")

    check_ai_awareness_sign_bug(stats["model_comparisons"])

    models = list(stats["rates_by_model"].keys())

    print(f"\n{'=' * 100}\nManuscript Table 4. AI Self-Awareness and Assistive Behavior rates.\n{'=' * 100}")
    ai_counts, ab_counts = {}, {}
    for m in models:
        overall = stats["rates_by_model"][m]["overall"]
        total = overall["total"]
        ai = overall["ai_self_reference"]
        ab = overall["assistant_orientation"]
        print(f"  {disp(m):18s}: AI-Awareness {ai['count']:3d}/{total} ({ai['count'] / total * 100:5.2f}%)   "
              f"Assistive Behavior {ab['count']:3d}/{total} ({ab['count'] / total * 100:5.2f}%)")
        ai_counts[short_name(m)] = (ai["count"], total)
        ab_counts[short_name(m)] = (ab["count"], total)

    pairwise_rate_table(ai_counts, "Supplementary Table S7. Pairwise comparisons, AI Self-Awareness rate.")
    pairwise_rate_table(ab_counts, "Supplementary Table S8. Pairwise comparisons, Assistive Behavior rate.")

    print(f"\n{'=' * 100}\nSupplementary Table S9 (summary rows). Assistive Behavior rate, overall vs. assistant_orientation-category only.\n{'=' * 100}")
    for m in models:
        overall = stats["rates_by_model"][m]["overall"]["assistant_orientation"]
        cat_block = stats["rates_by_model"][m]["by_category"]["assistant_orientation"]
        cat = cat_block["assistant_orientation"]
        print(f"  {disp(m):18s}: overall {overall['rate'] * 100:5.2f}%   "
              f"category-restricted {cat['count']}/{cat_block['total']} ({cat['rate'] * 100:5.2f}%)")

    print(f"\n{'=' * 100}\nSupplementary Table S9 (format rows). Assistive Behavior occurrence by format and model.\n{'=' * 100}")
    ab_fd = {}
    for m in models:
        by_format = stats["rates_by_model"][m]["by_format"]
        counts = {}
        for fmt, d in by_format.items():
            total = d["total"]
            count = round(d["assistant_orientation_rate"] * total)
            counts[fmt] = {"count": count, "total": total}
        formats = list(counts.keys())
        table = np.array([[counts[f]["count"], counts[f]["total"] - counts[f]["count"]] for f in formats])
        chi2, p, dof, _ = chi2_contingency(table)
        ab_fd[short_name(m)] = (chi2, p)
        print(f"  {disp(m):18s}:", "  ".join(f"{f}={counts[f]['count']}/{counts[f]['total']}" for f in formats))
        print(f"  {'':18s}  chi2={chi2:.3f}  df={dof}  p={p:.4g}  "
              f"-> {'DEPENDENT' if p < 0.05 else 'not significant'}")

    print(f"\n{'=' * 100}\nManuscript Figure 2. Format-dependence across Behavioral Assessment sub-dimensions.\n{'=' * 100}")
    ai_fd = {short_name(m): (v["chi2_statistic"], v["p_value"]) for m, v in stats["format_dependence_by_model"].items()}
    print(f"  {'Model':15s} {'Refusal (chi2,p)':>22s} {'AI Self-Aware. (chi2,p)':>26s} {'Assistive Behav. (chi2,p)':>28s}")
    for short in ["LlamaBase", "LlamaInstruct", "QwenInstruct", "QwenBase"]:
        r_chi2, r_p = refusal_fd[short]
        a_chi2, a_p = ai_fd[short]
        b_chi2, b_p = ab_fd[short]
        print(f"  {DISPLAY[short]:15s} "
              f"{r_chi2:>9.2f},p={r_p:<9.3g} "
              f"{a_chi2:>9.2f},p={a_p:<9.3g}  "
              f"{b_chi2:>9.2f},p={b_p:<9.3g}")


# ---------------------------------------------------------------------------
# Social Reasoning (Section 6.2, manuscript Figure 3; Supplementary Table S10)
# ---------------------------------------------------------------------------
def social_reasoning_tables(stats):
    print(f"\n{'#' * 100}\n# SOCIAL REASONING, FULL-SEQUENCE (manuscript Figure 3; Supplementary Table S10)\n{'#' * 100}")

    print(f"\n{'=' * 100}\nManuscript Figure 3. Social Reasoning loss and perplexity.\n{'=' * 100}")
    for m, v in stats["model_results"].items():
        print(f"  {disp(m):18s}: loss={v['mean_loss']:.4f}  PPL={v['perplexity']:.2f}  "
              f"95% CI=[{v['ci_95_lower']:.4f}, {v['ci_95_upper']:.4f}]")

    print(f"\n{'=' * 100}\nSupplementary Table S10. Pairwise comparisons, Social Reasoning full-sequence loss.\n{'=' * 100}")
    rows = list(stats["pairwise_statistics"].values())
    pvals = [v["p_value"] for v in rows]
    reject, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    print(f"{'Comparison':38s} {'Delta':>9s} {'d':>8s} {'p_raw':>12s} {'p_holm':>12s}  sig")
    for v, ph, rj in zip(rows, p_holm, reject):
        print(f"{disp(v['model1']):>15s} vs {disp(v['model2']):<15s} {v['mean_loss_diff']:>+9.4f} "
              f"{v['cohens_d']:>+8.3f} {v['p_value']:>12.4g} {ph:>12.4g}  {bool(rj)}")
    print("\n  These values summarize the full-sequence baseline result set. "
          "The primary clustered and Copeland single-reference results are "
          "summarized by scripts/social_reasoning/recompute_saved_results.py.")


# ---------------------------------------------------------------------------
def main():
    with open(REFUSAL_STATS) as f:
        refusal_stats = json.load(f)
    with open(STAGE2_STATS) as f:
        stage2_stats = json.load(f)
    with open(SOCIAL_REASONING_STATS) as f:
        sr_stats = json.load(f)

    refusal_fd = refusal_tables(refusal_stats)
    stage2_tables(stage2_stats, refusal_fd)
    social_reasoning_tables(sr_stats)

    print(f"\n{'#' * 100}")
    print("# Done. Behavioral summaries and full-sequence baseline statistics were computed")
    print("# from saved result files. Primary clustered and Copeland results are audited by")
    print("# scripts/social_reasoning/recompute_saved_results.py.")
    print(f"{'#' * 100}")


if __name__ == "__main__":
    main()
