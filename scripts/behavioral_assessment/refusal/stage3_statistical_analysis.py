#!/usr/bin/env python3
"""
Stage 3: Statistical Analysis

Performs rigorous statistical analysis on evaluation results:
1. Hypothesis testing (McNemar's test, Chi-square)
2. Effect size calculations (Cohen's d, Cramér's V)
3. Statistical power analysis
4. Multiple testing corrections (Bonferroni)
5. Format-dependence analysis

Outputs: statistical_analysis.json + analysis_report.txt

Usage:
  python stage3_statistical_analysis.py --results model_evaluations.json

Requirements:
  pip install scipy numpy statsmodels pandas
"""

import json, argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.contingency_tables import mcnemar

# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cramers_v(contingency_table: np.ndarray) -> float:
    """Calculate Cramér's V effect size for categorical data."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    
    if n == 0 or min_dim == 0:
        return 0.0
    
    return np.sqrt(chi2 / (n * min_dim))


def mcnemar_test(data: List[Tuple[bool, bool]]) -> Dict:
    """
    Perform McNemar's test for paired binary data.
    Each tuple is (model1_refused, model2_refused) for the same prompt.
    """
    # Build contingency table
    both_refuse = sum(1 for m1, m2 in data if m1 and m2)
    both_comply = sum(1 for m1, m2 in data if not m1 and not m2)
    m1_only = sum(1 for m1, m2 in data if m1 and not m2)
    m2_only = sum(1 for m1, m2 in data if not m1 and m2)
    
    table = np.array([[both_refuse, m1_only],
                      [m2_only, both_comply]])
    
    # McNemar's test (using exact test for small samples)
    result = mcnemar(table, exact=True)
    
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "table": table.tolist(),
        "significant": result.pvalue < 0.05,
    }


def power_analysis_binomial(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    """
    Calculate statistical power for detecting difference between two proportions.
    
    p1, p2: proportions for group 1 and 2
    n: sample size per group
    alpha: significance level
    """
    effect_size = cohens_h(p1, p2)
    
    try:
        power = tt_ind_solve_power(
            effect_size=effect_size,
            nobs1=n,
            alpha=alpha,
            ratio=1.0,
            alternative='two-sided',
        )
        return power
    except:
        return np.nan


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for difference between proportions."""
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return abs(phi1 - phi2)


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def extract_refusal_rates(results: Dict) -> Dict:
    """Extract refusal rates by category for each model."""
    
    rates = {}
    
    for model_path, data in results.items():
        model_rates = {
            "harmful": {},
            "harmless": None,
            "overall_harmful": None,
        }
        
        # Per-category rates
        total_harmful = 0
        total_harmful_refused = 0
        
        for category, evals in data["harmful"].items():
            n = len(evals)
            n_refused = sum(1 for e in evals if e["scores"]["binary_refusal"])
            rate = n_refused / n if n > 0 else 0
            
            model_rates["harmful"][category] = {
                "n": n,
                "n_refused": n_refused,
                "rate": rate,
            }
            
            total_harmful += n
            total_harmful_refused += n_refused
        
        # Overall harmful rate
        model_rates["overall_harmful"] = {
            "n": total_harmful,
            "n_refused": total_harmful_refused,
            "rate": total_harmful_refused / total_harmful if total_harmful > 0 else 0,
        }
        
        # Harmless rate
        evals = data["harmless"]
        n = len(evals)
        n_refused = sum(1 for e in evals if e["scores"]["binary_refusal"])
        
        model_rates["harmless"] = {
            "n": n,
            "n_refused": n_refused,
            "rate": n_refused / n if n > 0 else 0,
        }
        
        rates[model_path] = model_rates
    
    return rates


def compare_two_models(
    results: Dict,
    model1_path: str,
    model2_path: str,
) -> Dict:
    """Detailed statistical comparison between two models."""
    
    model1 = results[model1_path]
    model2 = results[model2_path]
    
    comparison = {
        "model1": model1_path,
        "model2": model2_path,
        "harmful": {},
        "harmless": None,
    }
    
    # Compare harmful prompts by category
    for category in model1["harmful"].keys():
        evals1 = model1["harmful"][category]
        evals2 = model2["harmful"][category]
        
        # Refusal rates
        rate1 = np.mean([e["scores"]["binary_refusal"] for e in evals1])
        rate2 = np.mean([e["scores"]["binary_refusal"] for e in evals2])
        
        # Refusal scores (continuous)
        scores1 = np.array([e["scores"]["refusal_score"] for e in evals1])
        scores2 = np.array([e["scores"]["refusal_score"] for e in evals2])
        
        # T-test on continuous scores
        t_stat, t_pval = stats.ttest_ind(scores1, scores2)
        
        # Chi-square on binary outcomes
        n1 = len(evals1)
        n2 = len(evals2)
        n1_refused = int(rate1 * n1)
        n2_refused = int(rate2 * n2)
        
        contingency = np.array([
            [n1_refused, n1 - n1_refused],
            [n2_refused, n2 - n2_refused],
        ])
        
        chi2, chi2_pval, _, _ = stats.chi2_contingency(contingency)
        cramers = cramers_v(contingency)
        
        # Effect size
        effect = cohens_d(scores1, scores2)
        
        # Power analysis
        power = power_analysis_binomial(rate1, rate2, min(n1, n2))
        
        comparison["harmful"][category] = {
            "n_prompts": n1,
            "refusal_rate_model1": float(rate1),
            "refusal_rate_model2": float(rate2),
            "rate_difference": float(rate1 - rate2),
            "ttest_statistic": float(t_stat),
            "ttest_pvalue": float(t_pval),
            "chi2_statistic": float(chi2),
            "chi2_pvalue": float(chi2_pval),
            "cohens_d": float(effect),
            "cramers_v": float(cramers),
            "statistical_power": float(power) if not np.isnan(power) else None,
        }
    
    # Compare harmless prompts
    evals1 = model1["harmless"]
    evals2 = model2["harmless"]
    
    rate1 = np.mean([e["scores"]["binary_refusal"] for e in evals1])
    rate2 = np.mean([e["scores"]["binary_refusal"] for e in evals2])
    
    scores1 = np.array([e["scores"]["refusal_score"] for e in evals1])
    scores2 = np.array([e["scores"]["refusal_score"] for e in evals2])
    
    t_stat, t_pval = stats.ttest_ind(scores1, scores2)
    
    n1, n2 = len(evals1), len(evals2)
    n1_refused = int(rate1 * n1)
    n2_refused = int(rate2 * n2)
    
    contingency = np.array([
        [n1_refused, n1 - n1_refused],
        [n2_refused, n2 - n2_refused],
    ])
    
    chi2, chi2_pval, _, _ = stats.chi2_contingency(contingency)
    cramers = cramers_v(contingency)
    effect = cohens_d(scores1, scores2)
    power = power_analysis_binomial(rate1, rate2, min(n1, n2))
    
    comparison["harmless"] = {
        "n_prompts": n1,
        "refusal_rate_model1": float(rate1),
        "refusal_rate_model2": float(rate2),
        "rate_difference": float(rate1 - rate2),
        "ttest_statistic": float(t_stat),
        "ttest_pvalue": float(t_pval),
        "chi2_statistic": float(chi2),
        "chi2_pvalue": float(chi2_pval),
        "cohens_d": float(effect),
        "cramers_v": float(cramers),
        "statistical_power": float(power) if not np.isnan(power) else None,
    }
    
    return comparison


def format_dependence_analysis(results: Dict, model_path: str) -> Dict:
    """Analyze if refusal depends on prompt format/variation."""
    
    model_data = results[model_path]
    
    # Extract format-specific refusal rates
    format_analysis = {}
    
    for category, evals in model_data["harmful"].items():
        # Group by variation template
        by_format = defaultdict(list)
        
        for e in evals:
            prompt = e["prompt"]
            refused = e["scores"]["binary_refusal"]
            
            # Infer format from prompt structure
            if prompt.endswith("Step 1:"):
                fmt = "step_format"
            elif prompt.endswith(":\n1."):
                fmt = "numbered_list"
            elif prompt.startswith("Write a detailed guide:"):
                fmt = "guide_frame"
            elif prompt.startswith("Provide instructions:"):
                fmt = "instruction_frame"
            elif prompt.startswith("Explain how to:"):
                fmt = "explanation_frame"
            elif "(for educational purposes)" in prompt:
                fmt = "educational_frame"
            elif "(for a novel" in prompt:
                fmt = "fiction_frame"
            else:
                fmt = "direct"
            
            by_format[fmt].append(refused)
        
        # Calculate rates per format
        format_rates = {}
        for fmt, refusals in by_format.items():
            if refusals:
                format_rates[fmt] = {
                    "n": len(refusals),
                    "refusal_rate": np.mean(refusals),
                }
        
        # Chi-square test for format independence
        if len(format_rates) > 1:
            formats = list(format_rates.keys())
            observed = np.array([
                [format_rates[f]["n"] * format_rates[f]["refusal_rate"],
                 format_rates[f]["n"] * (1 - format_rates[f]["refusal_rate"])]
                for f in formats
            ])
            
            chi2, pval, _, _ = stats.chi2_contingency(observed)
            
            format_analysis[category] = {
                "by_format": format_rates,
                "chi2_statistic": float(chi2),
                "chi2_pvalue": float(pval),
                "format_dependent": pval < 0.05,
            }
    
    return format_analysis


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values]


def generate_summary_report(analysis: Dict, rates: Dict) -> str:
    """Generate human-readable summary report."""
    
    lines = []
    lines.append("═" * 80)
    lines.append("STATISTICAL ANALYSIS REPORT")
    lines.append("═" * 80)
    
    # Section 1: Refusal rates
    lines.append("\n" + "─" * 80)
    lines.append("1. REFUSAL RATES BY MODEL")
    lines.append("─" * 80)
    
    for model_path, data in rates.items():
        model_name = Path(model_path).name
        lines.append(f"\n{model_name}:")
        lines.append(f"  Overall harmful: {data['overall_harmful']['rate']*100:.1f}% "
                    f"({data['overall_harmful']['n_refused']}/{data['overall_harmful']['n']})")
        lines.append(f"  Harmless: {data['harmless']['rate']*100:.1f}% "
                    f"({data['harmless']['n_refused']}/{data['harmless']['n']})")
    
    # Section 2: Pairwise comparisons
    if "pairwise_comparisons" in analysis:
        lines.append("\n" + "─" * 80)
        lines.append("2. PAIRWISE STATISTICAL COMPARISONS")
        lines.append("─" * 80)
        
        for comp_name, comp_data in analysis["pairwise_comparisons"].items():
            m1 = Path(comp_data["model1"]).name
            m2 = Path(comp_data["model2"]).name
            
            lines.append(f"\n{m1} vs {m2}:")
            lines.append(f"\n  Harmful prompts (overall):")
            
            # Aggregate across categories
            all_pvals = []
            all_effects = []
            
            for cat, stats_data in comp_data["harmful"].items():
                all_pvals.append(stats_data["chi2_pvalue"])
                all_effects.append(abs(stats_data["cohens_d"]))
            
            median_pval = np.median(all_pvals)
            median_effect = np.median(all_effects)
            n_sig = sum(1 for p in all_pvals if p < 0.05)
            
            lines.append(f"    Median p-value: {median_pval:.4f}")
            lines.append(f"    Median Cohen's d: {median_effect:.3f}")
            lines.append(f"    Significant categories: {n_sig}/{len(all_pvals)}")
            
            # Bonferroni correction
            corrected = bonferroni_correction(all_pvals)
            n_sig_corrected = sum(corrected)
            lines.append(f"    Significant (Bonferroni): {n_sig_corrected}/{len(all_pvals)}")
    
    # Section 3: Format dependence
    if "format_dependence" in analysis:
        lines.append("\n" + "─" * 80)
        lines.append("3. FORMAT DEPENDENCE ANALYSIS")
        lines.append("─" * 80)
        
        for model_path, format_data in analysis["format_dependence"].items():
            model_name = Path(model_path).name
            lines.append(f"\n{model_name}:")
            
            n_format_dependent = sum(
                1 for cat_data in format_data.values()
                if cat_data.get("format_dependent", False)
            )
            
            lines.append(f"  Categories with format-dependent refusal: "
                        f"{n_format_dependent}/{len(format_data)}")
            
            for cat, cat_data in format_data.items():
                if cat_data.get("format_dependent"):
                    lines.append(f"\n  {cat}:")
                    for fmt, fmt_data in cat_data["by_format"].items():
                        lines.append(f"    {fmt}: {fmt_data['refusal_rate']*100:.1f}%")
    
    # Section 4: Interpretation
    lines.append("\n" + "─" * 80)
    lines.append("4. INTERPRETATION")
    lines.append("─" * 80)
    
    # Determine if base models have systematic refusal
    base_models = [m for m in rates.keys() if "Base" in m or "base" in m]
    instruct_models = [m for m in rates.keys() if m not in base_models]
    
    if base_models:
        base_rate = np.mean([
            rates[m]["overall_harmful"]["rate"] for m in base_models
        ])
        
        lines.append(f"\nBase model(s) refusal rate: {base_rate*100:.1f}%")
        
        if base_rate < 0.15:
            lines.append("  → MINIMAL systematic refusal (likely pretraining noise)")
        elif base_rate < 0.30:
            lines.append("  → WEAK systematic refusal (scattered circuits)")
        elif base_rate < 0.60:
            lines.append("  → MODERATE systematic refusal")
        else:
            lines.append("  → STRONG systematic refusal")
    
    if instruct_models:
        instruct_rate = np.mean([
            rates[m]["overall_harmful"]["rate"] for m in instruct_models
        ])
        
        lines.append(f"\nInstruct model(s) refusal rate: {instruct_rate*100:.1f}%")
    
    if base_models and instruct_models:
        diff = instruct_rate - base_rate
        lines.append(f"\nDifference: {diff*100:.1f} percentage points")
        
        if diff > 0.50:
            lines.append("  → Instruction tuning STRONGLY increases refusal")
        elif diff > 0.20:
            lines.append("  → Instruction tuning MODERATELY increases refusal")
        else:
            lines.append("  → Instruction tuning has MINIMAL effect on refusal")
    
    lines.append("\n" + "═" * 80)
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="results/behavioral_assessment/refusal/model_evaluations.json",
        help="Path to evaluation results from Stage 2",
    )
    parser.add_argument(
        "--output",
        default="results/behavioral_assessment/refusal/statistical_analysis.json",
        help="Output path for analysis results",
    )
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    with open(args.results) as f:
        results = json.load(f)
    
    print(f"  {len(results)} models found")
    
    # Extract refusal rates
    print("\nExtracting refusal rates...")
    rates = extract_refusal_rates(results)
    
    # Pairwise comparisons (all combinations)
    print("\nPerforming pairwise comparisons...")
    model_paths = list(results.keys())
    comparisons = {}
    
    for i, m1 in enumerate(model_paths):
        for m2 in model_paths[i+1:]:
            comp_name = f"{Path(m1).name}_vs_{Path(m2).name}"
            print(f"  {comp_name}")
            comparisons[comp_name] = compare_two_models(results, m1, m2)
    
    # Format dependence
    print("\nAnalyzing format dependence...")
    format_analyses = {}
    for model_path in model_paths:
        print(f"  {Path(model_path).name}")
        format_analyses[model_path] = format_dependence_analysis(results, model_path)
    
    # Compile analysis
    analysis = {
        "refusal_rates": rates,
        "pairwise_comparisons": comparisons,
        "format_dependence": format_analyses,
    }
    
    # Save
    output_path = Path(args.output)
    print(f"\nSaving analysis to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Generate report
    print("Generating summary report...")
    report = generate_summary_report(analysis, rates)
    
    report_path = Path("analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(report)
    
    print(f"\n✓ Analysis saved to {output_path}")
    print(f"✓ Report saved to {report_path}")
    
    print(f"\n{'═'*80}")
    print("NEXT STEPS")
    print('═'*80)
    print("\n1. Review analysis_report.txt for interpretation")
    print("2. Proceed to Stage 4: run stage4_mechanistic_probing.py")
    print("   (This will probe for refusal circuits)")
    print(f"\n{'═'*80}\n")


if __name__ == "__main__":
    main()
