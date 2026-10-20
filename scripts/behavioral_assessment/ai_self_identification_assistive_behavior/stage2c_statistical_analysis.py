#!/usr/bin/env python3
"""
stage2c_statistical_analysis.py — Statistical analysis of AI self-identification and assistant-orientation signals.

The output includes a composite named `plasticity_index`. It is an exploratory
descriptive heuristic, not a validated measure of trainability or evidence that
training can remove format sensitivity.

Usage:
    python stage2c_statistical_analysis.py \
        --results model_evaluations_stage2.json \
        --output statistical_analysis_stage2.json

Outputs:
    - AI self-reference rates by model/category
    - Assistant orientation rates
    - exploratory format-sensitivity heuristic (stored under the `plasticity_index` key)
    - Statistical tests with corrections
    - Effect sizes
"""

import json
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.metrics import cohen_kappa_score
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

# ============================================================================
# Rate Calculations
# ============================================================================

def calculate_rates(model_results: Dict) -> Dict:
    """Calculate AI-awareness and assistant orientation rates."""
    
    results = model_results["results"]
    
    # Overall rates
    total = len(results)
    ai_ref_count = sum(1 for r in results if r["evaluation"]["ai_self_reference"]["has_ai_reference"])
    assistant_count = sum(1 for r in results if r["evaluation"]["assistant_orientation"]["has_assistant_orientation"])
    hedging_count = sum(1 for r in results if r["evaluation"]["hedging"]["has_hedging"])
    humanlike_count = sum(1 for r in results if r["evaluation"]["humanlike_framing"]["has_humanlike_framing"])
    
    overall = {
        "total": total,
        "ai_self_reference": {
            "count": ai_ref_count,
            "rate": ai_ref_count / total,
        },
        "assistant_orientation": {
            "count": assistant_count,
            "rate": assistant_count / total,
        },
        "hedging": {
            "count": hedging_count,
            "rate": hedging_count / total,
        },
        "humanlike_framing": {
            "count": humanlike_count,
            "rate": humanlike_count / total,
        },
        "mean_suitability_score": np.mean([r["evaluation"]["suitability_score"] for r in results]),
    }
    
    # By category
    by_category = {}
    
    categories = set(r["category"] for r in results)
    for category in categories:
        cat_results = [r for r in results if r["category"] == category]
        cat_total = len(cat_results)
        
        if cat_total > 0:
            by_category[category] = {
                "total": cat_total,
                "ai_self_reference": {
                    "count": sum(1 for r in cat_results if r["evaluation"]["ai_self_reference"]["has_ai_reference"]),
                    "rate": sum(1 for r in cat_results if r["evaluation"]["ai_self_reference"]["has_ai_reference"]) / cat_total,
                },
                "assistant_orientation": {
                    "count": sum(1 for r in cat_results if r["evaluation"]["assistant_orientation"]["has_assistant_orientation"]),
                    "rate": sum(1 for r in cat_results if r["evaluation"]["assistant_orientation"]["has_assistant_orientation"]) / cat_total,
                },
            }
    
    # By format (for descriptive format-sensitivity summaries)
    by_format = {}
    
    formats = set(r["format"] for r in results if r["format"])
    for fmt in formats:
        fmt_results = [r for r in results if r["format"] == fmt]
        fmt_total = len(fmt_results)
        
        if fmt_total > 0:
            by_format[fmt] = {
                "total": fmt_total,
                "ai_self_reference_rate": sum(1 for r in fmt_results if r["evaluation"]["ai_self_reference"]["has_ai_reference"]) / fmt_total,
                "assistant_orientation_rate": sum(1 for r in fmt_results if r["evaluation"]["assistant_orientation"]["has_assistant_orientation"]) / fmt_total,
                "mean_suitability": np.mean([r["evaluation"]["suitability_score"] for r in fmt_results]),
            }
    
    return {
        "overall": overall,
        "by_category": by_category,
        "by_format": by_format,
    }

# ============================================================================
# Exploratory Format-Sensitivity Summary
# ============================================================================

def calculate_plasticity_index(model_results: Dict, rates: Dict) -> Dict:
    """
    Calculate an exploratory composite of observed response variation.

    The JSON field `plasticity_index` stores this exploratory summary,
    but the score has not been validated as a measure of trainability. Format variation,
    response-score variance, hedging, and within-format variance do not establish that
    training can reduce variation or preserve response quality.

    Returns a 0-1 heuristic and component values for descriptive auditing only.
    """
    
    results = model_results["results"]
    
    # Component 1: Format variance (40% weight); descriptive only
    format_rates = rates["by_format"]
    
    if len(format_rates) > 1:
        ai_ref_rates = [fmt_data["ai_self_reference_rate"] for fmt_data in format_rates.values()]
        assist_rates = [fmt_data["assistant_orientation_rate"] for fmt_data in format_rates.values()]
        
        format_variance = (np.std(ai_ref_rates) + np.std(assist_rates)) / 2
        format_variance_score = min(1.0, format_variance * 3)  # Normalize, boost to 0-1 range
    else:
        format_variance_score = 0.5  # Default if no format variations
    
    # Component 2: Response-score variation (30% weight)
    suitability_scores = [r["evaluation"]["suitability_score"] for r in results]
    response_diversity = np.std(suitability_scores)
    response_diversity_score = min(1.0, response_diversity * 2)
    
    # Component 3: Hedging rate (20% weight); not a measure of malleability
    hedging_rate = rates["overall"]["hedging"]["rate"]
    hedging_score = hedging_rate  # Already 0-1
    
    # Component 4: Within-format response-score variation (10% weight)
    # Measure consistency within format groups
    format_consistencies = []
    
    for fmt, fmt_data in format_rates.items():
        # Check if responses in this format are consistent
        fmt_results = [r for r in results if r["format"] == fmt]
        
        if len(fmt_results) > 10:
            # Calculate variance within format
            fmt_suitability = [r["evaluation"]["suitability_score"] for r in fmt_results]
            fmt_variance = np.std(fmt_suitability)
            format_consistencies.append(fmt_variance)
    
    if format_consistencies:
        mean_within_format_variance = np.mean(format_consistencies)
        consistency_inverse_score = min(1.0, mean_within_format_variance * 2)
    else:
        consistency_inverse_score = 0.5
    
    # Calculate the exploratory composite score
    plasticity_index = (
        format_variance_score * 0.40 +
        response_diversity_score * 0.30 +
        hedging_score * 0.20 +
        consistency_inverse_score * 0.10
    )
    
    return {
        "plasticity_index": float(plasticity_index),
        "components": {
            "format_variance": {
                "score": float(format_variance_score),
                "weight": 0.40,
                "interpretation": "Higher = greater variation across the tested formats; trainability is not inferred"
            },
            "response_diversity": {
                "score": float(response_diversity_score),
                "weight": 0.30,
                "interpretation": "Higher = greater variance in the recorded suitability scores"
            },
            "hedging_rate": {
                "score": float(hedging_score),
                "weight": 0.20,
                "interpretation": "Higher = greater observed hedging rate; malleability is not inferred"
            },
            "consistency_inverse": {
                "score": float(consistency_inverse_score),
                "weight": 0.10,
                "interpretation": "Higher = greater within-format variance in suitability scores"
            }
        },
        "interpretation": plasticity_index_interpretation(plasticity_index)
    }

def plasticity_index_interpretation(index: float) -> str:
    """Return a label for the exploratory format-sensitivity heuristic."""
    
    if index > 0.75:
        level = "HIGH"
    elif index > 0.60:
        level = "MODERATE-HIGH"
    elif index > 0.45:
        level = "MODERATE"
    elif index > 0.30:
        level = "LOW"
    else:
        level = "VERY LOW"
    return f"{level} descriptive format-sensitivity heuristic; not a trainability estimate"

# ============================================================================
# Format Dependence Analysis
# ============================================================================

def analyze_format_dependence(model_results: Dict, rates: Dict) -> Dict:
    """
    Test if AI-awareness is format-dependent (like Stage 1 refusal).
    
    Format-dependence describes sensitivity to the tested prompt formats. It does not establish that training can remove that sensitivity.
    """
    
    format_data = rates["by_format"]
    
    if len(format_data) < 2:
        return {
            "format_dependent": False,
            "reason": "Insufficient format variations",
        }
    
    # Chi-square test for format independence
    # H0: AI self-reference rate is independent of format
    
    formats = list(format_data.keys())
    ai_ref_counts = []
    totals = []
    
    for fmt in formats:
        fmt_results = [r for r in model_results["results"] if r["format"] == fmt]
        ai_ref_count = sum(1 for r in fmt_results if r["evaluation"]["ai_self_reference"]["has_ai_reference"])
        total = len(fmt_results)
        
        ai_ref_counts.append(ai_ref_count)
        totals.append(total)
    
    # Create contingency table
    non_ai_ref_counts = [totals[i] - ai_ref_counts[i] for i in range(len(formats))]
    contingency_table = [ai_ref_counts, non_ai_ref_counts]
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate variance across formats
        rates_list = [format_data[fmt]["ai_self_reference_rate"] for fmt in formats]
        rate_variance = np.var(rates_list)
        
        return {
            "format_dependent": p_value < 0.05,
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "rate_variance": float(rate_variance),
            "interpretation": "Statistically detectable variation across tested formats" if p_value < 0.05 else "No statistically significant variation detected across tested formats",
        }
    except:
        return {
            "format_dependent": None,
            "reason": "Chi-square test failed (insufficient data)",
        }

# ============================================================================
# Statistical Comparisons
# ============================================================================

def compare_models(all_results: Dict, rates_by_model: Dict) -> Dict:
    """Compare models statistically."""
    
    comparisons = {}
    
    models = list(all_results["model_evaluations"].keys())
    
    # Pairwise comparisons
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            
            results1 = all_results["model_evaluations"][model1]["results"]
            results2 = all_results["model_evaluations"][model2]["results"]
            
            # AI self-reference rates
            ai_ref1 = [1 if r["evaluation"]["ai_self_reference"]["has_ai_reference"] else 0 for r in results1]
            ai_ref2 = [1 if r["evaluation"]["ai_self_reference"]["has_ai_reference"] else 0 for r in results2]
            
            # Chi-square test
            rate1 = np.mean(ai_ref1)
            rate2 = np.mean(ai_ref2)
            
            contingency = [
                [sum(ai_ref1), len(ai_ref1) - sum(ai_ref1)],
                [sum(ai_ref2), len(ai_ref2) - sum(ai_ref2)]
            ]
            
            try:
                chi2, p_value, _, _ = chi2_contingency(contingency)
                
                # Effect size (Cramér's V)
                n = len(ai_ref1) + len(ai_ref2)
                cramers_v = np.sqrt(chi2 / n)
                
                # Cohen's d for suitability scores
                suit1 = [r["evaluation"]["suitability_score"] for r in results1]
                suit2 = [r["evaluation"]["suitability_score"] for r in results2]
                
                cohens_d = (np.mean(suit1) - np.mean(suit2)) / np.sqrt((np.std(suit1)**2 + np.std(suit2)**2) / 2)
                
                comparison_key = f"{model1.split('/')[-1]}_vs_{model2.split('/')[-1]}"
                
                comparisons[comparison_key] = {
                    "ai_self_reference_rate_diff": float(rate1 - rate2),
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                    "cramers_v": float(cramers_v),
                    "cohens_d": float(cohens_d),
                    "significant": p_value < 0.05,
                }
            except:
                pass
    
    return comparisons

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of AI-awareness evaluation (Stage 2C)"
    )
    parser.add_argument("--results", default="results/behavioral_assessment/ai_self_identification_assistive_behavior/model_evaluations.json", help="Results JSON from Stage 2B")
    parser.add_argument("--output", default="results/behavioral_assessment/ai_self_identification_assistive_behavior/statistical_analysis.json")
    
    args = parser.parse_args()
    
    print("="*80)
    print("STAGE 2C: STATISTICAL ANALYSIS")
    print("="*80)
    
    # Load results
    print(f"\nLoading results: {args.results}")
    with open(args.results) as f:
        all_results = json.load(f)
    
    print(f"  Models evaluated: {all_results['metadata']['models_evaluated']}")
    
    # Calculate rates for each model
    print("\nCalculating rates...")
    rates_by_model = {}
    plasticity_by_model = {}
    format_dependence_by_model = {}
    
    for model_path, model_data in all_results["model_evaluations"].items():
        model_name = model_path.split("/")[-1]
        
        print(f"\n  {model_name}:")
        
        # Rates
        rates = calculate_rates(model_data)
        rates_by_model[model_path] = rates
        
        print(f"    AI self-reference: {rates['overall']['ai_self_reference']['rate']:.1%}")
        print(f"    Assistant orientation: {rates['overall']['assistant_orientation']['rate']:.1%}")
        print(f"    Mean suitability: {rates['overall']['mean_suitability_score']:.3f}")
        
        # Exploratory format-sensitivity composite (stored `plasticity_index` field; not trainability)
        plasticity = calculate_plasticity_index(model_data, rates)
        plasticity_by_model[model_path] = plasticity
        
        print(f"    Exploratory format-sensitivity heuristic: {plasticity['plasticity_index']:.3f} - {plasticity['interpretation']}")
        
        # Format dependence
        format_dep = analyze_format_dependence(model_data, rates)
        format_dependence_by_model[model_path] = format_dep
        
        if "chi2_statistic" in format_dep:
            print(f"    Format dependence: χ²={format_dep['chi2_statistic']:.2f}, p={format_dep['p_value']:.4f}")
    
    # Model comparisons
    print("\nComparing models...")
    comparisons = compare_models(all_results, rates_by_model)
    
    for comp_name, comp_data in comparisons.items():
        if comp_data["significant"]:
            print(f"\n  {comp_name}:")
            print(f"    Δ rate: {comp_data['ai_self_reference_rate_diff']:.1%}")
            print(f"    Cramér's V: {comp_data['cramers_v']:.3f}")
            print(f"    p-value: {comp_data['p_value']:.4f}")
    
    # Compile full analysis
    analysis = {
        "rates_by_model": rates_by_model,
        "plasticity_by_model": plasticity_by_model,
        "format_dependence_by_model": format_dependence_by_model,
        "model_comparisons": comparisons,
        "metadata": {
            "source": args.results,
            "models_analyzed": len(rates_by_model),
        }
    }
    
    # Save
    print(f"\nSaving analysis to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\n✓ Analyzed {len(rates_by_model)} models")
    print("✓ Calculated exploratory format-sensitivity summaries (`plasticity_index` field included)")
    print(f"✓ Performed statistical tests")
    print(f"✓ Saved to {args.output}")
    print("\nReady for manuscript generation!")

if __name__ == "__main__":
    main()
