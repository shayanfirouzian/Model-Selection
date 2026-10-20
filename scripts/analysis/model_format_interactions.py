#!/usr/bin/env python3
"""Recompute model-by-format interaction tests from saved Stage 2 evaluations."""
import json
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/behavioral_assessment/ai_self_identification_assistive_behavior/model_evaluations.json"
OUTPUT = ROOT / "results/behavioral_assessment/ai_self_identification_assistive_behavior/model_format_interactions.json"

def main():
    raw = json.loads(SOURCE.read_text())
    rows = []
    for model, model_data in raw["model_evaluations"].items():
        for result in model_data["results"]:
            if result["category"] == "control":
                continue
            ev = result["evaluation"]
            ai_score = ev["ai_self_reference"]["combined_score"]
            assist_score = ev["assistant_orientation"]["combined_score"]
            rows.append({
                "model": model,
                "family": result["category"] + "::" + result["base_prompt"],
                "format": result["format"],
                "ai_self_identification": int(ai_score > 0.4),
                "assistive_behavior": int(assist_score > 0.4),
                "category": result["category"],
            })
    df = pd.DataFrame(rows)
    # Only complete four-format experimental families are retained.
    counts = df.groupby("family")["format"].nunique()
    keep = counts[counts == 4].index
    df = df[df["family"].isin(keep)].copy()
    results = {}
    for outcome in ["ai_self_identification", "assistive_behavior"]:
        fit = smf.ols(f"{outcome} ~ C(model) * C(format)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["family"]}
        )
        names = fit.model.exog_names
        interaction = [i for i, name in enumerate(names)
                       if ":" in name and "C(model)" in name and "C(format)" in name]
        R = [[1.0 if j == idx else 0.0 for j in range(len(names))]
             for idx in interaction]
        test = fit.wald_test(R, use_f=True, scalar=True)
        results[outcome] = {
            "F_statistic": float(test.statistic),
            "df_num": len(interaction),
            "df_denom": int(df["family"].nunique() - 1),
            "p_value": float(test.pvalue),
        }
    adjusted = multipletests([results[k]["p_value"] for k in ["ai_self_identification", "assistive_behavior"]], method="holm")[1]
    for key, adj in zip(["ai_self_identification", "assistive_behavior"], adjusted):
        results[key]["holm_adjusted_p"] = float(adj)
    out = {
        "analysis": "Prompt-family-clustered linear probability model, Model x Format interaction",
        "matched_prompt_families": int(df["family"].nunique()),
        "direct_only_controls_excluded": 30,
        "results": results,
        "note": "Uses saved evaluation scores; does not run model inference."
    }
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
