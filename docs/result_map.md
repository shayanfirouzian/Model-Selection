# Results-to-Repository Map

| Analysis/result family | Inputs | Saved outputs | Recompute or audit command | Notes |
|---|---|---|---|---|
| Refusal prompt dataset | Dataset construction script | `data/behavioral_assessment/refusal/refusal_eval_dataset.json` | `python scripts/behavioral_assessment/refusal/stage1_build_dataset.py` | 1,280 harm-adjacent prompts across eight categories plus 320 harmless controls |
| Refusal model evaluations | Refusal dataset and four local checkpoints | `results/behavioral_assessment/refusal/model_evaluations.json` | `stage2_evaluate_models.py` (requires model checkpoints) | Includes saved responses and scores |
| Refusal statistics | Saved refusal evaluations | `results/behavioral_assessment/refusal/statistical_analysis.json` | `stage3_statistical_analysis.py` | Summary rates and format-dependence results |
| AI identity/assistive dataset | Dataset construction script | `data/behavioral_assessment/ai_self_identification_assistive_behavior/ai_awareness_eval_dataset.json` | `stage2a_build_dataset.py` | 630 prompts; control prompts occur only in direct format |
| AI identity/assistive evaluations | Dataset and four local checkpoints | `results/behavioral_assessment/ai_self_identification_assistive_behavior/model_evaluations.json` | `stage2b_evaluate_models.py` (requires model checkpoints) | Two classifications are applied to the same response set |
| AI identity/assistive statistics | Saved model evaluations | `results/behavioral_assessment/ai_self_identification_assistive_behavior/statistical_analysis.json` | `stage2c_statistical_analysis.py` | See `docs/known_limitations.md` for the `cohens_d` field caveat |
| Full-sequence Social Reasoning baseline | ProlificAI pairwise dataset and scoring notebook | `results/social_reasoning/full_sequence_baseline/baseline_results.json` | `scripts/analysis/summarize_baseline_statistics.py` | 393 scored examples for each checkpoint |
| Primary clustered Social Reasoning | ProlificAI pairwise dataset and primary notebook | `results/social_reasoning/primary_clustered_307/social_reasoning_results.json` | `scripts/social_reasoning/recompute_saved_results.py` | 307 question groups, four chosen-response records per question |
| Copeland single-reference Social Reasoning | ProlificAI dataset, Copeland responses, and sensitivity notebook | `results/social_reasoning/copeland_single_reference_393/social_reasoning_results.json` | Rebuild reference set, then audit saved results | 393 distinct questions, one response per question |
| Selected-question manifest and derived tables | Source dataset and saved scores | `results/social_reasoning/audit/` | `scripts/social_reasoning/recompute_saved_results.py` | Includes 124-question matched-overlap exploratory analysis |
| Baseline Social Reasoning figures | Full-sequence baseline result JSON | `figures/social_reasoning_baseline/` | `scripts/figures/generate_baseline_social_reasoning_figures.py` | Model-mean loss and paired-effect displays |
| Primary and sensitivity Social Reasoning figures | Primary and Copeland result JSONs | `figures/social_reasoning/` | `scripts/figures/generate_social_reasoning_figures.py` | Designs are labeled separately in figure titles and filenames |

Commands that require checkpoints perform model inference. Saved-result recomputation and figure generation operate on the numerical files included in the repository.
