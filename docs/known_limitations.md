# Known Data and Reproducibility Limitations

## Behavioral result files

The AI self-identification/assistive-behavior `statistical_analysis.json` file contains a `cohens_d` field in its `model_comparisons` objects that is computed from `suitability_score`, rather than directly from the rate difference named by the row. Do not use that field as the effect size for the corresponding behavioral rate. `scripts/analysis/summarize_baseline_statistics.py` recomputes pairwise proportion effect sizes from the relevant rate differences and applies Holm correction in its comparison tables.

The refusal statistical output reports per-category format-dependence tests. The summary script also computes a pooled format-dependence statistic by aggregating saved counts across categories. Assistive-behavior format-dependence is computed from saved per-format rates by the summary script.

## Behavioral construct and sample structure

The prompt datasets contain harm-adjacent/safety-policy probes as well as identity, human-like-response, and assistant-orientation probes. The included automatic scores are not supported by an independent human-annotation study. The repository therefore does not present them as validated measures of full persona fidelity or psychological realism.

The AI self-identification and assistive-behavior classifications are applied to the same 630 responses per model; they are two labels applied to the same response corpus, not independent samples. The 30 control prompts occur only in the direct format, creating an imbalance for full-corpus format comparisons.

## Format-sensitivity composite

The AI self-identification/assistive-behavior statistics contain a composite stored under the JSON field `plasticity_index` (nested under `plasticity_by_model`). The formula combines variation across prompt formats, variation in suitability scores, hedging rate, and within-format variance. It has not been validated as a measure of trainability and should not be used to claim that training can eliminate format sensitivity. Treat it as an exploratory format-sensitivity heuristic rather than a validated construct.

## Social Reasoning

The full-sequence baseline, primary clustered analysis, and Copeland single-reference sensitivity analysis use different sample and response-selection procedures and are not interchangeable. The primary analysis has 307 question-level units, each averaging four record losses. The Copeland analysis uses 393 distinct questions and one selected reference response per question. The two latter samples overlap on 124 exact question IDs; their matched-overlap analysis is post hoc and exploratory.

The model loss is computed over the entire plain-text question-plus-response sequence, and each model uses its own tokenizer. The primary analysis scores the `chosen` response from each of four pairwise records per sampled question; the sensitivity analysis scores one Copeland-ranked response per sampled question. These losses combine scenario/question modeling and response prediction. They are not calibrated across tokenizers, and the repository does not include independent validation against ratings of newly generated answers.

The repository does not include immutable checkpoint/tokenizer revision hashes or a complete lockfile for the inference environment. The saved-output workflow supports data checks, statistical auditing, and figure regeneration, but does not guarantee exact replay of model inference.
