# Model Selection for Generative Agent-Based Modeling: Code and Data

This repository contains the study's behavioral prompt datasets, saved model evaluations, Social Reasoning source data, analysis notebooks, statistical scripts, saved result files, and figures. The materials are organized by task and analysis design so that inputs, estimators, and outputs can be located directly.

## Quick start

The default workflow validates the included datasets and saved results, rebuilds the one-response-per-question reference set, recomputes statistics from saved losses, regenerates figures, and refreshes the file-integrity manifest. It does **not** download model weights or run model inference.

```bash
python -m pip install -r requirements.txt
bash scripts/run_analysis_checks.sh
```

The workflow's console logs are written under `logs/`, which is ignored by Git. Individual steps are documented in [`docs/repository_guide.md`](docs/repository_guide.md) and [`docs/result_map.md`](docs/result_map.md).

## Repository structure

| Path | Contents |
|---|---|
| `data/behavioral_assessment/` | Refusal, AI self-identification, and assistive-behavior prompt datasets |
| `data/social_reasoning/ProlificAI.jsonl` | 3,820 pairwise-preference records from the ProlificAI Social Reasoning dataset |
| `data/social_reasoning/reference_responses/most_preferred.jsonl` | One Copeland-ranked reference response for each of 955 distinct questions |
| `results/behavioral_assessment/` | Saved model evaluations and statistical summaries |
| `results/social_reasoning/full_sequence_baseline/` | Full-sequence baseline results for four checkpoints |
| `results/social_reasoning/primary_clustered_307/` | Primary clustered results: 307 question groups, four chosen-response records per group |
| `results/social_reasoning/copeland_single_reference_393/` | Single-reference sensitivity results for 393 distinct questions |
| `results/social_reasoning/paraphrase_sensitivity/` | Coverage counts and results for the tested paraphrase-sensitivity analyses |
| `results/social_reasoning/audit/` | Question manifest, pairwise tables, effect-size summaries, and audit report |
| `notebooks/social_reasoning/` | Model-scoring notebooks for each Social Reasoning analysis design |
| `scripts/` | Dataset construction, model evaluation, analysis, validation, and figure-generation scripts |
| `figures/` | Figures generated from the saved result files |
| `docs/` | Methods, result map, screening description, and limitations |
| `MANIFEST.sha256` | SHA-256 hashes of repository files, with relative paths |

## Social Reasoning analyses

The repository reports three separately identified analysis designs:

1. **Full-sequence baseline.** The saved results contain 393 scored examples for each of the four checkpoints. The scoring notebook and result JSON are provided under `notebooks/social_reasoning/full_sequence_baseline/` and `results/social_reasoning/full_sequence_baseline/`.
2. **Primary clustered analysis.** A fixed-seed sample of 307 exact-question groups is used. Each group contributes four chosen-response losses, averaged into one question-level outcome; inference is performed on the 307 paired question-level outcomes.
3. **Copeland single-reference sensitivity analysis.** One Copeland-ranked response is selected for each question, and a fixed-seed sample of 393 distinct questions is scored. This estimates loss against a different reference construction and is not pooled with the primary analysis.

The primary and Copeland samples share 124 question IDs. The matched-overlap analysis is exploratory. All Social Reasoning estimates use full-sequence loss over plain-text `question + " " + response` inputs and each checkpoint's native tokenizer. The measure therefore combines scenario/question modeling and response prediction; the saved results do not establish tokenizer-independent calibration or independently validate general social competence.

See [`docs/methods_social_reasoning.md`](docs/methods_social_reasoning.md) for the scoring and inference details, and [`docs/social_reasoning_primary_results.md`](docs/social_reasoning_primary_results.md) and [`docs/social_reasoning_copeland_results.md`](docs/social_reasoning_copeland_results.md) for the reported summaries.

## Model inference

Saved-result validation, statistical recomputation, and figure generation do not require model weights. Running model inference is optional and requires the appropriate checkpoints, compatible software, and suitable hardware. Set the checkpoint paths through `QWEN3_INSTRUCT_PATH`, `QWEN3_BASE_PATH`, `LLAMA31_BASE_PATH`, and `LLAMA31_INSTRUCT_PATH`, or edit the configuration cell in the relevant notebook. Model weights are not included.

The repository does not contain immutable checkpoint/tokenizer revision hashes or a complete lockfile for the inference environment. Exact bit-for-bit reproduction of model inference is therefore not guaranteed; the included workflow validates and recomputes summaries from the saved artifacts.

## Data sources and reuse

The Social Reasoning source file is available from [ProlificAI/social-reasoning-rlhf](https://huggingface.co/datasets/ProlificAI/social-reasoning-rlhf); its dataset card lists the dataset as MIT-licensed. That license statement applies to the upstream dataset and does not itself grant rights to every custom prompt, model output, or script in this repository. Check applicable terms before redistributing or reusing individual materials. The repository does not include a project-wide license.

## Additional sensitivity analyses

The saved behavioral results include prompt-family-clustered Model × Format interaction tests in `results/behavioral_assessment/ai_self_identification_assistive_behavior/model_format_interactions.json`. The corresponding analysis script is `scripts/analysis/model_format_interactions.py`. Paraphrase-screening coverage and sensitivity outcomes are recorded in `results/social_reasoning/paraphrase_sensitivity/`.
