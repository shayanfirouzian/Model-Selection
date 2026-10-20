# Repository Guide

## Organization

Files are grouped by task and analysis design. Source datasets are stored once; derived reference data, saved scores, summary statistics, and figures are kept in their corresponding result directories.

## Directory overview

```text
.
├── README.md
├── CITATION.cff
├── MANIFEST.sha256
├── requirements.txt
├── requirements-model-inference.txt
├── data/
│   ├── behavioral_assessment/
│   └── social_reasoning/
├── results/
│   ├── behavioral_assessment/
│   └── social_reasoning/
│       ├── full_sequence_baseline/
│       ├── primary_clustered_307/
│       ├── copeland_single_reference_393/
│       └── audit/
├── notebooks/social_reasoning/
│   ├── full_sequence_baseline/
│   ├── primary_clustered_307/
│   └── copeland_single_reference_393/
├── scripts/
│   ├── behavioral_assessment/
│   ├── analysis/summarize_baseline_statistics.py
│   ├── social_reasoning/
│   ├── figures/
│   ├── validate_repository.py
│   ├── generate_manifest.py
│   └── run_analysis_checks.sh
├── figures/
└── docs/
```

## Canonical inputs and outputs

- **Behavioral Assessment inputs:** `data/behavioral_assessment/`.
- **Behavioral Assessment saved outputs:** `results/behavioral_assessment/`.
- **Social Reasoning pairwise source data:** `data/social_reasoning/ProlificAI.jsonl`.
- **Copeland response set:** `data/social_reasoning/reference_responses/most_preferred.jsonl`; rebuild it from the pairwise source with `scripts/social_reasoning/build_copeland_reference_set.py`.
- **Full-sequence baseline:** `results/social_reasoning/full_sequence_baseline/baseline_results.json`.
- **Primary clustered results:** `results/social_reasoning/primary_clustered_307/social_reasoning_results.json`.
- **Copeland sensitivity results:** `results/social_reasoning/copeland_single_reference_393/social_reasoning_results.json`.
- **Derived Social Reasoning statistics:** `results/social_reasoning/audit/`; regenerate with `scripts/social_reasoning/recompute_saved_results.py`.

## Reproducibility levels

1. **File and dataset validation:** checks schemas, record counts, question grouping, selected-question alignment, and Python/notebook syntax.
2. **Saved-result analysis:** recomputes statistical summaries and audit tables from included per-record or per-question losses.
3. **Figure generation:** regenerates figures from saved result JSON files.
4. **Dataset construction:** reconstructs the prompt datasets and the Copeland reference set through included scripts.
5. **Model inference:** requires local checkpoints, appropriate hardware, and a compatible environment. Exact numerical replay is not guaranteed because immutable model/tokenizer revision hashes and a complete inference lockfile are not included.

## One-command workflow

From the repository root:

```bash
python -m pip install -r requirements.txt
bash scripts/run_analysis_checks.sh
```

The workflow validates the repository, reconstructs the Copeland reference set, computes saved-result statistics, generates figures, runs validation again, and refreshes `MANIFEST.sha256`. No model inference is performed. Console logs are written to `logs/`, which is excluded from version control.

## File integrity

`MANIFEST.sha256` lists SHA-256 digests using paths relative to the repository root. Regenerate it after an intentional file change with:

```bash
python scripts/generate_manifest.py
```
