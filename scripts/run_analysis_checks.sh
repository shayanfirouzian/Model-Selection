#!/usr/bin/env bash
# Revalidate and recreate derived statistics/figures from saved outputs.
# This script does NOT download model weights or perform model inference.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results/social_reasoning/audit

python scripts/validate_repository.py

python scripts/social_reasoning/build_copeland_reference_set.py \
  > logs/copeland_reference_rebuild.txt
python scripts/social_reasoning/recompute_saved_results.py \
  > logs/social_reasoning_audit.txt
python scripts/analysis/summarize_baseline_statistics.py \
  > logs/baseline_statistics.txt
python scripts/figures/generate_baseline_social_reasoning_figures.py \
  > logs/baseline_social_reasoning_figures.txt
python scripts/figures/generate_social_reasoning_figures.py \
  > logs/social_reasoning_figures.txt
python scripts/validate_repository.py
python scripts/generate_manifest.py

echo "Saved-result checks and figure generation completed. No model inference was run."
