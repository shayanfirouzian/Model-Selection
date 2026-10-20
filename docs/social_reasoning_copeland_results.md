# Copeland Single-Reference Social Reasoning Sensitivity Analysis

## Design

This analysis samples 393 distinct questions without replacement (seed 42) from the 955-question corpus reduced to one Copeland-ranked reference response per question. It is a separate operationalization from the primary four-record-per-question estimator. The target item count follows a two-independent-group planning approximation; inference in the saved analysis is paired by question.

## Model means

| Checkpoint | Questions | Mean full-sequence loss | 95% item-bootstrap CI |
|---|---:|---:|---|
| Qwen3 (8B) Base | 393 | 2.1802 | [2.1211, 2.2387] |
| Llama 3.1 (8B) Base | 393 | 2.2212 | [2.1663, 2.2755] |
| Llama 3.1 (8B) Instruct | 393 | 2.3284 | [2.2711, 2.3856] |
| Qwen3 (8B) Instruct | 393 | 2.4739 | [2.4118, 2.5358] |

All six paired contrasts are significant after Holm correction in the saved result. Qwen3 Base minus Llama Base is -0.0409 loss units (95% paired bootstrap CI [-0.0522, -0.0296]; paired $d_z=-0.359$).

## Scope

This analysis uses one Copeland-ranked response for each question and a sample that differs from the primary design. It is reported separately and is not pooled with the primary results. The score is full-sequence loss using each checkpoint's native tokenizer; the analysis does not provide independent construct validation.
