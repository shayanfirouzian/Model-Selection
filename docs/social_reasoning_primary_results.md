# Primary Social Reasoning Results: 307 Question Groups

## Design

The primary analysis samples 307 exact-question groups without replacement (seed 42). Each question contributes four chosen-response pairwise records. Four record losses are averaged within question, so the inferential unit is the question-level mean (307 units), not the 1,228 source rows.

## Model means

| Checkpoint | Questions | Records scored | Mean full-sequence loss | 95% bootstrap CI |
|---|---:|---:|---:|---|
| Qwen3 (8B) Base | 307 | 1,228 | 2.2579 | [2.2140, 2.3013] |
| Llama 3.1 (8B) Base | 307 | 1,228 | 2.3161 | [2.2738, 2.3575] |
| Llama 3.1 (8B) Instruct | 307 | 1,228 | 2.4138 | [2.3713, 2.4561] |
| Qwen3 (8B) Instruct | 307 | 1,228 | 2.5514 | [2.5049, 2.5965] |

All six paired checkpoint contrasts are significant after Holm correction in the saved analysis. The Qwen3 Base minus Llama Base difference is -0.0582 loss units (95% paired question-bootstrap CI [-0.0677, -0.0488]; paired $d_z=-0.682$).

## Scope

The score is loss over the full plain-text question-plus-chosen-response sequence using each checkpoint's native tokenizer. The measure includes both scenario modeling and reference-response prediction; it has not been independently validated against human ratings of generated social behavior.
