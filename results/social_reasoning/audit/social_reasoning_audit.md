# Saved Social Reasoning Results Audit

This report recomputes statistics from included per-item/per-question losses; it does not load model weights or rerun inference.

## Data integrity

- Source rows: 3820
- Exact question groups: 955
- Records per question: {4: 955}
- Source SHA-256: `ec624cb79f3ae1f07249ef0019f5f8f1862d990511bc2d7d92ab342d107767ea`
- Copeland references: 955, all responses match source candidates

## Analysis samples

- Primary: 307 question groups, four records per question; 1228 scored records per checkpoint.
- Copeland sensitivity: 393 distinct questions, one reference response per question.
- Shared question IDs: 124.

## Statistical verification

Primary model means, bootstrap intervals, paired mean differences, paired d_z, paired t-test p-values, and Holm-adjusted p-values were recomputed from the saved question-level scores and checked against the statistics embedded in the result JSON.
The same fields were recomputed for the Copeland analysis from its saved item-level losses and checked against the saved JSON. The matched-overlap analysis is recalculated from the 124 common items and is post hoc/exploratory.

## Interpretation and provenance

Both the primary and sensitivity analyses use plain-text question + one space + reference-response serialization and each checkpoint's native tokenizer. The score includes scenario/question and response tokens. The artifacts do not provide independent human validation of social competence or cross-tokenizer calibration. Immutable checkpoint/tokenizer revision hashes and a complete inference environment lockfile are absent, so the audit does not guarantee bit-for-bit replay of model inference.
