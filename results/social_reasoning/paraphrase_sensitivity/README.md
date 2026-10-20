# Paraphrase-sensitivity analysis

This supplementary sensitivity analysis evaluates whether the Social Reasoning model ordering depends on detected paraphrase links among source questions.

- Source corpus: 955 exact-question groups.
- Questions covered by screening: 424; not screened: 531.
- Candidate pairs reviewed: 1,066.
- Initial PARAPHRASE labels: 20 pairs, forming 16 connected groups.
- Stricter review retained: 10 likely paraphrase pairs.
- Primary 307-question sample covered by screening: 140; not screened: 167.
- Sensitivity variants: combine detected paraphrase-linked questions, use the broader initial links, and exclude affected questions.
- Result: the model ordering was unchanged in each tested sensitivity analysis; all six pairwise model comparisons remained significant after Holm correction.

The screening covers a subset of the source corpus. These results describe the tested sensitivity analyses only.
