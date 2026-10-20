# Social Reasoning: Data, Scoring, and Inference

## Source records

`data/social_reasoning/ProlificAI.jsonl` contains pairwise preference records with `question`, `chosen`, and `rejected` fields. The file has 3,820 non-empty JSONL records grouped into 955 distinct exact question strings, with four pairwise records per question. The preference labels are used as encoded in the source data and have not been independently re-annotated for this study.

## Primary clustered analysis

The primary analysis randomly samples 307 complete exact-question groups without replacement (seed 42), retaining all four records in each group. Each checkpoint is scored on the chosen response in each pairwise record. The four record losses are averaged within question. The independent inferential unit is therefore the question-level mean: 307 outcomes per checkpoint, formed from 1,228 scored source records. Paired checkpoint contrasts use the same question groups, paired tests, percentile-bootstrap confidence intervals, and Holm correction over the six pairwise contrasts.

The serialized input is plain text formed by `question + " " + chosen response`. The question and response are tokenized jointly with each checkpoint's native tokenizer. Loss is computed on the full token sequence, including question tokens. The scoring notebook does not silently truncate inputs at a fixed 512-token limit; it raises an error when an example exceeds the detected context length.

## Copeland single-reference sensitivity analysis

The source preference data do not provide a single canonical target answer per question. The included reference set derives one answer per question by whitespace-normalizing answers, building the directed graph of pairwise wins, and choosing the answer with the unique highest Copeland score (wins minus losses). A question is flagged when the top score is tied or the top-ranked answer belongs to a directed cycle. The builder reconstructs the 955 reference responses with no flagged questions and preserves their order.

The sensitivity analysis randomly samples 393 distinct questions from this 955-question reference set without replacement (seed 42), then scores one reference response per question. This is a distinct operationalization from the primary four-record estimator. The target sample size follows a two-independent-group planning approximation; the saved inferential comparisons are paired by question, so the planning calculation is not a prospective paired-design power guarantee.

## Matched overlap

The primary and Copeland samples share 124 question IDs. The audit script computes an exploratory, post hoc comparison on those shared items to illustrate how reference construction changes loss levels and pairwise differences. It is not an independent replication.

## Interpretation and reproducibility limits

The score is mean causal-language-model loss over the complete concatenated question and reference response. It combines scenario modeling and reference-response prediction. The models use different tokenizers, and the resulting values are not calibrated to a tokenizer-independent scale. The score has not been independently validated against human ratings of newly generated answers or against an independent social-intelligence criterion. Lower loss therefore describes likelihood under this implementation; it is not evidence, by itself, of general social competence.

The saved result JSONs contain sampled question IDs where available, per-record losses, question-level scores where applicable, summary statistics, and pairwise contrasts. They do not record immutable model/tokenizer revision hashes or a complete inference lockfile. The repository supports auditing and reanalysis of the saved outputs but does not guarantee bit-for-bit reproduction of model inference.
