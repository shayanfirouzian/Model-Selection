# Data Provenance and Licensing

## ProlificAI Social Reasoning preference data

Canonical local path: `data/social_reasoning/ProlificAI.jsonl`.

Upstream source: <https://huggingface.co/datasets/ProlificAI/social-reasoning-rlhf>.

The upstream dataset card describes pairwise records containing a question, a chosen answer, and a rejected answer, and lists the dataset as MIT-licensed. The local copy has 3,820 records grouped into 955 exact question strings. Its SHA-256 is recorded in `results/social_reasoning/audit/social_reasoning_audit.md`.

The file `data/social_reasoning/reference_responses/most_preferred.jsonl` is a derived artifact generated from the included source data. It contains one Copeland-ranked answer per question and can be rebuilt by `scripts/social_reasoning/build_copeland_reference_set.py`.

## Behavioral prompts and saved model outputs

The repository includes behavioral prompt datasets, model-generated response records, evaluation labels/scores, and statistical summaries. No project-wide license is included. The upstream dataset license does not automatically extend to these other materials. Check the applicable terms before redistributing saved model outputs or reusing custom prompts and code.

## Model weights and services

Model checkpoints are not bundled. Obtain checkpoints from their official repositories and use them under their respective terms. Saved files include model outputs and statistics; they do not include all model/tokenizer revision hashes needed for bit-for-bit re-inference.
