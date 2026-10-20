# Model Screening Procedure and Scope

## Search procedure

The study's candidate search used Hugging Face model listings with native filters for text-generating capability and fewer than 10 billion parameters. The pipeline tags were `text-generation`, `audio-text-to-text`, `image-text-to-text`, `video-text-to-text`, and `any-to-any`, with a February 2026 cutoff. Repository metadata and model-card information were used for model provenance and lineage checks, including rules and regular-expression filtering. A machine-assisted assessment considered whether a base/pretrained checkpoint and an officially documented reasoning-capable checkpoint or mode with a trace separable from the final answer were available. Qwen, OLMo, and Granite were carried forward for focused evaluation.

A timestamped scraper output and complete machine-readable candidate list are not included. Accordingly, this account documents the screening method and recorded outcomes but does not claim an exhaustive census of every eligible model.

## Criteria and scope

The criteria used in this study are design choices: open weights, first-party release, dense architecture, availability of a base/pretrained checkpoint, fewer than 10 billion parameters, and an officially documented reasoning mode/checkpoint that emits a separable trace. The parameter ceiling and dense-only choice are practical constraints, not general requirements for generative agent-based modeling. Documentation of a trace format establishes access to displayed output; it does not establish that the trace is faithful to internal computation or causally necessary for the final answer.

## Candidate families not retained

The recorded screening rationale identifies the following candidates as not carried forward under the output-channel requirement or the study's general-purpose scope:

| Family | Recorded reason |
|---|---|
| Tencent Hunyuan Dense | Base/Instruct and fast/slow thinking were noted; an explicit chain-of-thought emission channel or separable `<think>` trace was not verified. |
| Ministral 3 | Reasoning variants were noted; structured separation of reasoning and final answer was not verified. The 14B variant is also above the study's parameter ceiling. |
| ILSP Llama-Krikri | Base and Instruct releases were noted, but not an explicit thinking-mode separation or distinct reasoning-tagged checkpoint. |
| Xiaomi MiMo-7B | Reasoning-oriented training was noted, but an explicitly separable reasoning trace format was not verified. |
| ByteDance Seed-Coder | A reasoning variant was noted, but explicit chain-of-thought output separable from the final answer was not verified; its scope is specialized. |
| DeepSeek-Math 7B variants | Separable reasoning traces were noted for first-party sub-10B mathematics variants, but the models are purpose-specific rather than general-purpose candidates for the comparison. |
| Qwen3-VL | Multimodal reasoning traces are less directly comparable to the text-only empirical setting. |

These exclusions follow the study's criteria and intended scope; they are not claims that the listed families lack reasoning capability in general.
