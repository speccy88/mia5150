# Implementation review and remaining validation

## Checks implemented in code

- Exclusive output-file creation; new run directories, including resumes.
- Pinned source/model revisions and hashes; expected dataset counts and qrel referential integrity.
- No model-input query labels; type-safe canonical skill text; no SKILL.md execution.
- Exact/provenance query-family split checks; optional prior manifest validation.
- Natural candidates preserved on validation/test; all gold labels retained for metric denominators.
- Multi-positive categorical target construction; NONE for retrieval misses; position randomization.
- Efficient answer-letter-only logits from the last real prompt token; no free-text generation.
- Per-field truncation audit; fail rather than silently skip oversize final prompts.
- CUDA/BF16/no-quantization gate; LoRA-only trainable tensor check; finite/nonzero gradients.
- Zero-LoRA baseline parity; complete optimizer-boundary checkpoints; RNG/optimizer resume state.
- Baseline included in validation checkpoint selection; it can beat every trained checkpoint.
- Inference does not read test qrels; scorer rejects missing queries and changed candidates.
- Forced ranking and NONE decision success are separate; paired comparisons expose retrieval ceiling.
- Optional fresh-process adapter reload verification.

## CPU evidence that can be trusted here

The supplied test suite executes using `python -m pytest`, including negative controls and a tiny trainable LoRA. Tests are not invoked as ordinary Python files. Results/JUnit are saved under evidence/. A toy optimizer restart check confirms CPU resume math. These tests validate local data/loss/metric logic, not remote model behavior.

## Still requires RunPod validation

1. Dependency resolution on the actual CUDA image, including Transformers Qwen3_5ForCausalLM and PEFT behavior.
2. Loading the exact merged text checkpoint without missing critical tensors. Investigate unexpected/missing-weight warnings rather than accepting them blindly.
3. Real tokenizer/template parity, all 16 answer tokens, and truncation distribution on actual SkillRet rows.
4. Dense BGE-M3 normalization, text-limit counts, indexing latency and full source count checks.
5. Actual LoRA gradient flow, throughput, peak VRAM and GPU residency for the 4B model.
6. Finite/correct hybrid-attention backward kernels and real checkpoint reload/resume consistency.
7. Quality improvement, statistical uncertainty and complete end-to-end cost.

## Deliberate limitations

- One seed and one epoch initially; not a broad hyperparameter search.
- Query bootstrap, not fully family-aware confidence intervals.
- Exact provenance/text families do not prove semantic paraphrase disjointness.
- Official data can contain incompletely judged alternatives; NONE reflects qrels, not a universal truth.
- Metadata-only input differs from the paper's full-text protocol and may omit critical skill constraints.
- A 15-candidate shortlist caps achievable recall. A reranker cannot recover a missing skill.
- Uniform target probabilities are a training design, not calibrated usefulness probabilities.
- Different normalized skill aliases/near-duplicates may still appear across catalogs despite distinct IDs; inspect the pinned benchmark's own deduplication methodology and report limitations.
- Deadline checks are between operations, not a hardware/billing watchdog.
- No automatic Pod provisioning, account cleanup, public model upload, quantization, or deployment changes.
- No formal whole-program verification is claimed. The package is a tested scaffold with an explicit GPU pilot gate, not a guarantee of compatibility or performance.
