# Evidence status

Prepared October 2, 2026.

## Completed here

- Retrieved the prior CLM experiment's recorded results/method notes (not the original raw archive).
- Inspected official SkillRet schema/statistics, JevK5 prompt/runtime/training source, model revisions, PEFT concepts, and RunPod lifecycle documentation.
- Implemented data preparation, multi-positive conversion, BGE shortlist pipeline, LoRA training scaffold, checkpoint/resume, inference, strict scoring, cost extrapolation and evidence export.
- Executed CPU unit/negative-control tests and a tiny LoRA optimization/reload/resume test; see the actual evidence files.
- Created synthetic schema examples and two explanatory notebooks.

## Not completed here

- No real SkillRet JSONL download/conversion (source dataset revision must resolve on RunPod).
- No JevK5, Plumb or BGE model download/inference.
- No 4B LoRA training, no CUDA smoke test, no real model checkpoint, no test-set model score.
- No RunPod deployment, paid spending, price quote, billing cleanup or automated remote agent execution.
- No retrieval of the original CLM split-ID manifest/raw training package.

The starter ZIP contains runnable preparation/training code and local test evidence. It does not contain a trained SkillRet-JevK5 model. Never relabel synthetic/toy test results as real model benchmark results.
