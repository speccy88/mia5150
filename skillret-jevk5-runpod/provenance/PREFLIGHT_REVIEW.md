# SkillRet/JevK5 training-kit review

## Verification
- Original archive extracted to `/opt/data/reviews/skillret-jevk5/SkillRet_JevK5_RunPod` without executing bootstrap or training.
- 46 SHA256SUMS entries verified; zero mismatches.
- Fresh isolated Python 3.11 CPU environment: pytest 9.1.1, numpy 2.4.6, torch 2.14.1+cpu.
- Full included test suite: 37 passed. JUnit: `full-cpu-review-tests.xml`.
- No model weights downloaded, GPU training performed, production services changed, or paid resources allocated.
- This execution environment has no nvidia-smi; the physical RTX 2060 Super was not remotely inventoried.

## Hardware decision
The code is not RunPod-specific: training scripts create no pods, and can run on an appropriate local CUDA GPU. However, load_backbone and bootstrap explicitly require native BF16 and compute capability >=8. RTX 2060 Super is Turing (7.5), so it is rejected by the reference. It also has 8 GB nominal VRAM. Approximate 4-billion-parameter 16-bit weights alone consume 7.45 GiB, before CUDA, activations, LoRA tensors, gradients and optimizer state. Changing BF16 to FP16 would not fix the memory problem.

A separate 4-bit QLoRA + FP16 compute experiment could be explored on this card, but hybrid Qwen3.5 backward kernels, quantized projection targeting, numerical stability and peak VRAM must be tested. Existing attention-only module discovery checks isinstance(nn.Linear), so quantized module discovery needs verification. Smaller sequence caps, adapter rank and retriever batches may be necessary. Such changes require a distinct experiment/config/protocol and cannot be described as reproducing BF16 reference results. No fit/throughput guarantee is established.

Local CPU validation/data work is already supported; BGE build has --device cpu and batch size is configurable. BGE and JevK5 are used in separate stages, not simultaneously resident. Quantized inference on 8 GB is a separate, plausible downstream target, not validated by these tests.

## Implementation strengths
- Pinned model/retriever revisions, hashed data, validated family split, natural candidate lists with no validation/test gold injection.
- Appropriate multi-positive categorical loss and explicit NONE case.
- Efficient final-token letter-logit readout; avoids full sequence vocabulary logits.
- Base-parity, LoRA-only, finite gradient, weight-change and checkpoint reload checks.
- Baseline retained in checkpoint selection; negative result can win honestly.

## Issues before GPU run
1. GPU stack and actual hybrid-attention gradient flow are not validated by toy CPU tests. Requirements are ranges, not a tested environment lock.
2. Process deadline is checked between operations; not a hard timeout or billing watchdog. Initialization and pre-try admission failures do not write failure.json.
3. Resume best selection can refer to files in the prior output tree; old runs must remain accessible, not just the checkpoint directory.
4. Cached embedding folder is created before all files are written; interrupted construction can leave an unusable cache requiring quarantine/rebuild.
5. Metadata-only retrieval/reranking and query-level bootstrap limit interpretation: top-15 candidate ceiling, incomplete judgments, unverified semantic paraphrase disjointness, and one seed.

## Recommendation
Do not rent merely to validate this archive. CPU checks pass. For the reference experiment use a native-BF16 GPU with measured memory margin; the suggested 48 GB class is conservative, not a proven minimum. RunPod is one way to obtain it, not a dependency. Prefer the reference pilot for a reliable reproducible result; use a clearly separate local QLoRA feasibility branch if zero rental cost matters more than matching the reference.
