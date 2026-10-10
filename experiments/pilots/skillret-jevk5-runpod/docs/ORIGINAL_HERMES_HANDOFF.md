# Hermes / Codex execution handoff

Implement and validate this package as a NEW experiment on Charlie/RunPod. The user approved preparing a SkillRet-to-JevK5 LoRA pipeline and is considering RunPod. **Do not infer a current spending authorization from historical RunPod budgets.**

## First, inspect and preserve

Read START_HERE.md, PROTOCOL.md, STATUS.md, docs/CLM_LESSONS.md and the scripts. Inspect the existing CLM training package on Charlie if available. Recover exact dataset revision, metadata formatting, split-ID manifest, environment, metrics definitions and teardown practices. Retain copies/hashes; never overwrite original runs, models, notebooks, dashboard records or Hermes defaults.

The Library notes report CLM Recall@5 .0582 → .3174 while BGE-M3 was .5545. These are historical context, not a new independently verified result. Do not fabricate access to missing raw evidence.

## Before any paid operation

Present the current available GPU, current price and CAD equivalent, a target/cap, storage charges, exact teardown plan and scope. Ask Fred for an explicit budget/approval if not already supplied in this session. Prefer one 48 GB A40/A6000/L40S class GPU for the reference pilot; measure rather than guessing. No multi-GPU cluster, expensive H100 by default, always-on endpoints, or automatic second-model run.

Keep a controller-side deadline and check the exact Pod independently. A Python timeout is not a billing shutdown. Never print API keys or unrelated environment variables. Avoid public inference ports; SSH access is enough.

## Validate before spending on a full run

1. Run `python -m pytest`, verify a nonzero collected test count and the negative controls. Tests supplied here were CPU-tested; that is not CUDA validation.
2. Resolve/pin the dataset SHA; refuse unexpected counts. Do not choose a newer revision silently.
3. Reuse the original CLM split manifest only if verified. Otherwise record the new split and counts, without claiming exact prior comparability.
4. Build the pilot using BGE-M3 and NATURAL top15 lists plus NONE. Preserve every query with a retrieval miss. No gold injection into validation or test. Never execute SKILL.md contents.
5. Use merged JevK5 c4f7fdb... plus one NEW LoRA. Do not start from plain Qwen, double-apply the old adapter, or use an ordinary text-generation loss.
6. Validate the actual Transformers text-backbone class/loading warnings and LM-head readout. Confirm all required weights loaded; do not treat HTTP/file presence as success.
7. Check zero-LoRA logit agreement, trainable tensor names/counts, finite nonzero gradients, and actual weight updates. If an upstream API changed, make the smallest recorded patch in a NEW branch and add a regression test.
8. Start with reference kernels (FLA disabled). Do not silently enable an unvalidated backward kernel to make the benchmark faster.
9. Run the 100-update pilot; inspect tokens, memory, step times, baseline/adapted validation behavior, and fresh adapter reload. Preserve crashes too.
10. Estimate full train + validation + encoding + test + export costs. The provided cost script only estimates timed training/validation; add omitted costs explicitly.

If the 4B architecture fails the CUDA smoke test, document it and stop the paid run promptly. Do not silently switch to Strands, Julia, Plumb, a quantized base or a different objective and report it as JevK5 BF16.

## Full run only after the gate

Use a fresh merged-base initialization and the approved full config. Keep the pilot, unadapted baseline, selected checkpoint and final checkpoint distinct. Validation chooses the model, not test. The unadapted baseline may remain selected. A negative finding is valid.

Run Plumb only as a separately approved second experiment with the same frozen candidates/protocol. Not automatically.

## Final evidence and safety

After freezing selection, run official test zero-shot and selected inference with no labels in model inputs, then external scoring. Verify every query is present. Report:

- source/model/data revisions and hashes;
- real split counts and overlap checks;
- trainable parameters, hardware, environment, precision/kernels;
- BGE-only, BGE+unadapted, BGE+adapted Recall@1/5/10, Hit@k, MRR@15, nDCG, candidate recall/ceiling;
- abstention vs forced-ranking behavior, missing labels caveat;
- measured latency, token lengths, truncation, VRAM, actual/estimated charges;
- failures, positive/negative controls, adapter reload parity;
- whether any improvement survives paired uncertainty; one-seed limitation;
- explicit distinction from historical CLM full-catalog results.

Create ONE self-contained results ZIP, including scripts/patches/configs/tests/logs/raw predictions/checkpoints/manifests/reports and a START_HERE. Scan for credentials before transfer. Copy it off the Pod, verify SHA-256 and unzip integrity, then terminate only the exact experiment Pod. Confirm no experiment-owned compute/volumes/endpoints remain billing; preserve teardown evidence and any delayed billing uncertainty. Do not delete unrelated resources.

Do not push public models/datasets, change Charlie's defaults, or create production routing without a separate request. No unattended-agent promotion follows from skill-ranking accuracy alone.
