# RunPod operations and cost controls

This starter kit does not have or need a RunPod API key. It does not create or terminate Pods. A separate authorized operator/Hermes controller must manage billing and teardown.

## Before launch

Record current GPU availability, exact GPU count/type, native BF16 capability, advertised rate/currency, CAD conversion and its date if applicable, storage rate, spending target/hard cap, billing start time, experiment-owned Pod ID and an independent hard deadline. Historical USD3/USD5 CLM budgets are not a new authorization.

Prefer a normal single-GPU Linux/PyTorch Pod for one bounded training experiment. Allocate storage under /workspace or another explicitly persistent location on that Pod. A 48 GB class GPU is the conservative BF16 pilot recommendation; it is not a tested minimum. Do not advertise a precise training time until the pilot is measured.

Keep the RunPod account/API credential on the controller, not inside training logs or this ZIP. Bind any optional service to localhost; this pipeline does not require an inference API exposed to the internet. Do not run this on the production Charlie environment in place of working services.

## During execution

The quoted GPU rate applies while the Pod is allocated, not only while CUDA is busy. Include downloads, dependency installation, BGE indexing, tokenization, validation, export and idle debugging. `pilot_report.py` explicitly excludes these unmeasured costs; it is not a total invoice estimator.

A process time cap is best effort between operations. A blocked download, kernel hang, SIGKILL, disconnected client or crashed process can leave paid infrastructure alive. The controller must supervise independently. When a cap approaches, preserve a completed checkpoint and copy evidence before shutdown; do not silently raise the budget or switch GPU/precision.

Use a safe filename for every new run. Never overwrite completed run directories. Resume only a complete optimizer-boundary checkpoint, preserving its original run and source manifest. Hash checks do not make an untrusted pickle safe; load only locally produced training_state.pt files.

## After execution, including failures

1. Create a results ZIP with public source references, code, patches, prepared inputs, split IDs, logs, checkpoint(s), inference outputs and scoring.
2. Inspect for secrets. Do not include general environment dumps, HF tokens, SSH keys, RunPod credentials or unrelated user files.
3. Copy the ZIP to the controller / Charlie. Verify archive integrity and SHA-256 **off-Pod**.
4. Terminate only the exact experiment-owned Pod after this verified copy. Stopping a Pod may leave storage charges; termination may delete its ephemeral data.
5. Inspect the account again and record whether experiment-owned Pods/volumes/endpoints remain. Never bulk-delete all account resources.
6. Reconcile posted versus estimated charges; billing may be delayed. A successful process exit or GPU returning to idle is not proof of financial cleanup.

## Reproduction boundaries

- Source revisions are pinned. Python package ranges are a proposed compatible minor-version band, not a fully GPU-tested environment lock.
- `bootstrap.sh` records the resolved installation (including the JevK5 git commit from direct_url.json) and checks that CUDA PyTorch still works.
- Preserve the final pip freeze and exact Pod image digest after the successful pilot. Use that environment for the accepted full run; do not upgrade it mid-experiment.
- An incomplete download/cache directory fails visibly. Preserve it for diagnosis, then retry with a new directory/cache or quarantine it explicitly; do not silently reinterpret partial data as complete.
- Kernel optimizations, quantization, adapter merging and GGUF export are downstream experiments. A BF16 LoRA result is not evidence of identical 8 GB inference behavior on Charlie.

Primary operational references: https://docs.runpod.io/pods/manage-pods and https://docs.runpod.io/pods/storage/types (inspected October 2, 2026).
