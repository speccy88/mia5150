# MIA5150 SkillRet checkpoint — 2026-10-05

Paused new stages on parent request. Implementation is not complete. A healthy, bounded LoRA training job remains active.

## Location and running jobs

- Ubuntu-26.04 WSL project: `/home/user/code/mia5150-local`.
- Windows staging: `C:\Users\user\Documents\Codex\2026-10-05\task-2`.
- Active execution session: **29552**. WSL guard PID **11530**, training worker PID **11544**. Command: `python3 guard.py lora-hard-streamed .venv/bin/python lora_pilot.py hard`.
- Host telemetry execution session: **32275**, appending `host.jsonl` in Windows staging. Keep it active until GPU work ends. The guard fails closed if this file is stale.
- At checkpoint, training logged step 160 of at most 384; loss finite. Guest available RAM 3.34 GiB; host available RAM 2.27 GiB; GPU free 6,552 MiB; temperature 57 C. Swap counters unchanged throughout the healthy retry.
- Check `runs/lora-hard-streamed/exit.json`, `worker.log`, `guard-stop.json`, and `results/lora-hard-adapted.json` before resuming. Do not start another GPU job until this process finishes.

## Completed and measured

- Exact Library ZIP materialized into Ubuntu, readable at `plan/SkillRet_Research_Codex_Plan_2026-10-05.zip`, 182,489 bytes; SHA-256 `a4f2ca4604a1b7c6120f8533c80eca1d2fee540b75a6c6d1975287af8cf89a0d` verified. Its execution work orders, contract, proposal and relevant research reviewed.
- Official source pinned at `0cc508e6bfcfff1fe6352d90ff5957ba38a64bba`; dataset pinned at `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`. All six split counts match the authoritative plan. File hashes in `evidence/data-manifest.json`.
- Six metric/leakage tests passed, including official pytrec_eval MAP parity, duplicate ranks, missing-query denominator and MRR cutoff.
- Frozen public development panel: 128 queries, full 6,006-skill catalog. It is NOT an unseen final test.
- Header NDCG@10: official-version BM25 **0.598819**, base Qwen FP16 **0.528662**, SkillRet embedding FP16 **0.707372**, SkillRet+BM25 RRF **0.722561**. Full-document BM25 **0.472932**. These are separate evidence views; not an official 8B reproduction.
- Same 32 queries and fixed SkillRet top-20 pools: embedding NDCG@10 **0.713387**, SkillRet reranker **0.799176**. Total reranker time about **57.95 seconds**; native yes/no logits, header view, FP16.
- Clean supervised lexical adaptation: 512 training / 128 development queries, repository/name groups disjoint, crossing multi-skill queries and duplicate text excluded, development-group documents excluded from training negatives. NDCG@10: untrained RRF **0.626014**, hard-negative learned fusion **0.634344**, label smoothing **0.636961**. Deeper recall regresses. No promotion claim yet; confidence intervals pending.
- Native rank-4 LoRA toy overfit passed (32 optimization steps, final losses below 0.001). Real adapter checkpoint at step 7 was saved, deliberately cleared, restored exactly with optimizer/scaler/RNG, and training resumed at step 8. Evidence: `evidence/lora-native-smoke.json`, `checkpoints/lora-hard-resume-smoke/verification.json`.
- All eight Apache-licensed 0.6B baseline/specialist checkpoints downloaded and pinned: Qwen embed, SkillRet embed/rerank, SkillRouter embed/rerank, SkillReason embed/rerank, R3 rerank. R3 native chat template inspected and adapter corrected. Add its newly downloaded `chat_template.jinja` hash to its model manifest on resume.

## Implemented but not yet fully executed

`run_sequence.py` provides sequential guarded CPU, specialist, training, medium-model and scaling stages. Additional scripts include field TF-IDF, MMR, query-residual DPP/DSR ablation, domain routing, common-pool reranking, native quantized JevK5/Plumb pointwise plus separate 15+NONE order diagnostic, three-seed 10/100/1000/6006 nested-library experiments, paired bootstrap statistics, audit/notebook generation, and a local all-query classroom case viewer.

Run remaining stages only after inspecting the current LoRA outcome. `run_sequence.py training` will refuse existing run directories; do not rerun it wholesale. The replay LoRA variant has not started. Specialist inference, quantized SkillRet experiments, structural/field tests, scaling, viewer rendering, final report and portable ZIP remain unfinished. Full-catalog prompt scaling adapter still needs implementation. Full-body neural indexing and external transfer are not run.

## Important caveats and repairs

- Several earlier dependency/download/indexing attempts correctly stopped on active swap. Resource limits were never relaxed. File-specific cache hints and lazy corpus loading resolved current runs. Preserve their failure logs.
- Initial `results/adaptation-*.json` files without `clean` are superseded: early negative mining included development-group documents, and the initial `replay` label actually meant label smoothing. Do not report them as clean training evidence. Corrected files contain `adaptation-clean-*`.
- SRA-Bench access/license metadata verified (MIT, ungated; revision `6143f2634eb284955ce312213bac24b582d039f3`), but its data/labels have not been consumed. Cross-benchmark exposure still needs inspection.
- Jina 3.5 metadata/card reviewed; CC BY-NC 4.0 and custom remote-code path. Weights/code have not been executed; review/authorization gate remains.
- No paid API, RunPod, credentials, model uploads, Git push, security change or unrelated service termination. Speech work has not started.

## Resume commands

All commands run in `/home/user/code/mia5150-local`, using `.venv/bin/python`; use a fresh guard run name. Keep Windows host telemetry running. First inspect the active job, then proceed sequentially.

1. Finish/review current hard-negative LoRA base vs adapted outputs.
2. `python3 reclaim.py`; `python3 guard.py lora-replay .venv/bin/python lora_pilot.py replay`.
3. `python3 run_sequence.py specialists`, then `cpu`, `medium`, and `scale`, resolving any failed stage without overwriting evidence.
4. Generate comparison/statistics/audit/demo after completion; finish full-prompt and remaining bounded scope; package one ZIP with START_HERE, scripts, manifests, raw results, logs and explicit coverage ledger.

The final package has NOT been produced yet. This checkpoint is a resumption handoff, not a completion report.
