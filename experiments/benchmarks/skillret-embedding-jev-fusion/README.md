# SkillRet-0.6B full-text baseline and Jev RRF60 study

Dense retrieval has the highest pilot NDCG@10 point estimate. Jev reduced NDCG@10 by 0.136732; RRF60 changed it by -0.015567 (95% paired bootstrap CI [-0.037712, 0.006697]). RRF60 did not demonstrate an NDCG improvement, although its Recall@10 increased from 0.890000 to 0.903333. These conclusions apply to the fixed 200-query pilot.

The hosted smoke (10 queries) and pilot (200 queries) are complete with jev-1.13.0. All 200 pilot IDs and their fixed dense top-20 pools were retained. Pilot NDCG@10 is 0.830145 for dense retrieval, 0.693412 for Jev, and 0.814578 for fixed RRF60. The experiment has 779 usable scoring responses from 780 reserved transport attempts, including the 40 smoke responses reused without another charge. Known token usage is USD 0.403215288 at the published rate, or CAD 0.78871934447460000 after planning allowances. One original request lost during a WSL restart remains execution/billing unknown; its full USD 0.002752512 / CAD 0.00538411991040000 reservation is retained. An identical-payload recovery supplied the first available response without dropping a query. The 4,392-query hosted expansion was not run: its pilot-based projection is USD 8.85736023648 / CAD 17.32566092457261600, above the USD 5 / CAD 10 caps. Read [actual findings](results/hosted-study-findings-final/findings.md), [pilot evaluation](results/hosted-pilot-native-field-evaluation/evaluation.json), and [full expansion decision](results/full_expansion_decision.json).

The full local baseline is complete and passed the reproduction gate. All 4,392 test queries, 6,006 full-text skills and 7,187 binary relevance pairs were evaluated using the released ThakiCloud/SKILLRET-Embedding-0.6B checkpoint. NDCG@5/10/15 is 0.790523 / 0.810990 / 0.818848; the largest absolute difference from the official targets is 0.000323, within the 0.005 gate. Previous experiments and caches remain intact. Read `results/summary.md`, `results/baseline_gate.json` and the saved rankings for the evidence.

Two HTTP 200 responses exposed unsupported response-arithmetic assumptions. Both raw responses were preserved and reconciled without replay. The final policy validates structure, model/rubric, ranges, usage, hashes and ledger provenance while recording probability/score/confidence arithmetic as diagnostics. It preserves all native values and the original score/confidence/dense-rank ordering. See [response validation audit](SCORE_RESPONSE_AUDIT.md); the amendment was frozen before comparative quality metrics were inspected.

The portable archive and published CSV omit one query-text cell containing credential-shaped examples from the public benchmark. Query IDs, rankings and metrics are unchanged. The original full text and unmodified CSV remain in the local WSL experiment; the pinned dataset reconstructs the input. `results/secret_scan.json` and `results/artifact_manifest.json` record this export-only redaction and both hashes. An all-x documentation token is recorded as a verified placeholder. No environment secret is included.

## Protocol

Each skill is `name.strip() + ' | ' + description.strip() + ' | ' + skill_md.strip()`. Missing full text fails validation. Query input is exactly `Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: ` followed by the original query. The newline is an actual newline, and the prefix ends in one space. No relevance annotations enter the model.

The official evaluation limit is 32,768 tokens. Batch size one, FP16 weights, last-token pooling, float32 L2 normalization, and dot products retrieve top 50 from the full catalogue. This GPU lacks native BF16 support. The saved named query prompt is deliberately overridden. PyTorch memory-efficient SDPA is used with explicit repeated key/value heads, mathematically equivalent to grouped-query attention; quadratic math fallback is disabled on this 8 GB GPU.

All model files and test JSONL files are SHA256 checked against pinned manifests before execution. `protocol.json` binds source/runtime versions, document construction, token limits and cache identity. `SOURCE_AUDIT.md` gives pinned primary sources and explains why earlier metadata/1,024-token caches are not the official baseline.

## Local execution

Existing local asset root: `/home/user/code/mia5150-local`. No download, hosted service, or credential is needed for baseline execution. Keep the Windows host monitor running during WSL execution:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\host_monitor.ps1 -OutputPath "$PWD\host-telemetry.jsonl"
```

In WSL, from this experiment directory, choose a **new** run directory for each attempt:

```bash
../.venv/bin/python resource_guard.py \
  --host /mnt/c/Users/user/Documents/Codex/2026-10-06/task-3/skillret-embedding-jev-fusion/host-telemetry.jsonl \
  --run-dir runs/NEW-attempt \
  -- ../.venv/bin/python run_skillret_baseline.py --stage full --forward-cooldown 0.05
```

Stages are `audit`, `feasibility` (three longest skills), and `full`. Catalogue vectors are cached once and checkpointed after every document. On resume, recorded vectors are checked for shape, float32 dtype, finite unit norms, and per-row hashes. Corrupt arrays or journals stop the run; preserve them for investigation. An unjournaled last vector after interruption is safely recomputed. Queries have the same resume behavior. `results/dense_top50.jsonl` freezes candidate IDs and scores for the later study.

The supervisor requires fresh host telemetry (<8 seconds), at least 2 GiB available on both host and guest, unchanged swap-in/out counters, no more than 7,168 MiB device memory, GPU temperature at most 80 C, and no five consecutive host-CPU samples over 80%. It terminates only its own child process group on failure and preserves evidence. Windows monitor stop: create an empty file at `<host telemetry path>.stop`.

The runner defaults to a 50 ms inter-forward safety pause after corrected full attempt `06-full-correct-rope` reached the host-CPU guard. Resume verifies completed vectors and computes only missing ones. Reported active query latency excludes this pause and cache writes; run telemetry supplies actual wall time. `startup_seconds_this_run` measures model loading, separately from source hashing, token audit and package import time.

Run `07-full-cpu-cooldown` left 6,006 corrected document vectors and 961 query vectors before an unexplained interruption; it has no exit or guard-stop receipt. Run `08-resume-query-cache` resumed the verified cache, completed all 4,392 query vectors and rankings, and exited with code 0. The final ranking artifact and validated `baseline_gate.json` establish reproduction within the configured tolerance. After completion, GPU use was 0 MiB, utilization 0%, and temperature 50 C. See `COMPATIBILITY_FINDINGS.md` for preserved failed and interrupted attempts.

## Evaluation and future hosted stage

`evaluate.py` implements official TREC conventions, full query denominators, NDCG/Recall/Completeness/MAP/Precision at 5/10/15, MRR at 5/10, macro candidate recall and micro relevant-pair coverage at 20/50, deterministic SHA256 pilot200/smoke10 selection, fixed-pool RRF60 and paired query bootstrap. Offline tests cross-check TREC metrics against `pytrec_eval`.

The reproduction gate requires all full test rankings and maximum absolute NDCG deviation <=0.005 at 5/10/15. Larger differences require a documented investigation before hosted interpretation. A dense top-20 pool is fixed for A, B and C; Jev cannot recover missing candidates. Coverage@50 belongs to the upstream dense retrieval pool, not to a reranked 20-item result.

`run_jev_rerank.py` supports offline request preparation and structurally validated ingestion of genuine saved responses with arithmetic diagnostics; its `execute` command remains disabled and never reads credentials. The separate hosted transport defaults to disabled and reads only `TYPESAFE_API_KEY`; the specifically authorized saved-key handoff supplied that variable temporarily for the authorized model-list and scoring processes. The [hosted study runner](HOSTED_STUDY_RUNNER.md), `run_hosted_study.py`, supplies disabled-by-default orchestration, checkpoints, resume and ledger accounting. Its status mode makes no API call or credential access; paid mode requires genuine evidence receipts and explicit execution. The complete baseline, metric, preparation, client and runner suite passed 113 offline tests after the independent review fixes; see `results/final_offline_validation.json`.

Budget and process-only key handoff were explicitly approved. The user confirmed automatic refill is disabled and directed acceptance of reported USD 5 signup credit without a balance query. Current public pricing and FX receipts, an additional USD 5 consumption cap, and planning contingencies were enforced. These are not a live account-balance observation. No top-up, refill or RunPod action was taken. Historical billing assessments are superseded by the actual [stage status](results/hosted_stage_status.json).

The user-prespecified five-level rubric and fixed dense top-20 pool are frozen in request plans. Original dense ranks break exact score/confidence ties. Fusion uses equal weights and k=60 with no test-set tuning. Actual response version/usage/raw bytes must be saved before parsing. No Jev, RRF, usage, bootstrap, or comparative failure result is fabricated when calls have not run.

The provider documents rounded limits of 64k total context and 32k state plus the longest question without defining the exact boundary. The study interprets these locally as 64,000 and 32,000. For both a complete request and each same-state single-question request, the local check counts canonical UTF-8 JSON bytes plus 2,048 input-framing allowance and 2,048 output allowance. These are conservative study choices, not native Jev token counts, a proved token upper bound, or verified server boundaries. See the [context source audit](CONTEXT_SOURCE_AUDIT.md).

The original full-text-only preparation admitted 6 of 10 smoke queries and 104 of 200 pilot queries; 4 and 96 queries were blocked, involving 5 and 143 candidate admission failures respectively. Those plans remain as historical evidence. These were local estimate failures, not provider rejections or proof of native context overflow. Splitting a batch cannot fix an individual candidate that fails the same-state single-question check.

The user subsequently authorized deterministic batching and logged right truncation of `skill_md` when needed by this local check. The opt-in `truncate_skill_md_right` policy preserves every query, candidate ID, name, description and rubric, and retains the longest fitting Unicode prefix of `skill_md`. It preserves original text and records removed spans, counts and hashes. It never substitutes candidates or shrinks the fixed query denominator. If immutable content plus the required nonempty body cannot fit, the whole query still blocks. These changes affect future Jev input only; the validated dense baseline remains unchanged.

The completed [context preparation audit](results/context_preparation_audit.json) records all 10 smoke and all 200 pilot queries prepared, with zero blocked queries. The smoke has 200 query-candidate pairs: 5 truncated and 195 full text, across 40 planned requests. The pilot has 4,000 pairs: 143 truncated and 3,857 full text, across 779 planned requests; 96 queries and 103 unique skills are affected. The smoke is the fixed ten-query subset of the pilot, so these scope counts must not be added as independent usage. Retained `skill_md` bytes are 97.5853% for smoke and 95.7207% for pilot. Original selected IDs and top-20 pools are unchanged. An optional [context summary](results/context_preparation_summary.md) presents the same evidence. Local preparation readiness does not verify provider token admission, account credit/pricing/refill settings, the live model pin or other execution gates. Future quality analysis must distinguish full-text pairs from truncated pairs.

Prepare a new deterministic pilot snapshot locally under the authorized policy (use `--scope smoke` and a separate new output directory for its ten-query subset; preserve existing plans):

```bash
../.venv/bin/python run_jev_rerank.py prepare \
  --queries ../data/queries/test.jsonl --skills ../data/skills/test.jsonl \
  --dense results/dense_top50.jsonl --output-dir results/NEW-pilot-truncated \
  --scope pilot --model jev-latest --context-limit 64000 --pair-context-limit 32000 \
  --overhead-tokens 2048 --reserved-output-tokens 2048 \
  --oversize-policy truncate_skill_md_right
```

Generate the English report and diagnostics only after the baseline worker has exited and its result files are complete:

```bash
../.venv/bin/python report_results.py --experiment-dir .
../.venv/bin/python diagnostics.py --experiment-dir .
```

```bash
../.venv/bin/python -m pytest -q test_evaluate.py test_jev_prepare.py test_baseline.py test_hosted_jev_client.py test_hosted_study.py
```

The benchmark is public and has been inspected in prior studies. This is an exposed public test-set reproduction, not a newly blinded evaluation. Claims about Jev improvement require actual paired results and confidence intervals.

The pinned software, hardware and runtime adaptations are summarized in [ENVIRONMENT.md](ENVIRONMENT.md). The independent runner review is recorded in [RUNNER_REVIEW.md](RUNNER_REVIEW.md).

The subsequent billing-policy integration passed all 120 offline tests; see `results/billing_policy_offline_validation.json`. No real credit or refill receipt was fabricated.
