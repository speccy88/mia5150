# Local BM25 reranking diagnostic

Local JevK5 completed 80 finite-score evaluations over two fixed public development queries, two rubrics and 20 candidates. **Recall@5/10 remained 0.75**, already the pool ceiling. The constraint rubric improved NDCG@10 from **0.521912 to 0.622038** relative to BM25/original direct scoring. This two-query exposed diagnostic is not a confirmed benchmark improvement.

| Local JevK5 arm | Recall@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.750000 | 0.750000 | 0.521912 |
| Oracle | 0.750000 | 0.750000 | 0.806574 |
| original_direct | 0.750000 | 0.750000 | 0.521912 |
| original_fusion | 0.750000 | 0.750000 | 0.556574 |
| constraint_direct | 0.750000 | 0.750000 | 0.622038 |
| constraint_fusion | 0.750000 | 0.750000 | 0.556574 |

The same original/constraint rubric text, BM25 candidate pools, first two development IDs and stable retrieval-order ties were fixed before local inference. The existing native JevK5/SemIf four-option prompt and log-probability readout were used; the previously configured JevK5 Q5 temperature 1.22 was retained, not tuned on SkillRet. All 80 decisions contained every required option letter in the top256 token log-probabilities, so no probability-floor approximation was used. Model SHA256 and local adapter source hashes are pinned in the manifest. Hosted TypeSafe Jev and this local JevK5 model are different systems.

## Resource feasibility and remaining work

- **JevK5 Q5:** completed, 80/80 decisions valid, four complete query/rubric rankings.
- **Plumb Q4:** stopped by the resource guard during loading, before any prediction was saved. Windows available RAM was 2,129,534,976 bytes, below the unchanged 2GiB reserve. No retry. This is a feasibility failure, not measured retrieval quality.
- **Qwen3.5 Q4:** not loaded. After the Plumb pressure event, a conservative loading check required the model bytes plus 2GiB reserve and 128MiB staging overhead. Available host memory was 4.482GiB versus 4.678GiB required. No retry, limit relaxation, unrelated process termination or download.

All requested local models have an explicit execution status. Plumb and Qwen have no quality score; substituting a zero quality score would conflate resource availability with model accuracy. The complete failed-load guard log is preserved. Further local comparison needs more idle host RAM while keeping the same WSL limits, or a separately designed smaller workload; no such follow-up was silently launched.

JevK5 median wall time per 20-candidate rubric ranking: 8.303s. Input tokens across 80 pointwise decisions: 52,372. Total including model hash verification, startup, inference and shutdown: 47.879s. Cached BM25 retrieval is excluded. This differs from hosted multi-question batching, so no direct speedup ratio is claimed.

During the completed run: host minimum available 2.073GiB, guest 3.854GiB, GPU minimum free 4824MiB, GPU maximum 64C; no active swap observed. Guard samples cover host/guest/GPU reserves. Client-process RSS does not represent full server memory and is not used as model memory. WSL stays 6GiB RAM / 2 CPUs / 2GiB swap; llama.cpp uses one CPU thread and direct-I/O loading.

## Validation, cost and reproduction

67 offline tests pass, including finite answers, mock answer-order mapping, stable ranking, local endpoint restriction and maximum local-forward/context bounds. No external API, saved-key access, model upload or download was required by this diagnostic. New spend is zero; shared commitments remain USD1.87 after the separately approved hosted screen.

The one-shot local runner is `round2/local_rubric_screen.py`; completed/stopped output paths reject reruns. Offline report regeneration: `.venv/bin/python round2/report_local_rubrics.py`. Tests: `.venv/bin/python -m unittest discover -s round2 -p "test_*.py"`. Run from `/home/user/code/mia5150-local`.

The six-query hosted screen report is `round2/VARIED_SCREEN_RESULTS.md`. These two-query local metrics must not be merged with the six-query hosted summary or the earlier dense-retrieval confirmation.

Local protocol SHA256: `445bc052e58ca689a15148310df4b5b4e75c4e7137b461f57d25524692527cee`.
