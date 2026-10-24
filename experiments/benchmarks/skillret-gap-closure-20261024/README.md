# Matched local discovery study and proposal reconciliation

The 200-query primary and both local selector campaigns are complete with retained context exclusions and failures. Full-query retrieval is a separate stage. [Execution status](EXECUTION_STATUS.json) distinguishes actual outputs, preflight exclusions and unattempted cases. The original80 hosted model/method/size cells remain visible in [the crosswalk](proposal-80-cells.csv); local Qwen measurements are explicitly a reduced substitute.

The frozen primary comparison uses200 exposed official SkillRet queries, 98/78/24 with one/two/three positive labels, seed 42, five methods and catalog sizes10/100/1000/6006. A separate balanced 24-query panel supplies additional catalog seeds and a matched4B sensitivity. Twelve generated compositional probes retain silver labels and separate reporting. See [SPEC.md](SPEC.md), [protocol.json](protocol.json), [scope drift](scope-drift.csv), [label audit](label-audit.json), and [the independent baseline audit](AUDIT.md).

The selector is direct, constrained JSON metadata selection with no tools or host filesystem. It is not a new native SkillsMiddleware reproduction. Historical native integration remains in the separate pilot. Retrieval, selection, activation and task execution have separate meanings; no skill execution is claimed.

The official pinned BGE ONNX CPU export avoids the host-memory pressure observed with CUDA setup. Its synthetic parity fixture agrees with pinned PyTorch CPU within 1.23e-7 maximum absolute difference. [The embedding manifest](embedding-manifest.json) records hashes, tokenizer, CLS pooling, normalization, truncation and setup timings. The machine has8GB VRAM and 16GB host RAM; existing WSL limits/settings are unchanged. Preparation and GPU jobs now run fully sequentially, with a2GiB host/guest reserve and 1GiB free VRAM reserve. Resource stops preserve every attempted outcome and append later attempts only for unattempted cases.

[Implementation corrections](IMPLEMENTATION_CORRECTIONS.md) and [measurement corrections](MEASUREMENT_CORRECTIONS.md) explain resume accounting, routed-subset timing and unknown usage/cardinality. Existing baseline files and historical failures stay intact. All recent public queries are exposed; this is not blind evaluation or proof of generalization to independently authored requests.

## Reproduction

Use an isolated Python3.12 environment and [the pinned requirements](requirements-runtime.txt). Keep runtime/model assets outside the public checkout. Populate `runtime-config.example.json` privately with local paths. Fetch the immutable assets with `prepare.py`; selector download receipts and complete GGUF hashes are in `model-provenance.json`. Reuse llama.cpp b11425 and verify its binary hash against execution receipts. No downloaded skill instructions are executed.

```sh
python -m pytest -q
python prepare.py --runtime-root /absolute/private/runtime-data
python run.py freeze --runtime-config /absolute/private/runtime-config.json
python run.py embed --runtime-config /absolute/private/runtime-config.json
python supervise.py --runtime-config /absolute/private/runtime-config.json
# After both selector stages, run CPU preparation/retrieval sequentially:
python run.py embed-full --runtime-config /absolute/private/runtime-config.json
python run.py retrieval --runtime-config /absolute/private/runtime-config.json
python aggregate.py
python plot.py
```

The Windows host monitor writes `host-memory.json` in the workspace containing the checkout. Keep it running during local work; stale/missing telemetry blocks execution. Supervisor waits for 4.5GiB headroom before loading and stops after 15 minutes without recovery. It never changes Windows/WSL settings or stops other services. The runner's source snapshots bind measured cases to actual implementation. Bulk vectors are local assets; archived per-case outputs and public pinned inputs support offline aggregate regeneration.

Tables/charts and closure status are regenerated from saved ledgers. Read report/RESEARCH_DRAFT.md and GAP_CLOSURE.md. Human authoring/adjudication, a fresh external panel, original hosted cells and execution/ordering remain distinct pending evidence. The [annotation guide](ANNOTATION_GUIDE.md) specifies the human work without inventing labels.

The [independent primary validation](audit/INDEPENDENT_VALIDATION_README.md) reproduces all 4,000 primary cases, 20 weighted cells and eight paired intervals against immutable commit ed12adad. It does not claim validation of later sensitivity or full-query retrieval.


Regenerate supplementary diagnostics with sensitivity_analysis.py (query-clustered balanced 24 intervals and seed SD), error_summary.py, resource_summary.py, then plot.py. The paper/Beamer source compilation could not be confirmed by the app compiler and no terminal TeX installation is present; sources remain research drafts for review.


On Windows, run monitor_host_memory.ps1 before the Linux stages. It writes atomic host-memory telemetry in the checkout's parent workspace until that workspace's host-monitor.stop marker appears. Use an isolated workspace without a previous stop marker. Only the benchmark workload stops on guard failure. Compressed full-retrieval evidence preserves exact raw bytes and can be rescored without model assets.


The [completed-local independent audit](audit/INDEPENDENT_SENSITIVITY_README.md) verifies all 5,680 canonical cases, official qrels/generated unions, 100 summary cells and 80 seed cells within its explicit limits. This external review does not cover human labels, full-query retrieval or hosted execution. The full-query streaming arithmetic and overlap check is a separate owner-run verification.


The [final independent retrieval and supplementary audit](audit/INDEPENDENT_FULL_RETRIEVAL_README.md) targets immutable data commit 0f4e9818b6cefb94e09da30ad76b9ff5ac97999c. Its standalone verifiers and unchanged passed receipts cover 52,704 retrieval cases, 158,112 ranker metric rows, 36 aggregates, 2,400 primary overlaps, 26,352 additional-seed full-catalog rank comparisons, and the supplementary bootstrap/error tables. Recorded timing arithmetic was checked; timing was not independently remeasured. Exposed-query, human-label, hosted and execution limitations remain explicit.


[Visual gap analysis and results summary (PDF)](report/MIA5150_Gap_Analysis_and_Results_2026-10-24.pdf) — eight validated pages, preserved unchanged. The PDF cites data commit 0f4e9818b6cefb94e09da30ad76b9ff5ac97999c; later documentation/audit commits do not change that data identity.


Hosted follow-up after the immutable PDF/data snapshot: authentication succeeded, the original failed DeepSeek canary now carries an approved permanent US$0.14 UNKNOWN hold, and the frozen 24-query/N=100 diverse-model runner passed offline checks. Its first Gemma canary returned an upstream HTTP 429 rate limit with no ID/output/usage, causing another immediate stop. Two total generation attempts and zero benchmark cases are recorded; actual charges remain unknown and US$0.140744 is held. The [follow-up controls and API/cost ledger](hosted/README.md) preserve all failures, the shared US$2/600 ceilings, and all unattempted denominators. Private request identifiers and the live ledger remain outside the public repository.

