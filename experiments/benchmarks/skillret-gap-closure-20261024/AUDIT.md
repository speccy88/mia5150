# Independent MIA5150 gap audit
Audit date: 24 October 2026. Read-only baseline: `c4f4889fb3ec6864898d3ba07d08c30c438e3e12` (main at audit start). This is an evidence audit, not a new model experiment.

## Decision first
The project has substantial valid retrieval evidence, including full-public-test and matched scaling studies. The largest remaining research gap is a sufficiently informative **matched final-selection comparison**, not missing implementations of every method. Do not restart 16 studies or pretend the original 80 cells are completed. Preserve the original proposal as history and state the reduced, feasible scope.

**Immediate corrections before new inference:**
1. Remove the obsolete claim that 4,192 public IDs remain untouched. All 4,392 public queries have since been exposed.
2. Preserve historical failures. A constrained domain-routing arm already exists; extend it under a new protocol rather than replacing old failures.
3. Reconcile original versus revised factors and choose one primary matched local design.
4. Avoid extending the legacy name-based listing benchmark unchanged. It has identity-collision, single-target, prompt-only isolation, 300-character-view, and panel-wide gold-union constraints.
5. Record planned logical cases, attempted stages, physical model calls, valid outputs, failures, and cached replays separately.
6. Keep human annotation and genuinely independent evaluation pending. Neither new random selection from exposed SkillRet nor assistant-authored examples creates a human/untouched panel.

## Inputs and evidence boundaries
All three supplied inputs were read fully through Library: `CORRECTIVE_ACTIONS.md` (515 lines), `MIA5150_Corrective_Action_Plan.md` (369 lines), and `proposal.pdf` (all four extracted pages). Their full extracted text is retained privately under `inputs/`. The supplied PDF contains private student identity information; do **not** publish that extraction or the original PDF into the public repository.

GitHub source files were read through the connected GitHub tool and retained under `repo/`, not cloned or executed. All evidence links below pin the baseline commit. Absence means not found in the inspected catalog/navigation/source directories, not proof that the team owns no private artifact. No new inference, paid calls, human annotation, or independent holdout was performed in this audit.

## Proposal comparison
| Dimension | Supplied proposal | Revised repository proposal | Audit conclusion |
|---|---|---|---|
| Methods | Listing, BM25, Vector, Domain Vector, Hybrid | Listing, BM25, Vector, Domain Vector | Hybrid is an original factor and already a substantial extension; reduced scope needs disclosure, not retroactive rewriting. |
| Sizes | 10/100/1,000/6,006 | 50/250/1,000 | Original set already appears in CPU matched32; scaling uses their union. A superset is optional, not necessary to rescue a claim. |
| Models | gpt-4o-mini, gpt-4.1-mini, gpt-4.1-nano, DeepSeek V4 | One proposed DeepSeek V4.1 Flash/provider | Exact hosted arms are not closed by local Qwen. Record not-run/substitution status; no new spend is implied. |
| Panel | Full local 4,392 and roughly 2,000 hosted | Planned 20 new IDs after 200 exploratory | The old reservation is obsolete after later full-public studies. Any new SkillRet panel is exposed exploratory/confirmatory-to-protocol, not blind. |
| Library construction | Not fully specified | Per-query, gold-conditioned nested distractors | Distinguish conditional distractor scaling from natural catalog coverage. |
| Scenario | Single/multi-domain, overlap, cascading | Dependency graphs/chains out of scope | Gold count/category co-membership are proxies, not semantic overlap or dependency labels. |
| Integration | Skill discovery focus | Native SkillsMiddleware plus separate bounded reads | Plain local JSON selection is useful but not exact native integration reproduction. Activation/read is not execution success. |
| Budget | About CAD20/model, simplified input-token calculation | US3/CAD5 and 300-call historic ledger; new authorization required | Cost assumptions require exact endpoint/rates and full call matrix; historical ledgers remain immutable. |

The supplied proposal explicitly allows iterative refinement. The revised source itself conditions seeds 43/44 on budget review; those seeds are not an unconditional promise.

## Prioritized acceptance criteria: feasible local 8 GB
### P0 — validity and reporting before new inference
- Publish a scope/deviation table for all 80 original method-size-model cells with evidence links and status. Do not count retrieval results as selector completion.
- Freeze a protocol, query-ID panel, labels, exact rendered document bytes, candidate catalogs/order, method definitions, model/tokenizer hashes, context settings, failure/retry policy and primary contrasts before the new run.
- Mark the entire existing public SkillRet test split exposed. Record overlap with prior development panels. No claim of fresh independent generalization.
- Use unique skill IDs/opaque aliases and collision-free virtual paths. Reject unknown IDs, duplicates and overlength selections rather than silently repairing a primary result. Separate any tolerant parser ablation.
- Validate metrics with hand-calculated fixtures: 1/2/3-positive cases, missing gold, empty ranking, duplicate/unknown predictions, tie order, rank 5 versus 6, invalid output, and missing/unattempted cases. Recent positive-qrel scoring was inspected; no demonstrated numeric aggregate bug was established by this audit.
- Preserve macro query recall versus micro positive-pair coverage; set exact match versus completeness; MRR cutoff; valid-response versus all-planned rates. For genuine empty-gold tasks, define a new convention explicitly rather than silently applying nonempty-qrel code.
- Keep every planned case, but distinguish structural preflight infeasibility, route attempt, selection attempt, provider/local inference attempt, failed, valid, unattempted and cache-hit. Unknown tokens/costs/timings are null, not zero.

### P1 — one informative matched local comparison
- Prefer one verified local selector that fits 8 GB, a single pinned small embedder, BM25 and predeclared RRF; all five discovery methods on a common panel and size set. A 20-query panel is a bounded diagnostic; 200 does not automatically guarantee power. Choose n from a precision/runtime target and report the achievable uncertainty.
- Choose the original four-size grid or the six-size union; do not add sizes merely to check boxes. Construct gold-conditioned libraries per query, not the union of all panel golds. Rebuild BM25 IDF inside each library.
- Freeze single-domain routing with no fallback for the proposal arm. Reuse the existing constrained grammar as a clearly labeled protocol. Compute routing parse validity, selected-domain size, any-positive coverage, all-positive coverage and the oracle one-domain ceiling for multi-domain gold sets.
- Add a small label-independent natural-catalog sensitivity and/or retain full-catalog as the natural condition. Report catalog coverage separately from ranking quality. A natural catalog with no gold is not automatically a real no-skill-needed request.
- Tokenize the actual full listing with its actual wrapper and output reserve. If infeasible, record a preflight failure with token evidence; do not truncate/chunk it secretly or waste repeated inference on a known overflow. Exact model capacity is not inferred from a wrapper-compaction error.
- Fresh timing must use the same machine, selector, panel, concurrency and cache conditions. Separate setup/indexing/query embedding/routing/selection and end-to-end wall time. Report failure latency separately from successful latency, and both from replay latency.
- Relative end-to-end latency must use BM25 + the same selector at the same size, not retrieval-only BM25 at full corpus as denominator. Always show absolute timings too.
- Record whole-system/child-process memory, context/KV, startup, temperature and swap. One GPU workload at a time; retain resource-stopped arms. Do not expand context based on weight-file fit alone.
- Generate all tables from the case ledger. Bootstrap paired query differences (cluster related tasks when appropriate); resample the same query across sizes/seeds jointly. Repeated size/seed/model rows are not independent new queries. Predeclare primary contrasts; label screened results exploratory.

### P2 — synthesis and human/hosted items
- Produce a canonical report and presentation using traceable results; preserve proposal and historical evidence. Distinguish observed inference, cached reanalysis, offline contract tests, and planned work.
- A genuine human-authored/adjudicated scenario panel requires actual humans, independent labels and provenance. Draft guides/templates can be completed locally; human agreement statistics and labels must remain pending until provided.
- A genuinely untouched independent panel cannot be manufactured by excluding 200 old IDs. It needs independently sourced requests kept separate from development and appropriately validated labels.
- Exact OpenAI/DeepSeek model comparisons require live access, a rate snapshot, a spend cap, and authorized hosted execution. The local study can be complete in its reduced scope while these remain not run.
- Native integration can be checked locally with a compatible runtime and safe bounded reads. Offline scripted canaries validate wiring only; successful live selection, actual read traces, and execution/task success are distinct outcomes.

## Corrections to potentially misleading prescriptions
- Retrieval has **no API bill** in a local run, but is not computationally free. Full query-by-size-by-seed BM25 rebuilding and encoding have wall-time, memory and energy cost.
- Do not claim listing inherently cannot scale or route failure proves poor model reasoning; the observed limit depends on representation, wrapper, configured context and output budget.
- Do not use same-major labels as proof of overlapping capability, or multiple qrels as proof of an ordered cascade.
- Do not infer causal project prioritization from the presence of extensions: “came at the expense of” the original plan is unsupported.
- Fixed top-five retrieval is not a final selected set: because gold sets contain 1–3 skills, always choosing five makes exact match mechanically impossible.
- For listing, candidate coverage of the entire library and ordered top-five final selection are different quantities; do not fabricate a retrieval Recall@5 for an unranked full catalog.
- Existing metadata evidence already has three library seeds (5150/5151/5152) and hard-overlap sensitivity. Fresh seed evidence is still needed for a new primary selector design, but “no seed variance exists” is false.

## Primary-source claims
See `primary-source-claims.md`. Both named DeepSeek models exist. The proposal input-price and BGE-M3 rate have official-source support, but live pages/provider variants disagree and direct legacy aliases can resolve to newer models. The cost arithmetic is correct only for the stated input subtotal, not a whole factorial study. 6,006 skills/4,392 queries describe the test split, not all dataset rows. The 1,024-character specification constrains description, not total header/body/tokens. Exact pinned dataset bytes were not newly verified through web in this audit.

## How to use the attached matrix
`gap_matrix.json` and `gap_matrix.md` enumerate each numbered assertion in the first gap list, the five categories of the second list, and additional methodological risks. Status refers to evidence at the pinned audit baseline, not to subsequent work by the implementation agent. Reassess only against the final candidate commit and its new receipts.

