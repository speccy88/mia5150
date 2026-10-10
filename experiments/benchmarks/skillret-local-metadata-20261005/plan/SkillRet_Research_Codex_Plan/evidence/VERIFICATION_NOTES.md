# What was actually verified

## Completed in this research response

- Read the supplied four-page proposal using its complete extracted text; preserved the original PDF unchanged in inputs/.
- Searched current public literature and followed references from agent-skill surveys, the SkillRet paper/repository, and relevant primary author/model pages.
- Registered 54 unique papers/technical reports: 37 agent-skill-related works including surveys/benchmarks/background/security, five transferable tool-retrieval papers, and 12 general IR/model-method papers.
- Checked primary landing pages and recorded inspection depth per entry. Most were abstract/metadata screened; this is not a claim of full critical review of every paper. Selected methods/results sections (SkillRet, field-aware retrieval, diversity-aware routing, SkillReason, SkillSeek) and relevant official model documentation were inspected more closely.
- Cross-checked the main published SkillRet results against its official README and reranker model card. Recorded the oracle-insertion distinction and old-versus-corrected-pool mismatch.
- Checked current native interfaces/documentation for Jev, JevK5, Plumb, Qwen/SkillRet reranking and Jina; checked official OpenAI model/pricing pages. Older cached OpenAI search results differed from the live pages, so the live pages were used for the proposed API roles.
- Verified released SkillRouter, R3-rerank and SkillReason model identities; inspected SkillReason's older SkillRet pool statistics and native non-generative scoring interface.
- Created an experimental specification (not a benchmark implementation), model/method registry, phase prompts, and an offline package-consistency validator. See PLAN_VALIDATION.json for the validator's actual output.

## Not completed or claimed

No SkillRet corpus bytes or checkpoints were downloaded here. No user's repository was audited. No local/RunPod GPU or API account was accessed. No embedding inference, retrieval benchmark, reranking benchmark, paid API call or training run was performed. No literature result was independently reproduced. Model fit, provider access, exact bill, runtime and final winners remain unknown until Codex executes approved jobs.

A container-side request to fetch arXiv metadata failed because DNS/network access was unavailable; public research was performed through the web tool instead. The package does not contain offline copies of all paper PDFs, model weights or a finished benchmarking implementation. The original proposal and generated planning materials are self-contained; running the experiments will require source/model downloads and allocated resources.

Historical benchmark hints were treated as leads, not as independently re-inspected evidence. A Files search for previous SkillRet/Jev result artifacts returned unrelated project books rather than relevant raw runs; those books were not used as evidence for this plan. This is why the first Codex job audits historical results and exposure rather than assuming a clean holdout.

## Interpretation warnings to preserve

1. Published values are not measured results from this package.
2. An author benchmark on JevBench, BEIR, SkillRouter, SRA or an old SkillRet release does not establish a ranking on corrected SkillRet.
3. A gold-insertion pool is an oracle diagnostic, not an ordinary retrieval result or a perfect-ranking ceiling.
4. Model card capability, calibration, latency and commercial claims are not independent validation on Fred's workloads.
5. The research catalogue is a scoped sweep through 2026-10-05, not an exhaustive search certificate or a peer-review judgement.
