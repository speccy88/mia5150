# Sources and design boundaries

## Requested basis

The supplied project brief asks for a two-day feasibility trial (open the data, run a baseline, report a number) and a controlled Benchmark-category project. The notebook follows the workflow discussed with the user: library size → discovery method → same LLM selector → quality and token usage.

The uploaded `mia5150-main.zip` was inspected as text/JSON only. No original code, model, or test was executed. Its dataset manifest supplies revision `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`; its embedding manifest supplies `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The exact pilot-query IDs are copied without changing the reserved split.

## External documentation inspected for the new code

These are official documentation or upstream source pages, inspected during preparation. Live source pages can change; the implementation's main dependency pins follow the supplied pilot rather than an unbounded latest-version install.

- SkillRet loading, schemas, qrels and documented full-document representation: https://huggingface.co/datasets/ThakiCloud/SKILLRET
- Deep Agents skill frontmatter, source directories and progressive disclosure: https://docs.langchain.com/oss/python/deepagents/skills
- SkillsMiddleware implementation, including required custom-template slots: https://raw.githubusercontent.com/langchain-ai/deepagents/main/libs/deepagents/deepagents/middleware/skills.py
- Functional middleware hooks and execution order: https://docs.langchain.com/oss/python/langchain/middleware/custom
- ChatOllama integration: https://docs.langchain.com/oss/python/integrations/chat/ollama
- ChatOllama source, including reasoning, usage metadata and structured-output handling: https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/partners/ollama/langchain_ollama/chat_models.py
- BM25S interface and defaults: https://pypi.org/project/bm25s/0.3.11/
- Deep Agents dependency release: https://pypi.org/project/deepagents/0.7.19/
- LangChain dependency release: https://pypi.org/project/langchain/1.4.2/
- BGE model card: https://huggingface.co/BAAI/bge-small-en-v1.5

These sources support API choices; inspecting documentation is not an execution test.

## Deliberate simplifications

| Original production-style pilot | Teaching notebook |
|---|---|
| Many commands, reports, ledgers and model backends | One notebook and one local model |
| Full Deep Agents factory and harness exclusions | LangChain create_agent plus native SkillsMiddleware only |
| In-state filesystem and full skill bodies | Temporary candidate-only metadata projection files |
| UUID-name output convention | Short display aliases mapped outside the LLM |
| Larger intended library sizes | 10 and 50 skills by default |
| 200-query retrieval pilot | First 20 for the initial baseline; configurable up to the same 200 |
| Extensive metadata processing and hashes | Explicit simplified metadata cap; do not assume numerical parity |
| Earlier single-domain routing | Optional query-only routing to up to two categories |
| Multiple paid providers | Ollama localhost only |
| Extensive cost/context policies | Small token preflight and raw usage; not a production budget system |

Metadata-only retrieval and full-document retrieval are different conditions. Do not relabel their metrics or pool this demo's new results with old reports. The catalogue sampling is a new, explicit teaching implementation: it preserves the nested-membership property but does not guarantee identical distractor draws to the old harness.

## Interpretation

Candidate recall measures discovery. Final F1/exact match measure the model's chosen set. A correct candidate does not prove the model selected it; a correct selection does not prove it can execute the skill.

Labels influence only evaluator-side catalogue construction and scoring. The selected-query object drops the dataset's answer-bearing columns before inference.

Any final paper should use a locked protocol and an untouched evaluation set, report repeated catalogue seeds, distinguish context-budget rejection from model errors, and discuss the synthetic-query limitation. This notebook is not a substitute for that final study.
