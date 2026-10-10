# Start here

Open **`proposal_workflow.ipynb`** in VS Code or Jupyter. It is a separate, readable demonstration of the proposal workflow, not a replacement for the two notebooks in your existing repository.

For a read-only preview without Jupyter, open `proposal_workflow.html` in a browser. It shows the code and explanations but cannot execute them. The optional `support/` folder contains the notebook-generation source; it is not required to use the notebook.

## What it does

The notebook walks through loading the pinned SkillRet data, computing a full-document BM25 baseline, building small nested libraries, comparing listing/BM25/vector candidate sets, and asking the same local LLM to select skills. Domain → vector is an optional extra at the end. It records retrieval recall, final selection F1/exact match, and actual LLM input-token usage when available.

The functions are in the notebook. There is no hidden project package or custom class hierarchy. The code does not import or execute your old repository.

## Run on your Mac

Extract the entire ZIP into its own folder and open that folder in VS Code. Create a new environment rather than change the environment used by the existing benchmark:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Select `.venv` as the notebook kernel. Run the cells from top to bottom. The first run downloads the dataset and the small BGE embedding model; later runs reuse the normal Hugging Face cache. Download sizes, package installation time, and runtime have not been measured here.

The dependency versions were taken from the uploaded pilot. They are a direct-dependency specification, not a newly validated Mac lockfile. Installation could not be tested in this authoring environment. Keep any installation error for diagnosis rather than silently changing versions mid-experiment.

### First pass: no LLM calls

The defaults are:

```python
BASELINE_QUERIES, DEMO_QUERIES = 20, 3
LIBRARY_SIZES = (10, 50)
RUN_LLM = False
RUN_DOMAIN = False
```

This pass computes real retrieval results when run on your machine, without using Ollama or any paid API. No mock model is substituted.

The notebook ships the exact 200 development-query IDs from the supplied pilot. Only the configured first 20 are used for the initial baseline; the first 3 are used in the small scaling demonstration. The remaining reserved final-test queries are not evaluated.

### Second pass: use your local model

Start Ollama and check the model name:

```bash
ollama list
```

The notebook is configured for your existing `qwen3.5:4b-mlx`. Set `RUN_LLM = True` and run from the top again. No API key is needed, and the model endpoint is localhost only. The notebook requests non-thinking generation, temperature zero, 16,384 context tokens, and a 256-token output cap. The corresponding Hugging Face tokenizer is downloaded only for this live pass.

This is not a speed or memory guarantee for your Mac. A conservative preflight refuses long prompts, and large tokenizer/server usage discrepancies are flagged. Inspect those cases instead of counting them as clean model-quality results.

Set `RUN_DOMAIN = True` only after the basic selector works. It adds one category-routing call for each query/library pair; the category limit is two and the router's input tokens are included in that method's total.

### Reuse the existing data download

Set `LOCAL_DATA` to the existing `skillret-pilot/data` directory. The notebook checks those files against the hashes in the original uploaded manifest. When `LOCAL_DATA` is `None`, it uses Hugging Face `load_dataset` at the pinned revision instead.

## Where results go

Each run creates its own `runs/<timestamp>_<suffix>/` folder containing:

- `bm25_full_document.jsonl`: initial BM25 rankings and metrics;
- `decisions.jsonl`: candidate IDs, local LLM responses and prompt traces, statuses, and metrics;
- `results.csv` and `summary.csv`: tables for inspection;
- `manifest.json`: settings, query IDs, dataset/model references, and package versions.

A stopped run retains rows already written. This teaching notebook does not provide the resume, billing, or extensive auditing machinery of the original project.

## What was verified here

**32 offline checks passed** on notebook structure and small helper functions. This included nested-library membership, scoring, alias validation, and token-total arithmetic. All code cells compiled. The supplied repository and its notebooks were left unchanged.

**Not executed here:** full SkillRet downloads, the BM25/BGE library paths, and native LangChain/Deep Agents/Ollama inference. Required dependencies could not be installed because the environment could not resolve external package/model hosts. The notebook is deliberately delivered with empty outputs, not invented benchmark results. See `evidence/VALIDATION.md`.

## Scope and academic use

The selector uses LangChain `create_agent` and native Deep Agents `SkillsMiddleware`. It intentionally does not instantiate the full `create_deep_agent` stack. No downloaded skill script or full instruction body is executed. This is selection, not agent task completion or skill activation.

The optional source notes explain differences from the original harness. Keep the two original notebooks intact; use this standalone demo for explanation or move its cells into them when appropriate. Check the code and retain the course's required AI-assistance disclosure in the report appendix.
