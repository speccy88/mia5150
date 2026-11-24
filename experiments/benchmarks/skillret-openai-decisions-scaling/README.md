# SkillRet Decisions API scaling benchmark

The frozen experiment compares BM25, BGE-small vector retrieval and hybrid rank fusion on the same 200 queries, seven catalog sizes and three seeds. Each retrieval method contributes a top-20 pool for a separate OpenAI Decisions reranking condition.

The local retrieval matrix is measured and independently reproducible. The live API phase is pending. See the [results report](REPORT.md) and [experimental protocol](PROTOCOL.md).

## Reproduce the saved results

Use Python 3.12 and run the commands from this directory.

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python evaluate.py
python plot.py
python validate.py --allow-incomplete
```

These commands evaluate the frozen rankings without API calls. The repository checkout and delivery ZIP contain the historical source files at their original paths. The validator reports each incomplete phase. Without `--allow-incomplete`, an incomplete result exits with status 2 and an inconsistency exits with status 1.

## Prepare and execute the benchmark

```sh
python prepare.py --stage full --work-dir /path/to/external-embedding-cache
python validate.py --inputs-only --report evidence/preflight_validation.json
python run_benchmark.py plan
python run_benchmark.py run --budget-usd 15
python evaluate.py
python plot.py
python validate.py
```

The default preparation downloads the checksum-pinned encoder assets from Hugging Face when they are absent from the external cache. It verifies the raw asset hashes, creates candidate pools and freezes the manifest. When a full manifest already exists, preparation verifies the recorded files and reuses them.

The live command reads `OPENAI_API_KEY` from the process environment. It runs a synthetic compatibility request, ten preselected benchmark cases and an automatic cost projection before submitting the remaining work. `--smoke-only` stops after the compatibility, case and budget checks. The live runner supports configurable concurrency, request rate, HTTP attempt cap and timeouts. Keep `results/.decisions-runtime/` private and preserve it to resume a paid run with its existing cache and reservations.

## Results and evidence

| Artifact | Contents |
|---|---|
| [experiment_manifest.json](results/experiment_manifest.json) | Frozen query IDs, encoding provenance, case counts, smoke selection and hashes |
| [runs and candidate evidence](results/) | Catalog membership, candidate rankings, canonical requests, response ledger, outcomes and binary qrels |
| [plot_data.csv](results/plot_data.csv) | One row per catalog size and retrieval method |
| [plot_data_long.csv](results/plot_data_long.csv) | Named series with metrics, populations and uncertainty fields |
| [presentation_plot_data.csv](results/presentation_plot_data.csv) | Exact new and historical data points used for the presentation figure |
| [figures](results/figures/) | PNG at 300 DPI or higher, SVG and PDF exports |
| [validation.json](results/validation.json) | Independent audit of saved results and completion status |
