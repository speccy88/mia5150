# OpenRouter Benchmark

Skill discovery benchmark using the [OpenRouter](https://openrouter.ai) API.
Two scripts work together: a **runner** that sends queries to LLMs and
accumulates results, and a **report generator** that produces plots and
summary tables from those results.

> **This is a paid service** -- each run incurs API costs.  Use `--dry-run`
> to validate connectivity before committing to a full run.

## Prerequisites

1. Set `OPENROUTER_API_KEY` in `.env` at the project root:

   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   ```

2. Python dependencies (already in the project venv):
   `openai`, `pandas`, `matplotlib`, `tabulate`, `python-dotenv`

3. Set a spending limit on your OpenRouter key at
   https://openrouter.ai/workspaces/default/keys to avoid surprises.

---

## Script 1 -- Runner

```
benchmark_models_testplan_openrouter.py <result_dir> [options]
```

Sends skill-discovery queries to one or more OpenRouter models across
multiple library sizes (N=50 to N=5000) and writes per-query results to
`<result_dir>/details.csv`.

### Quick start

```bash
# Validate connectivity (single request, not persisted, ~$0.0005)
python benchmark_models_testplan_openrouter.py tests_openrouter \
    --models deepseek/deepseek-v4-flash-0731 --dry-run

# Full run
python benchmark_models_testplan_openrouter.py tests_openrouter \
    --models deepseek/deepseek-v4-flash-0731
```

### Adding models incrementally

Each `--models` call merges new models into the existing run config.
Already-completed combos (model + library size) are skipped automatically.
Previous results are never overwritten.

```bash
# Session 1
python benchmark_models_testplan_openrouter.py tests_openrouter \
    --models deepseek/deepseek-v4-flash-0731

# Session 2 -- DeepSeek results are preserved, only Gemini runs
python benchmark_models_testplan_openrouter.py tests_openrouter \
    --models google/gemini-2.5-flash

# Session 3 -- add multiple models at once
python benchmark_models_testplan_openrouter.py tests_openrouter \
    --models openai/gpt-6-luna,qwen/qwen3.8-flash
```

### Resumability

The runner is resumable at the **query level**. If a run is interrupted
(Ctrl-C, network error, spending limit), re-run the same command. The script
detects which queries already have clean results and picks up where it
left off.

Ctrl-C during a combo saves the partial results collected so far -- they
are not lost.

### Re-running a specific combo

Use `--rerun` to drop all rows for a model/size combination and re-run it:

```bash
python benchmark_models_testplan_openrouter.py tests_openrouter \
    --rerun anthropic/claude-sonnet-5.5:2000

# Multiple combos
python benchmark_models_testplan_openrouter.py tests_openrouter \
    --rerun moonshotai/kimi-k2.5:50,moonshotai/kimi-k2.5:100
```

### Check status

```bash
python benchmark_models_testplan_openrouter.py tests_openrouter --status
```

### Runner options

| Flag | Default | Description |
|------|---------|-------------|
| `<result_dir>` | *(required)* | Directory for results (created if needed) |
| `--models` | `deepseek/deepseek-v4-flash-0731` | Comma-separated OpenRouter model slugs |
| `--lib-sizes` | `50,100,250,500,1000,2000,5000` | Library sizes to test |
| `--num-queries` | `20` | Queries per combo |
| `--top-k` | `10` | Ask model for top-K ranked skills |
| `--delay` | `1.0` | Seconds between API requests |
| `--seed` | `42` | Random seed for reproducibility |
| `--dry-run` | off | Single request to the first model, not persisted |
| `--status` | off | Show progress and exit |
| `--rerun` | -- | Comma-separated `model:size` combos to drop and re-run |

---

## Script 2 -- Report generator

```
benchmark_models_testplan_openrouter_report.py <result_dir> [options]
```

Reads `details.csv` from the result directory and generates plots and a
summary table.

### Usage

```bash
# Default -- only models with complete data (20 queries per combo)
python benchmark_models_testplan_openrouter_report.py tests_openrouter

# Include models with partial data (fewer than 20 queries in some combos)
python benchmark_models_testplan_openrouter_report.py tests_openrouter --all

# Exclude specific models
python benchmark_models_testplan_openrouter_report.py tests_openrouter \
    --exclude moonshotai/kimi-k2.5

# Only fully completed combos (no rate-limit or skip errors)
python benchmark_models_testplan_openrouter_report.py tests_openrouter --completed-only
```

### Report options

| Flag | Default | Description |
|------|---------|-------------|
| `<result_dir>` | *(required)* | Directory containing `details.csv` |
| `--all` | off | Include models with incomplete data (partial query counts) |
| `--exclude` | -- | Comma-separated model slugs to exclude |
| `--completed-only` | off | Only plot data from fully completed combos |

### Output files

Generated in the result directory:

| File | Description |
|------|-------------|
| `comparison_accuracy.png` | Accuracy vs library size per model |
| `comparison_latency.png` | Latency with p5-p95 bands |
| `comparison_dashboard.png` | 4-panel dashboard (accuracy, latency, outcome distribution, model summary) |
| `comparison_cost.png` | Cost per model |

---

## Model registry

The runner includes a registry (`OPENROUTER_MODELS` dict) with context
window sizes, pricing, and per-model settings. Unknown models passed via
`--models` default to 128K context and zero cost tracking.

For accurate context-window checks and cost estimates, add the model to the
registry in `benchmark_models_testplan_openrouter.py`:

```python
"provider/model-name": {
    "display": "Friendly Name",
    "context_window": 128_000,
    "temperature": 0,
    "supports_max_tokens": True,
    "cost_input_per_1m": 0.50,    # USD per 1M input tokens
    "cost_output_per_1m": 1.50,   # USD per 1M output tokens
},
```

### Optional per-model keys

| Key | Type | Description |
|-----|------|-------------|
| `"reasoning"` | `True` / `False` | Override reasoning mode. `True` sends `reasoning: {enabled: true}` (needed for Kimi K2.5). `False` sends `reasoning: {enabled: false}` (needed for DeepSeek). Omit to let the model use its default. |
| `"provider"` | `dict` | Pin to a specific OpenRouter provider, e.g. `{"only": ["deepinfra/fp8"], "allow_fallbacks": False}` |

Find model specs and pricing at https://openrouter.ai/models.

### Models tested

| Model | Slug | Context | Input $/M | Output $/M |
|-------|------|---------|-----------|------------|
| DeepSeek V4 Flash | `deepseek/deepseek-v4-flash-0731` | 164K | $0.20 | $0.60 |
| Kimi K2.5 | `moonshotai/kimi-k2.5` | 262K | $0.49 | $2.50 |
| Qwen 3.8 Flash | `qwen/qwen3.8-flash` | 1M | $0.15 | $0.47 |
| GPT-6 Luna | `openai/gpt-6-luna` | 1M | $0.10 | $0.50 |
| GPT-6.1 Terra | `openai/gpt-6.1-terra` | 1M | $1.00 | $5.00 |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | 1M | $0.30 | $2.50 |
| Claude Sonnet 5.5 | `anthropic/claude-sonnet-5.5` | 1M | $2.00 | $10.00 |

---

## Result directory structure

```
tests_openrouter/
  run_config.json         -- models, sizes, seed, run history
  test_queries.json       -- the 20 test queries (deterministic)
  details.csv             -- per-query results (all models, all runs)
  metrics.json            -- aggregate metrics
  comparison_table.md     -- summary table
  comparison_accuracy.png -- generated by report script
  comparison_latency.png
  comparison_dashboard.png
  comparison_cost.png
```

---

## Cost notes

- **Prompt caching**: OpenRouter automatically caches repeated prompts.
  Since all 20 queries per combo share the same system prompt (the skill
  listing), 19 of 20 queries typically hit the cache. Observed cache hit
  rate: ~84%, reducing actual costs to roughly 1/3 of the estimated price.

- **Reasoning tokens**: Models with mandatory reasoning (e.g. Claude
  Sonnet 5.5) bill thinking tokens as output tokens. This significantly
  increases cost for large library sizes.

- **Spending limits**: Set a limit on your OpenRouter API key to avoid
  unexpected charges. The runner handles 402/403 limit errors gracefully
  (marks them as errors in the CSV). Clean error rows and re-run to
  resume after increasing the limit.

---

## Differences from testplanA (CX AI Playground)

| | testplanA | OpenRouter |
|---|---|---|
| API | CX AI Playground (internal) | OpenRouter (paid) |
| Script | `benchmark_models_testplan.py` | `benchmark_models_testplan_openrouter.py` |
| Report | (built into runner) | `benchmark_models_testplan_openrouter_report.py` |
| Result location | `results_models/run_<name>/` | User-specified directory |
| Description truncation | 300 chars | 1024 chars (per spec) |
| Reasoning control | N/A | Per-model (`reasoning` key in registry) |
| Cost tracking | Estimated | Estimated (per model registry); actual billing via OpenRouter dashboard |
