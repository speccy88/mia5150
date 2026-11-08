# OpenRouter Benchmark Report

**Date:** November 2026
**Test parameters:** 20 queries per combo, 7 library sizes (N=50 to N=5000), seed=42
**Descriptions:** Full (up to 1024 chars, not truncated)

---

## Model Summary

| Model | Context Limit | Sizes Completed | Avg Accuracy | Avg Latency | Queries | Est. Cost | Actual Cost | Savings |
|-------|--------------|-----------------|-------------|-------------|---------|-----------|-------------|---------|
| Qwen 3.8 Flash | 1,000,000 | 50-5000 (all 7) | 76.4% | 3.19s | 140 | $1.39 | $0.22 | 84% |
| DeepSeek V4 Flash | 163,840 | 50-2000 (6 of 7) | 79.2%* | 1.65s | 140 | $0.79 | $0.22 | 72% |
| Claude Sonnet 5.5 | 1,000,000 | 50-5000 (all 7)** | 81.7%** | 2.02s | 104 | $11.55 | $11.90 | -3% |
| Gemini 2.5 Flash | 1,048,576 | 50-5000 (all 7) | 74.3% | 1.31s | 140 | $2.84 | $1.14 | 60% |
| GPT-6 Luna | 1,050,000 | 50-5000 (all 7) | 72.1% | 1.76s | 140 | $0.88 | $0.14 | 84% |
| GPT-6.1 Terra | 1,050,000 | 50-5000 (all 7) | 66.4% | 2.12s | 140 | $8.80 | $1.00 | 89% |
| Kimi K2.5 | 262,144 | 50-2000 (6 of 7) | 71.7%* | 51.04s | 140 | $2.39 | $2.36 | 1% |
| **Total** | | | | | **944** | **$28.65** | **$17.00** | **41%** |

\* Accuracy computed over non-context-limited queries only. DeepSeek and Kimi hit their context limit at N=5000 (20 queries each auto-skipped).
\*\* Claude Sonnet 5.5 has partial data: 20 queries for N=50-500, 13 for N=1000, 6 for N=2000, 5 for N=5000 (spending limit hit mid-run).

---

## Estimated vs Actual Cost

The estimated cost is calculated by the runner script using list prices from the
OpenRouter model registry (per 1M input/output tokens). The actual cost is what
OpenRouter billed, as shown on the dashboard.

| Model | Input $/M | Output $/M | Est. Cost | Actual Cost | Ratio |
|-------|-----------|------------|-----------|-------------|-------|
| Qwen 3.8 Flash | $0.15 | $0.47 | $1.39 | $0.22 | 0.16x |
| DeepSeek V4 Flash | $0.20 | $0.60 | $0.79 | $0.22 | 0.28x |
| Gemini 2.5 Flash | $0.30 | $2.50 | $2.84 | $1.14 | 0.40x |
| GPT-6 Luna | $0.10 | $0.50 | $0.88 | $0.14 | 0.16x |
| GPT-6.1 Terra | $1.00 | $5.00 | $8.80 | $1.00 | 0.11x |
| Kimi K2.5 | $0.49 | $2.50 | $2.39 | $2.36 | 0.99x |
| Claude Sonnet 5.5 | $2.00 | $10.00 | $11.55 | $11.90 | 1.03x |
| **Total** | | | **$28.65** | **$17.00** | **0.59x** |

### Why actual cost is lower than estimated

**Prompt caching** is the primary factor. OpenRouter automatically caches
repeated prompts. In this benchmark, all 20 queries per combo share the
identical system prompt (the skill listing), so 19 of 20 queries hit the
cache. The observed cache hit rate was **83.9%** across all models.

Cached input tokens are billed at a fraction of the full price (typically
1/10th). For a model like GPT-6.1 Terra ($1.00/M input, $0.05/M cached),
this is a 20x reduction on ~95% of input tokens.

**Exceptions:**
- **Kimi K2.5** (ratio 0.99x): Minimal caching benefit. Kimi's high
  latency (51s avg) suggests it processes each request fully.
- **Claude Sonnet 5.5** (ratio 1.03x): Actual cost slightly *exceeded*
  the estimate. Claude has mandatory reasoning ("thinking") tokens billed
  as output at $10/M. The runner's estimate does not account for these
  hidden output tokens.

---

## Context Window Limitations

| Model | Context Window | Max Viable N | N=5000 Status |
|-------|---------------|-------------|---------------|
| DeepSeek V4 Flash | 163,840 | ~2,000 | Context-limited (all 20 queries skipped) |
| Kimi K2.5 | 262,144 | ~2,000 | Context-limited (all 20 queries skipped) |
| Qwen 3.8 Flash | 1,000,000 | 5,000 | OK |
| GPT-6 Luna | 1,050,000 | 5,000 | OK |
| GPT-6.1 Terra | 1,050,000 | 5,000 | OK |
| Gemini 2.5 Flash | 1,048,576 | 5,000 | OK |
| Claude Sonnet 5.5 | 1,000,000 | 5,000 | OK (partial data) |

At N=5000, the prompt contains ~305K tokens. Models with <300K context
cannot process these queries.

---

## Accuracy by Library Size

| Library Size | DeepSeek | Gemini | Qwen | GPT-6 Luna | GPT-6.1 Terra | Kimi | Claude* |
|-------------|----------|--------|------|------------|---------------|------|---------|
| N=50 | 90.0% | 90.0% | 90.0% | 80.0% | 70.0% | 85.0% | 90.0% |
| N=100 | 85.0% | 85.0% | 85.0% | 85.0% | 70.0% | 70.0% | 85.0% |
| N=250 | 85.0% | 80.0% | 80.0% | 80.0% | 70.0% | 75.0% | 85.0% |
| N=500 | 80.0% | 75.0% | 75.0% | 75.0% | 65.0% | 75.0% | 65.0% |
| N=1000 | 75.0% | 70.0% | 75.0% | 65.0% | 65.0% | 60.0% | 84.6% |
| N=2000 | 60.0% | 65.0% | 65.0% | 60.0% | 60.0% | 65.0% | 83.3% |
| N=5000 | ctx-limit | 55.0% | 65.0% | 60.0% | 65.0% | ctx-limit | 80.0% |

\* Claude N=1000 based on 13 queries, N=2000 on 6 queries, N=5000 on 5 queries.

---

## Key Observations

1. **Best accuracy**: Claude Sonnet 5.5 (81.7%) leads overall, but with
   partial data and at 15-50x the cost of cheaper models. Among complete
   runs, Qwen 3.8 Flash (76.4%) and DeepSeek V4 Flash (79.2%) lead.

2. **Best value**: GPT-6 Luna ($0.14 actual) and DeepSeek V4 Flash ($0.22)
   deliver 72-79% accuracy for under $0.25. Qwen 3.8 Flash ($0.22)
   matches DeepSeek's cost with broader context coverage.

3. **Best latency**: Gemini 2.5 Flash (1.31s avg) is the fastest, followed
   by DeepSeek (1.65s) and GPT-6 Luna (1.76s). Kimi K2.5 is an outlier
   at 51s average.

4. **GPT-6.1 Terra underperforms**: Despite being 10x the price of GPT-6
   Luna, Terra scores lower (66.4% vs 72.1%). The premium model does not
   justify its cost for this task.

5. **Caching dramatically reduces costs**: Actual spend was 41% below
   estimates overall. Models with cheap cached-input pricing (GPT-6.1
   Terra, GPT-6 Luna) saw the largest savings (84-89%).
