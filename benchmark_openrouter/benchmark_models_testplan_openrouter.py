#!/usr/bin/env python3
"""
Skill Discovery Benchmark -- OpenRouter Runner
===============================================
MIA5150 Project: Agent Skill Discovery at Scale

Runs the skill-discovery benchmark through OpenRouter's paid API.
Results are written to the directory specified as the first argument.

Unlike the CX AI Playground runner, this script:
  - Uses OPENROUTER_API_KEY from .env
  - Takes a result directory path as the positional argument
  - Does NOT truncate skill descriptions (up to 1024 chars per spec)
  - Supports --dry-run for a single cheap validation request
  - Separates reporting into benchmark_models_testplan_openrouter_report.py

Usage
-----
  # Run a benchmark (creates/resumes in the given directory)
  python benchmark_models_testplan_openrouter.py tests_openrouter

  # Single-request dry-run (no results persisted)
  python benchmark_models_testplan_openrouter.py tests_openrouter --dry-run

  # Check status
  python benchmark_models_testplan_openrouter.py tests_openrouter --status

  # Specify models and sizes
  python benchmark_models_testplan_openrouter.py tests_openrouter \\
      --models deepseek/deepseek-v4-flash-0731,google/gemini-2.5-flash \\
      --lib-sizes 50,100,250,500,1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import openai as openai_module
from tabulate import tabulate

# ── Re-use shared logic from benchmark_models (read-only imports) ────────────

from benchmark_models import (
    SEED,
    DEFAULT_NUM_QUERIES,
    DEFAULT_TOP_K,
    DEFAULT_LIB_SIZES,
    DEFAULT_DELAY_S,
    K_VALUES,
    load_skills,
    create_test_queries,
    _clean_text,
    estimate_tokens,
    parse_ranked_response,
    parse_single_response,
    compute_per_query_retrieval,
    SYSTEM_PROMPT_SINGLE,
    SYSTEM_PROMPT_RANKED,
)

# ── OpenRouter defaults ──────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3
BACKOFF_BASE_S = 10        # shorter backoff -- OpenRouter rate limits are per-minute
CONSECUTIVE_429_ABORT = 5
DESC_MAX_CHARS = 1024      # per spec: do NOT truncate below 1024

# ── OpenRouter model registry ────────────────────────────────────────────────
# Add models as needed.  The runner also accepts unknown models via --models
# (defaults to 128K context).

OPENROUTER_MODELS: dict[str, dict] = {
    "deepseek/deepseek-v4-flash-0731": {
        "display": "DeepSeek V4 Flash",
        "context_window": 163_840,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 0.20,
        "cost_output_per_1m": 0.60,
        "reasoning": False,  # Prevent providers from enabling reasoning
    },
    "deepseek/deepseek-v4.1-flash-1017": {
        "display": "DeepSeek V4.1 Flash",
        "context_window": 1_048_576,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 0.15,
        "cost_output_per_1m": 0.60,
        "reasoning": False,  # Prevent providers from enabling reasoning
    },
    "moonshotai/kimi-k2.5": {
        "display": "Kimi K2.5",
        "context_window": 262_144,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 0.49,
        "cost_output_per_1m": 2.50,
        "reasoning": True,  # Kimi thinks in content unless reasoning is enabled
    },
    "moonshotai/kimi-k3.1-pro": {
        "display": "Kimi K3.1 Pro",
        "context_window": 1_048_576,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 0.40,
        "cost_output_per_1m": 7.88,
        "reasoning": True,  # Kimi models need reasoning enabled
    },
    "qwen/qwen3.8-flash": {
        "display": "Qwen 3.8 Flash",
        "context_window": 1_000_000,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 0.15,
        "cost_output_per_1m": 0.47,
    },
    "openai/gpt-6-luna": {
        "display": "GPT-6 Luna",
        "context_window": 1_050_000,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 0.10,
        "cost_output_per_1m": 0.50,
    },
    "openai/gpt-6.1-terra": {
        "display": "GPT-6.1 Terra",
        "context_window": 1_050_000,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 1.00,
        "cost_output_per_1m": 5.00,
    },
    "google/gemini-2.5-flash": {
        "display": "Gemini 2.5 Flash",
        "context_window": 1_048_576,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 0.30,
        "cost_output_per_1m": 2.50,
    },
    "anthropic/claude-sonnet-4": {
        "display": "Claude Sonnet 4",
        "context_window": 1_000_000,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 3.00,
        "cost_output_per_1m": 15.00,
    },
    "anthropic/claude-sonnet-5.5": {
        "display": "Claude Sonnet 5.5",
        "context_window": 1_000_000,
        "temperature": 0,
        "supports_max_tokens": True,
        "cost_input_per_1m": 2.00,
        "cost_output_per_1m": 10.00,
    },
    # Add more models here as needed.  Unknown models passed via --models
    # default to 128K context and zero cost tracking.  To get accurate cost
    # estimates and context-window checks, add the model entry here first.
    #
    # Optional per-model keys:
    #   "provider": {"only": ["deepinfra/fp8"], "allow_fallbacks": False}
    #   "reasoning": True   -- enable reasoning (default: False)
}

# ── Prompt construction (full descriptions, up to 1024 chars) ────────────────

def build_skill_listing_full(skills: list[dict]) -> str:
    """Build a numbered listing with full descriptions (up to 1024 chars)."""
    lines: list[str] = []
    for i, s in enumerate(skills, 1):
        name = s["name"]
        desc = _clean_text(s.get("description", "No description.")[:DESC_MAX_CHARS])
        lines.append(f"{i}. {name}: {desc}")
    return "\n".join(lines)


# ── OpenRouter API call with retry ───────────────────────────────────────────

def call_openrouter(
    client: OpenAI,
    model: str,
    system_content: str,
    query: str,
    temperature: float = 0.0,
    supports_max_tokens: bool = True,
    max_tokens_value: int = 100,
    timeout: float = 120.0,
    provider: dict | None = None,
    reasoning: bool | None = None,
) -> dict:
    """Send a chat-completion to OpenRouter with retry on 429.

    Returns dict with: response, latency_s, error, context_limit,
    rate_limited, rate_limit_type, input_tokens, output_tokens.
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if supports_max_tokens:
        kwargs["max_tokens"] = max_tokens_value

    # Build extra_body for OpenRouter-specific parameters
    extra: dict = {}
    if reasoning is not None:
        extra["reasoning"] = {"enabled": reasoning}
    if provider:
        extra["provider"] = provider
    if extra:
        kwargs["extra_body"] = extra

    last_exc: Exception | None = None
    total_latency = 0.0

    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            latency = time.time() - t0
            total_latency += latency

            raw = resp.choices[0].message.content if resp.choices else None
            content = raw.strip() if raw else ""
            usage = resp.usage

            return {
                "response": content,
                "latency_s": round(total_latency, 3),
                "error": None,
                "context_limit": False,
                "rate_limited": False,
                "rate_limit_type": "",
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
            }

        except openai_module.RateLimitError as exc:
            latency = time.time() - t0
            total_latency += latency
            last_exc = exc

            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_S * (2 ** attempt)
                print(f"    429 rate limit -- waiting {wait}s (retry {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            # fall through to final return

        except Exception as exc:
            latency = time.time() - t0
            total_latency += latency
            last_exc = exc
            err_str = str(exc).lower()

            is_rate = "429" in err_str or "rate limit" in err_str
            if is_rate and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_S * (2 ** attempt)
                print(f"    Rate limited (from error) -- waiting {wait}s (retry {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            context_limit = any(kw in err_str for kw in [
                "context_length_exceeded", "context window",
                "input too long", "request too large",
                "too many tokens", "maximum context length",
            ])
            return {
                "response": "",
                "latency_s": round(total_latency, 3),
                "error": str(exc)[:500],
                "context_limit": context_limit,
                "rate_limited": False,
                "rate_limit_type": "",
                "input_tokens": None,
                "output_tokens": None,
            }

    # Exhausted retries
    return {
        "response": "",
        "latency_s": round(total_latency, 3),
        "error": str(last_exc)[:500] if last_exc else "rate_limited",
        "context_limit": False,
        "rate_limited": True,
        "rate_limit_type": "unknown",
        "input_tokens": None,
        "output_tokens": None,
    }


# ── Context-limit placeholder row ───────────────────────────────────────────

def _ctx_limit_row(
    model: str, lib_size: int, q: dict, est_tokens: int, reason: str
) -> dict:
    row = {
        "model": model,
        "library_size": lib_size,
        "query": q["query"][:120],
        "gold_skill": q["gold_skill"],
        "selected_skill": "",
        "ranked_skills": "[]",
        "num_valid_skills": 0,
        "num_invalid_skills": 0,
        "invalid_skills": "[]",
        "rank_of_gold": -1,
        "exact_match": 0,
        "fuzzy_match": 0,
        "latency_s": 0.0,
        "llm_latency_s": 0.0,
        "retrieval_latency_s": 0.0,
        "e2e_latency_s": 0.0,
        "error": reason,
        "context_limit": True,
        "rate_limited": False,
        "rate_limit_type": "",
        "input_tokens": est_tokens,
        "output_tokens": 0,
        "total_tokens": est_tokens,
        "est_cost_usd": 0.0,
    }
    for k in K_VALUES:
        row[f"recall@{k}"] = 0.0
        row[f"precision@{k}"] = 0.0
        row[f"ndcg@{k}"] = 0.0
        row[f"mrr@{k}"] = 0.0
        row[f"map@{k}"] = 0.0
    row["candidate_completeness"] = 0.0
    return row


# ── Single-model evaluation (OpenRouter variant) ────────────────────────────

def evaluate_model_openrouter(
    client: OpenAI,
    model_name: str,
    queries: list[dict],
    all_skills: list[dict],
    library_size: int,
    context_window: int,
    temperature: float = 0.0,
    supports_max_tokens: bool = True,
    top_k: int = DEFAULT_TOP_K,
    delay: float = DEFAULT_DELAY_S,
    seed: int = SEED,
    gold_names_override: set[str] | None = None,
    model_info: dict | None = None,
) -> pd.DataFrame:
    """Evaluate one model at one library size across all queries via OpenRouter."""
    rng = random.Random(seed)
    gold_names = (gold_names_override if gold_names_override is not None
                  else {q["gold_skill"] for q in queries})
    if model_info is None:
        model_info = {}

    # Build library -- gold skills always included
    gold_skills = [s for s in all_skills if s["name"] in gold_names]
    other = [s for s in all_skills if s["name"] not in gold_names]
    n_fill = min(library_size - len(gold_skills), len(other))
    if n_fill < 0:
        print(f"  SKIP: library_size {library_size} < {len(gold_skills)} gold skills")
        return pd.DataFrame()
    library = gold_skills + rng.sample(other, n_fill)
    rng.shuffle(library)

    library_names = {s["name"] for s in library}

    # Build the prompt body (full descriptions, up to 1024 chars)
    skill_listing = build_skill_listing_full(library)
    if top_k == 1:
        prompt_prefix = SYSTEM_PROMPT_SINGLE
    else:
        prompt_prefix = SYSTEM_PROMPT_RANKED.format(top_k=top_k)
    system_content = f"{prompt_prefix}\n\nAvailable skills:\n{skill_listing}"
    est_tokens = estimate_tokens(system_content)

    use_reasoning = model_info.get("reasoning", False) if model_info else False
    # Reasoning models need more tokens (reasoning budget + content)
    if use_reasoning:
        max_tokens_value = 4096
    else:
        max_tokens_value = 100 if top_k == 1 else max(100, 50 * top_k)

    print(f"  Library: {len(library)} skills | ~{est_tokens:,} prompt tokens (est)")

    # Pre-flight context-window check
    if est_tokens > int(context_window * 0.90):
        print(
            f"  CONTEXT LIMIT (estimated): ~{est_tokens:,} tokens "
            f"> 90% of {context_window:,} window"
        )
        rows = []
        for q in queries:
            rows.append(_ctx_limit_row(model_name, len(library), q, est_tokens,
                                       reason="CONTEXT_LIMIT_ESTIMATED"))
        return pd.DataFrame(rows)

    # Run queries
    rows: list[dict] = []
    consecutive_429 = 0

    for i, q in enumerate(queries):
        t_e2e_start = time.time()

        result = call_openrouter(
            client, model_name, system_content, q["query"],
            temperature=temperature,
            supports_max_tokens=supports_max_tokens,
            max_tokens_value=max_tokens_value,
            provider=model_info.get("provider") if model_info else None,
            reasoning=model_info.get("reasoning", None) if model_info else None,
        )

        llm_latency = result["latency_s"]
        e2e_latency = round(time.time() - t_e2e_start, 3)

        # Parse response
        if result["error"]:
            ranked_skills: list[str] = []
            invalid_skills: list[str] = []
            selected = ""
        elif top_k == 1:
            selected = parse_single_response(result["response"])
            ranked_skills = [selected] if selected else []
            invalid_skills = (
                [selected] if selected and selected not in library_names else []
            )
            if selected and selected not in library_names:
                match = next((n for n in library_names
                              if n.lower() == selected.lower()), None)
                if match:
                    ranked_skills = [match]
                    invalid_skills = []
                    selected = match
        else:
            ranked_skills, invalid_skills = parse_ranked_response(
                result["response"], library_names
            )
            selected = ranked_skills[0] if ranked_skills else ""

        # Exact / fuzzy match
        exact = int(selected == q["gold_skill"])
        fuzzy = int(selected.lower().strip() == q["gold_skill"].lower().strip()
                     if selected else False)

        # Retrieval metrics
        ret_metrics = compute_per_query_retrieval(
            ranked_skills, q["gold_skill"], K_VALUES
        )

        # Cost estimation
        in_tok = result["input_tokens"] or est_tokens
        out_tok = result["output_tokens"] or 0
        cost_in = model_info.get("cost_input_per_1m", 0) * in_tok / 1_000_000
        cost_out = model_info.get("cost_output_per_1m", 0) * out_tok / 1_000_000
        est_cost = round(cost_in + cost_out, 6)

        row = {
            "model": model_name,
            "library_size": len(library),
            "query": q["query"][:120],
            "gold_skill": q["gold_skill"],
            "selected_skill": selected[:120],
            "ranked_skills": json.dumps(ranked_skills[:top_k]),
            "num_valid_skills": len(ranked_skills),
            "num_invalid_skills": len(invalid_skills),
            "invalid_skills": json.dumps(invalid_skills[:10]),
            "rank_of_gold": (
                ranked_skills.index(q["gold_skill"]) + 1
                if q["gold_skill"] in ranked_skills else -1
            ),
            "exact_match": exact,
            "fuzzy_match": fuzzy,
            "latency_s": llm_latency,
            "llm_latency_s": llm_latency,
            "retrieval_latency_s": 0.0,
            "e2e_latency_s": e2e_latency,
            "error": result["error"] or "",
            "context_limit": result["context_limit"],
            "rate_limited": result.get("rate_limited", False),
            "rate_limit_type": result.get("rate_limit_type", ""),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "est_cost_usd": est_cost,
        }
        row.update(ret_metrics)
        rows.append(row)

        # Console output
        if result.get("rate_limited"):
            tag = "429"
            consecutive_429 += 1
        elif result["error"]:
            tag = "CTX" if result["context_limit"] else "ERR"
            consecutive_429 = 0
        elif fuzzy:
            tag = "OK "
            consecutive_429 = 0
        else:
            tag = "MISS"
            consecutive_429 = 0

        rank_str = f"@{row['rank_of_gold']}" if row["rank_of_gold"] > 0 else ""
        print(
            f"  [{i + 1:2d}/{len(queries)}] {tag}  "
            f"gold={q['gold_skill'][:25]:25s}  sel={selected[:30]:30s}  "
            f"{rank_str:4s}  ({llm_latency:.1f}s)"
        )

        # Context limit from API
        if result["context_limit"]:
            print("  >> Context limit hit from API -- skipping remaining queries")
            for rq in queries[i + 1:]:
                rows.append(
                    _ctx_limit_row(model_name, len(library), rq, est_tokens,
                                   reason="CONTEXT_LIMIT_API_SKIPPED")
                )
            break

        # Abort after consecutive 429s
        if consecutive_429 >= CONSECUTIVE_429_ABORT:
            print(
                f"  >> {CONSECUTIVE_429_ABORT} consecutive 429s "
                f"-- skipping remaining queries for this combo"
            )
            for rq in queries[i + 1:]:
                r_row = _ctx_limit_row(model_name, len(library), rq, est_tokens,
                                       reason="RATE_LIMIT_SKIPPED")
                r_row["rate_limited"] = True
                r_row["rate_limit_type"] = result.get("rate_limit_type", "")
                r_row["context_limit"] = False
                rows.append(r_row)
            break

        # Pacing
        if i < len(queries) - 1:
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                print("\n  >> Interrupted! Returning partial results ...")
                df = pd.DataFrame(rows)
                df._interrupted = True
                return df

    return pd.DataFrame(rows)


# ── CSV / plan helpers ───────────────────────────────────────────────────────

def _is_row_clean(error_val: str, rate_limited: bool) -> bool:
    err = str(error_val).strip() if pd.notna(error_val) else ""
    if rate_limited:
        return False
    if not err or err == "nan":
        return True
    if "CONTEXT_LIMIT" in err:
        return True
    return False


def load_run_config(run_dir: str) -> dict | None:
    path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def save_run_config(run_dir: str, cfg: dict) -> None:
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)


def get_completed_combos(run_dir: str, num_queries: int = 20) -> set[tuple[str, int]]:
    csv_path = os.path.join(run_dir, "details.csv")
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return set()

    completed: set[tuple[str, int]] = set()
    for (model, lib_size), group in df.groupby(["model", "library_size"]):
        rate_limited = group["rate_limited"].any() if "rate_limited" in group else False
        err_col = group["error"].fillna("").astype(str)
        err_col = err_col.replace("nan", "")
        has_skip_errors = err_col.str.contains(
            "RATE_LIMIT_SKIPPED|HOURLY_STALL_SKIPPED|DAILY_LIMIT",
            na=False, regex=True
        ).any()
        # Count clean (non-error) rows: empty error or context-limit
        is_clean = err_col.str.len().eq(0) | err_col.str.contains("CONTEXT_LIMIT", na=False)
        n_clean = is_clean.sum()
        all_errors = n_clean == 0
        # Complete only if we have enough clean rows
        has_enough = n_clean >= num_queries
        if not rate_limited and not has_skip_errors and not all_errors and has_enough:
            completed.add((str(model), int(lib_size)))
    return completed


def get_clean_queries_for_combo(
    run_dir: str, model: str, lib_size: int
) -> set[str]:
    csv_path = os.path.join(run_dir, "details.csv")
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return set()
    mask = (df["model"] == model) & (df["library_size"] == lib_size)
    combo = df[mask]
    if combo.empty:
        return set()
    clean = set()
    for _, row in combo.iterrows():
        if _is_row_clean(row.get("error", ""), bool(row.get("rate_limited", False))):
            clean.add(str(row["query"]))
    return clean


def drop_dirty_combo_rows(run_dir: str, model: str, lib_size: int) -> int:
    csv_path = os.path.join(run_dir, "details.csv")
    if not os.path.exists(csv_path):
        return 0
    df = pd.read_csv(csv_path)
    combo_mask = (df["model"] == model) & (df["library_size"] == lib_size)
    combo = df[combo_mask]
    if combo.empty:
        return 0
    dirty_idx = []
    for idx, row in combo.iterrows():
        if not _is_row_clean(row.get("error", ""), bool(row.get("rate_limited", False))):
            dirty_idx.append(idx)
    if dirty_idx:
        df = df.drop(dirty_idx)
        df.to_csv(csv_path, index=False)
    return len(dirty_idx)


def drop_all_combo_rows(run_dir: str, model: str, lib_size: int) -> int:
    """Drop ALL rows for a combo (used by --rerun to force full re-evaluation)."""
    csv_path = os.path.join(run_dir, "details.csv")
    if not os.path.exists(csv_path):
        return 0
    df = pd.read_csv(csv_path)
    combo_mask = (df["model"] == model) & (df["library_size"] == lib_size)
    n_dropped = combo_mask.sum()
    if n_dropped > 0:
        df = df[~combo_mask]
        df.to_csv(csv_path, index=False)
    return int(n_dropped)


def append_results(run_dir: str, new_df: pd.DataFrame) -> None:
    csv_path = os.path.join(run_dir, "details.csv")
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)


# ── Dry-run ──────────────────────────────────────────────────────────────────

def run_dry_run(
    client: OpenAI,
    model_name: str,
    skills: list[dict],
    queries: list[dict],
    model_info: dict,
    top_k: int,
) -> None:
    """Execute a single query at N=50 to validate connectivity and format."""
    print(f"\n{'=' * 70}")
    print(f"DRY RUN  |  Model: {model_info.get('display', model_name)}")
    print(f"{'=' * 70}")

    q = queries[0]
    rng = random.Random(SEED)
    gold_skills = [s for s in skills if s["name"] == q["gold_skill"]]
    other = [s for s in skills if s["name"] != q["gold_skill"]]
    n_fill = min(49, len(other))
    library = gold_skills + rng.sample(other, n_fill)
    rng.shuffle(library)

    library_names = {s["name"] for s in library}

    skill_listing = build_skill_listing_full(library)
    if top_k == 1:
        prompt_prefix = SYSTEM_PROMPT_SINGLE
    else:
        prompt_prefix = SYSTEM_PROMPT_RANKED.format(top_k=top_k)
    system_content = f"{prompt_prefix}\n\nAvailable skills:\n{skill_listing}"
    est_tokens = estimate_tokens(system_content)

    print(f"  Query:      {q['query'][:80]}")
    print(f"  Gold skill: {q['gold_skill']}")
    print(f"  Library:    {len(library)} skills | ~{est_tokens:,} est tokens")
    print(f"  Desc cap:   {DESC_MAX_CHARS} chars (full, not truncated)")
    print()

    result = call_openrouter(
        client, model_name, system_content, q["query"],
        temperature=model_info.get("temperature", 0),
        supports_max_tokens=model_info.get("supports_max_tokens", True),
        max_tokens_value=max(100, 50 * top_k),
        provider=model_info.get("provider"),
        reasoning=model_info.get("reasoning", None),
    )

    if result["error"]:
        print(f"  ERROR: {result['error'][:300]}")
        return

    print(f"  Response ({result['latency_s']:.2f}s):")
    print(f"    {result['response'][:500]}")
    print()

    # Parse
    if top_k == 1:
        selected = parse_single_response(result["response"])
        ranked = [selected] if selected else []
    else:
        ranked, invalid = parse_ranked_response(result["response"], library_names)
        selected = ranked[0] if ranked else ""

    exact = selected == q["gold_skill"]
    print(f"  Selected:   {selected}")
    print(f"  Exact match: {exact}")
    if ranked:
        print(f"  Ranked list: {ranked[:5]}")
    if result["input_tokens"]:
        cost_in = model_info.get("cost_input_per_1m", 0) * result["input_tokens"] / 1_000_000
        cost_out = model_info.get("cost_output_per_1m", 0) * (result["output_tokens"] or 0) / 1_000_000
        print(f"  Tokens:     {result['input_tokens']} in / {result['output_tokens']} out")
        print(f"  Est. cost:  ${cost_in + cost_out:.6f}")

    print(f"\n  Dry run complete. No results persisted.")


# ── Status display ───────────────────────────────────────────────────────────

def print_status(run_dir: str, cfg: dict) -> None:
    model_list = cfg["models"]
    lib_sizes = cfg["library_sizes"]
    num_queries = cfg["num_queries"]
    completed = get_completed_combos(run_dir, num_queries)
    total = len(model_list) * len(lib_sizes)

    print(f"\nResults directory: {run_dir}")
    print(f"Models:    {', '.join(OPENROUTER_MODELS.get(m, {}).get('display', m) for m in model_list)}")
    print(f"Sizes:     {lib_sizes}")
    print(f"Queries:   {num_queries}  (seed={cfg['seed']})")
    print(f"Top-K:     {cfg.get('top_k', DEFAULT_TOP_K)}")
    print(f"Runs:      {len(cfg.get('runs', []))}")
    print(f"Progress:  {len(completed)}/{total} combos completed")

    if cfg.get("runs"):
        print(f"\nRun history:")
        for run in cfg["runs"]:
            print(f"  {run['started']}  completed={run.get('combos_done', '?')}  "
                  f"reason={run.get('stop_reason', '?')}")

    print(f"\nCombo status:")
    for model in model_list:
        display = OPENROUTER_MODELS.get(model, {}).get("display", model)
        for lib_size in lib_sizes:
            if (model, lib_size) in completed:
                status = "done"
            else:
                clean_q = get_clean_queries_for_combo(run_dir, model, lib_size)
                if clean_q:
                    status = f"PARTIAL {len(clean_q)}/{num_queries}"
                else:
                    status = "PENDING"
            print(f"  {display:25s}  N={lib_size:5d}  [{status}]")

    remaining = total - len(completed)
    if remaining == 0:
        print(f"\nAll combos COMPLETE.")
    else:
        print(f"\n{remaining} combo(s) remaining.")


# ── Summary output (table + metrics.json, no plots) ─────────────────────────

def save_summary(run_dir: str, cfg: dict) -> None:
    """Save metrics.json and comparison_table.md from details.csv."""
    csv_path = os.path.join(run_dir, "details.csv")
    if not os.path.exists(csv_path):
        print("\nNo results to summarize.")
        return

    combined = pd.read_csv(csv_path)
    if combined.empty:
        print("\nNo results to summarize.")
        return

    model_list = cfg["models"]
    lib_sizes = cfg["library_sizes"]

    summary = combined.groupby(["model", "library_size"]).agg(
        exact_match=("exact_match", "mean"),
        fuzzy_match=("fuzzy_match", "mean"),
        avg_latency=("latency_s", "mean"),
        context_limited=("context_limit", "sum"),
        rate_limited=("rate_limited", "sum"),
        errors=("error", lambda s: int(s.astype(str).replace("nan", "").ne("").sum())),
        total_queries=("query", "count"),
    ).reset_index()

    summary["model_display"] = summary["model"].map(
        lambda m: OPENROUTER_MODELS.get(m, {}).get("display", m)
    )

    completed = get_completed_combos(run_dir, cfg["num_queries"])
    total_combos = len(model_list) * len(lib_sizes)
    done_combos = len(completed)

    print(f"\n{'=' * 80}")
    print(f"OpenRouter Benchmark  |  Progress: {done_combos}/{total_combos} combos")
    print(f"{'=' * 80}\n")

    show = ["model_display", "library_size", "exact_match", "fuzzy_match",
            "avg_latency", "context_limited", "rate_limited", "errors",
            "total_queries"]
    hdrs = ["Model", "Lib Size", "Exact", "Fuzzy", "Avg Lat(s)",
            "Ctx Limit", "Rate Lim", "Errors", "Total"]
    print(tabulate(summary[show], headers=hdrs,
                   tablefmt="github", floatfmt=".3f", showindex=False))

    # Aggregate metrics
    from benchmark_models import aggregate_metrics

    all_agg: list[dict] = []
    for (model, lib_size), group in combined.groupby(["model", "library_size"]):
        rows_dicts = group.to_dict("records")
        # Patch model_display lookup for aggregate_metrics
        agg = aggregate_metrics(rows_dicts, str(model), int(lib_size), K_VALUES)
        if agg:
            # Override display name for OpenRouter models
            agg["model_display"] = OPENROUTER_MODELS.get(
                str(model), {}
            ).get("display", str(model))
            all_agg.append(agg)

    # Compute total cost
    total_cost = combined["est_cost_usd"].sum() if "est_cost_usd" in combined else 0

    metrics = {
        "run_info": {
            "provider": "openrouter",
            "method": "listing_multi_model",
            "top_k": cfg.get("top_k", DEFAULT_TOP_K),
            "models": model_list,
            "num_queries": cfg["num_queries"],
            "seed": cfg["seed"],
            "library_sizes": lib_sizes,
            "delay_s": cfg.get("delay", DEFAULT_DELAY_S),
            "k_values": K_VALUES,
            "desc_max_chars": DESC_MAX_CHARS,
            "model_context_windows": {
                m: OPENROUTER_MODELS.get(m, {}).get("context_window", 128_000)
                for m in model_list
            },
            "completion": f"{done_combos}/{total_combos}",
            "completed_combos": [list(c) for c in sorted(completed)],
            "total_est_cost_usd": round(total_cost, 4),
            "runs": cfg.get("runs", []),
        },
        "per_model_size": all_agg,
        "summary_table": json.loads(summary.to_json(orient="records")),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(os.path.join(run_dir, "comparison_table.md"), "w") as fh:
        fh.write(f"## OpenRouter Benchmark\n\n")
        fh.write(f"Progress: {done_combos}/{total_combos} combos completed\n")
        fh.write(f"Total estimated cost: ${total_cost:.4f}\n\n")
        fh.write(tabulate(summary[show], headers=hdrs,
                          tablefmt="github", floatfmt=".3f", showindex=False))
        fh.write("\n")

    remaining = total_combos - done_combos
    print(f"\n{'=' * 70}")
    print(f"Results in {run_dir}/")
    print(f"  details.csv           -- per-query rows")
    print(f"  metrics.json          -- comprehensive metrics")
    print(f"  comparison_table.md   -- Markdown table")
    print(f"  test_queries.json     -- test query set")
    if total_cost > 0:
        print(f"  Total est. cost:      ${total_cost:.4f}")
    if remaining > 0:
        print(f"\n  {remaining} combo(s) remaining -- run again to continue")
    else:
        print(f"\n  All combos COMPLETE")
    print(f"  Run benchmark_models_testplan_openrouter_report.py {run_dir} to generate plots")
    print(f"{'=' * 70}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Skill Discovery Benchmark -- OpenRouter Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("result_dir",
                    help="Directory to store results (created if needed)")
    ap.add_argument("--dataset", default="dataset_skills/test.jsonl",
                    help="Skills JSONL file (default: dataset_skills/test.jsonl)")
    ap.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    ap.add_argument("--lib-sizes",
                    default=",".join(str(s) for s in DEFAULT_LIB_SIZES))
    ap.add_argument("--models", default="",
                    help="Comma-separated OpenRouter model slugs "
                         "(default: all in registry)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--status", action="store_true",
                    help="Show status and exit (no API calls)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Single-request validation (not persisted)")
    ap.add_argument("--rerun", type=str, default="",
                    help="Force re-run combo(s): model:size,model:size")
    args = ap.parse_args()

    run_dir = args.result_dir
    os.makedirs(run_dir, exist_ok=True)

    # Determine model list
    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
    else:
        model_list = list(OPENROUTER_MODELS.keys())

    # Register unknown models
    for mn in model_list:
        if mn not in OPENROUTER_MODELS:
            print(f"  INFO: model '{mn}' not in registry -- using 128K context default")
            OPENROUTER_MODELS[mn] = {
                "display": mn.split("/")[-1] if "/" in mn else mn,
                "context_window": 128_000,
                "temperature": 0,
                "supports_max_tokens": True,
                "cost_input_per_1m": 0,
                "cost_output_per_1m": 0,
            }

    # Load or create run config
    cfg = load_run_config(run_dir)

    if cfg is None:
        cfg = {
            "created": datetime.now().isoformat(),
            "provider": "openrouter",
            "models": model_list,
            "library_sizes": sorted(int(x) for x in args.lib_sizes.split(",")),
            "num_queries": args.num_queries,
            "seed": args.seed,
            "top_k": args.top_k,
            "delay": args.delay,
            "dataset": args.dataset,
            "desc_max_chars": DESC_MAX_CHARS,
            "runs": [],
        }
        save_run_config(run_dir, cfg)
        run_only_models = None  # New run: execute all models
        print(f"Created new run config in {run_dir}/")
    else:
        print(f"Resuming (created {cfg['created']})")
        # Merge in new models from --models if specified
        run_only_models = None  # None = run all models in config
        if args.models:
            requested = [m.strip() for m in args.models.split(",")]
            existing = set(cfg["models"])
            added = [m for m in requested if m not in existing]
            if added:
                cfg["models"].extend(added)
                save_run_config(run_dir, cfg)
                print(f"  Added models: {', '.join(added)}")
            # Filter execution to only the requested models
            run_only_models = set(requested)
        model_list = cfg["models"]

    # Status only
    if args.status:
        print_status(run_dir, cfg)
        return

    # Validate API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY must be set in .env or environment.")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )

    # Load dataset
    dataset_path = cfg.get("dataset", args.dataset)
    print(f"\nLoading skills from {dataset_path} ...")
    skills = load_skills(dataset_path)
    print(f"  {len(skills)} skills loaded")

    max_possible = len(skills)
    lib_sizes = [s for s in cfg["library_sizes"] if s <= max_possible]

    # Generate queries (deterministic)
    queries = create_test_queries(skills, cfg["num_queries"], cfg["seed"])
    print(f"  {len(queries)} queries (seed={cfg['seed']})")

    # Save queries
    queries_path = os.path.join(run_dir, "test_queries.json")
    if not os.path.exists(queries_path):
        with open(queries_path, "w") as fh:
            json.dump(queries, fh, indent=2)

    # Dry-run mode
    if args.dry_run:
        # Use first model in list
        mn = model_list[0]
        info = OPENROUTER_MODELS[mn]
        run_dry_run(client, mn, skills, queries, info, cfg.get("top_k", DEFAULT_TOP_K))
        return

    # Handle --rerun (drops ALL rows for specified combos, not just dirty ones)
    force_rerun: set[tuple[str, int]] = set()
    if args.rerun:
        for spec in args.rerun.split(","):
            parts = spec.strip().split(":")
            if len(parts) == 2:
                model_slug, lib_sz = parts[0], int(parts[1])
                force_rerun.add((model_slug, lib_sz))
                n = drop_all_combo_rows(run_dir, model_slug, lib_sz)
                print(f"  Will re-run: {model_slug} @ N={lib_sz} ({n} rows dropped)")

    # Determine pending combos
    completed = get_completed_combos(run_dir, cfg["num_queries"])
    all_combos = [(m, s) for m in model_list for s in lib_sizes]
    pending = [c for c in all_combos if c not in completed or c in force_rerun]
    # If --models was specified on resume, only run those models' pending combos
    if run_only_models:
        pending = [c for c in pending if c[0] in run_only_models]

    print(f"\n  Total combos:     {len(all_combos)}")
    print(f"  Already complete: {len(completed)}")
    print(f"  To run this session: {len(pending)}")

    if not pending:
        print("\nAll combos already completed!")
        save_summary(run_dir, cfg)
        return

    # Run pending combos
    run_record = {
        "started": datetime.now().isoformat(),
        "combos_attempted": [],
        "combos_done": 0,
        "stop_reason": "completed",
    }

    combos_done_this_session = 0

    for combo_i, (model_name, lib_size) in enumerate(pending, 1):
        info = OPENROUTER_MODELS[model_name]

        # Query-level resume
        clean_queries = get_clean_queries_for_combo(run_dir, model_name, lib_size)
        remaining_queries = [q for q in queries if q["query"][:120] not in clean_queries]
        n_dropped = drop_dirty_combo_rows(run_dir, model_name, lib_size)

        print(f"\n{'=' * 70}")
        print(
            f"[{combo_i}/{len(pending)}]  "
            f"Model: {info['display']}  |  Library: N={lib_size}  |  "
            f"Context: {info['context_window']:,} tokens"
        )
        print("=" * 70)

        if clean_queries:
            print(
                f"  Resuming: {len(clean_queries)}/{len(queries)} queries "
                f"already done, {len(remaining_queries)} remaining"
                + (f" ({n_dropped} dirty rows dropped)" if n_dropped else "")
            )

        if not remaining_queries:
            print("  All queries already completed for this combo!")
            combos_done_this_session += 1
            run_record["combos_attempted"].append(f"{model_name}:{lib_size}")
            continue

        all_gold_names = {q["gold_skill"] for q in queries}

        result_df = evaluate_model_openrouter(
            client, model_name, remaining_queries, skills,
            library_size=lib_size,
            context_window=info["context_window"],
            temperature=info.get("temperature", 0),
            supports_max_tokens=info.get("supports_max_tokens", True),
            top_k=cfg.get("top_k", DEFAULT_TOP_K),
            delay=cfg.get("delay", DEFAULT_DELAY_S),
            seed=cfg["seed"],
            gold_names_override=all_gold_names,
            model_info=info,
        )
        interrupted = getattr(result_df, '_interrupted', False)

        if not result_df.empty:
            append_results(run_dir, result_df)
            run_record["combos_attempted"].append(f"{model_name}:{lib_size}")

            acc = result_df["exact_match"].mean()
            valid_lat = result_df[
                (~result_df["context_limit"]) & (result_df["error"] == "")
            ]["latency_s"]
            avg_lat = valid_lat.mean() if len(valid_lat) > 0 else 0
            ctx_n = int(result_df["context_limit"].sum())
            rate_n = int(result_df["rate_limited"].sum())

            new_clean = sum(
                1 for _, r in result_df.iterrows()
                if _is_row_clean(r.get("error", ""), bool(r.get("rate_limited", False)))
            )
            total_clean = len(clean_queries) + new_clean

            # Cost for this combo
            combo_cost = result_df["est_cost_usd"].sum()

            print(
                f"\n  >> Accuracy={acc:.1%}  Avg Latency={avg_lat:.2f}s  "
                f"Context Limits={ctx_n}/{len(result_df)}  "
                f"Rate Limited={rate_n}/{len(result_df)}  "
                f"Progress={total_clean}/{len(queries)} queries  "
                f"Cost=${combo_cost:.4f}"
            )

            if total_clean >= len(queries):
                combos_done_this_session += 1

        if interrupted:
            print("  >> Stopping benchmark (Ctrl-C). Partial results saved.")
            break

    run_record["combos_done"] = combos_done_this_session
    run_record["finished"] = datetime.now().isoformat()
    cfg.setdefault("runs", []).append(run_record)
    save_run_config(run_dir, cfg)

    save_summary(run_dir, cfg)


if __name__ == "__main__":
    main()
