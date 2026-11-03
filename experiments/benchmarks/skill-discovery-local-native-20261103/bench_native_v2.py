"""Skill discovery v1. Standard library only; inference is opt-in via `run`."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
LOCAL_PROTOCOL = "v1-native-final-answer-parser-v2"
SOURCE_HASHES = {
    "test.jsonl": "f53aad53cfb44c7a3a26edb8550e8a64080b0b9916c17d8e61bb79f60c6617c5",
    "test_queries.json": "7ef69814dd8f93f4a8dadc290567f158e78986898aa76a33bc95cf883716c28b",
}
SIZES = [50, 100, 250, 500, 1000, 2000, 5000]
SYSTEM = (
    "You are evaluating skill discovery performance.\n"
    "Given a user task, identify the most relevant skills from the list below.\n\n"
    "RULES:\n"
    "- Reply with a numbered list of the top 10 most relevant skills, ranked from most to least relevant.\n"
    "- Use ONLY exact skill names as they appear after the number in the list.\n"
    "- Format each line as: NUMBER. SKILL_NAME\n"
    "- Do NOT add explanations, quotes, or extra formatting.\n"
    "- If fewer than 10 skills are relevant, list only the relevant ones.\n\n"
    "SKILLS:\n"
)
COLUMNS = (
    "model,library_size,query,gold_skill,selected_skill,ranked_skills,"
    "num_valid_skills,num_invalid_skills,invalid_skills,rank_of_gold,exact_match,"
    "fuzzy_match,latency_s,llm_latency_s,retrieval_latency_s,e2e_latency_s,error,"
    "context_limit,rate_limited,rate_limit_type,input_tokens,output_tokens,"
    "total_tokens,est_cost_usd,recall@1,precision@1,ndcg@1,mrr@1,map@1,"
    "recall@3,precision@3,ndcg@3,mrr@3,map@3,recall@5,precision@5,ndcg@5,"
    "mrr@5,map@5,recall@10,precision@10,ndcg@10,mrr@10,map@10,candidate_completeness"
).split(",")


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load_data(dataset, queries, strict=True):
    # Full skill bodies remain data. Only the three metadata fields are retained.
    skills = []
    with Path(dataset).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            skills.append({k: item[k] for k in ("id", "name", "description")})
    qs = json.loads(Path(queries).read_text(encoding="utf-8-sig"))
    if strict and (len(skills) != 6006 or len(qs) != 20):
        raise ValueError("Expected exactly 6006 skills and the 20 supplied queries")
    ids = {s["id"]: s for s in skills}
    if len(ids) != len(skills):
        raise ValueError("Duplicate IDs; source must be investigated")
    if len({q["query"] for q in qs}) != len(qs):
        raise ValueError("Duplicate query texts; row keys must be unambiguous")
    for s in skills:
        if not all(isinstance(s[k], str) for k in ("id", "name", "description")):
            raise ValueError("Metadata must contain strings")
    for q in qs:
        if q["gold_id"] not in ids or ids[q["gold_id"]]["name"] != q["gold_skill"]:
            raise ValueError("Gold ID/name mismatch")
    return skills, qs


def make_plan(skills, queries, sizes=SIZES):
    by_id = {s["id"]: s for s in skills}
    trials = []
    for n in sizes:
        # Fresh Random(42) per size: identical base library for all 20 queries.
        base = random.Random(42).sample(skills, n)
        for qi, q in enumerate(queries):
            sampled = list(base)
            replaced = not any(s["id"] == q["gold_id"] for s in sampled)
            if replaced:
                sampled[-1] = by_id[q["gold_id"]]
            system = SYSTEM + "\n".join(
                f"{i}. {s['name']}: {s['description'][:300]}"
                for i, s in enumerate(sampled, 1)
            )
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": q["query"]}]
            trials.append({"key": f"{n}:{qi}", "n": n, "qi": qi,
                           "query": q, "candidates": sampled, "messages": messages,
                           "gold_replaced_last": replaced,
                           "prompt_sha256": digest(canonical(messages).encode())})
    return trials


def clean_name(text):
    return text.strip().strip("`*\"' ").strip()


def reasoning_prefix(prompt):
    """Recognize only a reasoning marker already supplied by the chat template."""
    if re.search(r"<think>\s*\Z", prompt):
        return "<think>\n"
    if re.search(r"<\|channel>thought\s*\Z", prompt):
        return "<|channel>thought\n"
    return ""


def final_answer(text, prefix=""):
    """Keep internal delimited reasoning out of ranks; never promote invalid final slots."""
    combined = (prefix + text).replace("<|channel>thought", "<think>").replace("<channel|>", "</think>")
    inside, cursor, parts = False, 0, []
    for match in re.finditer(r"</?think>", combined):
        opening = match.group() == "<think>"
        if opening:
            if inside:
                return "", "MALFORMED_REASONING_BOUNDARY"
            parts.append(combined[cursor:match.start()])
            inside = True
        else:
            if not inside:
                return "", "MALFORMED_REASONING_BOUNDARY"
            inside = False
            cursor = match.end()
    if inside:
        return "", "REASONING_NO_FINAL_ANSWER"
    parts.append(combined[cursor:])
    return "".join(parts), None


def parse_ranked(text, candidates):
    """Never remove an invalid rank and promote the next item to top-1."""
    names = {s["name"] for s in candidates}
    folded = {}
    for name in names:
        folded.setdefault(name.casefold(), []).append(name)
    # Strip think blocks; malformed/unclosed reasoning produces invalid output.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        m = re.match(r"^(?:[-*]\s+)?(?:\*\*)?(\d+)[.)]\s*(.*)$", line)
        if not m:
            # Non-list text is an invalid slot; never silently ignore an answer.
            entries.append({"raw": line, "name": None, "exact": False})
            continue
        label = clean_name(m[2])
        # Exact full name takes priority over description suffix removal.
        variants = [label]
        variants.extend(clean_name(x) for x in re.split(r"\s+(?:[-–—]|:)\s+|:\s+", label, maxsplit=1)[:1])
        match, exact = None, False
        for val in variants:
            if val in names:
                match, exact = val, True
                break
        if match is None:
            for val in variants:
                possible = folded.get(val.casefold(), [])
                if len(possible) == 1:
                    match = possible[0]
                    break
        entries.append({"raw": label, "name": match, "exact": exact})
    entries = entries[:10]
    # Duplicate predictions consume their ranks and are reported invalid.
    seen = set()
    for e in entries:
        if e["name"] in seen:
            e["name"], e["exact"] = None, False
        if e["name"] is not None:
            seen.add(e["name"])
    return entries


def metrics(rank):
    result = {}
    for k in (1, 3, 5, 10):
        hit = bool(rank and rank <= k)
        result.update({f"recall@{k}": int(hit), f"precision@{k}": 1/k if hit else 0,
                       f"ndcg@{k}": 1/math.log2(rank + 1) if hit else 0,
                       f"mrr@{k}": 1/rank if hit else 0,
                       f"map@{k}": 1/rank if hit else 0})
    return result


def empty_row(config, trial):
    q = trial["query"]
    row = {k: "" for k in COLUMNS}
    row.update(model=config["model"], library_size=trial["n"], query=q["query"],
               gold_skill=q["gold_skill"], ranked_skills="[]", invalid_skills="[]",
               num_valid_skills=0, num_invalid_skills=0, rank_of_gold=0,
               exact_match=False, fuzzy_match=False, context_limit=False,
               rate_limited=False, candidate_completeness=1.0)
    row.update(metrics(0))
    for k in ("latency_s", "llm_latency_s", "retrieval_latency_s", "e2e_latency_s"):
        row[k] = "0.000000"
    if config["backend"] == "llama":
        row["est_cost_usd"] = "0.000000"  # Local inference service cost, excludes electricity.
    return row


def score(row, text, candidates):
    ranked = parse_ranked(text, candidates)
    valid = [e for e in ranked if e["name"] is not None]
    invalid = [e["raw"] for e in ranked if e["name"] is None]
    row.update(ranked_skills=json.dumps([e["name"] or e["raw"] for e in ranked], ensure_ascii=False),
               selected_skill=ranked[0]["name"] or ranked[0]["raw"] if ranked else "",
               num_valid_skills=len(valid), num_invalid_skills=len(invalid),
               invalid_skills=json.dumps(invalid, ensure_ascii=False))
    gold = row["gold_skill"]
    # Strict retrieval scores require exact spelling after format normalization.
    rank = next((i for i, e in enumerate(ranked, 1) if e["name"] == gold and e["exact"]), 0)
    row["rank_of_gold"] = rank
    row["exact_match"] = rank == 1
    row["fuzzy_match"] = bool(ranked and ranked[0]["name"] == gold)
    row.update(metrics(rank))
    if not ranked:
        row["error"] = "EMPTY_OUTPUT"
    elif not valid:
        row["error"] = "INVALID_OUTPUT"
    return ranked


class BenchError(Exception):
    def __init__(self, code, status=None, provider_code=None, request_id=None, retry_after_s=None):
        super().__init__(code)
        self.code, self.status = code, status
        self.provider_code = provider_code
        self.request_id = request_id
        self.retry_after_s = retry_after_s


def safe_error_code(value):
    return value if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,100}", value) else None


def classify_error(message, status=None):
    low = message.lower()
    if any(x in low for x in ("out of memory", "cuda error", "cuda malloc", "allocat")):
        return "HARDWARE_OOM"
    if any(x in low for x in ("context_length", "context window", "context length", "context size", "too many tokens", "input too long")):
        return "CONTEXT_LIMIT_ACTUAL"
    if status == 429:
        return "RATE_LIMITED"
    return f"HTTP_{status}" if status else "TRANSPORT_ERROR"


def request(url, body=None, token=None, timeout=180, stream=False):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise BenchError("HTTP_REDIRECT_REJECTED")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers)
    try:
        response = urllib.request.build_opener(NoRedirect).open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = e.read(32768).decode("utf-8", "replace")
        # Provider text is never echoed into logs/CSV: it can contain prompts or secrets.
        try:
            payload = json.loads(raw)
            detail = payload.get("error", {}) if isinstance(payload, dict) else {}
            provider_code = safe_error_code(detail.get("code")) if isinstance(detail, dict) else None
        except (ValueError, TypeError):
            provider_code = None
        rid = e.headers.get("x-request-id", "") if e.headers else ""
        rid = rid if re.fullmatch(r"[A-Za-z0-9_-]{1,150}", rid) else None
        after = e.headers.get("retry-after", "") if e.headers else ""
        after = int(after) if after.isdigit() else None
        raise BenchError(classify_error(raw, e.code), e.code, provider_code, rid, after) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise BenchError("TRANSPORT_ERROR") from None
    if stream:
        return response
    with response:
        return json.load(response)


def read_sse(response, deadline=None, audit=None):
    text, completed, usage = [], False, None
    data = []
    def consume(lines):
        if not lines:
            return None
        payload = "\n".join(lines)
        return None if payload == "[DONE]" else json.loads(payload)
    for raw in response:
        if deadline is not None and time.perf_counter() > deadline:
            raise BenchError("STREAM_DEADLINE_EXCEEDED")
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            event, data = consume(data), []
            if not event:
                continue
            typ = event.get("type")
            if typ == "response.output_text.delta":
                text.append(event.get("delta", ""))
            elif typ == "response.completed":
                completed = True
                result = event.get("response", {})
                usage = result.get("usage")
                if audit is not None:
                    audit.update(server_model=result.get("model"), response_id=result.get("id"),
                                 reasoning_effort=(result.get("reasoning") or {}).get("effort"),
                                 token_source="response.completed.usage")
            elif typ in ("response.refusal.delta", "response.refusal.done"):
                raise BenchError("MODEL_REFUSAL")
            elif typ in ("response.failed", "error", "response.incomplete"):
                error = event.get("error") or event.get("response", {}).get("error") or {}
                provider_code = safe_error_code(error.get("code")) if isinstance(error, dict) else None
                kind = "RATE_LIMITED" if provider_code and "usage_limit_exceeded" in provider_code else "RESPONSE_FAILED"
                raise BenchError(kind, provider_code=provider_code)
    if data:
        raise BenchError("SSE_INCOMPLETE_EVENT")
    if not completed:
        raise BenchError("SSE_NOT_COMPLETED")
    return "".join(text), usage


def check_gpu_proof(config):
    proof_path = Path(config["gpu_proof"])
    proof = json.loads(proof_path.read_text(encoding="utf-8-sig"))
    if not proof.get("all_layers_gpu") or not proof.get("kv_gpu"):
        raise BenchError("GPU_PROOF_INVALID")
    if proof["context_window"] != config["context_window"] or proof["model"] != config["model"]:
        raise BenchError("GPU_PROOF_CONFIG_MISMATCH")
    # Logs grow after startup; hash the sealed startup prefix only.
    with Path(proof["log_path"]).open("rb") as f:
        if digest(f.read(proof["log_bytes"])) != proof["log_sha256"]:
            raise BenchError("GPU_PROOF_LOG_MISMATCH")
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-Command",
               f"$p = Get-Process -Id {int(proof['pid'])} -ErrorAction Stop; "
               "$p.StartTime.ToUniversalTime().ToString('o')"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode or p.stdout.strip() != proof["process_start"]:
            raise BenchError("GPU_SERVER_NOT_RUNNING")
    props = request(config["base_url"].rstrip("/") + "/props", timeout=10)
    actual_ctx = props.get("default_generation_settings", {}).get("n_ctx")
    if actual_ctx != config["context_window"]:
        raise BenchError("SERVER_CONTEXT_MISMATCH")
    # Alias binds proof to this process's endpoint, avoiding accidental Ollama/other server use.
    models = request(config["base_url"].rstrip("/") + "/v1/models", timeout=10)
    if config["model"] not in [m["id"] for m in models.get("data", [])]:
        raise BenchError("SERVER_MODEL_MISMATCH")
    return proof


def infer(config, trial):
    base, backend = config["base_url"].rstrip("/"), config["backend"]
    timeout = config.get("timeout_s", 300)
    token = None
    if backend == "chatgpt-oauth":
        from oauth_chatgpt import access_token, OAuthError
        try:
            token = access_token()
        except OAuthError:
            raise BenchError("OAUTH_AUTH_REQUIRED") from None
    elif backend == "openai-compatible":
        token = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"))
        if not token:
            raise BenchError("API_KEY_MISSING")
    t0 = time.perf_counter()
    measured_input = None
    response_audit = {}
    if backend == "llama":
        check_gpu_proof(config)
        templated = request(base + "/apply-template", {"messages": trial["messages"]}, timeout=timeout)
        response_audit["reasoning_prefix"] = reasoning_prefix(templated["prompt"])
        tokens = request(base + "/tokenize", {"content": templated["prompt"], "add_special": True,
                                             "parse_special": True}, timeout=timeout)["tokens"]
        measured_input = len(tokens)
        if measured_input + 500 > config["context_window"]:
            raise BenchError("CONTEXT_LIMIT_TOKENIZED")
        t1 = time.perf_counter()
        r = request(base + "/completion", {"prompt": tokens, "temperature": 0,
                    "n_predict": 500, "seed": 42, "stream": False, "cache_prompt": False}, timeout=timeout)
        if r.get("truncated"):
            raise BenchError("PROTOCOL_PROMPT_TRUNCATED")
        text = r.get("content", "")
        usage = {"input_tokens": r.get("tokens_evaluated", measured_input),
                 "output_tokens": r.get("tokens_predicted")}
        if usage["input_tokens"] != measured_input:
            raise BenchError("PROTOCOL_TOKEN_COUNT_MISMATCH")
    elif backend == "openai-compatible":
        t1 = time.perf_counter()
        r = request(base + "/chat/completions", {"model": config["model"],
                    "messages": trial["messages"], "temperature": 0, "max_tokens": 500,
                    "stream": False}, token=token, timeout=timeout)
        text = r["choices"][0]["message"].get("content") or ""
        u = r.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}
    elif backend == "chatgpt-oauth":
        t1 = time.perf_counter()
        body = {"model": config["model"], "instructions": trial["messages"][0]["content"],
                "input": [trial["messages"][1]], "store": False, "stream": True}
        with request(base + "/responses", body, token=token, timeout=timeout, stream=True) as response:
            text, usage = read_sse(response, deadline=t1+timeout, audit=response_audit)
    else:
        raise ValueError("Unknown backend")
    return {"text": text, "usage": usage or {}, "llm_s": time.perf_counter() - t1,
            "e2e_s": time.perf_counter() - t0, "preflight_tokens": measured_input,
            "response_audit": response_audit}


class RunLock:
    def __init__(self, folder):
        self.file = (Path(folder)/"run.lock").open("a+b")
    def __enter__(self):
        self.file.seek(0)
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self
    def __exit__(self, *_):
        if getattr(self, "db", None) is not None:
            self.db.close()
        self.file.close()


def export_csv(db, folder, plan=None, config=None, not_run_error=None):
    saved = {r[0]: json.loads(r[1]) for r in db.execute("SELECT key, row_json FROM trials WHERE row_json IS NOT NULL ORDER BY ordinal")}
    if plan is not None and not_run_error:
        rows = []
        for trial in plan:
            if trial['key'] in saved:
                rows.append(saved[trial['key']])
            else:
                row = empty_row(config, trial)
                row['error'] = not_run_error
                rows.append(row)
    else:
        rows = list(saved.values())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(Path(folder)/"details.csv", buffer.getvalue())


def evaluate(config, trial):
    row = empty_row(config, trial)
    # Required v1 estimate gate, kept independent from measured tokenizer counts.
    if trial["n"]*44.5 + 200 > .9*config["context_window"]:
        row.update(error="CONTEXT_LIMIT_ESTIMATED", context_limit=True)
        return row, {"context_estimate": trial["n"]*44.5 + 200}
    if config.get("startup_error"):
        row["error"] = config["startup_error"]
        return row, {"inference_attempted": False, "startup_error": config["startup_error"]}
    started = time.perf_counter()
    try:
        result = infer(config, trial)
        parsed_text, reasoning_error = final_answer(result["text"],
            result.get("response_audit", {}).get("reasoning_prefix", ""))
        if reasoning_error:
            row["error"] = reasoning_error
            ranked = []
        else:
            ranked = score(row, parsed_text, trial["candidates"])
        row.update(llm_latency_s=f"{result['llm_s']:.6f}", latency_s=f"{result['e2e_s']:.6f}",
                   e2e_latency_s=f"{time.perf_counter()-started:.6f}")
        u = result["usage"]
        tin, tout = u.get("input_tokens"), u.get("output_tokens")
        usage_available = all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (tin, tout))
        if usage_available:
            row.update(input_tokens=tin, output_tokens=tout, total_tokens=tin+tout)
            prices = config.get("usd_per_million")
            if prices:
                row["est_cost_usd"] = f"{(tin*prices['input']+tout*prices['output'])/1e6:.9f}"
        elif config.get('require_token_usage'):
            row['error'] = 'TOKEN_USAGE_MISSING'
        # Prompt hashes, normalized ranks and a response hash support local auditing.
        audit = {"response_sha256": digest(result["text"].encode()),
                 "token_usage_available": usage_available,
                 "preflight_tokens": result["preflight_tokens"],
                 "ambiguous_returned_names": sorted({e["name"] for e in ranked if e["name"] and
                     sum(s["name"] == e["name"] for s in trial["candidates"]) > 1})}
        audit.update(result.get("response_audit", {}))
        audit.update(final_answer_sha256=digest(parsed_text.encode()), reasoning_error=reasoning_error)
        if config.get("raw_response_dir"):
            raw_path = Path(config["raw_response_dir"])/(trial["key"].replace(":", "_")+"-"+audit["response_sha256"][:12]+".json")
            atomic_write(raw_path, json.dumps({"text":result["text"], "usage":u,
                "reasoning_prefix":result.get("response_audit",{}).get("reasoning_prefix","")},ensure_ascii=False))
            audit["raw_response_path"] = str(raw_path)
        return row, audit
    except BenchError as e:
        row["error"] = e.code
        row["context_limit"] = e.code.startswith("CONTEXT_LIMIT")
        row["rate_limited"] = e.code == "RATE_LIMITED"
        row["rate_limit_type"] = "provider" if row["rate_limited"] else ""
        if not row["context_limit"]:
            row["latency_s"] = row["e2e_latency_s"] = f"{time.perf_counter()-started:.6f}"
        return row, {"http_status": e.status, "provider_code": e.provider_code,
                     "request_id": e.request_id, "retry_after_s": e.retry_after_s}


def run(config, skills, queries, folder, resume=False, resolve=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    trials = make_plan(skills, queries)
    spec = {"protocol": LOCAL_PROTOCOL if config["backend"] != "chatgpt-oauth" else "v1-oauth-variant",
            "config": config, "dataset_sha256": file_hash(ROOT/"data/test.jsonl"),
            "queries_sha256": file_hash(ROOT/"data/test_queries.json"),
            "implementation_sha256": {n: file_hash(ROOT/n) for n in (Path(__file__).name,'oauth_chatgpt.py')},
            "plan": [{"key": t["key"], "prompt_sha256": t["prompt_sha256"],
                      "candidate_ids": [s["id"] for s in t["candidates"]],
                      "gold_replaced_last": t["gold_replaced_last"]} for t in trials]}
    fingerprint = digest(canonical(spec).encode())
    with RunLock(folder) as lock:
        existed = (folder/"state.sqlite3").exists()
        if existed and not resume:
            raise ValueError("Output exists; use --resume or a new output directory")
        db = sqlite3.connect(folder/"state.sqlite3")
        lock.db = db
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS meta (fingerprint TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS trials (key TEXT PRIMARY KEY, ordinal INTEGER, row_json TEXT, audit TEXT)")
        saved = db.execute("SELECT fingerprint FROM meta").fetchone()
        if saved and saved[0] != fingerprint:
            raise ValueError("Resume configuration/data/plan changed; use a new directory")
        if not saved:
            db.execute("INSERT INTO meta VALUES (?)", (fingerprint,))
            db.commit()
            atomic_write(folder/"manifest.json", json.dumps(spec, ensure_ascii=False, indent=2))
            atomic_write(folder/"notes.md", run_notes(config))
        pending = {r[0] for r in db.execute("SELECT key FROM trials WHERE row_json IS NULL")}
        if pending and not resolve:
            raise ValueError("Uncertain previous request; inspect locally, then --resolve-inflight-as-error. Never auto-replayed.")
        stop = None
        for i, trial in enumerate(trials):
            prev = db.execute("SELECT row_json FROM trials WHERE key=?", (trial["key"],)).fetchone()
            if prev and prev[0] is not None:
                continue
            if trial["key"] in pending:
                row, audit = empty_row(config, trial), {"uncertain_previous_request": True}
                row["error"] = "INTERRUPTED_REQUEST_OUTCOME_UNKNOWN"
            else:
                db.execute("INSERT INTO trials VALUES (?, ?, NULL, NULL)", (trial["key"], i))
                db.commit()  # Durable intent BEFORE any network request.
                row, audit = evaluate(config, trial)
            db.execute("UPDATE trials SET row_json=?, audit=? WHERE key=?",
                       (canonical(row), canonical(audit), trial["key"]))
            db.commit()
            export_csv(db, folder)
            print(f"{i+1}/140 N={trial['n']} query={trial['qi']+1} {row['error'] or 'OK'}", flush=True)
            # Account/auth failures stop the run; no fallback or repeated rejected calls.
            fatal = row["error"] in ("HTTP_401", "HTTP_403", "RATE_LIMITED", "API_KEY_MISSING", "OAUTH_AUTH_REQUIRED")
            if config.get('stop_on_any_error') and row['error'] and row['error']!='CONTEXT_LIMIT_ESTIMATED':
                fatal = True
            if fatal:
                stop = {'error':row['error'], 'at_trial':trial['key'], **audit}
                atomic_write(folder/'stop.json', json.dumps(stop, indent=2))
                print("Stopped; remaining trials are pending. Resume only after resolving account access.")
                break
            pause = config.get('inter_request_delay_s', 0)
            if pause and not row['context_limit']:
                time.sleep(min(float(pause), 60))
        completed = db.execute('SELECT COUNT(*) FROM trials WHERE row_json IS NOT NULL').fetchone()[0]
        status = {'model':config['model'],'protocol':spec['protocol'],'recorded_trials':completed,
                  'planned_trials':len(trials),'not_run_trials':len(trials)-completed,'stop':stop,
                  'complete':completed==len(trials)}
        export_csv(db, folder, trials, config, 'NOT_RUN_AFTER_MODEL_STOP' if stop else None)
        atomic_write(folder/'status.json',json.dumps(status,indent=2))
        db.close()
        return status


def run_notes(config):
    return f"""# Run notes

Protocol: {'v1-oauth-variant' if config['backend']=='chatgpt-oauth' else LOCAL_PROTOCOL}
Model: {config['model']}; backend: {config['backend']}; context: {config['context_window']}.
Quantization: {config.get('quantization', 'provider managed / unknown')}.
Hardware: NVIDIA RTX 2060 SUPER, 8192 MiB; observed driver 610.88 (preparation, 2026-11-03).
Actual local startup proof is bound to the server process, model file hash and log prefix.

140 trials = seven sizes x supplied 20 queries. Candidate source order retained.
Fresh random.Random(42).sample per size; identical base library for all queries at that size.
Absent gold ID replaces the last skill. Descriptions are sliced to 300 Python characters.
This preserves v1 placement bias. Confirm RNG reset scope against the original hosted runner
before claiming byte-identical reproduction; that runner was not provided.

The 6006 IDs have 5801 distinct names. Nothing is deduplicated. Scores resolve names,
not IDs: duplicate-name identity cannot be recovered from the required output format.
Candidate IDs, prompt hashes and per-response ambiguous-name audits remain local.
Some queries contain name hints, and 20 queries provide a small, partly lexical sample.

The v1 context estimate (44.5*N+200 > 90% window) creates explicit blocked rows;
actual tokenized overflow is separate. Estimated counts are NEVER token usage.
Local /apply-template + /tokenize uses the loaded model's real tokenizer, then passes
the resulting token IDs to native /completion (temperature=0, n_predict=500, seed=42).
GPU-only means all transformer/output layers and KV on CUDA; CPU tokenization/control
is expected. Partial offload is rejected. No context shifting or automatic memory fit.
500 output tokens are reserved before local inference. No truncation is accepted.
No thinking-mode override is applied; generation can exhaust its 500-token allowance.
Native parser v2 retains a prefilled <think> marker as parsing state only. It is
not added to the model prompt a second time. Delimited reasoning is excluded from
rank scoring; unfinished reasoning has REASONING_NO_FINAL_ANSWER with real usage.
Invalid and repeated slots in the final answer retain their positions. Raw model
responses are kept in the private response directory, excluded from Git export.

Compatible API: system/user chat messages, temperature=0, max_tokens=500. Unsupported
parameters are errors, with no silent retry or replacement model.
OAuth variant: same system text in instructions, user input array, streaming Responses,
store=false. temperature and max_output_tokens MUST be omitted on this route.
It is a separate variant, without a guaranteed 500-token output cap or fixed temperature.
OAuth context comes from the authenticated account catalog, not the larger public API
maximum. Actual input/output token counts come from response.completed.usage; the
44.5 tokens/skill gate remains an estimate, not a claimed tokenizer measurement.
Reasoning effort is not sent; the actual server value is kept in each SQLite audit.
If an error stops a model, CSV placeholders use NOT_RUN_AFTER_MODEL_STOP. These rows
were never attempted and must not be interpreted as observed zero-accuracy results.

Ranking preserves invalid/duplicate slots; bad rank 1 is never promoted away.
Markdown/quotes/description suffixes are normalized. Case-only matching is a recorded
fallback (fuzzy_match); strict exact/retrieval metrics require exact name spelling.
P@K denominator is always K; one gold: recall hit, NDCG=1/log2(rank+1), MRR=AP=1/rank.
candidate_completeness=1 because the gold ID is assured in every candidate library;
it is not response-list length. rank_of_gold=0 means absent, never rank 0 relevance.

Token counts come from server usage/tokenizer only; unavailable fields remain blank.
Unknown hosted monetary cost remains blank. Local service cost=0 excludes electricity.
Latency: retrieval=0 (listing method); llm measures inference HTTP roundtrip; e2e includes
preflight/template/tokenization/parsing, excluding run setup and SQLite/CSV persistence.
Estimated context skips have zero latency/metrics. Other errors retain elapsed time.
No automatic inference retries. In-flight interrupted calls require explicit resolution,
so a crash cannot silently repeat a possibly charged request. SQLite is authoritative;
CSV is rebuilt atomically after each transaction. Use unique output directories.

No inference ran during script preparation. Tests use synthetic data/mock HTTP only.
No full prompts, API keys or OAuth tokens are printed to console or stored in run outputs.
CSV contains supplied queries/names and invalid output lines: review before sharing.
OAuth credentials use Windows DPAPI CurrentUser in this package's private/. No Hermes
or Codex credential files are read, copied, or modified. Do not publish private/ or data/.
"""


def validate_config(config):
    backend = config["backend"]
    url = urllib.parse.urlparse(config["base_url"])
    if config["model"].startswith("REPLACE_"):
        raise ValueError("Replace the example model slug with an account-verified choice first")
    if config["context_window"] <= 500:
        raise ValueError("context_window must exceed 500")
    if backend == "llama":
        if url.hostname != "127.0.0.1" or url.scheme != "http":
            raise ValueError("Local backend must use HTTP 127.0.0.1")
    elif backend == "chatgpt-oauth":
        if config["base_url"].rstrip("/") != "https://api.openai.com/v1":
            raise ValueError("OAuth tokens can only go to the documented OpenAI API origin")
    elif backend == "openai-compatible":
        if url.scheme != "https":
            raise ValueError("Hosted API requires HTTPS")
    else:
        raise ValueError("Unknown backend")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "plan", "run"])
    parser.add_argument("--config", default=str(ROOT/"configs/native-v2/qwen3-0.6b-q8.json"))
    parser.add_argument("--out", default=str(ROOT/"results/local-native-v2/qwen3-0.6b-q8"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resolve-inflight-as-error", action="store_true")
    parser.add_argument("--allow-hosted-inference", action="store_true")
    parser.add_argument("--startup-error-log", help="Record a verified local startup OOM without inference")
    args = parser.parse_args()
    for filename, expected in SOURCE_HASHES.items():
        if file_hash(ROOT/"data"/filename) != expected:
            raise ValueError("Source bytes differ from the supplied v1 data; stop before inference")
    skills, qs = load_data(ROOT/"data/test.jsonl", ROOT/"data/test_queries.json")
    if args.command == "validate":
        info = {"records": len(skills), "unique_names": len({s['name'] for s in skills}),
                "queries": len(qs), "gold_ids_valid": True,
                "dataset_sha256": file_hash(ROOT/"data/test.jsonl"),
                "queries_sha256": file_hash(ROOT/"data/test_queries.json")}
        atomic_write(ROOT/"data/validation.json", json.dumps(info, indent=2))
        print(json.dumps(info, indent=2))
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if config.get("gpu_proof"):
        proof = Path(config["gpu_proof"])
        config["gpu_proof"] = str((ROOT/proof).resolve()) if not proof.is_absolute() else str(proof)
    validate_config(config)
    if args.command == "plan":
        plan = make_plan(skills, qs)
        info = {"trials": len(plan), "sizes": SIZES, "estimated_blocked_trials":
                sum(t["n"]*44.5+200 > .9*config["context_window"] for t in plan),
                "model": config["model"], "backend": config["backend"], "inference_called": False}
        print(json.dumps(info, indent=2))
        return
    if args.startup_error_log:
        if config["backend"] != "llama":
            raise ValueError("Startup OOM evidence is only for local models")
        log_path = Path(args.startup_error_log)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if classify_error(text) != "HARDWARE_OOM":
            raise ValueError("Startup log does not contain recognizable hardware OOM evidence")
        config.update(startup_error="HARDWARE_OOM", startup_error_log_sha256=file_hash(log_path))
    if config["backend"] != "llama" and not args.allow_hosted_inference:
        raise ValueError("Hosted inference disabled; explicitly pass --allow-hosted-inference to consume usage")
    if config["backend"] == "llama" and not config.get("startup_error"):
        proof = check_gpu_proof(config)
        config["local_artifact_identity"] = {k: proof[k] for k in
            ("model_sha256", "executable_sha256", "kv_type", "quantization")}
    run(config, skills, qs, args.out, args.resume, args.resolve_inflight_as_error)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, BenchError, OSError) as exc:
        print(f"Stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
