#!/usr/bin/env python3
"""Small SkillRet pilot: data/retrieval, evaluation and command-line orchestration.

Downloaded skill text is data only. No skill code, links or instructions execute.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from reference_metrics import retrieval_metrics, selection_metrics, make_nested_catalogs


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text())


def save_json(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(p)


def append_row(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rows(path):
    p = ROOT / path
    # JSON strings may contain literal U+2028; only LF delimits JSONL records.
    return [json.loads(x) for x in p.read_text().split("\n") if x.strip()] if p.exists() else []


def config(path="pilot.defaults.json"):
    return read_json(path)


def doctor():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)  # Contents are never printed or logged.
    packages = ["deepagents", "langchain", "langchain-core", "langchain-openrouter",
                "datasets", "huggingface-hub", "sentence-transformers", "bm25s",
                "numpy", "pytest", "nbclient"]
    info = {"python": sys.version.split()[0], "env_file_present": (ROOT / ".env").is_file(),
            "key_present": bool(os.getenv("OPENROUTER_API_KEY")),
            "paid_permission": os.getenv("ALLOW_PAID_RUNS") == "1",
            "versions": {p: importlib.metadata.version(p) for p in packages}}
    save_json("evidence/doctor.json", info)
    return info


def download():
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset
    manifest_path = ROOT / "manifests/dataset.json"
    repo = config()["dataset"]["repo_id"]
    sha = read_json("manifests/dataset.json")["revision"] if manifest_path.exists() else HfApi().dataset_info(repo).sha
    expected = read_json("manifests/dataset.json").get("files", {}) if manifest_path.exists() else {}
    manifest = {"repo_id": repo, "revision": sha, "split": "test", "files": {}}
    if not manifest_path.exists():
        save_json("manifests/dataset.json",manifest)  # Pin before the first download.
    for subset in ["skills", "queries", "qrels"]:
        target = ROOT / f"data/{subset}.jsonl"
        if not target.exists():
            ds = load_dataset(repo, subset, split="test", revision=sha,
                              cache_dir=str(ROOT / ".cache/datasets"))
            target.parent.mkdir(exist_ok=True)
            temporary=target.with_suffix(".download")
            with temporary.open("w") as f:
                for row in ds:
                    f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            temporary.replace(target)
        data = target.read_bytes()
        if subset in expected and hashlib.sha256(data).hexdigest()!=expected[subset]["sha256"]:
            raise ValueError("Downloaded/local dataset checksum mismatch: "+subset)
        manifest["files"][subset] = {"path": str(target.relative_to(ROOT)),
                                     "sha256": hashlib.sha256(data).hexdigest(),
                                     "bytes": len(data), "rows": len(data.splitlines())}
    card = hf_hub_download(repo, "README.md", repo_type="dataset", revision=sha)
    (ROOT / "evidence/dataset_card.md").write_bytes(Path(card).read_bytes())
    save_json("manifests/dataset.json", manifest)
    return manifest


def load_data():
    manifest = read_json("manifests/dataset.json")
    out = []
    for name in ["skills", "queries", "qrels"]:
        p = ROOT / manifest["files"][name]["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError("Dataset checksum mismatch: " + name)
        out.append(rows(p))
    return out


def gold_map(qrels):
    gold = defaultdict(set)
    for r in qrels:
        if r["relevance"] > 0:
            gold[r["query_id"]].add(r["skill_id"])
    return gold


def inspect_data():
    skills, queries, qrels = load_data()
    requirements = [(skills, {"id", "name", "description", "skill_md", "major", "sub", "domain"}),
                    (queries, {"id", "query", "skill_ids", "skill_names", "k"}),
                    (qrels, {"query_id", "skill_id", "relevance"})]
    for records, required in requirements:
        if not records or any(not required.issubset(r) for r in records):
            raise ValueError("SkillRet schema drift; required fields missing")
    sids, qids = {s["id"] for s in skills}, {q["id"] for q in queries}
    gold = gold_map(qrels)
    checks = {"duplicate_skill_ids": len(skills)-len(sids),
              "duplicate_query_ids": len(queries)-len(qids),
              "duplicate_qrel_pairs": len(qrels)-len({(r["query_id"],r["skill_id"]) for r in qrels}),
              "orphan_qrels": sum(r["query_id"] not in qids or r["skill_id"] not in sids for r in qrels),
              "empty_gold": sum(not gold[q["id"]] for q in queries),
              "query_qrel_disagreement": sum(set(q["skill_ids"]) != gold[q["id"]] for q in queries),
              "k_disagreement": sum(q["k"] != len(gold[q["id"]]) for q in queries)}
    if any(checks.values()):
        save_json("results/data_integrity_failure.json", checks)
        raise ValueError("Data integrity audit failed; see results/data_integrity_failure.json")
    rng = random.Random(42)
    ids = sorted(qids)
    rng.shuffle(ids)
    pilot_ids = ids[:200]
    singles = [q for q in pilot_ids if len(gold[q]) == 1]
    multis = [q for q in pilot_ids if len(gold[q]) > 1]
    live = [x for pair in zip(singles[:10], multis[:10]) for x in pair]
    live += [q for q in pilot_ids if q not in live][:20-len(live)]
    for path, value in [("pilot_query_ids.json",pilot_ids),("live_query_ids.json",live),
                        ("final_query_ids.json",ids[200:])]:
        if (ROOT / "manifests" / path).exists() and read_json("manifests/"+path) != value:
            raise ValueError("Refusing to change existing query reservation")
        save_json("manifests/"+path,value)
    import numpy as np
    summary = {"revision": read_json("manifests/dataset.json")["revision"],
               "counts": {"skills":len(skills),"queries":len(queries),"qrels":len(qrels)},
               "schemas": {name: {k:type(v).__name__ for k,v in rs[0].items()}
                           for name,rs in zip(["skills","queries","qrels"],[skills,queries,qrels])},
               "checks":checks, "major":dict(Counter(s["major"] for s in skills)),
               "gold_sizes":dict(Counter(len(gold[q["id"]]) for q in queries)),
               "missing_skill_text": {k:sum(not (s.get(k) or "").strip() for s in skills)
                                      for k in ["name","description","skill_md"]},
               "duplicate_names":len(skills)-len({s["name"] for s in skills}),
               "missing_query_text":sum(not q["query"].strip() for q in queries),
               "body_chars_quantiles": dict(zip(["min","median","p95","max"],
                    np.quantile([len(s["skill_md"]) for s in skills],[0,.5,.95,1]).tolist())),
               "license_fields":{k:dict(Counter(str(s[k]) for s in skills))
                                 for k in skills[0] if "license" in k.lower()},
               "pilot_queries":200,"live_queries":len(live),"final_reserved":len(ids)-200,
               "provenance":"Public GitHub skills; synthetic queries. Metadata/queries Apache-2.0; source skill licenses retained. See pinned card."}
    qby = {q["id"]:q for q in queries}
    sby = {s["id"]:s for s in skills}
    examples = [{"query":qby[q],"relevant_skill_metadata":[{k:v for k,v in sby[s].items() if k!="skill_md"}
                                                           for s in sorted(gold[q])] } for q in live[:3]]
    save_json("results/dataset_summary.json",summary)
    save_json("results/pilot_examples.json",examples)
    return summary


def full_document(skill):
    return " | ".join((skill.get(k) or "").strip() for k in ["name","description","skill_md"])


def bm25_rank(texts, ids, query_texts, k=10):
    """CPU port of pinned skillret/eval.py eval_bm25; positive scores only."""
    import bm25s
    index = bm25s.BM25()
    index.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
    rankings = []
    for query in query_texts:
        tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        if len(tokens.vocab) == 0:
            tokens = bm25s.tokenize("NONE", stopwords=[], show_progress=False)
        hits, scores = index.retrieve(tokens, corpus=ids, k=min(k,len(ids)), show_progress=False)
        rankings.append([(str(h), float(s)) for h,s in zip(hits[0],scores[0]) if s>0])
    return rankings


def baseline(n=200):
    if not 1 <= n <= 200:
        raise ValueError("Baseline may use only the 200 reserved pilot queries")
    skills, queries, qrels = load_data()
    ids = read_json("manifests/pilot_query_ids.json")[:n]
    qby = {q["id"]:q for q in queries}
    gold = gold_map(qrels)
    spec = {"dataset_revision":read_json("manifests/dataset.json")["revision"],
            "source_commit":"0cc508e6bfcfff1fe6352d90ff5957ba38a64bba",
            "representation":"name | description | skill_md (full text)",
            "bm25s":importlib.metadata.version("bm25s"),"tokenization":"bm25s.tokenize(stopwords='en'), default regex/lowercase; no stemming",
            "parameters":{"method":"lucene","k1":1.5,"b":0.75,"delta":0.5},
            "query_ids":ids,"protocol":"partial-protocol reproduction on 200-query pilot subset; local metrics"}
    run_id = digest(spec)
    old = [r for r in rows("results/baseline_summary.jsonl") if r["run_id"]==run_id]
    if old:
        return old[-1]
    start = time.monotonic()
    rankings = bm25_rank([full_document(s) for s in skills],[s["id"] for s in skills],[qby[q]["query"] for q in ids])
    metrics = []
    for q,ranked in zip(ids,rankings):
        m = {**retrieval_metrics([r[0] for r in ranked],gold[q],5),
             **retrieval_metrics([r[0] for r in ranked],gold[q],10)}
        metrics.append(m)
        append_row("results/baseline_rows.jsonl",{"run_id":run_id,"query_id":q,"ranking":ranked,
                   "evaluator_only_gold":sorted(gold[q]),"metrics":m})
    summary = {"run_id":run_id,"spec":spec,"query_count":n,"corpus_size":len(skills),
               "metrics":{k:sum(m[k] for m in metrics)/n for k in metrics[0]},
               "wall_seconds":time.monotonic()-start}
    append_row("results/baseline_summary.jsonl",summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["doctor","download","inspect","baseline","offline-check","vectors","smoke","run","activation-smoke","report","package","ollama-smoke","ollama-run","ollama-report"])
    p.add_argument("--queries",type=int,default=200)
    p.add_argument("--config",default="pilot.defaults.json")
    p.add_argument("--live",action="store_true")
    p.add_argument("--quick",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--model",default=None,help="Installed local model name for Ollama commands")
    a = p.parse_args()
    if a.command.startswith("ollama-"):
        import local_runner
        result={"results":local_runner.report() if a.command=="ollama-report" else
                (local_runner.smoke if a.command=="ollama-smoke" else local_runner.run)(a.model)}
    elif a.command in ["doctor","download","inspect","baseline"]:
        result = {"doctor":doctor,"download":download,"inspect":inspect_data,"baseline":lambda:baseline(a.queries)}[a.command]()
    else:
        from trial import dispatch
        result = dispatch(a)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if a.command=='smoke' and a.live and not result.get('canary_correct',False):
        raise SystemExit(2)
    if a.command=='run' and a.live and not result.get('prerequisites_ok',False):
        raise SystemExit(2)
    if a.command=='activation-smoke' and a.live and any(s!='valid' for s in result.get('statuses',[])):
        raise SystemExit(2)
    if a.command=='ollama-smoke' and any(not r['correct'] for r in result['results']):
        raise SystemExit(2)
    if a.command=='ollama-run' and any(r.get('status')=='canary_failed' or r.get('valid',0)<20 for r in result['results']):
        raise SystemExit(2)
    if result.get("status")=="BLOCKED" or (a.live and result.get("blocker")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
