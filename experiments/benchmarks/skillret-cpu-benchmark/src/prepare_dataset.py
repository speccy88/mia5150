#!/usr/bin/env python3
"""Download pinned public SkillRet test data and recreate the CPU evaluation inputs.

Only Python's standard library is required. This recreates inputs, not prior scores.
Run: python src/prepare_dataset.py [--offline] [--root /path/to/project]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "ThakiCloud/SKILLRET"
REVISION = "6583d7d2ed07644d0fb8938ed8178f3a7dc42a12"
QUERY_SALT = "skillret-cpu-query-subset-v1"
CATALOG_SALT = "skillret-cpu-catalog-v1"
MAIN_QUERY_COUNT = 64
TIMING_QUERY_COUNT = 32
ORIGINAL_QUERY_COUNT = 256
DESCRIPTION_CHARS = 1024
SOURCES = {
    "data/skills/test.jsonl": (117606037, "1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81"),
    "data/queries/test.jsonl": (6053507, "4d07a35f40e597f8468c6dab480c95a1f9b40ecbde80274081ba2e7c643d0f6d"),
    "data/qrels/test.jsonl": (661204, "edbfae0b8109de460d54e27e6235b1b47316f34956744db0571ad1b4ddb860af"),
    "data/taxonomy.json": (5372, "8fb8e8e8eeb22b8781615d455350a1d9d03e7fe10ca26940993fc37b0cdacf48"),
    "README.md": (10763, "f4a387d14dcfe53a1a2495d36bb5ee47c1208c2a5812d9440c63aa03aa55edb0"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def hash_key(*parts: str) -> bytes:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).digest()


def source_url(relative: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{relative}"


def ensure_source(raw: Path, relative: str, offline: bool) -> dict:
    path = raw / relative
    expected_bytes, expected_sha256 = SOURCES[relative]
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Missing source in offline mode: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(source_url(relative) + "?download=true", timeout=180) as response, partial.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        require(partial.stat().st_size == expected_bytes, f"Source size mismatch: {relative}")
        require(sha256_file(partial) == expected_sha256, f"Source SHA-256 mismatch: {relative}")
        partial.replace(path)
    require(path.stat().st_size == expected_bytes, f"Source size mismatch: {relative}")
    require(sha256_file(path) == expected_sha256, f"Source SHA-256 mismatch: {relative}")
    return {"path": "data/raw/" + relative, "repository_path": relative,
            "url": source_url(relative), "bytes": expected_bytes, "sha256": expected_sha256}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def unique_index(rows: list[dict], field: str, label: str) -> dict[str, dict]:
    require(all(isinstance(row.get(field), str) and row[field] for row in rows), f"Invalid {label} ID")
    result = {row[field]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {label} IDs")
    return result


def make_inputs(root: Path, offline: bool) -> dict:
    raw = root / "data/raw"
    out = root / "data/processed"
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        sources = list(executor.map(lambda relative: ensure_source(raw, relative, offline), SOURCES))
    write_json(raw / "source_manifest.json", {"repository": REPO, "revision": REVISION, "files": sources})
    source_skills = read_jsonl(raw / "data/skills/test.jsonl")
    source_queries = read_jsonl(raw / "data/queries/test.jsonl")
    source_qrels = read_jsonl(raw / "data/qrels/test.jsonl")
    skills_by_id = unique_index(source_skills, "id", "skill")
    queries_by_id = unique_index(source_queries, "id", "query")
    require((len(source_skills), len(source_queries), len(source_qrels)) == (6006, 4392, 7187), "Unexpected pinned test counts")
    skill_ids = sorted(skills_by_id)
    skill_id_set = set(skill_ids)
    gold_by_query = collections.defaultdict(set)
    qrel_pairs = set()
    for row in source_qrels:
        qid, sid = row["query_id"], row["skill_id"]
        require(row["relevance"] == 1, "Qrels must be binary positives")
        require(qid in queries_by_id and sid in skills_by_id, "Qrel join failed")
        require((qid, sid) not in qrel_pairs, "Duplicate qrel pair")
        qrel_pairs.add((qid, sid))
        gold_by_query[qid].add(sid)
    require(set(gold_by_query) == set(queries_by_id), "Query without gold qrels")
    for qid, row in queries_by_id.items():
        require(len(row["skill_ids"]) == len(set(row["skill_ids"])), f"Duplicate query skill IDs: {qid}")
        require(set(row["skill_ids"]) == gold_by_query[qid], f"Query/qrel gold mismatch: {qid}")
        require(1 <= len(gold_by_query[qid]) <= 3, f"Invalid gold count: {qid}")
        require(row["k"] == len(gold_by_query[qid]), f"Query k mismatch: {qid}")
        require(isinstance(row["query"], str) and bool(row["query"]), f"Invalid query text: {qid}")

    skills = []
    for sid in skill_ids:
        row = skills_by_id[sid]
        require(all(isinstance(row[field], str) for field in ["name", "description", "major", "domain"]), f"Invalid skill fields: {sid}")
        skills.append({"skill_id": sid, "name": row["name"], "description": row["description"],
                       "text": row["name"] + "\n" + row["description"][:DESCRIPTION_CHARS],
                       "major": row["major"], "domain": row["domain"]})

    queries = []
    for qid in sorted(queries_by_id):
        gold = sorted(gold_by_query[qid])
        queries.append({"query_id": qid, "text": queries_by_id[qid]["query"], "gold_skill_ids": gold,
                        "major": sorted({skills_by_id[sid]["major"] for sid in gold}),
                        "domain": sorted({skills_by_id[sid]["domain"] for sid in gold})})
    processed_queries = {row["query_id"]: row for row in queries}
    ordered_query_ids = sorted(queries_by_id, key=lambda qid: (hash_key(QUERY_SALT, qid), qid))
    selected = ordered_query_ids[:MAIN_QUERY_COUNT]
    timing = ordered_query_ids[:TIMING_QUERY_COUNT]
    original = ordered_query_ids[:ORIGINAL_QUERY_COUNT]
    selected_set = set(selected)
    catalogs = []
    for qid in selected:
        gold = sorted(gold_by_query[qid])
        negatives = sorted(skill_id_set - set(gold), key=lambda sid: (hash_key(CATALOG_SALT, qid, sid), sid))
        candidates = (gold + negatives)[:1000]
        require(len(candidates) == len(set(candidates)) == 1000, f"Invalid catalog: {qid}")
        require(set(candidates) <= skill_id_set, f"Catalog join failed: {qid}")
        require(candidates[:len(gold)] == gold, f"Gold prefix mismatch: {qid}")
        for size in [10, 100, 1000]:
            require(set(gold) <= set(candidates[:size]), f"Missing gold in {size}-pool: {qid}")
        require(set(candidates[:10]) <= set(candidates[:100]) <= set(candidates) <= skill_id_set, "Non-nested catalog")
        catalogs.append({"query_id": qid, "candidate_skill_ids": candidates})

    qrels = sorted(source_qrels, key=lambda row: (row["query_id"], row["skill_id"]))
    selected_qrels = [row for row in qrels if row["query_id"] in selected_set]
    write_jsonl(out / "skills.jsonl", skills)
    write_jsonl(out / "allqueries.jsonl", queries)
    write_jsonl(out / "queries.jsonl", [processed_queries[qid] for qid in selected])
    write_jsonl(out / "qrels.jsonl", selected_qrels)
    write_jsonl(out / "qrels_selected64.jsonl", selected_qrels)
    write_jsonl(out / "qrels_all.jsonl", qrels)
    write_jsonl(out / "catalogs.jsonl", catalogs)
    write_json(out / "full_catalog.json", skill_ids)
    write_json(out / "expensive_query_ids.json", original)
    write_json(out / "selected_query_ids.json", selected)
    write_json(out / "latency_query_ids.json", timing)
    write_json(out / "pilot32_query_ids.json", timing)
    write_json(out / "taxonomy.json", json.loads((raw / "data/taxonomy.json").read_text(encoding="utf-8")))

    # Read the written semantic views back to check the actual artifacts, not just local variables.
    saved_skills = read_jsonl(out / "skills.jsonl")
    allowed_skill_fields = {"skill_id", "name", "description", "text", "major", "domain"}
    require(all(set(row) == allowed_skill_fields for row in saved_skills), "Unexpected semantic skill field")
    require(all(row["text"] == row["name"] + "\n" + row["description"][:DESCRIPTION_CHARS] for row in saved_skills), "Semantic text mismatch")
    require([row["skill_id"] for row in saved_skills] == skill_ids, "Skills not sorted by ID")
    require([row["query_id"] for row in read_jsonl(out / "allqueries.jsonl")] == sorted(queries_by_id), "All queries not sorted by ID")
    require([row["query_id"] for row in read_jsonl(out / "queries.jsonl")] == selected, "Selected query order mismatch")
    require(timing == selected[:32] and selected == original[:64], "Cohort nesting mismatch")

    artifacts = {}
    names = ["skills.jsonl", "allqueries.jsonl", "queries.jsonl", "qrels.jsonl", "qrels_selected64.jsonl",
             "qrels_all.jsonl", "catalogs.jsonl", "full_catalog.json", "expensive_query_ids.json",
             "selected_query_ids.json", "latency_query_ids.json", "pilot32_query_ids.json", "taxonomy.json"]
    for name in names:
        path = out / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "protocol_version": "skillret-cpu-recovery-v1", "dataset_repository": REPO, "dataset_revision": REVISION,
        "recovery_statement": "Exact input protocol recreation from pinned public sources; no old raw model results are restored or invented.",
        "source_manifest": "data/raw/source_manifest.json", "source_counts": {"skills": 6006, "queries": 4392, "qrels": 7187},
        "cohort_counts": {"main": 64, "latency_and_fallback": 32, "preserved_original_hash_subset": 256},
        "qrels_jsonl_scope": "selected 64 query IDs only", "qrels_selected_count": len(selected_qrels),
        "qrels_all_jsonl_scope": "all 4392 test queries and all 7187 binary-positive qrels",
        "query_selection": {"salt": QUERY_SALT, "key": "(SHA256(UTF8(salt + NUL + query_id)), query_id)", "ordering": "ascending raw digest; equivalent to ascending hexadecimal digest", "main_prefix": 64, "latency_prefix": 32, "preserved_original_prefix": 256},
        "candidate_selection": {"salt": CATALOG_SALT, "stored_order": "sorted complete gold skill IDs, then non-gold IDs sorted by (SHA256(UTF8(salt + NUL + query_id + NUL + skill_id)), skill_id); first 1000 IDs", "memberships": {"10": "candidate_skill_ids[:10]", "100": "candidate_skill_ids[:100]", "1000": "candidate_skill_ids[:1000]", "full": "all sorted 6006 test skill IDs"}, "scoring_order": "Sort candidate skill IDs lexicographically before scoring at every pool size. Positive-first storage must never be used as score/rank order.", "rank_tie_break": "skill_id ascending after score descending"},
        "semantic_input": {"skill": "name + LF + description[:1024]", "description_limit_unit": "Python Unicode code points", "query": "raw query text only", "body_and_skill_md": "excluded", "skill_metadata": "Pre-existing major tags may be used for an index only; never append labels to semantic text.", "query_gold_skill_ids_major_domain": "Evaluation/audit labels only. Forbidden as retrieval, routing, prompt, scoring, or candidate-ranking inputs."},
        "artifacts": artifacts,
    }
    write_json(out / "manifest.json", manifest)
    audit = {
        "passed": True, "source_hashes_and_sizes_verified": True,
        "source_counts_exact": True, "duplicate_skill_ids": 0, "duplicate_query_ids": 0, "duplicate_qrel_pairs": 0,
        "qrel_join_failures": 0, "query_qrel_gold_mismatches": 0, "queries_without_gold": 0,
        "gold_count_histogram_all": dict(sorted(collections.Counter(len(gold_by_query[qid]) for qid in queries_by_id).items())),
        "gold_count_histogram_selected64": dict(sorted(collections.Counter(len(gold_by_query[qid]) for qid in selected).items())),
        "max_gold_count": max(map(len, gold_by_query.values())), "all_pool_sizes_preserve_complete_gold": True,
        "catalogs_unique_valid_and_nested": True, "semantic_fields_exact_and_body_excluded": True,
        "semantic_text_formula_verified_from_written_artifacts": True, "cohort_nesting_32_64_256_verified": True,
        "selected_query_ids_first8": selected[:8], "processed_manifest_sha256": sha256_file(out / "manifest.json"),
        "scoring_order_status": "Input protocol requires lexicographic skill ID order. Inference/scorer execution must be audited separately; this script does not run models.",
        "label_leakage_status": "Query labels retained only for audit/evaluation; this script does not route queries or perform inference.",
    }
    write_json(root / "protocol/data_audit.json", audit)
    atomic_write(root / "protocol/DATA_PROTOCOL.md", protocol_markdown(manifest, audit))
    return {"root": str(root), "source_counts": manifest["source_counts"], "cohort_counts": manifest["cohort_counts"],
            "selected_qrels": len(selected_qrels), "audit_passed": True, "manifest": str(out / "manifest.json")}


def protocol_markdown(manifest: dict, audit: dict) -> str:
    return f"""# SkillRet CPU input recreation protocol

This is an exact recreation of the specified input protocol from public pinned sources. It is not recovery of any lost historical model outputs. No model inference is performed here.

## Source and integrity

- Dataset: https://huggingface.co/datasets/{REPO}/tree/{REVISION}
- Revision: `{REVISION}`
- Test split: 6,006 skills, 4,392 queries, 7,187 binary-positive qrels
- Raw file byte sizes, URLs and SHA-256 values: `data/raw/source_manifest.json`
- Processed artifact sizes and SHA-256 values: `data/processed/manifest.json`
- Checks: `protocol/data_audit.json`
- Reproduce online: `python src/prepare_dataset.py`
- Reproduce from verified local raw data: `python src/prepare_dataset.py --offline`
- Only the Python standard library is needed. Existing raw files are hash-checked, never silently trusted.

## Frozen query memberships

Sort all query IDs by `(SHA256(UTF8('skillret-cpu-query-subset-v1' + NUL + query_id)), query_id)`, ascending. The NUL is one zero code point, not two printable characters. SHA digest byte ordering equals hexadecimal lexical ordering.

- `selected_query_ids.json`: the first 64 IDs, the main cohort
- `latency_query_ids.json`: the first 32 IDs, frozen timing and fallback cohort
- `pilot32_query_ids.json`: the identical first-32 alias
- `expensive_query_ids.json`: the original first-256 membership preserved for traceability, not a claim that 256 queries were evaluated
- `queries.jsonl`: selected 64 rows in hash order
- `allqueries.jsonl`: all test queries in lexical query-ID order
- `qrels.jsonl` and `qrels_selected64.jsonl`: selected 64 only, {manifest['qrels_selected_count']} positive rows
- `qrels_all.jsonl`: all 7,187 positive rows

## Frozen candidate memberships and ranking safety

For each selected query, concatenate the complete lexically sorted gold skill IDs with all non-gold IDs sorted by `(SHA256(UTF8('skillret-cpu-catalog-v1' + NUL + query_id + NUL + skill_id)), skill_id)`. Store the first 1,000 IDs in `catalogs.jsonl`.

The 10, 100 and 1,000 pools are prefixes of that stored list. The full pool is all 6,006 sorted test skill IDs from `full_catalog.json`. All pools are nested and contain every positive (at most three). This is a gold-preserving controlled candidate-set experiment; small pools are not unassisted full-catalog retrieval.

Crucially, the positive-first arrangement is storage only. Every scorer must sort the selected candidate IDs lexically before scoring. Never treat storage order as a rank. Sort final results by descending score and ascending skill ID for deterministic ties. Any scorer must verify this itself; the preparation script does not run inference.

## Semantic inputs and metadata restrictions

Every processed skill has exactly `skill_id`, `name`, `description`, `text`, `major`, and `domain`. Semantic `text` is exactly `name + '\\n' + description[:1024]`, with slicing in Unicode code points and no stripping or other normalization. The preserved description field is not the semantic input if it exceeds 1,024 characters. Full `body` and `skill_md` are excluded.

Query semantic input is raw query text only. Query `gold_skill_ids`, `major`, and `domain` are evaluation/audit labels, derived from the joined gold skills, and must never be used to route, score, rank, or build a query prompt. Pre-existing skill `major` tags are allowed for index construction only, not as added semantic text. Any routing decision must come from the query text and eligible index, never query ground-truth labels.

## Validation performed

Verified all source checksums and expected counts; no duplicate skill IDs, query IDs, or qrel pairs; all qrels join; query skill sets exactly match qrels; `k` equals the gold count; each query has 1–3 positives; all selected catalogs contain valid unique IDs, preserve all gold, and nest correctly; cohort 32/64/256 nesting; exact field whitelist and semantic formula in written artifacts; correct output row order.

All-data gold-count histogram: {json.dumps(audit['gold_count_histogram_all'], sort_keys=True)}. Selected-64 histogram: {json.dumps(audit['gold_count_histogram_selected64'], sort_keys=True)}.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--offline", action="store_true", help="Require all pinned raw sources locally; do not download")
    args = parser.parse_args()
    print(json.dumps(make_inputs(args.root.resolve(), args.offline), indent=2))


if __name__ == "__main__":
    main()
