#!/usr/bin/env python3
"""Evaluate ranked SkillReason-Bench predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def ranked_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("retrieved_skill_ids") or row.get("retrieved") or []
    result = []
    for value in values:
        skill_id = value.get("skill_id") if isinstance(value, dict) else value
        skill_id = str(skill_id or "").strip()
        if skill_id and skill_id not in result:
            result.append(skill_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("data/queries.jsonl"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    queries = {str(row["query_id"]): set(map(str, row["positive_ids"])) for row in read_jsonl(args.queries)}
    predictions: dict[str, list[str]] = {}
    for row in read_jsonl(args.predictions):
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in predictions:
            raise ValueError(f"Missing or duplicate prediction query_id: {query_id!r}")
        predictions[query_id] = ranked_ids(row)

    unknown = set(predictions) - set(queries)
    missing = set(queries) - set(predictions)
    if unknown:
        raise ValueError(f"Predictions contain {len(unknown)} unknown query IDs")
    if missing and not args.allow_missing:
        raise ValueError(f"Predictions are missing {len(missing)} queries")

    ks = (1, 5, 10, 20)
    recall = {k: 0.0 for k in ks}
    ndcg10 = 0.0
    mrr20 = 0.0
    for query_id, gold in queries.items():
        ranked = predictions.get(query_id, [])
        for k in ks:
            recall[k] += len(gold.intersection(ranked[:k])) / len(gold)
        first_rank = next((index + 1 for index, skill_id in enumerate(ranked[:20]) if skill_id in gold), None)
        if first_rank is not None:
            mrr20 += 1.0 / first_rank
            if first_rank <= 10:
                ndcg10 += 1.0 / math.log2(first_rank + 1)

    total = len(queries)
    metrics = {f"Recall@{k}": recall[k] / total for k in ks}
    metrics.update(
        {
            "NDCG@10": ndcg10 / total,
            "MRR@20": mrr20 / total,
            "num_queries": total,
            "prediction_coverage": (total - len(missing)) / total,
        }
    )
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
