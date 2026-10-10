#!/usr/bin/env python3
"""Offline consistency checks for the research plan, not a model benchmark."""
from __future__ import annotations
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool) -> None:
        checks.append({"check": name, "passed": bool(passed)})

    try:
        papers = load("registries/papers.json")
        sources = load("registries/engineering_sources.json")
        models = load("registries/models.json")
        plan = load("registries/experiments.json")
        reference = load("evidence/published_reference_results.json")
        paper_ids = {p["id"] for p in papers}
        source_ids = {p["id"] for p in sources}
        model_ids = {p["id"] for p in models}
        phases = plan["phases"]
        phase_ids = {p["id"] for p in phases}
        check("54 unique paper IDs", len(papers) == len(paper_ids) == 54)
        check("unique primary paper URLs", len({p["primary_url"] for p in papers}) == len(papers))
        check("18 unique engineering sources", len(sources) == len(source_ids) == 18)
        check("25 unique model/method roles", len(models) == len(model_ids) == 25)
        check("seven unique work orders", len(phases) == len(phase_ids) == 7)
        check("primary URLs are HTTPS", all(urlparse(p["primary_url"]).scheme == "https" for p in papers))
        check("inspection depth recorded", all(p.get("inspection_depth") for p in papers))
        check("no literature reproduction claim", all(p["reproduced_in_this_work"] is False for p in papers))
        check("no model test or account-access claim", all(m["loaded_or_tested_here"] is False and m["access_confirmed"] is False for m in models))
        check("model source references resolve", all(m["source_id"] in source_ids | paper_ids for m in models))
        check("paid calls default disabled", plan["allow_paid_calls"] is False)
        check("pod creation default disabled", plan["allow_pod_creation"] is False)
        check("training default disabled", plan["allow_training"] is False)
        check("unallocated budgets remain null", plan["approved_budget_cad"] is None and plan["approved_gpu_hours"] is None)
        check("proposal library sizes preserved", plan["library_sizes"] == [10, 100, 1000, 6006])
        check("official metric cutoffs preserved", plan["official_metric_k"] == [5, 10, 15])
        check("proposal MRR cutoff preserved", plan["proposal_extra_metric"] == "MRR@5")
        check("corrected test counts recorded", plan["official_expected_counts"]["test_skills"] == 6006 and plan["official_expected_counts"]["test_queries"] == 4392 and plan["official_expected_counts"]["test_qrels"] == 7187)
        check("phase dependency references resolve", all(set(p["depends_on"]) <= phase_ids for p in phases))
        check("phase paper references resolve", all(set(p["paper_ids"]) <= paper_ids for p in phases))
        check("phase model references resolve", all(set(p["model_ids"]) <= model_ids for p in phases))
        check("phase prompt files exist", all((ROOT / p["prompt"]).is_file() for p in phases))
        pending = {p["id"]: set(p["depends_on"]) for p in phases}
        visited: set[str] = set()
        while pending:
            ready = {k for k, deps in pending.items() if deps <= visited}
            if not ready:
                break
            visited.update(ready)
            for k in ready:
                pending.pop(k)
        check("work-order graph is acyclic", not pending)
        check("published references marked not reproduced", reference["status"] == "published_reference_only_not_reproduced")
        numeric = [v for row in reference["rows"] for k, v in row.items() if k != "model"]
        check("reference metrics use valid 0-100 scale", all(isinstance(v, (float, int)) and 0 <= v <= 100 for v in numeric))
        check("main published anchor recorded correctly", reference["rows"][-1]["ndcg_10"] == 87.74 and reference["rows"][-1]["completeness_10"] == 88.96)
        proposal = (ROOT / "inputs/proposal_1.pdf").read_bytes()
        check("original proposal is a PDF", proposal.startswith(b"%PDF"))
        check("original proposal byte hash preserved", hashlib.sha256(proposal).hexdigest() == "7fb118f08cbfa36370c9d6dd897d79489d13d1c9545cc5946abb5265d5a263a6")
        required = ["START_HERE.md", "EXPERIMENT_PLAN.md", "BENCHMARK_CONTRACT.md", "PROPOSAL_ALIGNMENT.md", "RESOURCE_PLAN.md", "RESEARCH_CATALOGUE.md", "CODEX/MASTER_HANDOFF.md", "evidence/VERIFICATION_NOTES.md", "evidence/SEARCH_SCOPE.md"]
        check("required planning documents exist", all((ROOT / p).is_file() for p in required))
        manifest_path = ROOT / "MANIFEST_SHA256.json"
        if manifest_path.is_file():
            manifest = load("MANIFEST_SHA256.json")
            check("packaged byte hashes match", all((ROOT / p).is_file() and hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in manifest["files"].items()))
        report = {"scope": "offline planning-package consistency only; no model/API/benchmark execution", "success": all(c["passed"] for c in checks), "checks_passed": sum(bool(c["passed"]) for c in checks), "checks_total": len(checks), "paper_count": len(papers), "paper_categories": dict(Counter(p["category"] for p in papers)), "model_method_roles": len(models), "jobs": len(phases), "checks": checks}
        print(json.dumps(report, indent=2))
        return 0 if report["success"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"success": False, "error": str(error), "scope": "planning-package validation"}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
