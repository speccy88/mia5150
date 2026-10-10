"""Offline controls for the additional scaling scheduler; no generation calls."""

import numpy as np
import pytest
from scripts.benchmark_openrouter import select_candidates, build_plan
from pilot import rows, digest, make_nested_catalogs


def test_domain_filter_has_no_global_fallback_and_stable_ties():
    library = ["b", "a", "c"]
    metadata = {
        s: {"metadata": s, "major": d} for s, d in zip(library, ["X", "Y", "X"])
    }
    vectors = {
        "docs": np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "queries": np.array([[1.0, 0.0]]),
    }
    indices = {"b": 0, "a": 1, "c": 2}
    qindices = {"q": 0}
    assert (
        select_candidates(
            "vector", library, "query", "q", metadata, vectors, indices, qindices
        )
        == library
    )
    assert select_candidates(
        "domain_vector",
        library,
        "query",
        "q",
        metadata,
        vectors,
        indices,
        qindices,
        "Y",
    ) == ["a"]
    assert (
        select_candidates(
            "domain_vector",
            library,
            "query",
            "q",
            metadata,
            vectors,
            indices,
            qindices,
            "missing",
        )
        == []
    )


def test_three_query_scaling_fits_remaining_request_cap_without_resetting_ledger(
    reference_json,
):
    # Check recorded reuse with its original source identity. Formatting the GitHub
    # edition changes current hashes, and must not erase historical evidence.
    cfgs, _, fresh_spec, _ = build_plan()
    spec = reference_json("manifests/openrouter_scaling.json")
    reused = 0
    cached_routes = set()
    for label in spec["models"]:
        rid = digest(spec["sources"][label])
        reused += len(
            {
                (r["query_id"], r["N"], r["strategy"])
                for r in reference_json(
                    f"results/openrouter/{label}/trial.jsonl", jsonl=True
                )
                if r["run_id"] == rid and r["query_id"] in spec["query_ids"]
            }
        )
        for r in reference_json(f"results/openrouter/{label}/routes.jsonl", jsonl=True):
            if r["query_id"] in spec["query_ids"] and r["case_key"] == digest(
                [rid, r["query_id"], "router"]
            ):
                cached_routes.add((label, r["query_id"]))
    # Worst case before exclusions: one call for each uncached selector/route.
    additional = 180 - reused + 15 - len(cached_routes)
    assert additional == 132
    # Baseline ledger has 147 attempts; the new study never needs a higher ceiling.
    assert (
        147 + additional
        <= cfgs[spec["models"][0]]["budget"]["max_completion_requests"]
        == 300
    )
    # A fresh three-query run also fits: selectors + routers + one canary/model.
    assert len(fresh_spec["query_ids"]) * 3 * 4 * 5 + 3 * 5 + 5 <= 300


def test_catalog_size_extension_preserves_previous_50_skill_order():
    ids = [str(i) for i in range(1200)]
    small = make_nested_catalogs(ids, {"1", "2"}, [50], "q")
    large = make_nested_catalogs(ids, {"1", "2"}, [50, 250, 1000], "q")
    assert small[50] == large[50]
    assert set(large[50]) < set(large[250]) < set(large[1000])


@pytest.mark.parametrize(
    "usage,settled,expected",
    [
        (0.381476739, 0.3814767505, True),
        (0.3814757505, 0.3814767505, False),
        (0.32430197, 0.3814767505, False),
        (0.40, 0.3814767505, False),
    ],
)
def test_reconciliation_rounding_never_clears_usage_lag_or_material_difference(
    usage, settled, expected
):
    from scripts.reconcile_rejection import usage_matches

    assert usage_matches(usage, settled) == expected
