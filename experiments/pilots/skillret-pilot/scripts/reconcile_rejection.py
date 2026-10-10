"""Reconcile a known pre-generation rejection against the key's cumulative usage.

Never clears a timeout, server failure, missing response, or ambiguous generation.
Only suitable for this dedicated key with no unrelated concurrent usage.
"""

from pathlib import Path
import argparse
import os
import sys
import time
import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import rows, save_json
from agent_runner import Budget


def usage_matches(usage, settled):
    """Allow at most 0.1 microdollar of cumulative billing rounding.

    Applies only after explicit pre-generation 400/429 rejection validation.
    A provider usage lag or any meaningful unexplained charge still fails closed.
    """
    from decimal import Decimal

    difference = abs(Decimal(str(usage)) - Decimal(str(settled)))
    return difference.is_finite() and difference <= Decimal("0.0000001")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_id")
    args = parser.parse_args()
    failure = next(
        r
        for r in rows("evidence/traces/failures.jsonl")
        if r["request_id"] == args.request_id
    )
    raw = failure.get("raw_response") or {}
    code = raw.get("error", {}).get("code")
    assert code in {400, 429} and not raw.get("id") and not raw.get("choices")
    if code == 400:
        assert (
            raw["error"]["message"]
            == "Reasoning is mandatory for this endpoint and cannot be disabled."
        )
    if code == 429:
        meta = raw["error"]["metadata"]
        assert meta.get("provider_error_code") in {
            "rate_limit_exceeded",
            "engine_overloaded",
        }
        assert meta.get("limit_source") == "upstream_provider_shared_pool"
        assert "temporarily rate-limited upstream" in meta.get("raw", "")
    events = rows("results/budget.jsonl")
    assert not any(
        r["event"] == "settle" and r["request_id"] == args.request_id for r in events
    )
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    response = httpx.get(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
        timeout=30,
    )
    response.raise_for_status()
    usage = response.json()["data"]["usage"]
    total = Budget().totals()
    assert usage_matches(
        usage, total["actual_usd"]
    ), "Provider usage does not reconcile; reservation retained."
    evidence = {
        "request_id": args.request_id,
        "http_code": code,
        "provider_key_usage": usage,
        "ledger_actual_usd": total["actual_usd"],
        "reconciled_actual_usd": 0,
        "time": time.time(),
        "difference_usd": usage - total["actual_usd"],
        "tolerance_usd": 0.0000001,
        "reason": "Pre-generation rejection with no completion ID; cumulative key usage matches settled ledger costs within the logged rounding tolerance.",
    }
    save_json(
        "evidence/openrouter-reconciliation-" + str(time.time_ns()) + ".json", evidence
    )
    Budget().settle(args.request_id, 0)
    print(evidence)


if __name__ == "__main__":
    main()
