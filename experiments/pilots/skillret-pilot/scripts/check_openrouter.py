"""One native Deep Agents credential canary; persistent paid guard and resume."""

from pathlib import Path
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import ROOT, config, digest, rows, append_row, save_json
from agent_runner import (
    require_permission,
    preflight,
    GuardedModel,
    run_agent,
    fixtures,
    Blocked,
    sanitize,
    Budget,
)

require_permission()
cfg = config()
pin = preflight(
    cfg,
    model="qwen/qwen3.8-27b",
    manifest_path="manifests/openrouter/qwen-qwen3.8-27b.json",
)
key = digest(
    {
        "phase": "credential_canary",
        "pin": {k: v for k, v in pin.items() if k not in {"snapshot", "pricing"}},
        "code": (ROOT / "agent_runner.py").read_text(),
        "config": cfg,
    }
)
previous = [
    r
    for r in rows("results/openrouter/credential_canary.jsonl")
    if r["case_key"] == key
]
if previous:
    result = previous[-1]
else:
    model = GuardedModel(pin=pin, cfg=cfg, case_id=key)
    try:
        result = run_agent("Add two numbers.", fixtures()[:2], model)
    except Blocked as exc:
        result = {
            "status": "blocked",
            "selected": [],
            "calls": sanitize(model.calls),
            "error": str(exc),
        }
    result.update(
        case_key=key,
        model=pin["model"],
        provider=pin["provider_name"],
        purpose="Real OpenRouter credential/integration canary; not a SkillRet benchmark",
    )
    result["canary_correct"] = result["status"] == "valid" and result["selected"] == [
        fixtures()[0]["id"]
    ]
    append_row("results/openrouter/credential_canary.jsonl", result)
total = Budget(cfg).totals()
summary = {
    "model": pin["model"],
    "provider": pin["provider_name"],
    "status": result["status"],
    "canary_correct": result["canary_correct"],
    "reused": bool(previous),
    "usage": [r.get("usage") for r in result["calls"]],
    "completion_ids": [r.get("completion_id") for r in result["calls"]],
    "budget": total,
    "estimated_cad_actual": total["actual_usd"]
    * float(os.getenv("CAD_PER_USD", "1.5")),
}
save_json("reports/OPENROUTER_KEY_CHECK.json", summary)
print(json.dumps(summary, indent=2))
raise SystemExit(0 if result["canary_correct"] else 2)
