"""Sequential five-model pilot. Resumes exact cases; never repairs/retries outputs."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pilot import config, append_row
from agent_runner import Budget, Blocked, sanitize
from trial import smoke, run_trial, activation_smoke

MODELS = [
    "qwen38-27b",
    "qwen38-flash",
    "deepseek41-flash",
    "mimo-v2-6-pro",
    "glm53-flash",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["canaries", "trial", "activation", "all"], default="all"
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    failed = False
    for label in args.models:
        cfg = config(str(ROOT / "manifests/configs" / f"{label}.json"))
        stages = (
            ["canaries", "trial", "activation"] if args.stage == "all" else [args.stage]
        )
        for stage in stages:
            if Budget(cfg).totals()["unresolved_reserved_usd"]:
                print(
                    "BLOCKED: unresolved reservation; reconcile before more paid calls.",
                    flush=True,
                )
                return 2
            print(
                json.dumps({"model": label, "stage": stage, "event": "start"}),
                flush=True,
            )
            started = time.monotonic()
            try:
                if stage == "canaries":
                    raw = smoke(True, cfg)
                    result = {
                        k: raw.get(k)
                        for k in [
                            "case_key",
                            "status",
                            "canary_correct",
                            "error_category",
                        ]
                    }
                    ok = bool(raw.get("canary_correct"))
                elif stage == "trial":
                    result = run_trial(cfg, quick=True, live=True)
                    ok = result["prerequisites_ok"]
                else:
                    result = activation_smoke(True, cfg)
                    ok = result["tasks"] == 5 and all(
                        s == "valid" for s in result["statuses"]
                    )
            except Exception as exc:
                result = {
                    "status": "blocked",
                    "error_category": sanitize(str(exc)),
                    "error_type": type(exc).__name__,
                }
                ok = False
            record = {
                "model": label,
                "stage": stage,
                "result": result,
                "exit_code": 0 if ok else 2,
                "wall_seconds": time.monotonic() - started,
                "time": time.time(),
            }
            append_row("evidence/openrouter_stages.jsonl", record)
            print(json.dumps(record), flush=True)
            failed |= not ok
            if stage == "canaries" and not ok:
                break
    print(json.dumps({"budget": Budget().totals()}), flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
