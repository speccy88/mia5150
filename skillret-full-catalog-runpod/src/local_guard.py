"""Detached Mac-side RunPod cost guard; stops only this run's created pod.

The API key is loaded from the existing local .env into the CLI subprocess
environment. It never enters argv, stdout, logs, the pod, or the export ZIP.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dotenv import dotenv_values


def due(start_balance: float, current_balance: float, deadline: float, now: float,
        target_spend: float = 5.75) -> bool:
    return now >= deadline or current_balance <= start_balance - target_spend


def atomic_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def cli(command: list[str], key: str, binary: str):
    p=subprocess.run([binary,*command],capture_output=True,text=True,timeout=30,
                     env={**os.environ,"RUNPOD_API_KEY":key})
    if p.returncode:
        raise RuntimeError(f"runpodctl {command[0]} {command[1]} failed with exit {p.returncode}")
    return json.loads(p.stdout) if p.stdout.strip().startswith("{") else None


def run(control: Path):
    state=json.loads(control.read_text())
    assert state["created_by_this_run"] and state["pod_id"] and state["pod_name"]
    root=control.parents[3]
    key=dotenv_values(root/".env").get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    assert key, "local RUNPOD_API_KEY absent"
    binary=str(root/".tools/runpodctl-v2.14.0-darwin-arm64")
    log=control.parent/"guard_events.jsonl"
    def event(**kw):
        with log.open("a") as f:
            f.write(json.dumps({"utc":datetime.now(timezone.utc).isoformat(),**kw})+"\n")
    caffeinate=subprocess.Popen(["caffeinate","-i","-w",str(os.getpid())],
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) if sys.platform=="darwin" else None
    failed=0
    try:
        while True:
            state=json.loads(control.read_text())
            if state.get("cleanup_complete"):
                event(action="exit_after_cleanup")
                return
            try:
                pod=cli(["pod","get",state["pod_id"]],key,binary)
                if pod is None or pod.get("id")!=state["pod_id"] or pod.get("name")!=state["pod_name"]:
                    raise RuntimeError("pod identity mismatch")
                user=cli(["user"],key,binary)
                balance=float(user["clientBalance"])
                event(action="read",balance=balance,pod_status=pod.get("runtimeStatus"),
                      deadline=state["deadline_epoch"])
                failed=0
                if due(float(state["starting_balance"]),balance,float(state["deadline_epoch"]),time.time()):
                    cli(["pod","stop",state["pod_id"]],key,binary)
                    event(action="stop",reason="target_spend_or_deadline",balance=balance)
                    return
            except Exception as exc:
                failed+=1;event(action="read_error",error_type=type(exc).__name__,consecutive=failed)
                if failed>=3:
                    try:
                        cli(["pod","stop",state["pod_id"]],key,binary)
                        event(action="stop",reason="three_consecutive_control_read_errors")
                    except Exception as stop_exc:
                        event(action="stop_failed",error_type=type(stop_exc).__name__)
                    return
            time.sleep(60)
    finally:
        if caffeinate: caffeinate.terminate()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("control",type=Path);a=p.parse_args()
    run(a.control)
