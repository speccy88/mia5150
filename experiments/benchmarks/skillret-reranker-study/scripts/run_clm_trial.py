#!/usr/bin/env python3
"""Prepare locally; optionally run a temporary Runpod pod and retrieve evidence.

No API key is sent to the pod. All pod mutations are limited to the ID created
by this invocation. A detached local watchdog and finally block delete it.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clm_reranker import (append_jsonl, new_run, prepare_trial, read_jsonl, sha256,
                          summarize_clm_results, write_csv, write_new)

API = "https://api.runpod.io/v2"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"


def credentials():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    missing = []
    if not os.environ.get("RUNPOD_API_KEY"):
        missing.append("RUNPOD_API_KEY")
    if os.environ.get("ALLOW_PAID_RUNS") != "1":
        missing.append("ALLOW_PAID_RUNS=1")
    return missing


def api_client():
    import httpx
    if credentials():
        raise RuntimeError("Missing local prerequisites: " + ", ".join(credentials()))
    return httpx.Client(base_url=API, headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]},
                        timeout=30, trust_env=False)


def api(client, method, path, **kwargs):
    r = client.request(method, path, **kwargs)
    r.raise_for_status()
    return r.json() if r.content else None


def safe_pod(pod):
    return {k: pod.get(k) for k in ["id", "name", "status", "cost", "createdAt", "startedAt", "image", "gpu", "cudaVersion", "dataCenterId", "ports", "disk", "mounts"]}


def budget_plan(hourly, previous_reserved, max_seconds=5400):
    # Independent CLM budget; prior OpenRouter/Jev spending is not reset or reused.
    hard_usd, hard_cad, fx = 5.0, 7.5, 1.5
    remaining = min(3.0, hard_usd - previous_reserved, hard_cad / fx - previous_reserved)
    if not 0 < hourly or remaining <= .25:
        raise RuntimeError("persistent CLM experiment budget exhausted")
    # $0.10/hour storage reserve, 25% compute margin and 5-minute cleanup reserve.
    conservative_rate = hourly * 1.25 + .10
    seconds = min(max_seconds, int((remaining - .10) / conservative_rate * 3600) - 300)
    if seconds < 900:
        raise RuntimeError("insufficient remaining budget for bounded pod run")
    reserved = conservative_rate * (seconds + 300) / 3600 + .10
    return {"deadline_seconds": seconds, "reserved_usd": reserved, "reserved_cad": reserved * fx,
            "hard_usd": hard_usd, "hard_cad": hard_cad, "cad_per_usd": fx,
            "fx_status": "planning assumption, not verified exchange rate", "hourly_compute_usd": hourly,
            "storage_reserve_usd_per_hour": .10, "compute_reserve_multiplier": 1.25}


def terminate_created(client, state, out):
    if not state.get("created_by_this_run") or not state.get("pod_id"):
        raise ValueError("refusing cleanup without creation attribution")
    pod_id = state["pod_id"]
    events = []
    for attempt in range(6):
        try:
            r = client.get("/pods/" + pod_id)
            if r.status_code == 404 or (r.status_code == 200 and r.json().get("status") == "TERMINATED"):
                result = {"pod_id": pod_id, "terminated": True, "verified_epoch": time.time(), "get_status": r.status_code,
                          "events": events, "compute_charges_after_termination": "none; actual invoice not measured"}
                write_new(out, result)
                return result
            r.raise_for_status()
            if r.json().get("name") != state["pod_name"]:
                raise ValueError("pod ownership/name mismatch")
            d = client.delete("/pods/" + pod_id)
            events.append({"delete_status": d.status_code, "time": time.time()})
            if d.status_code != 404:
                d.raise_for_status()
        except Exception as exc:
            events.append({"error_type": type(exc).__name__, "time": time.time()})
        time.sleep(min(2 ** attempt, 20))
    write_new(out, {"pod_id": pod_id, "terminated": False, "events": events})
    raise RuntimeError("Pod termination could not be verified: " + pod_id)


def watchdog(run):
    """Independent local process survives runner failure; never receives a remote key."""
    write_new(run / "watchdog_ready.json", {"pid": os.getpid(), "started": time.time()})
    state_path = run / "pod_state.json"
    while not state_path.exists():
        intent_path = run / "provision_intent.json"
        if intent_path.exists():
            intent = json.loads(intent_path.read_text())
            if time.time() >= intent["deadline"]:
                # A lost create response must not orphan a paid pod. This name was
                # randomly time-versioned and recorded before our single POST.
                with api_client() as client:
                    pods = api(client, "GET", "/pods")["pods"]
                    found = [p for p in pods if p["name"] == intent["pod_name"]]
                    if len(found) > 1:
                        raise RuntimeError("ambiguous create recovery; inspect matching pods")
                    if found:
                        recovered = {**intent, "pod_id": found[0]["id"], "created_by_this_run": True}
                        terminate_created(client, recovered, run / "watchdog_cleanup.json")
                    else:
                        write_new(run / "watchdog_cleanup.json", {"terminated": True, "matching_pods": 0, "verified_epoch": time.time()})
                return 0
        elif (run / "provisioning_done.json").exists():
            return 0
        time.sleep(1)
    state = json.loads(state_path.read_text())
    while time.time() < state["deadline"]:
        cleanup = run / "cleanup.json"
        if cleanup.exists() and json.loads(cleanup.read_text()).get("terminated"):
            return 0
        time.sleep(min(5, max(0, state["deadline"] - time.time())))
    with api_client() as client:
        terminate_created(client, state, run / "watchdog_cleanup.json")
    return 0


def ssh_args(endpoint, key, known_hosts, scp=False):
    host = endpoint["host"]
    if not host or host.startswith("-") or not endpoint["username"].replace("-", "").isalnum():
        raise ValueError("invalid SSH endpoint")
    flags = ["-i", str(key), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
             "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=" + str(known_hosts),
             "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
             "-P" if scp else "-p", str(int(endpoint["port"]))]
    return flags, endpoint["username"] + "@" + host


def remote_bundle(run, deadline):
    versions = json.loads((ROOT / "evidence/clm_sources/versions.json").read_text())
    bundle_dir = run / "transfer"
    bundle_dir.mkdir(exist_ok=False)
    for src, name in [(ROOT / "clm_reranker.py", "clm_reranker.py"), (ROOT / "reference_metrics.py", "reference_metrics.py"),
                      (ROOT / "scripts/clm_remote_worker.py", "clm_remote_worker.py"), (run / "inputs.json", "inputs.json"),
                      (ROOT / "evidence/clm_sources/versions.json", "versions.json")]:
        shutil.copy2(src, bundle_dir / name)
    # The setup includes only trusted source/pins, never dataset text or credentials.
    setup = "\n".join([
        "#!/bin/bash", "set -euo pipefail", "export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1",
        "export HF_HOME=\"$PWD/model-cache\"", "python3 -m venv .venv",
        ".venv/bin/python -m pip install --no-cache-dir " + shlex.quote("vllm==" + versions["vllm_version"]) + " httpx huggingface-hub",
        "git clone --filter=blob:none --no-checkout https://github.com/Contrastive-LM/CLM.git official-clm",
        "git -C official-clm checkout --detach " + shlex.quote(versions["clm_commit"]),
        ".venv/bin/python -m pip install --no-cache-dir -e ./official-clm",
        "exec .venv/bin/python -u clm_remote_worker.py --deadline " + str(deadline), ""])
    (bundle_dir / "setup.sh").write_text(setup)
    bundle = run / "transfer.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        for p in sorted(bundle_dir.iterdir()):
            tf.add(p, arcname=p.name)
    return bundle


def run_pod(run):
    if credentials():
        write_new(run / "blocker.json", {"missing": credentials(), "pod_created": False, "status": "BLOCKED_CREDENTIALS"})
        return 2
    if os.environ.get("MIA5150_RUNPOD_POD_ID"):
        # Explicit existing IDs are inspected, never silently replaced or mutated.
        with api_client() as client:
            existing = api(client, "GET", "/pods/" + os.environ["MIA5150_RUNPOD_POD_ID"])
            write_new(run / "existing_pod.json", safe_pod(existing))
        write_new(run / "blocker.json", {"status": "EXISTING_POD_REQUIRES_SEPARATE_ISOLATION", "pod_created": False,
                   "reason": "This temporary-pod runner protects existing work. Unset the ID to create a dedicated pod."})
        return 2
    ledger = ROOT / "results/clm_budget.jsonl"
    lock = (ROOT / "results/clm_budget.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state, endpoint, process = None, None, None
    awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]) if platform.system() == "Darwin" else None
    started = time.time()
    key = ROOT / ".tools/clm/keys" / run.name
    key.parent.mkdir(parents=True, exist_ok=True)
    key.parent.chmod(0o700)
    known = key.with_suffix(".known_hosts")
    remote_path = "/workspace/" + run.name
    sync_errors = []
    with api_client() as client:
        try:
            catalog = api(client, "GET", "/catalog/gpus", params={"include": "AVAILABILITY", "product": "POD", "cloud": "SECURE", "count": 1, "minCudaVersion": "13.0"})
            options = [g for g in catalog["gpus"] if g.get("manufacturer") == "NVIDIA" and 40 <= g["memory"] <= 80
                       and g.get("availability") in ["LOW", "MEDIUM", "HIGH"] and g.get("price", {}).get("secure", 0) > 0]
            if not options:
                raise RuntimeError("No suitable BF16 GPU available; no paid pod created")
            chosen = min(options, key=lambda g: g["price"]["secure"])
            prior = read_jsonl(ledger) if ledger.exists() else []
            # Reservations remain charged conservatively across reruns, even if actual spend was lower.
            plan = budget_plan(chosen["price"]["secure"], sum(x.get("reserved_usd", 0) for x in prior))
            template = api(client, "GET", "/templates/runpod-torch-v280")
            if template["image"] != IMAGE:
                raise RuntimeError("official template image drift; inspect before provisioning")
            write_new(run / "runpod-selection.json", {"gpu": chosen, "budget": plan, "catalog": catalog,
                       "image": IMAGE, "created_by_this_run": True, "reason": "lowest-price available secure NVIDIA 40-80GB; BF16 headroom"})
            print(json.dumps({"will_create": chosen["id"], "USD_per_hour": chosen["price"]["secure"], "max_seconds": plan["deadline_seconds"]}), flush=True)
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-f", str(key), "-N", "", "-C", "mia5150-clm-ephemeral"], check=True, timeout=15)
            pub = key.with_suffix(".pub").read_text().strip()
            watchdog_log = (run / "watchdog.log").open("x")
            watcher = subprocess.Popen([sys.executable, str(Path(__file__)), "--watchdog", str(run)],
                                       stdout=watchdog_log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(100):
                if (run / "watchdog_ready.json").exists():
                    break
                if watcher.poll() is not None:
                    raise RuntimeError("watchdog failed before provisioning")
                time.sleep(.1)
            else:
                raise RuntimeError("watchdog startup timeout")
            deadline = time.time() + plan["deadline_seconds"]
            name = "mia5150-clm-" + run.name.removeprefix("clm_runpod_")
            body = {"name": name, "image": IMAGE, "gpu": {"id": chosen["id"], "count": 1, "minCudaVersion": "13.0"},
                    "cloud": "SECURE", "disk": 80, "ports": ["22/tcp"], "startSsh": True, "startJupyter": False,
                    "env": {"PUBLIC_KEY": pub}}
            write_new(run / "create_request.json", {k: v for k, v in body.items() if k != "env"})
            write_new(run / "provision_intent.json", {"pod_name": name, "deadline": deadline,
                       "created_by_this_run": True, "started_epoch": started})
            append_jsonl(ledger, {"run": run.name, "time": time.time(), **plan})
            try:
                pod = api(client, "POST", "/pods", json=body)
            except Exception:
                # Do not retry ambiguous creation. Recover only this unique request's name.
                candidates = api(client, "GET", "/pods")["pods"]
                matches = [p for p in candidates if p["name"] == name]
                if len(matches) == 1:
                    pod = matches[0]
                else:
                    raise
            state = {"pod_id": pod["id"], "pod_name": name, "created_by_this_run": True,
                     "deadline": deadline, "started_epoch": started, "parent_pid": os.getpid()}
            write_new(run / "pod_state.json", state)
            write_new(run / "pod_created.json", safe_pod(pod))
            if pod.get("cost", 0) > plan["hourly_compute_usd"] * 1.25:
                raise RuntimeError("actual pod rate exceeds reserved price margin")
            ready_limit = min(deadline - 300, time.time() + 480)
            while time.time() < ready_limit:
                pod = api(client, "GET", "/pods/" + pod["id"])
                endpoint = (pod.get("ssh") or {}).get("direct")
                if endpoint:
                    flags, target = ssh_args(endpoint, key, known)
                    probe = subprocess.run(["ssh", *flags, target, "true"], capture_output=True, timeout=20)
                    if probe.returncode == 0:
                        break
                time.sleep(5)
            else:
                raise TimeoutError("SSH did not become ready within 8 minutes")
            flags, target = ssh_args(endpoint, key, known)
            scpflags, _ = ssh_args(endpoint, key, known, scp=True)
            bundle = remote_bundle(run, deadline - 180)
            subprocess.run(["ssh", *flags, target, "mkdir -p " + shlex.quote(remote_path)], check=True, timeout=30)
            subprocess.run(["scp", *scpflags, str(bundle), target + ":" + remote_path + "/transfer.tar.gz"], check=True, timeout=60)
            command = "cd " + shlex.quote(remote_path) + " && tar xzf transfer.tar.gz && bash setup.sh"
            log = (run / "remote-install-and-run.log").open("x")
            process = subprocess.Popen(["ssh", *flags, target, command], stdout=log, stderr=subprocess.STDOUT)
            def sync():
                r = subprocess.run(["scp", *scpflags, "-r", target + ":" + remote_path + "/out", str(run / "remote")],
                                   capture_output=True, text=True, timeout=45)
                if r.returncode:
                    sync_errors.append({"time": time.time(), "stderr": r.stderr})
            (run / "remote").mkdir()
            while process.poll() is None:
                if time.time() > deadline - 120:
                    raise TimeoutError("stopping for retrieval and pod cleanup deadline")
                time.sleep(10)
                sync()
            sync()
            return process.returncode
        except Exception:
            (run / "orchestration_failure.txt").write_text(traceback.format_exc())
            return 2
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            # Recover an ambiguous creation before cleanup; never retry POST.
            if state is None and (run / "provision_intent.json").exists():
                intent = json.loads((run / "provision_intent.json").read_text())
                try:
                    matches = [p for p in api(client, "GET", "/pods")["pods"] if p["name"] == intent["pod_name"]]
                    if len(matches) == 1:
                        pod = matches[0]
                        state = {**intent, "pod_id": pod["id"]}
                        write_new(run / "pod_state.json", state)
                except Exception:
                    # Detached watchdog retains the intent and tries at deadline.
                    pass
            # A final best-effort retrieval always precedes deletion.
            if endpoint and state:
                try:
                    scpflags, target = ssh_args(endpoint, key, known, scp=True)
                    (run / "remote").mkdir(exist_ok=True)
                    result = subprocess.run(["scp", *scpflags, "-r", target + ":" + remote_path + "/out", str(run / "remote")],
                                            capture_output=True, text=True, timeout=45)
                    if result.returncode:
                        sync_errors.append({"time": time.time(), "stderr": result.stderr})
                except Exception as exc:
                    sync_errors.append({"error_type": type(exc).__name__})
            write_new(run / "retrieval_errors.json", sync_errors)
            try:
                if state:
                    cleanup = terminate_created(client, state, run / "cleanup.json")
                    write_new(run / "cost_estimate.json", {"elapsed_seconds": cleanup["verified_epoch"] - started,
                               "approximate_compute_usd": (cleanup["verified_epoch"] - started) / 3600 * pod.get("cost", plan["hourly_compute_usd"]),
                               "basis": "elapsed wall clock times reported hourly rate; invoice/storage not measured"})
            finally:
                write_new(run / "provisioning_done.json", {"finished_epoch": time.time()})
                if awake:
                    awake.terminate()
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()


def report(run):
    baseline = json.loads((run / "bm25_baseline.json").read_text())
    rawpath = run / "remote/out/bm25_clm_raw.jsonl"
    raw = read_jsonl(rawpath) if rawpath.exists() else []
    summary, evaluated = summarize_clm_results(json.loads((run / "evaluation_only.json").read_text()), raw)
    if not (run / "bm25_clm_summary.json").exists():
        write_new(run / "bm25_clm_summary.json", summary)
        write_csv(run / "bm25_clm_summary.csv", [summary])
        write_new(run / "paired_evaluation.json", evaluated)
    return {"baseline": baseline, "clm": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "runpod", "mac"], default="auto")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--watchdog", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.watchdog:
        return watchdog(args.watchdog)
    run = args.run_dir.resolve() if args.run_dir else new_run(ROOT)
    if not (run / "run_manifest.json").exists():
        prepare_trial(ROOT, run)
    print("Results: " + str(run), flush=True)
    if args.prepare_only:
        print((run / "bm25_baseline.json").read_text())
        return 0
    if args.backend == "mac":
        write_new(run / "blocker.json", {"status": "UNSUPPORTED_MAC_REFERENCE", "reason": "official vLLM NVIDIA BF16 reference cannot run on this Apple Silicon Mac"})
        print(json.dumps(report(run), indent=2))
        return 2
    # Do not retry or overwrite a previously attempted pod run.
    if any((run / n).exists() for n in ["create_request.json", "blocker.json", "pod_state.json"]):
        raise SystemExit("Run already attempted or blocked; omit --run-dir for a fresh versioned attempt")
    def interrupted(signum, frame):
        raise KeyboardInterrupt("termination signal: " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    code = run_pod(run)
    print(json.dumps(report(run), indent=2))
    from package_clm_trial import deliver
    print(json.dumps(deliver(run), indent=2))
    if code:
        print("Run incomplete; inspect saved blocker/failure evidence. No metrics were invented.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
