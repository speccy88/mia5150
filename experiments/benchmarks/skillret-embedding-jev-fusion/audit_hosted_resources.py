"""Read only named resource-guard telemetry/receipts; never API, keys or worker logs.

Repeat --run-dir for each attempt. JSON goes to stdout, or an exclusively created
--output file. An absent exit receipt means partial/unknown, never completed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys


GIB = 1024 ** 3
THRESHOLDS = {
    "minimum_host_and_guest_available_gib": 2,
    "maximum_gpu_used_mib": 7168,
    "maximum_gpu_temperature_c": 80,
    "host_cpu_threshold_pct": 80,
    "host_cpu_guard_consecutive_samples": 5,
    "host_telemetry_age_seconds": "0 <= age < 8 at sampling time",
    "swap_guard": "Any counter change relative to the unlogged prelaunch sample",
}


def strict_json(raw):
    def reject(value):
        raise ValueError("Non-finite JSON constant: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def snapshot(path):
    """Capture a fixed byte prefix so a growing telemetry file cannot extend a read."""
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            raw = stream.read(size)
            after_size = os.fstat(stream.fileno()).st_size
    except FileNotFoundError:
        return None, {"path": str(path), "present": False}
    return raw, {"path": str(path), "present": True,
                 "captured_bytes": len(raw), "initial_bytes": size,
                 "bytes_after_read": after_size,
                 "changed_size_during_read": after_size != size,
                 "sha256_captured_bytes": hashlib.sha256(raw).hexdigest()}


def numeric(value, label, minimum=0, maximum=None, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)
            or (integer and not isinstance(value, int))):
        raise ValueError("Invalid " + label)
    return value


def validate_row(row):
    numeric(row["utc"], "sample timestamp")
    host = row["host"]
    numeric(host["utc"], "host timestamp")
    numeric(host["available_bytes"], "host available bytes", integer=True)
    numeric(host["cpu_pct"], "host CPU", maximum=100)
    numeric(row["guest_available_bytes"], "guest available bytes", integer=True)
    numeric(row["gpu_used_mib"], "GPU used MiB", integer=True)
    numeric(row["gpu_temp_c"], "GPU temperature", integer=True)
    for name in ("pswpin", "pswpout"):
        numeric(row["swap"][name], name, integer=True)
    return row


def summarize(rows):
    if not rows:
        return {"samples": 0, "threshold_violations": [], "metrics": None}
    longest = current = over = 0
    for row in rows:
        current = current + 1 if row["host"]["cpu_pct"] > 80 else 0
        longest = max(longest, current)
        over += row["host"]["cpu_pct"] > 80
    ages = [row["utc"] - row["host"]["utc"] for row in rows]
    timestamps = [row["utc"] for row in rows]
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    swaps = {}
    for key in ("pswpin", "pswpout"):
        values = [row["swap"][key] for row in rows]
        changes = [b - a for a, b in zip(values, values[1:])]
        swaps[key] = {"first": values[0], "last": values[-1],
                      "minimum": min(values), "maximum": max(values),
                      "net_change": values[-1] - values[0],
                      "positive_increments": sum(max(0, change) for change in changes),
                      "counter_decreases": sum(change < 0 for change in changes),
                      "changed_intervals": sum(change != 0 for change in changes)}
    metrics = {
        "first_sample_utc": timestamps[0], "last_sample_utc": timestamps[-1],
        "observed_span_seconds": timestamps[-1] - timestamps[0],
        "maximum_sample_gap_seconds": max(gaps) if gaps else None,
        "non_increasing_sample_intervals": sum(gap <= 0 for gap in gaps),
        "minimum_host_available_gib": min(r["host"]["available_bytes"] for r in rows) / GIB,
        "minimum_guest_available_gib": min(r["guest_available_bytes"] for r in rows) / GIB,
        "maximum_gpu_used_mib": max(r["gpu_used_mib"] for r in rows),
        "maximum_gpu_temperature_c": max(r["gpu_temp_c"] for r in rows),
        "maximum_host_cpu_pct": max(r["host"]["cpu_pct"] for r in rows),
        "host_cpu_samples_above_80_pct": over,
        "maximum_consecutive_host_cpu_samples_above_80_pct": longest,
        "host_telemetry_age_seconds_minimum": min(ages),
        "host_telemetry_age_seconds_maximum": max(ages),
        "host_telemetry_age_invalid_samples": sum(not 0 <= age < 8 for age in ages),
        "swap_counters": swaps,
    }
    violations = []
    if metrics["minimum_host_available_gib"] < 2: violations.append("Host available memory below 2 GiB")
    if metrics["minimum_guest_available_gib"] < 2: violations.append("Guest available memory below 2 GiB")
    if metrics["maximum_gpu_used_mib"] > 7168: violations.append("GPU memory above 7168 MiB")
    if metrics["maximum_gpu_temperature_c"] > 80: violations.append("GPU temperature above 80 C")
    if longest >= 5: violations.append("Host CPU above 80% for at least five consecutive samples")
    if metrics["host_telemetry_age_invalid_samples"]: violations.append("Host telemetry age outside [0, 8) seconds")
    if any(s["changed_intervals"] for s in swaps.values()): violations.append("Observed swap counters changed")
    return {"samples": len(rows), "threshold_violations": violations, "metrics": metrics}


def audit_run(path):
    artifacts, errors, receipts = {}, [], {}
    # Read exit first: an attempt that finishes during capture remains conservatively partial.
    for name in ("exit.json", "guard-stop.json"):
        try:
            raw, artifacts[name] = snapshot(path / name)
            if raw is not None:
                value = strict_json(raw)
                if not isinstance(value, dict): raise ValueError("Receipt is not an object")
                numeric(value["utc"], name + " timestamp")
                if name == "exit.json":
                    numeric(value["returncode"], "supervisor returncode", minimum=-100000, integer=True)
                    receipts[name] = {"utc": value["utc"], "returncode": value["returncode"]}
                else:
                    if not isinstance(value.get("error"), str): raise ValueError("Guard error is not text")
                    receipts[name] = {"utc": value["utc"], "error": value["error"]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"file": name, "error": str(exc)})
    rows, segments, segment = [], [], []
    unterminated = False
    try:
        raw, artifacts["telemetry.jsonl"] = snapshot(path / "telemetry.jsonl")
        if raw is not None:
            unterminated = bool(raw) and not raw.endswith(b"\n")
            for number, line in enumerate(raw.splitlines(), 1):
                try:
                    row = validate_row(strict_json(line))
                    rows.append(row); segment.append(row)
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append({"file": "telemetry.jsonl", "line": number, "error": str(exc)})
                    segments.append(segment); segment = []
            segments.append(segment)
    except OSError as exc:
        errors.append({"file": "telemetry.jsonl", "error": str(exc)})
    logged = summarize(rows)
    if rows:
        # Invalid rows break consecutive-sample evidence, rather than join two streaks.
        logged["metrics"]["maximum_consecutive_host_cpu_samples_above_80_pct"] = max(
            (summarize(s)["metrics"]["maximum_consecutive_host_cpu_samples_above_80_pct"] for s in segments if s), default=0)
        cpu_violation = "Host CPU above 80% for at least five consecutive samples"
        if logged["metrics"]["maximum_consecutive_host_cpu_samples_above_80_pct"] < 5:
            logged["threshold_violations"] = [v for v in logged["threshold_violations"] if v != cpu_violation]
    guard = receipts.get("guard-stop.json")
    recovered = None
    if guard:
        for prefix in ("less than 2 GiB host/guest reserve: ", "GPU memory/temperature guard: "):
            if guard["error"].startswith(prefix):
                try:
                    recovered = validate_row(strict_json(guard["error"][len(prefix):]))
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append({"file": "guard-stop.json", "error": "Embedded failed sample: " + str(exc)})
                break
    exit_receipt = receipts.get("exit.json")
    has_exit = exit_receipt is not None
    changed = any(a.get("changed_size_during_read") for a in artifacts.values())
    complete_evidence = has_exit and bool(rows or recovered) and not errors and not unterminated and not changed
    observed = summarize(rows + ([recovered] if recovered else []))
    if not has_exit:
        status = "partial_or_unknown_no_exit_receipt"
    elif guard:
        status = "supervisor_finished_after_guard_stop"
    elif exit_receipt["returncode"] != 0:
        status = "supervisor_finished_with_failed_worker_exit"
    else:
        status = "supervisor_finished_with_zero_worker_exit"
    result = {
        "run_dir": str(path), "run_name": path.name, "status": status,
        "exit_receipt": exit_receipt, "guard_stop_receipt": guard,
        "worker_returncode": exit_receipt["returncode"] if has_exit and not guard else None,
        "study_completion_verified": False,
        "resource_evidence_status": ("observed_violation_or_guard_stop" if guard or observed["threshold_violations"]
                                     else "incomplete_evidence" if not complete_evidence
                                     else "no_observed_threshold_violation"),
        "logged_telemetry": logged,
        "embedded_failed_sample": recovered,
        "extrema_including_recovered_failed_sample": observed if recovered else None,
        "unterminated_last_telemetry_line": unterminated,
        "data_quality_errors": errors, "artifacts": artifacts,
        "interpretation": (
            "Exit code belongs to the supervisor; without a guard-stop receipt it relays the worker code. "
            "With a guard stop, the worker's own exit code is unavailable. No study-completion claim is made. "
            "Extrema cover captured samples only. Prelaunch swap counters were not saved; observed deltas "
            "start at the first logged sample. Host age is measured at sample time, not at audit time. "
            "Five consecutive CPU samples are not necessarily five seconds. Missing exit means partial/unknown."
        ),
    }
    return result


def build_audit(paths):
    runs = [audit_run(Path(path)) for path in paths]
    return {
        "schema_version": "hosted-resource-audit-v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "network_calls": 0, "credentials_accessed": False,
        "read_scope": ["telemetry.jsonl", "exit.json", "guard-stop.json"],
        "thresholds_from_resource_guard_py": THRESHOLDS,
        "runs": runs,
        "summary": {
            "attempts": len(runs),
            "partial_or_unknown_attempts": [r["run_name"] for r in runs if r["exit_receipt"] is None],
            "failed_supervisor_exits_preserved": [
                {"run_name": r["run_name"], "returncode": r["exit_receipt"]["returncode"]}
                for r in runs if r["exit_receipt"] and r["exit_receipt"]["returncode"] != 0],
            "guard_stop_attempts": [r["run_name"] for r in runs if r["guard_stop_receipt"]],
            "all_attempts_have_exit_receipts": all(r["exit_receipt"] is not None for r in runs),
            "study_completion_verified": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, help="Create a new JSON file; never overwrite evidence")
    args = parser.parse_args()
    text = json.dumps(build_audit(args.run_dir), indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
