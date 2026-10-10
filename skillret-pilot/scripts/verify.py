"""Run pytest as a module, save exact output and preserve its return code."""

from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "evidence").mkdir(exist_ok=True)
log = ROOT / "evidence" / f"pytest-{time.time_ns()}.log"
with log.open("w") as f:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        stdout=f,
        stderr=subprocess.STDOUT,
    )
record = {
    "command": "python -m pytest -q",
    "exit_code": result.returncode,
    "log": str(log.relative_to(ROOT)),
}
with (ROOT / "evidence/commands.jsonl").open("a") as f:
    f.write(json.dumps(record) + "\n")
print(log.read_text())
print(json.dumps(record))
raise SystemExit(result.returncode)
