"""Execute both notebooks in this interpreter's environment; preserve failures."""

import json
import os
from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
# Automated notebook validation is always offline, even when a developer's
# terminal has live opt-ins. Interactive notebooks retain explicit opt-in cells.
kernel_env = {
    **os.environ,
    "ALLOW_PAID_RUNS": "0",
    "ALLOW_PAID_NOTEBOOK_RUNS": "0",
    "ALLOW_LOCAL_NOTEBOOK_RUNS": "0",
}
(ROOT / "evidence").mkdir(exist_ok=True)
results = []
for name in ["01_data_and_retrieval.ipynb", "02_deepagents_trial.ipynb"]:
    start = time.monotonic()
    book = nbformat.read(ROOT / name, as_version=4)
    try:
        NotebookClient(
            book,
            timeout=600,
            resources={"metadata": {"path": str(ROOT)}},
            kernel_name="python3",
        ).execute(env=kernel_env)
        row = {
            "notebook": name,
            "status": "passed",
            "code_cells": sum(c.cell_type == "code" for c in book.cells),
        }
    except Exception as exc:
        row = {
            "notebook": name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    nbformat.write(book, ROOT / name)
    row["wall_seconds"] = time.monotonic() - start
    results.append(row)
(ROOT / "evidence/notebook-execution.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
raise SystemExit(0 if all(r["status"] == "passed" for r in results) else 1)
