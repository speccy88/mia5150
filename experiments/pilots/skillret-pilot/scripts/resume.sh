#!/bin/sh
# Run from the extracted/project root after private setup_local_key.py.
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
"$PY" pilot.py doctor
"$PY" pilot.py download
"$PY" pilot.py inspect
"$PY" pilot.py baseline --queries 200
"$PY" pilot.py vectors
"$PY" pilot.py smoke --live
"$PY" pilot.py run --quick --resume --live
"$PY" pilot.py run --config pilot.defaults.json --resume --live
"$PY" pilot.py activation-smoke --live
"$PY" scripts/verify.py
"$PY" scripts/execute_notebooks.py
"$PY" pilot.py report
"$PY" pilot.py package
