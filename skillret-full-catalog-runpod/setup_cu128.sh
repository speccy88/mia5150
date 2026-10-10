#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv --system-site-packages /tmp/mia5150-cu128-venv
mkdir -p piptmp
TMPDIR="$PWD/piptmp" /tmp/mia5150-cu128-venv/bin/python -m pip install --no-cache-dir \
  'vllm==0.10.2' 'datasets' 'sentence-transformers==5.1.1' 'transformers==4.55.4' 'rank-bm25' \
  'reportlab' 'matplotlib' 'pymupdf' 'nbformat' 'pytest' 'huggingface-hub' 'requests'
/tmp/mia5150-cu128-venv/bin/python -m pip install --no-cache-dir -e ./clm_source
/tmp/mia5150-cu128-venv/bin/python -c 'import torch,vllm; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.__version__,torch.version.cuda,vllm.__version__)'
/tmp/mia5150-cu128-venv/bin/python -m pip check
PYTHONPATH="$PWD/src:$PWD" /tmp/mia5150-cu128-venv/bin/python -m pytest -q tests > logs/pytest_cu128.txt 2>&1
PYTHONPATH="$PWD/src:$PWD" /tmp/mia5150-cu128-venv/bin/python -m pip freeze > evidence/requirements.freeze.txt
