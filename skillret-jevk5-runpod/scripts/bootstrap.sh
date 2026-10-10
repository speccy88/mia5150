#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v git >/dev/null
command -v nvidia-smi >/dev/null
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'Start from a CUDA-enabled PyTorch Pod image.'
assert torch.cuda.is_bf16_supported() and torch.cuda.get_device_properties(0).major >= 8, 'This reference needs native BF16 support.'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
if [[ -e .venv ]]; then echo '.venv exists. Reuse it explicitly or create a new experiment directory; not overwriting.' >&2; exit 1; fi
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps 'jevk5 @ git+https://github.com/allebee/jevk5.git@v0.3.0'
mkdir -p workspace
python -m pip freeze > workspace/pip-freeze.txt
python - <<'PY'
import torch, transformers, peft, importlib.metadata as m, json
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert hasattr(transformers, 'Qwen3_5ForCausalLM')
from jevk5.prompt import messages, decision_options
print('Torch:', torch.__version__, 'Transformers:',transformers.__version__,'PEFT:',peft.__version__)
d=m.distribution('jevk5')
print('Pinned runtime install provenance:', d.read_text('direct_url.json'))
with open('workspace/runtime-install.json','x') as f:
    json.dump({'jevk5_direct_url':json.loads(d.read_text('direct_url.json'))},f,indent=2)
PY
python -m pytest -q tests --junitxml=workspace/bootstrap-tests.xml
printf '\nActivate with: source .venv/bin/activate\nDo not start training before confirming budget and shutdown supervision.\n'
