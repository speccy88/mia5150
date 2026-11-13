#!/usr/bin/env bash
# Run on RunPod only, inside the declared compatible CUDA/PyTorch image.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$HERE/../manifests/pip-freeze.txt" ]]; then
  echo 'Missing reconciled runtime lock; do not guess dependency versions.' >&2
  exit 1
fi
python -m pip install -r "$HERE/../manifests/pip-freeze.txt"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
