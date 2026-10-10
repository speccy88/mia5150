# Exact model downloads and isolated runtimes

Model downloads and inference are optional. BM25 and synthetic tests need no model weights. The seven sources below are exactly pinned in `docs/model_sources.json`; the supplied downloader uses those revisions and its recorded file-selection policy. No moving `main` revision or substitute checkpoint is used.

| Local directory under models/ | Repository | Immutable revision | Runtime environment |
| --- | --- | --- | --- |
| bge-small | `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | `.venv` |
| e5-small | `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | `.venv` |
| embeddinggemma-int8 | `onnx-community/embeddinggemma-300m-ONNX` | `5090578d9565bb06545b4552f76e6bc2c93e4a66` | `.venv` |
| gliner | `fastino/gliner2.5-small-v1` | `7132dc4561c3f94563c6147e75ffa8ef34c4964a` | `.venv-gliner` |
| verdict | `heman10x/rlcd-modernbert-151m` | `8af2496eb63c7fa66d7d234e1f62629380030eb4` | `.venv` |
| laya | `convaiinnovations/laya-multilingual` | `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` | `.venv` |
| julia-1 | `SupersonicLabs/Julia-1` | `a85b127321d580d65176c89ced8273f305745d85` | `.venv-julia` |

## Privacy correction before any new model process

The newly provided `setup_privacy.sh` sets `ORT_DISABLE_TELEMETRY=1` before a new process can initialize ONNX Runtime. Source it before optional model commands:

```sh
. ./setup_privacy.sh
```

This is a **prospective correction, not the measured configuration**. The executed scorer/adapters and their recorded hashes remain unchanged. The setup was syntax-checked only; no ORT initialization, inference, denial retry or alternative execution was performed. It does not authorize resuming the stopped run. See [the privacy addendum](docs/PRIVACY_ADDENDUM.md) for official ONNX Runtime1.29/1.30 documentation and the final common32 evidence boundary.

## Install the optional model environments

Linux x86-64 and Python3.12 were used. These are observed direct dependency pins, not complete portable lockfiles. CUDA is not required. On other platforms, compatible wheels/instruction sets are not guaranteed; Windows support is not asserted.

```sh
python3.12 -m venv .venv
python3.12 -m venv .venv-gliner
python3.12 -m venv .venv-julia
.venv/bin/python -m pip install -r requirements-main.txt
.venv-gliner/bin/python -m pip install -r requirements-gliner.txt
.venv-julia/bin/python -m pip install -r requirements-julia.txt

# Exact executed downloader: fetches all seven pinned rows above, into models/.
.venv/bin/python src/download_models.py

# Install Julia's Python runtime from its pinned official model repository.
.venv-julia/bin/python -m pip install --no-deps ./models/julia-1
```

Julia uses Transformers5.0.0 and the official `supersonic-julia` local package. GLiNER uses `gliner2[local]==2.0.0` with Transformers4.57.6. The main environment uses Transformers5.18.0 for BGE/E5/Gemma/Verdict/Laya; E5 is Torch float32 and Gemma is the selected public INT8 ONNX conversion. Keep the environments separate.

The downloader writes model-file SHA256s to `docs/download_receipts.json`. The included receipt is provenance from the executed downloads, not weights. Check licensing in `docs/model_sources.json`, including Gemma's terms. The original gated Google Gemma repository is not substituted for the pinned ONNX-community checkpoint.

No new convenience downloader wrapper is included: `src/download_models.py` is the unmodified script used for this run. Full clean-room installation/download/inference was not rerun merely to validate this code archive.
