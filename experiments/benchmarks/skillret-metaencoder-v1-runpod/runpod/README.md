# RunPod reproduction

Run all commands below on the authorized RunPod GPU, not a local or controller computer. Provision an 80-GB GPU with sufficient persistent storage. This repository does not provision paid resources automatically.

Install the exact versions recorded in `../manifests/pip-freeze.txt` in a compatible CUDA/PyTorch image. Model wrapper source is obtained by the pinned snapshot download rather than vendored here. The audit documents the model API requirements.

```sh
export STUDY="$PWD"
export ROOT=/workspace/metaencoder
python "$STUDY/src/runner2.py" setup --root "$ROOT"
PYTHONPATH="$STUDY/src" python -m unittest discover -s "$STUDY/tests"
python "$STUDY/src/runner2.py" smoke --root "$ROOT" --batch 1
python "$STUDY/src/runner2.py" documents --root "$ROOT" --batch 1 --max-seconds 120
```

Inspect the measured pilot, current price and remaining approved budget before admitting the full run. The run's continuation implementation validates cached ID, input hash, instruction, limit, shape, finiteness and normalization, then embeds missing inputs. Choose an absolute UTC deadline with headroom for artifact export and transfer:

```sh
python "$STUDY/src/resume_fast.py" --root "$ROOT" \
  --deadline "$APPROVED_UTC_DEADLINE" --batch 4 --secondary-seconds 180
```

The deadline is a process deadline, not automatic pod termination or a guarantee of bounded billing. The controller must stop the pod before exceeding the approved cap. Secondary duration is a bounded attempt, not a claim of completion. Verify stage outputs before proceeding.

After both primary arrays are complete, measured diagnostics and figures may be regenerated on RunPod without new model inference. The export program requires the full primary array shapes and validates lossless array contents on RunPod:

```sh
python "$STUDY/src/export_artifacts.py" --root "$ROOT"
```

Transfer the lossless exports to the verified Drive destination. Keep raw arrays, model caches, checkpoints and authentication material out of Git. Preserve source code and the exact executed configuration with published results. Stop and terminate rented GPU resources only after safe artifact delivery; check billing/storage state separately.
