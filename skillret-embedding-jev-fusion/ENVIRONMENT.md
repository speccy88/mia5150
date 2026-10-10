# Reproduction environment

The validated run used the existing project environment at `/home/user/code/mia5150-local/.venv` under WSL2 Ubuntu-26.04 on a Windows host. It reused the local model, dataset and verified embedding caches; it did not download a replacement model or use metadata-only vectors.

Python is 3.12.3. The baseline's immutable `protocol.runtime` receipt records PyTorch 2.8.0+cu128, Transformers 4.57.1, NumPy 2.5.3, tokenizers 0.22.2, accelerate 1.15.0 and safetensors 0.8.0. `requirements.txt` also pins the metric reference and test packages. Model, dataset, official code and individual source-file hashes are embedded in `results/baseline_skillret_06b.json` and documented in `SOURCE_AUDIT.md`.

The recorded hardware is an NVIDIA GeForce RTX 2060 SUPER with 8,192 MiB VRAM and driver 610.88. The WSL kernel is `6.18.40.1-microsoft-standard-WSL2`; the guest has two logical CPUs, 6,217,752,576 bytes total memory and 2,147,483,648 bytes configured swap. See `results/hardware.json`. Configured swap is not permission to use active swapping.

The baseline uses one CPU thread, FP16 weights and batch size one, with a 32,768-token right-truncation limit. Runtime adaptations explicitly restore `rope_theta=1000000` and repeat key/value heads before memory-efficient CUDA attention; the quadratic math fallback is disabled. These changes and the preserved failed attempts are explained in `COMPATIBILITY_FINDINGS.md`.

The resource supervisor requires fresh Windows telemetry within eight seconds, at least 2 GiB available on both host and guest, no swap-counter increase, at most 7,168 MiB VRAM, at most 80 C GPU temperature, and no five consecutive host CPU samples above 80%. Run commands and the Windows monitor are in `README.md`. Reported query latency excludes the configured 50 ms inter-forward safety pause.

The local baseline and offline preparation require no API credentials. The hosted client remains disabled by default and accepts only `TYPESAFE_API_KEY` when explicitly enabled after its evidence gates. Do not put a key in source, configuration receipts, command arguments or exported artifacts. No `.env` file is needed for local reproduction, and the program has no `.env` loader.

To reconstruct the software environment, use a new project-local virtual environment and the pinned requirements; this is an installation recipe, not an action performed by the study:

```bash
python3.12 -m venv .venv-reproduction
.venv-reproduction/bin/python -m pip install -r requirements.txt
```

Do not install globally or replace the existing environment to inspect these results. Cached source files, the exact protocol hashes and compatibility settings must also match before reusing an embedding cache.
