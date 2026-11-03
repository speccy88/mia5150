# Local skill discovery, native final-answer adapter (3 November 2026)

Start with [LOCAL_REPORT.md](LOCAL_REPORT.md), the [cross-route comparison](COMPARISON.md)
and [publication transformations](PUBLICATION_TRANSFORMS.json).

Final closure adds [200 separate Qwen8/Mistral profile measurements](profiles-v3/PROFILE_REPORT.md).
There are now 16 distinct local models measured across the two series (1,320
responses), while Llama 3.2 and Gemma 3 remain unexecuted at the user's request.
The original 32K/f16 blocks remain historical evidence; they are not overwritten.

The canonical `results/local-native-v2/` contains 18 x 140 rows: 14 models were
executed (80 measured responses each), four were blocked before inference, and
every model has 60 estimated-context exclusions. Of 1,120 measured outputs, 542
have no error. Nanbeige's 80 reasoning-only outputs are measured failures; the four
blocked models have no measured accuracy. Never treat all 2,520 rows as inferences.

The v1 archive and VRAM-oversubscription diagnostic remain separate. `native-v2`
names the final-answer parser revision; it does not implement the proposed unbiased
scientific protocol v2. The golds remain at positions 1-20, at every library size.

## Offline verification

Use Python 3.12. Install the test dependencies in an environment outside the public
tree, or in the ignored `.venv` directory. No credentials, model weights or network
are required to recompute CSV/SQLite metrics:

```powershell
python -m pip install -r requirements-test.txt -r requirements-analysis.txt
python -m unittest discover -s tests -v
python -c "from review_local_results import review; review('local-native-v2')"
python validate_public.py
```

The distributed SQLite files are rebuilt audit exports with row JSON preserved,
cleaned audit metadata, no old free pages or WAL/SHM, and an explicitly unusable
resume fingerprint. They are not resumable live inference state. The private raw
response captures are not distributed; their source hashes and historical audit
summaries remain, so full normalization replay needs the retained private captures.

## New measurements

Source and acquisition helpers are included. Rebuild artifacts from pinned official
sources in `evidence/model-artifact-provenance.json`, checking provider terms in
`evidence/model-source-licenses.json`. Weights and third-party model cards are not
redistributed. Gated models remain gated; no acceptance or substitute is inferred.
`local-models.json` uses study-relative model paths; adjust local paths if needed
without changing the pinned hashes or quantization. The two Qwen3.5 artifacts are
identified as Ollama exports, rather than official publisher GGUFs.

Create a new working directory for inference and use the native adapter, 32K
configured window, 500 output tokens and original server settings. Start the server
with `Start-LocalServer.ps1` and a freshly generated proof; historical GPU summaries
are not admission proof. The full-layer, KV and physical-VRAM checks reject CPU
fallback. No hardware inference has been repeated for publication preparation.
The measured profile places transformer/output layers and KV on CUDA, while
llama.cpp b11960 retains CPU_Mapped input/embedding weights and CPU preparation.
It must not be described as computation with zero CPU participation. See
[the exact GPU scope](evidence/NATIVE_GPU_SCOPE.md) and buffer audit.
Read `README_LOCAL_ARCHIVED.md` and the source before an intentional new run;
historical launch helpers default to the already-existing results and refuse replay.

## Data and attribution

`data/test.jsonl` is a prompt-equivalent metadata derivative, not the 117.6 MB full
corpus. It preserves IDs, names, source order and the first 300 description characters.
`data/test_queries.json` is unchanged. The public runner accepts the derivative hash;
the exact core measured sources under `source-measured/` retain the original hash.
`data/ATTRIBUTIONS.jsonl` gives every source ID's author handle, repository, received
MIT/Apache label and original links; [attribution notes](data/ATTRIBUTION.md) explain
the scope. No attribution field is inserted into a prompt.

Current publication file hashes are in `FILE_MANIFEST.json`. Original archive hashes,
per-file original/public hashes, exclusions and metadata transformations are retained
in `PUBLICATION_TRANSFORMS.json`. The original delivery ZIPs remain unchanged privately.
