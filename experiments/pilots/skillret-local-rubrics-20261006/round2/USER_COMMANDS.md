# Round 2 user-run helpers

## Save keys once in WSL

```bash
python3 /home/user/code/mia5150-local/round2/secret_store.py
```

Enter each key at its hidden prompt, or press Enter to skip it. The setup performs no network calls and never overwrites an existing file.

Storage is `/home/user/.local/share/mia5150-secrets/`: directory mode 0700, owned by Fred; `jev.key`, `openrouter.key`, and `runpod.key` are separate plaintext UTF-8 files, mode 0600. They are outside the project, Windows mounts, result directories, and project ZIPs. Helpers reject unsafe permissions, symlinks, and hard-linked credential files. No environment variables or shell history entries are needed.

This is local access control, not encryption. Fred's WSL account, root, compromised code running as Fred, and whole-WSL backups can access the keys. Do not upload this directory. Existing keys are preserved; rotation requires a separately reviewed user action. The setup does not inspect arbitrary `.env` files.

The auth and benchmark helpers automatically use their own provider's saved file if configured. Otherwise they prompt for hidden input. They never copy keys into results, request payloads, a pod, or child process arguments. Paid probes still require local spending and data-transfer confirmation.

## OpenRouter one-shot probe

```bash
python3 /home/user/code/mia5150-local/round2/openrouter_probe.py
```

This sends the 12 frozen public SkillRet query texts and their 20 candidate names/descriptions to OpenRouter/GMICloud using Qwen3-235B-A22B-Instruct-2507 FP8. No labels, private files, weights, or tools are sent. One fixed route, no fallbacks, retries, or truncation. Output is a permutation of local aliases. This is exposed-development exploration, not a blind benchmark.

The helper verifies the public route catalog before requesting a key, with maximum rates $0.10/input-million and $0.40/output-million. Request bounds are 32,000 input tokens and 512 output tokens per query, with a planning usage ceiling of $0.0408576. A conservative **$0.50 all-in reservation** must fit the shared **$4** ledger. Existing balance only, with the user verifying fees and taxes. No credit purchase or auto-refill. Quote changes, uncertainty, errors, malformed rankings, or model/provider changes stop the batch. The reservation is not automatically refunded.

Current ledger carries $0.10 Jev plus $1 external, totaling $1.10 pending billing reconciliation. Enter only *additional* external spend not already recorded. If no further external spend exists, approving this probe would bring commitments to $1.60.

The run directory is fixed (`openrouter-listwise-12-v1`); its existence prevents automatic reruns. No live OpenRouter inference has been executed by the implementation agent.

## RunPod diagnostic (deferred until user setup and diagnosis)

```bash
python3 /home/user/code/mia5150-local/round2/runpod_auth_check.py
```

Uses the official v2 endpoint with `limit=1`. Successful pod response bodies are never read. For 403 only, diagnostics classify JSON versus HTML and allowlist fixed authorization words; arbitrary server text is not displayed. A 403 alone does not prove an invalid key. Do not use the legacy option to bypass the denial or broaden permissions. No pod is created, started, stopped, or changed.

## Reproduce Jev evaluation offline

```bash
cd /home/user/code/mia5150-local
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python round2/score_jev.py
```

Results: `round2/jev-typed-12-v1/evaluation.json` and `REPORT.md`.

## Validation

29 offline unit tests passed with dummy credentials only. All 12 OpenRouter requests prepared offline successfully. No real credential store was created or read during validation. Original local-results ZIP is unchanged.
