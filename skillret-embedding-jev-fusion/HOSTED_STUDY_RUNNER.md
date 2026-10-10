# Hosted study runner

`run_hosted_study.py` is disabled by default. Status mode reads the frozen plan and any existing hash-chained spend ledger. It does not read credential variables, create files, load receipt configuration, or contact an API.

```bash
python run_hosted_study.py --plan results/jev-plan-smoke-context-v1/preparation_manifest.json --run-dir results/hosted-execution
```

The parent must supply genuine, hashed evidence receipts required by `HostedJevClient`: validated baseline, human approval, pricing, FX and fees, available existing credit, provider contract, model pin, and (for pilot/full) measured expansion projection. This runner does not create or infer these claims. A gate configuration is a JSON object with the `GateBundle` field names. Every value is an object containing `path` and `sha256`; optional fields may be null. Relative receipt paths resolve relative to the configuration file. The configuration must not contain credentials.

```bash
python run_hosted_study.py --plan results/jev-plan-smoke-context-v1/preparation_manifest.json --run-dir results/hosted-execution --gates results/receipts/gate-config.json --execute
```

Only `HostedJevClient` accesses `TYPESAFE_API_KEY`, after its evidence gates. The runner never loads a saved key, changes the process environment, purchases credit, looks up models, resets a ledger, or automatically advances to a larger scope. All scopes must use the same `--run-dir`, retaining model-list evidence, `budget_ledger.jsonl`, raw responses, and semantic caches. Model aliases can reuse the existing client's verified semantic cache; accounting always comes from unique ledger attempts, so reused smoke responses are not counted as another pilot/full charge.

The whole selected scope must be prepared before any call. A blocked query, changed request hash, suspect credential-shaped content, missing candidate, or altered partition of the twenty dense candidates stops execution. No selected query is silently removed. Calls run sequentially. Every successful batch produces a durable per-query checkpoint and concise accounting progress. Query completion requires all twenty candidates and validated saved response caches.

Before skipping a cached query, a read-only check joins each envelope to its unique ledger settlement, original reservation, saved-response event, original request bytes, and unmodified transport response body and metadata. Model, usage, latency, and costs must agree. An alias cache also needs its ledger reuse event, semantic cache index, and equivalent original payload. This check never calls `client.request()` or reads a credential. Copied or external envelopes without this experiment's provenance cannot claim completion or publish an ingestion snapshot; the original files remain intact for inspection.

`--max-queries N` is an optional checkpoint boundary: stop after visiting N previously incomplete queries in this invocation. Already complete queries are skipped. The result is explicitly `interrupted` until the full selected scope completes; snapshots preserve all 10, 200, or 4392 IDs, with empty incomplete B/C rankings. Repeating the same command resumes remaining work and reuses completed responses.

The cumulative admission and transport-start deadline is at most 21600 seconds from the first paid reservation in the experiment ledger, including previous invocations and scopes. `--max-seconds` can lower this bound, never raise it. The runner checks before each client request and immediately before transport; socket timeouts shrink to the remaining allowance. A socket timeout is not a hard wall-clock cancellation guarantee for an in-flight request, so an already-started response can outlast this bound. Its provider-contract receipt must specify one attempt per request so internal retries cannot cross the admission boundary. If a deadline is reached after a durable reservation but before transport, no request is sent and the reservation remains held for manual reconciliation. Every new or unreviewed ambiguous request stops subsequent paid calls; the exact reviewed recovery exception below does not change the general retry policy or deadline.

One exceptional recovery is supported through a hash-pinned `GateBundle.recovery` receipt under `reviewed-wsl-reboot-recovery-v1`. The client verifies the exact original request, model, plan, reservation hash, and a saved reboot/process observation before allowing the specifically named replacement attempt `-2`. The normal contract remains `max_attempts=1`; this is one reviewed replacement, not an automatic retry loop. No third attempt is permitted. A different request cannot be dispatched until the replacement settles. The original attempt remains unresolved and its full reservation remains held alongside the separate replacement reservation and eventual charge. A reboot proves the old local process cannot resume; it does not prove the provider cancelled the original request or left it unbilled.

Runner checkpoints record `reviewed_unknowns`, `reviewed_unknown_attempts`, and `unreviewed_unresolved_attempts` while retaining the total `unresolved_attempts` and all held amounts. Only exact IDs validated by the client may be exempted from admission blocking; an invalid receipt, unrelated unknown, or lost replacement still stops execution. Recovery does not clear locks or reset the ledger automatically. Successful completion of every selected query establishes evaluation coverage, **not billing reconciliation**: reports must continue to disclose the original unknown charge and retained allowance rather than present it as a known charge or release.

Checkpoint history is immutable under `study_progress/<plan-sha>/<invocation-id>/`; `latest.json` is an atomically replaced status pointer. Final ingestion always targets a new `study_snapshots/<plan-sha>/<invocation-id>/` directory and uses `run_jev_rerank.ingest`. Interruptions, errors, and partial runs retain the original selected denominator. A crash can leave `study_runner.lock`; recovery is manual after confirming no prior process or request remains active. Existing ledger locks and reservations are never cleared by this runner.

Monetary progress separates settled USD/CAD from unresolved reserved USD/CAD. CAD ledger values include the client's recorded FX, fee multiplier, and buffer, but exclude any separately configured experiment fixed fee; the client still includes that fee in every budget admission check. Reserved attempts are conservatively counted as possible API attempts, not proof that every reservation reached the provider. Error output contains only exception class names and fixed runner reason codes; request text, HTTP bodies, and credentials stay out of stdout/stderr.

Offline validation uses injected synthetic transport and synthetic credentials only:

```bash
python -m unittest test_hosted_study.StudyRunnerTests -v
```
