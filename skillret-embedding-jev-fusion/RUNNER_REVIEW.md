# Independent offline hosted-runner review

Review date: 2026-10-07 UTC. Scope: `run_hosted_study.py`, its tests, and the client/cache/ingestion paths it calls. No real credential was inspected and no TypeSafe API was called. Reproductions used temporary synthetic fixtures, injected transport, and a mock that denied credential access. Existing baseline and experiment evidence were not modified.

Reviewed production runner SHA256: **`2afcbe24a7598b40a993343084ec71a4e93062f481873a0dc9b8120b7432e757`**.

## Finding and fix verification

**Material finding, corrected:** the original runner's initial cache scan checked response-envelope hashes and schema, but did not require ledger settlement or original transport evidence. Fully cached queries were skipped before the client's stronger checks. This allowed an orphaned cache set, such as responses restored without their ledger, to appear complete while recorded spend was zero.

An independent synthetic reproduction created ten well-formed response envelopes in an empty experiment ledger. The original runner returned:

```text
status=completed; complete_queries=10; selected_queries=10
ledger_attempts=0; settled_usd=0; mock_transport_calls=0
credential access denied by mock
```

The repaired runner adds read-only provenance validation. A cached response must connect to a unique original settlement, reservation and saved-response event; original request bytes, response body and transport metadata must match their hashes and model/accounting data. Alias reuse additionally requires its ledger reuse event and semantic-cache binding. The checks run before skipping cached work and before final ingestion, without invoking the client request path merely to validate a cache.

The original reproduction against the repair returned:

```text
status=interrupted; reason=snapshot_failed
complete_queries=0; selected_queries=10; ledger_attempts=0
mock_transport_calls=0; no completed snapshot published
credential access denied by mock
```

I then independently ran all **16 `StudyRunnerTests` tests: passed in 6.607 seconds**, using the existing WSL Python environment and the shared Windows source. Coverage includes orphan caches, altered original response bodies, valid alias reuse without new spend, partial resume, fixed denominators, unresolved reservations, deadline persistence, pre-transport deadline checks, disabled defaults and manual lock recovery. The author subsequently extended synthetic Score fixtures for the separate strict-response-parser work; the parent should include its final combined test run in the release evidence.

## Budget, deadline and denominator assessment

No additional blocking runner defect was identified. New paid dispatch still passes the client's receipt, credit and CAD-cap checks before a credential is read or a reservation is created. Unresolved attempts stop later paid work. Budget accounting counts unique ledger attempts rather than cache reuse. A separately configured experiment fixed fee is excluded from the progress subtotal but remains included in the client's admission check; the output states this distinction.

Partial runs keep every selected query ID. Incomplete Jev and RRF rankings remain empty in the snapshot rather than shrinking the comparison denominator. Completing a query requires all twenty original candidates.

Remaining limitations:

- The deadline prevents new dispatch after the recorded cumulative limit and shrinks transport timeouts. The standard-library HTTP timeout is not a hard wall-clock cancellation guarantee for an already dispatched request; an in-flight operation may finish after the nominal deadline.
- Full-scope runtime projections should include filesystem, ledger and immutable-checkpoint work. Per-batch checkpoints serialize the selected study state, so 4,392-query wall time and disk use do not necessarily scale linearly from ten API response latencies. The repair indexes ledger events within each cache scan, reducing repeated provenance-search work.
- This review verifies local behavior with synthetic evidence. It does not verify live account credit, pricing, tax treatment, model access, real service responses or provider billing.

No source code or tests were edited by this reviewer. The runner author implemented the fix; this document records independent reproduction, source inspection and offline verification.
