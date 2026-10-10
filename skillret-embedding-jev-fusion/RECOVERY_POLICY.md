# Reviewed recovery after the WSL reboot

Policy: `reviewed-wsl-reboot-recovery-v1`. This is one explicitly reviewed recovery within the already approved experiment budget. It is not a general retry policy, a claim that the lost request was unexecuted, or a new authorization to purchase credit.

The only eligible original attempt is `f340d90743f01aac474aed7d9135f15599da30e73aa1f6bb08fed1cf6eb746dd-1`. Its identical request may receive one replacement reservation and dispatch, ending in `-2`. The original's entire USD and CAD reservations remain held as unknown execution/billing. No usage or score is invented for it, and no settlement or no-execution release is appended for it.

`GateBundle.recovery` is an optional SHA256-bound `Artifact`. Without it, every unknown attempt continues to block new paid work. `HostedJevClient.reviewed_unresolved_attempt_ids(rows=None)` validates the receipt and returns only that exact original attempt ID. It performs no writes, credential access, transport or lock removal. Supplied rows must come from the hash-verifying ledger reader.

The receipt requires these fields:

- `schema_version="reviewed-wsl-reboot-recovery-v1"`, `experiment_id="newskillret-embedding-jev-fusion"`, `verified=true`, `verified_at_unix` no more than 24 hours old, nonempty `source` and `reviewed_by` identifying the explicit operator review.
- `request_sha256`, `original_attempt_id` and `replacement_attempt_id` matching the exact request and attempts above.
- `actual_model_version`, `plan_sha256` and `original_reservation_event_sha256` matching the unique original reservation.
- `retain_original_reservation=true`, `additional_attempts=1`, `no_automatic_retry=true`.
- `reboot_proof={"path": "<relative JSON path inside the experiment ledger directory>", "sha256": "<actual proof hash>"}`.

The proof records `kind="wsl_reboot_observation"`, `observed_at_unix`, `uptime_seconds`, the observed nonempty `boot_id`, and integer `matching_live_process_count=0`. Observation time must precede or equal review time. `observed_at_unix - uptime_seconds` must be later than the lost reservation's timestamp, establishing that the observed boot followed the reservation. These fields record an actual external observation; the client does not invent them or pretend to inspect current processes itself.

The original request file must exist and hash to the frozen request ID. Original attempt `-1` must have no saved-response ledger event, body, metadata or response cache. After replacement settlement, the same request's cache may exist only if its attempt ID and response hash match the unique settled `-2` record. Request bytes, original plan, model, candidates, rubric and context policy remain unchanged.

Before replacement dispatch, normal evidence and budget gates still run. Both reservations count against CAD 10 and the applicable USD consumption cap. The replacement reservation and cache include `reviewed_recovery_of_attempt_id`, `recovery_receipt_sha256` and `recovery_policy`. This permits exactly `-2` even when the ordinary contract has `max_attempts=1`; it does not enlarge the attempt limit for other requests. No distinct new request may dispatch before the replacement settles. Cached successful work remains readable.

Any ambiguity on attempt `-2` remains unreviewed and blocks every further paid request, including different queries. No third dispatch is permitted. A successful replacement allows remaining distinct requests while the original reservation remains held; reporting must distinguish known settled costs from that unknown commitment. Its first received scores are used without selection against an unavailable original result.

Only the coordinating operator may preserve and remove the two proven-stale lock files after reviewing the reboot/process evidence. This implementation never removes stale locks automatically. Receipt validity and content hashes prove local integrity, not the truth of arbitrary operator-written assertions.

Offline tests use temporary synthetic ledgers, proof receipts and mocked transport. They cover missing receipts; forged, stale and mismatched receipts; changed or invalid process proof; absent/altered original request bytes; existing original response evidence; exactly one `-2` settlement with the original reserve held; recovery-before-other-requests ordering; second-ambiguity blocking; and budget checks accounting for both attempts. No real credential, provider call, ledger mutation or lock removal was performed to implement these tests.
