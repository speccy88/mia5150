# Host swap guard stop

The strict any-swap-activity guard stopped this invocation after the host
wrote one additional 4 KiB page to swap. The final sample still reported
4.47 GiB available system RAM, 5,274 MiB device VRAM and no OOM. The guard
observes the whole host, so this does not establish which process owned the
page. The benchmark model was unloaded and the event retained.

The next invocation keeps the same model, context, chunking, microbatch,
cache identity and strict guard policy. It checks and reuses existing vector
checkpoints. Its swap baseline starts from the current host counters; the
prior event remains in telemetry and is not removed from performance history.
