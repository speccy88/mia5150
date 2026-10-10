# Prospective ONNX Runtime telemetry opt-out

ONNX Runtime 1.29 introduced POSIX telemetry for enabled builds. Microsoft's 1.30 privacy documentation says official supported-platform builds enable telemetry and documents a process-lifetime non-Windows opt-out: set `ORT_DISABLE_TELEMETRY=1` before ONNX Runtime initializes. Its API-level event switch may be called only after an initialization event, so it is not the equivalent pre-initialization precaution. Sources: [1.29 release notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0) and [1.30 privacy documentation](https://github.com/microsoft/onnxruntime/blob/v1.30.0/docs/Privacy.md#disabling-telemetry).

For a new, separately authorized run, before any Python process that could import or initialize ONNX Runtime:

```sh
. ./setup_privacy.sh
# Equivalent prospective setting: export ORT_DISABLE_TELEMETRY=1
```

This is a newly added privacy setup, not a configuration used in the measured runs. `setup_privacy.sh` only sets the variable; it launches no process. It was syntax-checked but not exercised with ONNX Runtime or inference. No measured scorer, model adapter, model manifest or recorded result was changed. No denial was retried and no alternate inference was run.

The benchmark stopped during the attempted Verdict 48-query extension. Final primary quality therefore uses the exact common 32 queries; completed Julia/GLiNER 48-query results remain separate supplements. The attempted 48-query plan is preserved. This stop does not establish which telemetry events, if any, were transmitted in earlier runs.

This precaution addresses ONNX Runtime telemetry only. It is not a blanket guarantee about all dependencies, downloads, networking or another platform. Review the source documentation and local permissions before starting a new run. It does not authorize resuming this stopped benchmark.

## Installed-source verification

The installed ONNX Runtime1.30.0 `Privacy.md` was read without importing the package: 4,696 bytes, SHA256 `9703b86132bbe407a7138f9a45689e6996a8ab0ac670529d3699ce753f67538e`. It contains the documented pre-initialization opt-out. `docs/ort_privacy_source.json` records this static provenance; no ORT process was started.
