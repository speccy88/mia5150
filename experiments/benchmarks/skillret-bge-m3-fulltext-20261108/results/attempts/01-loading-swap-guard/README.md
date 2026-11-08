# Model loading swap guard

The first CUDA feasibility invocation loaded the model on CUDA, then stopped
before its first encoding because whole-host swap traffic exceeded 64 MiB.
One-second telemetry recorded 894.4 MiB additional swap-out, 4.63 GiB minimum
available host RAM and 1,206 MiB peak GPU use. There was no CUDA OOM and no
retrieval result. Cgroup memory/high/swap limits were unrestricted.

The next loader maps the backbone directly to CUDA through the official
Transformers device_map argument, avoiding an intermediate CPU model copy.
The model weights, FP16 inference, document plans and resource guard are
unchanged. The retained event is not erased or counted as a quality outcome.
