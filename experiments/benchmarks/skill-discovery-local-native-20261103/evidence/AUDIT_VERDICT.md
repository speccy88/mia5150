# Sampling and gold-position verdict

PASS for the requested construction: `random.Random(42).sample(all_skills, N)`
over the 6,006 rows, copied for each question, then replacement of the final item
only when the gold ID is absent. The code adds no gold prefix.

The independent reproduction reads the original JSON directly, without calling
`bench.make_plan`. It confirms that the 20 supplied gold IDs already match the
first 20 draws with seed 42. Gold for question i therefore occupies position i at
every size: 50, 100, 250, 500, 1000, 2000 and 5000. No swaps are needed for these
data. This position bias is real and must accompany every conclusion; a test with
balanced positions would be a separate variant.

The executed `bench.py` has SHA-256
`992ac9549bcada4ba29018ede5e7c932a78a98578092eaab4d10792a8d378e9c`.
The inspected local manifests and three OAuth manifests have identical candidate
plans and prompt hashes. The original comparison runner was unavailable, so an
exact reproduction of that separate run could not be established.

Evidence: `gold-position-independent-check.json` and `gold-position-audit.json`.

Study: `./`. Historical dynamic status: `results/local-v1/suite-status.json`.
Summary: `LOCAL_REPORT.md` (dated snapshot; regenerate after a new suite).

In the historical diagnostic run, Qwen3-0.6B has 140 rows: 80 measured responses
and 60 estimated context exclusions. Qwen3-1.7B has 140 rows: 66 measured responses,
an OOM on request 67, 13 subsequent server-stopped rows and 60 estimated exclusions.
An OOM is not summarized as measured zero accuracy. Original results are preserved;
no inference was replayed for this audit.
