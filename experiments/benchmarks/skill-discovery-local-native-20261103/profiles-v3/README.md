# Final supplement: adjusted contexts and KV cache, separate v3 series

Read [PROFILE_REPORT.md](PROFILE_REPORT.md), [the final comparison](../COMPARISON.md)
and [publication transformations](PUBLICATION_TRANSFORMS.json).

200 responses: each of Qwen3-8B and Mistral v0.3 has 80 f16 responses at sizes
50/100/250/500, followed by 20 Q8_0 responses at the same context as f16 for N=500.
Q4_K_M weights are unchanged. Here Q8_0 denotes the KV cache, rather than a new
weight quantization. Admission-only probes are not quality measurements.

The original estimation gate with a 10% margin is replaced by actual input tokens
+500, rounding n_ctx up to a multiple of 256. Candidate lists, questions,
descriptions, temperature 0, the 500-token output cap and scoring are unchanged.
This context change is explicit: do not merge this series into the historical
32K/f16 suite that blocked these models. No N>=1000 was measured. The gold-position
bias at positions 1–20 remains.

## Public offline verification

```powershell
python -m unittest discover -s tests -v
python validate_profiles_public.py
```

The validator recomputes metrics from CSVs and checks SQLite exports, tokenizer
preflights, plans/hashes and paired comparisons. Raw responses remain private:
`review_profiles.py` is the historical audit code requiring those captures for full
replay. Public databases cannot be resumed. The minimal dataset produces identical
prompts; each skill's attribution and license are in [the parent catalogue](../data/ATTRIBUTION.md).
Weight sources/licenses are in [parent provenance](../evidence/model-artifact-provenance.json).

Archived GPU observations are sanitized historical evidence. Transformer/output
layers and KV are CUDA, with documented CPU_Mapped input embeddings. All 14 loads
were checked locally against sealed log prefixes and CUDA/WDDM counters. Full logs
and process identifiers are not redistributed. Counter limitations remain as
described in the report.

To repeat measurements, use a fresh directory, independently verified weights and
new process evidence. Source paths and the derivative-data hash are adapted to
this tree without changing measured parameters. `source-measured/` preserves the
original runner and launcher versions. Llama/Gemma were not run before closure, as requested: see
[CLOSURE.md](CLOSURE.md).
