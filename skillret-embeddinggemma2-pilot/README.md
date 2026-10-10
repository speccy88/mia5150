# EmbeddingGemma 2 bounded retrieval pilot

Start with [EmbeddingGemma2-RTX2060S-Report.md](EmbeddingGemma2-RTX2060S-Report.md). This is a text-only FP32 compatibility and restricted-pool retrieval pilot, including a 20-query/28-document bilingual fixture. It is not a full SkillRet evaluation. Failed compatibility attempts and the successful configuration are retained separately.

Source, fixtures, evaluation outputs/vectors, configuration/license text, dependency manifests, and immutable model metadata are included. The downloaded base weights, dependency overlay, wheels, and runtime environment are omitted. Rebuild the model using model-metadata.json/model-manifest.json and eg2_download.py at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`. Audit dependencies before executing; evaluation scripts load models and were not rerun.
