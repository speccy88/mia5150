# Seed-5150 local scaling, canonical version 2

Start with [the audited report](scaling5150/report/REPORT.md), [protocol](scaling5150/protocol.json), and [provenance receipt](scaling5150/provenance-audit/receipt.json). Twenty exposed public queries span six nested libraries; selector and reranker diagnostics use two queries. This is audited preliminary evidence, not blind confirmation.

The original local ZIP exactly matches Library version 2, SHA-256 `35fe8071ed1ecdbd7327e09c130424f9c23d657a27575a9a59f8367239007b6c`. Its expanded code/raw evidence/public inputs are preserved here; no duplicate ZIP is committed. Public credential-shaped examples in one query-input export were redacted; see the root publication audit for original/export hashes. The original CONTENT_SHA256.json describes original package bytes.

Offline tests: from this folder use the installed Linux environment and `python -m unittest discover -s scaling5150 -p 'test_*.py'`. No model loading is required. The original validate_artifacts.py also expects the original delivery ZIP and is not a portable archive-validation entrypoint. The root audit supplies current exported-file hashes and parsing checks.

Vector arrays and complete skill bodies are not included in the canonical ZIP. Rebuild using pinned inputs and the recorded scripts; preserve original `index-*/fingerprint.json`, model-provenance, and audit receipts. Historical EG2/Jina producer provenance remains incomplete. Do not claim a fresh independent vector replay from this archive alone.
