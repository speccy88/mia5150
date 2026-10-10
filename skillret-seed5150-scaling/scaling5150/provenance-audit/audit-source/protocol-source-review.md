# Read-only source review for scaling5150

Scope: inspected existing Windows source and summary artifacts only. No source code was imported or executed; no WSL, GPU, model, inference, training, network, install, or service calls were made. Frozen source/result files were not changed. This review does not certify arrays that were not available in the inspected Windows directories.

All source paths below are relative to C:/Users/user/Documents/Codex/2026-10-05/task-2.

## Reuse and configuration

- scaling5150/index_new_small.py:44,53,58: the actual EG2/Jina reuse gate is matching ID and text, followed by direct row copying. It does not compare the producing run's immutable model revision, tokenizer hashes, runtime, native prefix, dtype, normalization, or max length before reuse.
- Lines 48–49 replace historical entries with pilot vectors while supplying current text. The code does not verify the pilot manifest's view/config before the overlay. The audit should compare pilot view_sha256, model_manifest_sha256, dtype, attention, max_tokens, batch and ordered row bindings against the full index and historical evidence.
- Current EG2 configuration at lines 15–18,30–38: text-only SentenceTransformer; float32; eager; max_seq_length 512; batch 1; native SearchQuery/Document prompts; normalize_embeddings=True; 768-dimensional outputs checked finite and unit-normalized. The actual prefix strings come from model.prompts, not a literal in this source.
- Current Jina nano configuration at lines 21–38: reviewed Python source hash check; AutoModel float32/eager; retrieval adapter; query/document prompts; 1024 max tokens; batch 1; 768-dimensional outputs checked finite and unit-normalized. Source declares Query: / Document: for token counting, but normalization/prompt behavior must also be verified against the reviewed native encode implementation.
- deliverables/scaling5150/source-audit.json records EG2 revision 914f7f89142e33e77833254d9c9b90c3cef7303b and Jina nano revision 8a7f00aac812071b69403df470f1038ec85f8925. These are manifest pins, not proof of the older producer's loaded bytes.
- That artifact reports, for each model, full-index reuse of 299 document and 4 query vectors; pilot generation of 30 new document and 1 new query vectors with 2 document and 1 query vectors reused. An immediate-source label of new_pilot does not mean all pilot rows were newly encoded.
- scaling5150/provenance_receipt.py:85 already states the material gap accurately: historical per-run records do not bind model revision, tokenizer hashes, and immutable producer source to vector files. No re-encoding is warranted under the requested scope.

## Thirteen changed headers and other cached encoders

- scaling5150/protocol.py:16 renders (name.strip() + " | " + description.strip())[:1024].
- harness.py:9–11 produces the old uncapped header. scaling5150/prepare_cached_index.py:10–18 asserts historical text/query hashes, identifies exactly 13 changed rows, refreshes only those, and copies the other 5,993 document rows plus the 20 selected rows of a 128-query cache.
- Existing runtime implementation neural.py:4,10–23 uses role-specific query prompts, empty document prefix, left padding, 1024-token truncation, FP16 model/SDPA, last-token pooling, then L2 normalization and float32 array output. Historical document batches are 4; changed headers use batch 1.
- Receipt verification should independently derive all 13 IDs from source corpus and current documents, compare each model's changed_document_ids, hash old/new UTF-8 header text and source/output vector rows, then require all 5,993 unchanged rows and all 20 query rows to match bitwise. IDs themselves cannot be certified from source code alone.

## Score-to-ranking replay

- scaling5150/retrieval.py:13,58,68: sort case IDs lexically, gather matching rows, compute scores = docs @ query_vector, sort by (-float(score), original_skill_id), retain first 100.
- harness.py:47–51 and retrieval.py:76: hybrid sums 1/(60 + one_based_rank) for each document in each of the SkillRet top100 and BM25 top100 channels, then sorts by (-fused_score, original_skill_id).
- Reranker tie behavior is different and intentional: scaling5150/rerank_local.py:12–14 sorts by score descending then frozen candidate-pool position.
- Selection domain-vector scoring (scaling5150/selection.py:59–61) searches the entire selected domain within the library, not only the saved global top100.
- provenance_receipt.py:88–94 replays six encoders x two fixed queries x sizes 50/6006 = 24 cases. Exact matches establish saved array-to-ranking consistency, not independent encoder-production provenance.

## Jev output provenance

- scaling5150/rerank_local.py:39–58 imports the existing reviewed JevK5GGUF adapter, uses temperature 1.22 and top_k 256, rejects missing option letters, and returns answer['score'] plus the processed decide answer.
- Lines 70–73 save that processed answer under a field named raw. This is not the original /completion response or pre-normalization logits. The subclass at lines 40–45 does not capture the completion response for persistence.
- Existing source and Apache 2.0 license can be included only after their local bytes/license are located and checked. The code's /home/user/code/routing-pilot/source reference is not evidence those files were available in this Windows review.
- Root jina_native.py concerns a different Jina joint-candidate reranker, not Jina nano embedding. It cannot supply missing nano reuse provenance.

## Audit/report guardrails

- provenance_receipt.py checks guest MemAvailable only (lines 11–14). Host RAM guard and low process priority must be supplied by the launcher. Its file hashing is 1 MiB chunks; array finite/norm chunks are 64 rows.
- The receipt copies qrels/source queries but does not itself compare all evaluator gold to upstream qrels. scaling5150/independent_audit.py performs that separate comparison and library reconstruction. Preserve the distinction between source-backed verification and archived self-consistency.
- Merely finding provenance_receipt.py or finish_provenance.py is not evidence that either ran successfully. finish_provenance.py contains fixed narrative counts; validate the actual receipt before using that narrative.
- The exact two-sided paired sign-flip test with four positive nonzero differences and sixteen ties yields 2/16 = 0.125. The bootstrap interval remains descriptive; no general-winner or untouched-confirmation claim.
