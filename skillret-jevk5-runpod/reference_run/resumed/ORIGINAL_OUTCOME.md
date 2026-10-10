# SkillRet / JevK5 resumed pilot — completed and independently verified

## Outcome
2026-10-02: resumed the original frozen pilot from the complete step-50 adapter/optimizer/RNG checkpoint and completed the requested 100-update schedule. Step 100 won validation selection. Fresh-process selected-adapter reload passed: maximum logit difference 0.0, full ranking agreement 1.0. Original incomplete run/failure evidence retained separately. No official test or full-dataset training ran. No production promotion.

## Held-out validation (256 queries)
Metadata-only, natural BGE-M3 top-15 candidates plus NONE, full relevance denominator; 2,000 training groups, seed 42. Held out from dataset train split, NOT the official paper full-text protocol.
- BGE-M3: Recall@5 0.430339; MRR@15 0.576059.
- Unadapted JevK5: Recall@5 0.473307; MRR@15 0.616340.
- Selected step-100 LoRA: Recall@5 0.509766; MRR@15 0.735584; Hit@1 0.695312.
- Recall@5 gain vs unadapted: 3.646 percentage points; query-bootstrap 95% CI [2.083, 5.469] points. Family dependence not corrected; validation also selected checkpoint, so this is not independent test evidence.
- Candidate Recall@15 ceiling 0.514974; false abstention 0.058594 vs unadapted 0.011719. Improved forced ranking does not remove abstention trade-offs.
- Resume training stage 1971.2s; peak allocated GPU memory 9956569088 bytes. Native BF16 A40, pinned backbone alibiserikbay/JevK5 @ c4f7fdb3aeab5582336406e78d3bef11bf98833d. Reference kernels, no FLA. Timing includes resumed updates/validation, not original training; no controlled speedup claim.

## Export, teardown, cost
- All 157 export members hash-matched against the stable idle pod and extracted Lexar files; ZIP CRC and source/destination archive SHA-256 passed. Selected adapter manifest hashes verified; 256 unique reload predictions scored.
- Evidence: `/media/LEXAR256/SkillRet-JevK5-RunPod/resumed-pilot-20261002`; archive `SkillRet_JevK5_Resumed_Pilot_Results.zip`.
- Archive SHA-256: `6a470fe563e0ac607e5a4c5931b15652af34f579234db5231913265f39824758`.
- Excluded pretrained download cache, virtual environments and Python/test caches; retained source, data, embeddings, checkpoints/optimizer/RNG state, predictions, metrics, configs and logs. Finalization receipts/scripts live beside the archive. Credential patterns only in checksum-verified upstream public dataset examples; no runtime matches elsewhere. Archive not publicly uploaded; pattern scans cannot guarantee absence of arbitrary secrets.
- Exact pod 37rdyis3kwmomk terminated; provider absence verified at 2026-10-02T11:49:54+00:00. No separately created network volume was attached.
- Estimated resumed compute US$0.622, storage US$0.018; aggregate original+resume GPU/storage C$1.880, within original C$10 cap. Approved FX 1.4243. Estimates, not settled billing; tax excluded.

## Next gate
A one-seed small pilot is not a generalization guarantee or a full-training approval. Preserve results and evaluate retrieval coverage/abstention before promotion. Observed full-workload projection: ~37.24 hours / C$25.99 compute; conservative 2x C$51.98, excluding setup, mining, reload, export, storage and tax. Any full run needs separate approval.

Related original run: [[SkillRet JevK5 A40 Pilot 2026-10-02]].
#ai #skillret #runpod
