# SkillRet / JevK5 A40 pilot — incomplete schedule, verified export and teardown

## Outcome
2026-10-02: approved 100-update pilot stopped at update 67 after its configured 2,700-second training-process limit. Failure type: TimeoutError. The last complete checkpoint is step 50; step-0 and step-50 adapter/tokenizer/optimizer/RNG states, raw predictions, source, configs, dataset, embeddings and logs were preserved. No official test or full training ran. No retry, model/protocol change, extra paid resource, deadline extension or budget increase was made. The time-limit guard worked; bypassing it is not a bug fix.

## Real validation results
256 held-out queries from SKILLRET's train split, 2,000 training groups, natural BGE-M3 top-15 candidates plus NONE, seed 42. Not official test results and not a completed/accepted 100-step pilot.

- BGE-M3: Recall@5 0.430339; MRR@15 0.576059.
- Unadapted JevK5: Recall@5 0.473307; MRR@15 0.616340.
- Step-50 LoRA: Recall@5 0.505859; MRR@15 0.730219; Hit@1 0.691406; false abstention 0.078125.
- Step-50 Recall@5 change vs unadapted: 3.255 percentage points. Candidate Recall@15 ceiling: 0.514974.
- CUDA BF16 unquantized backbone admitted; 14,376,960 trainable parameters; peak allocated GPU memory 10,071,471,616 bytes. Logs show reference PyTorch kernels because causal_conv1d and flash-linear-attention are absent; this is a performance limitation, not proof of the sole timeout cause.

## Export and teardown
- Exact pod x2y8qdw44usgdl / charlie-skillret-jevk5-pilot-20261002 terminated; provider readback verified absence at 2026-10-02T06:25:37+00:00.
- Results directory: `/media/LEXAR256/SkillRet-JevK5-RunPod/pilot-20261002`.
- Archive: `/media/LEXAR256/SkillRet-JevK5-RunPod/pilot-20261002/SkillRet_JevK5_Pilot_Results.zip`.
- SHA-256: `fd042ff7c72a36ae5be003ccb7e083033a68048b974d45ea210919fd69b97dd2`.
- Independently verified SHA-256 equality, ZIP CRC/testzip, 103 archive entries, actual result files, and credential-pattern scan. Matches were confined to the checksum-verified upstream public skills dataset (documentation placeholders and example/test credentials); no runtime credential patterns detected elsewhere. Raw evidence was preserved unchanged and the archive was not publicly uploaded. This scan is not a guarantee that arbitrary secrets cannot exist.
- Receipt: `teardown-receipt.json`; independent checks: `independent-export-verification.json` alongside the archive.
- Estimated GPU cost US$0.661 / C$0.942; estimated running storage US$0.019. Based on 1.349 hours, US$0.49/hour and approved FX 1.4243; these are estimates, NOT provider-confirmed billing.

## Interpretation / next action
Step-50 validation improves over the unadapted baseline on this small held-out sample, but retrieval coverage limits recall, and this single incomplete run does not justify full training or promotion. Any resume, optimized-kernel investigation, matched repeat or full run requires separate approval. Preserve the export and failure evidence; no historical rows were rewritten.

#ai #skillret #runpod
