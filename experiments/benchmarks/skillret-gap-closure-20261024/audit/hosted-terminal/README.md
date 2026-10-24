# Independent terminal hosted audit

**Passed; no report-release blocker.** Snapshot `d76981132a4510804f3858e3bb4366e99144054c` of https://github.com/speccy88/mia5150.

The previously audited Qwen raw cases, performance summary, paired intervals and frozen inputs are byte-identical. All 57 terminal evidence manifest hashes pass. The complete 303-event budget history independently reconstructs 150 unique generation attempts; every final per-attempt amount agrees with the public HTTP ledger. The maximum reconstructed guarded total remains within US$2.

## Verified terminal accounting

| Item | Verified value |
|---|---:|
| Generation POST attempts | 150 |
| Successful Qwen generation calls | 147: 3 canaries, 24 routing, 120 selection |
| Failed provider canaries | 3: DeepSeek, Gemma, Mistral |
| Additionally logged read-only GETs | 9: 6 authentication, 3 availability |
| Merged public request rows | 159 |
| Budget history events, **not calls** | 303 |
| Known API-reported cost | US$0.02411946 |
| Conservative unknown-cost holds | US$0.19: 0.14 + 0.02 + 0.03 |
| Known cost plus holds | US$0.21411946 |
| Actual total cost | **Unknown** |

Older incompletely instrumented read-only HTTP calls remain an unknown count. Unknown holds are not measured charges. The provider/account invoice was not accessed.

The Mistral first canary is the only new generation attempt since the Qwen snapshot. Its exact pinned request hash reconstructs correctly, with no fallback/tools/plugins. The frozen full-context/full-completion upper bound is `131072 × 0.00000009 + 32768 × 0.00000030 = US$0.02162688`, below the upward-rounded US$0.03 hold. This raises liability while retaining STOP; it does not authorize continuation. The public HTTP ledger records 429 and no Retry-After header. The owner-preserved error receipt reports Parasail and `upstream_provider_shared_pool`; the private raw error body was not accessed. No Mistral selector, router, replay or further model appears in the final ledger.

Original selector plan: 360 = 120 completed Qwen + 240 unattempted DeepSeek/Gemma. Later exploratory Mistral addition: 120 unattempted. Combined: 480 planned, 120 completed, 360 unattempted. Unattempted cells remain unmeasured, not zero-performance results. Only one model has benchmark-quality evidence; a completed multi-model comparison cannot be claimed.

The erroneous failed-canary cost label is corrected. The missing timing-scope metadata has an explicit immutable-parent fallback resolved before network activity. The first canary failed before any benchmark case, so no lost-response recovery or timing reconstruction was needed. Terminal generation guards are present at call and transport boundaries. Runtime stop-file existence and the owner's offline test execution remain owner evidence; this independent audit is read-only and made no live calls.

## Sources and reproduction

Terminal input: `MIA5150_Final_Hosted_Evidence_2026-10-24.zip`, Library `libfile_a2b8ce995a5481918b2e8429e071e01c`, 357,359 bytes, SHA-256 `13b204dd4526f64b7e90f665feec384000e3d036776763bae05c1fd5117596a8`.

Prior input: `MIA5150_Hosted_Qwen_Evidence_2026-10-24.zip`, Library `libfile_dcf5cfd747e08191baa551b8a27aa109`, SHA-256 `9b211bcec32cf0cb649691590452fae6fa1ca95fd5b99a04a21769bf1e67497d`.

```sh
python independent_terminal_audit.py \
  --evidence /path/to/extracted/final-hosted-evidence \
  --qwen-evidence /path/to/extracted/qwen-hosted-evidence \
  --output independent_terminal_validation.json
```

Standard-library Python only. No project imports, network access or live tests. This sanitized packet contains the script, JSON receipt and this README only. Prior limitations remain: 24 exposed balanced queries, gold-conditioned N=100 catalogs, qrels-based selection rather than human semantic adjudication/native skill execution, nominal paired uncertainty, and no causal local-versus-hosted model-size inference.
