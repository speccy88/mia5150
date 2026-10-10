> Historical assessment. The later authorized smoke and pilot are complete. Current evidence and budget outcome are in [BUDGET_STATUS.md](BUDGET_STATUS.md) and [actual findings](results/hosted-study-findings-final/findings.md). Earlier pending-account requirements below are superseded.

# Billing gate reassessment — 2026-10-07

The earlier demand for an independently queried account balance was overly restrictive. The coordinating user instruction says to accept the reported USD 5 signup credit without querying its balance. That statement can support an additional USD 5 experiment-consumption ceiling; it must not be relabeled a verified current balance. The CAD 10 all-in authorization remains in force, with no purchase or refill-setting change authorized.

## Verified evidence

- Current public pricing, rechecked at approximately 01:45 UTC: Jev `jev-1.13.0`, USD 0.042 per million input tokens, free output. Both public aliases currently resolve to this version. The documented context limits remain rounded 64k total and 32k state plus longest question. [Official model/pricing page](https://docs.typesafe.ai/models)
- The already authorized authenticated model-list call returned HTTP 200, `jev-latest` and `jev-preview`. It did not return an inference-version or billing receipt. Its immutable raw body and receipt are in `results/hosted-execution/model_list*`. No scoring request has been made.
- The inspected [API reference](https://docs.typesafe.ai/api) and [documentation index](https://docs.typesafe.ai/llms.txt) expose no documented balance/refill lookup. The [account console](https://console.typesafe.ai) is the documented place to view credit. No private billing route was guessed.
- A fresh browser inventory returned exactly `{"apps":[],"browsers":[]}`. Earlier attempts returned `No browser is available` and `Browser is not available: iab`. This is an execution-environment limitation, not a TypeSafe authentication or billing denial. No 401, 402 or account-credit denial was observed.

## What the ledger can bound

At the public rate, reserving 65,536 input tokens per attempt gives USD 0.002752512 per request. This is a deliberately generous reservation based on the documented rounded context size, not a measured native token count or server-side monetary cap. The 40-request smoke reserves USD 0.11010048; the 779-request pilot including smoke reserves USD 2.144206848. At the verified reference rate of CAD 1.4226/USD with a 10% planning buffer, the pilot reserve is CAD 3.35538353. These are limits under the stated rate/context assumptions, not actual usage or charges.

Sequential reservations, persistent settlements, no blind retry and held ambiguous attempts bound this experiment's modeled consumption independently of a live balance query. The operator can stop on insufficient-credit responses rather than purchase credit. A public-rate assumption is reasonable for the reported signup-credit account absent contrary account evidence; no custom-rate agreement has been observed. Planning allowances must be labeled assumptions, not verified fees.

## One remaining account fact

The [customer agreement §8.2](https://typesafe.ai/legal/mca) says promotional credits are consumed first. An account previously opted into automatic refill can purchase its selected refill amount at zero balance or a configured threshold; it does not say refill is enabled by default. A local request ledger does not cap that separate purchase.

Consequently, no balance audit or repeated budget approval is requested. The smallest factual question is: **Have you previously enabled automatic credit refills on this TypeSafe account?** This asks about the existing setting, not permission to change it. An answer that refill is not enabled permits the reported-credit route without a balance query. If it is enabled, establish a safe existing-credit margin above its threshold before dispatch; do not require disabling it or infer that a small request cannot trigger a larger purchase.

Paid calls remain zero pending this fact. The baseline, frozen smoke/pilot IDs, fixed top-20 pools, truncation policy and 5/200 smoke and 143/4000 pilot truncated-pair counts are unchanged.
