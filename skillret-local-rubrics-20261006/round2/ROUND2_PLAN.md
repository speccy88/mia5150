# Small diverse SkillRet probes — approval-ready plan

Status: local preparation only. No paid requests, uploads, credential creation, downloads, training or inference started. The other task owns this PC's compute resources. The previous verified ZIP and all raw results remain unchanged.

The shared ceiling is **US$4 total**, including provider usage, storage, mandatory fees and taxes. Existing balance only; no top-up, auto-refill or subscription. Suggested API operating target: at most $0.60 combined; preserve the remainder. Access, balance, actual prices and permission to transmit the exact payload must be verified first.

## First: native Jev typed relevance, not Jev Router

Freeze `exploration-freeze.json`: 12 already-exposed public-development queries, each with the existing SkillRet top20 pool. These exclude the prior eight-query local GGUF cohort. Reuse cached dense and native reranker controls; do not rerun their inference. No gold insertion or changed candidate coverage.

One configuration, at most12 calls to `https://api.typesafe.ai/v1/systemone`, pinned `jev-1.13.0`. A single call shares canonical query text across20 Noul relevance questions and20 ordinal Score questions; each question contains just its own candidate's public name/description. No qrels, expected skill counts, private project files, checkpoints, credentials or source instructions are sent as control instructions. Output is two distinct relevance formulations on identical evidence. No secondary LLM routing.

Official current price: $0.042 per million input tokens, output free. At the documented64k maximum input per call,12 calls cost at most **$0.032256 usage** if that price and accounting apply. This is a planning bound, not measured spend. Reserve **$0.10 all-in** after confirming account terms; no automatic retries. Persist a reservation before transmission, count uncertain/failed requests against the reserve, stop on unknown usage/model drift. `probe_contract.py` implements offline request/response and serial reservation contracts; it contains no network client. Six synthetic unit tests passed; this is not live API validation.

Sources: [TypeSafe models](https://docs.typesafe.ai/models), [native API schema](https://docs.typesafe.ai/api).

## Second: one small non-thinking listwise contrast

Only if OpenRouter already has usable credit/access: use the same12 queries and20 candidate headers, one frozen listwise prompt, one pinned provider/model route, no fallbacks and no repeated prompt search. Candidate: Qwen3-235B-A22B-Instruct-2507, subject to an actual current route quote. At6000 input/300 output tokens per query and the parent-researched $0.0875/$0.35 per million rates, estimated usage is **$0.00756 for12 calls**, not a measured or guaranteed price. Reserve at most **$0.50 all-in**; charge context errors, malformed lists and failed calls in the denominator. No call is ready to execute until price, input/output caps and account access are verified. This listwise client is proposed, not implemented or tested.

Source to recheck: [OpenRouter model listing](https://openrouter.ai/qwen/qwen3-235b-a22b-2507). Do not purchase new credits under the $4 cap if minimum purchase exceeds it.

## Evaluation and stopping

Before scoring, record top20 candidate recall/completeness and oracle relevance ceiling. Score both Jev formulations and any listwise output on the exact same query/pool intersection; retain failures and original tie order. Report per-query NDCG@10 deltas, Recall@10, completeness, latency, actual usage and exploratory paired uncertainty. Twelve cases are for finding distinct failure modes, not confident model promotion. Stop after this screen; do not automatically expand to4392 queries or tune on its labels. Selective escalation may reuse these saved outputs without more paid calls, but cannot be called measured end-to-end latency.

All existing SkillRet panels have exposure. None can be relabelled untouched. Genuine project-held-out confirmation needs a separately frozen, repository/family-disjoint query/label panel that was not used in selection; preferably human-authored/independently annotated. Existing SRA/SkillReason public panels can provide frozen external diagnostics, but checkpoint exposure remains unknown. Do not claim independent confirmation merely by changing the random seed or hiding already-inspected labels.

## Optional single RunPod experiment — lower priority

Only if the distinct API probes justify adaptation: one corrected replay run, not a sweep, preserving all91 unique positive pairs in the saved384-example protocol. Prefer a fresh same-hardware FP16 base control or verify numerical parity before using saved outputs; never compare a BF16 remote run as identical to the localFP16 control. Existing interrupted adapter weights stay local unless a specific upload is separately approved. A fresh run can download the pinned public base/data directly on the pod.

Planning example: one A40,90 minutes inclusive of provisioning/download/export,60GB ephemeral container disk, no network/retained volume. At the published $0.49/hour and $0.10/GB/month disk rates, estimated compute is **$0.735** and disk about **$0.013** (720-hour month approximation). Reserve **$1.25 all-in**, only after the actual quote, mandatory charges, at least one hour's required credit and usable balance are verified. This is not a runtime forecast or launch authorization for a particular offer.

Proposed outgoing material: only a reviewed allowlist of experiment-specific code/config/manifests and public benchmark query/skill metadata. Exclude the62MB project ZIP, private-fork notes, local paths/history, secrets and model/adapter/optimizer binaries. The exact allowlist, destination pod/account and transfer approval remain prerequisites.

Require a tested control-plane watchdog independent of the training worker before launch, deadline below the reserved budget, periodic verified local exports, and confirmed termination of this exact newly created pod. Job exit does not stop billing; a stopped retained volume still costs money. No open-ended process, no auto-refill, no persistent storage and no deletion of unique outputs. If reliable export/shutdown cannot be established, do not launch. No deployer/watchdog has been implemented or tested yet.

Sources: [RunPod published prices](https://www.runpod.io/pricing), [billing and storage](https://docs.runpod.io/pods/pricing), [stop/terminate behavior](https://docs.runpod.io/pods/manage-pods).

## Access findings and secure next step

No RunPod/OpenRouter/TypeSafe/Jev connector is exposed. Standard key variables are absent in the checked Windows and WSL environments. RunPod CLI and its expected config are absent. The four explicitly checked project `.env` paths do not exist; no secret-bearing file contents were inspected. Browser sessions/account balances have not been checked, so account ownership or existing credit cannot be ruled out.

First establish whether the user's Jev access is direct TypeSafe, OpenRouter or another service. Then the user can provide an existing key through an approved secure local environment/secret mechanism, never chat, with its endpoint/model/balance and permission status verified using a harmless authenticated read such as TypeSafe `GET /v1/models`. A successful model list would prove authentication, not credit availability. Do not create/configure a new credential without the required approval/user handoff. No secret setup, authenticated call or transfer has happened.

Automatic approval review rejected an earlier proposed `.env` assignment inspection. That code was removed before execution; the safe replacement checks file existence only. There was no workaround or secret read.
