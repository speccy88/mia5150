> Historical assessment. The later authorized smoke and pilot are complete. Current evidence and budget outcome are in [BUDGET_STATUS.md](BUDGET_STATUS.md) and [actual findings](results/hosted-study-findings-final/findings.md). Earlier pending-account requirements below are superseded.

# TypeSafe existing-credit and fee gate

Public-source review: 2026-10-07 UTC. No account, credential or authenticated endpoint was accessed by this audit. No gate was changed.

## What the sources establish

The published Jev 1.13 price is USD **0.042 per million input tokens**, with output free. No separate per-evaluation or payment-processing surcharge is stated on that rate card. This is the public rate, not a verified account-specific rate. [Current models and pricing](https://docs.typesafe.ai/models)

The MCA, updated September 23, 2026, says:

- Section 8.1: fees are in USD unless the customer's order specifies otherwise.
- Section 8.2: submitted inputs consume purchased or promotional credits; consumption rates can depend on account settings. The balance is visible in the account.
- Section 8.2(a): an opted-in automatic refill can trigger below a configured threshold, not merely at zero. Without opt-in, the provider may decline output after credits are consumed.
- Section 8.4: applicable taxes are additional to fees.

It does **not** specify whether taxes are collected solely when buying credits, establish a separate tax debit per evaluation, or promise that every account has no additional charges. It supplies no card-fee or currency-conversion schedule. [MCA, section 8](https://typesafe.ai/legal/mca)

The reviewed public HTTP/SDK interfaces expose evaluation and model listing, not a documented credit-balance or billing endpoint. A successful model-list response cannot establish credit, refill settings or tax treatment. Do not invent a balance URL. [API reference](https://docs.typesafe.ai/api), [pinned SDK endpoints](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/src/typesafe_sdk/_core/endpoints.py)

## How to represent the CAD 10 calculation truthfully

Track credit consumption separately from new cash payment. Under an existing-credit-only plan with no purchase/refill, there is no documented new card transaction for each API call; this is an inference from the prepaid mechanism, not observed account behavior. Do not manufacture a per-call card fee or describe consumption as a new card charge.

The coordinator independently verified CAD/USD **1.4226** for 2026-10-06. This audit did not repeat that FX verification. At that conversion, the public rate corresponds to CAD **0.0597492 per million input tokens**, before any contingencies. This conversion values usage for the study; it is not the customer's actual historical exchange rate or bank settlement rate.

A conservative conversion, tax/fee allowance and fixed buffer can legitimately be recorded as **planning assumptions**. They do not become verified actual fees. For example, the ledger can explicitly name `fx_stress_allowance`, `tax_fee_contingency`, and `safety_buffer_cad`, while preserving the verified public USD rate and FX source. Any chosen percentages must be visible rather than invented as account facts. An assumed allowance alone cannot prove an unbounded unknown charge is covered.

## Remaining evidence and gate decision

The practical account evidence remains available credit, the applicable rate or confirmation of standard pricing, and the automatic-refill setting/threshold. A refill must not be triggered by this experiment. Account evidence can also clarify whether any further taxes or charges apply to spending the existing balance. Do not require proof of a hypothetical per-call processing fee merely because historical credit purchases might have involved one.

Keep actual fee verification and conservative budget assumptions as distinct fields. Public-source verification is complete; account credit/refill/rate verification remains outside this audit. **Do not set an account/fee gate to verified on the strength of this document, and do not claim an observed balance.**
