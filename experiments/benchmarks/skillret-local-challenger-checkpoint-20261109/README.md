# Local SkillRet challenger

This campaign trains a query-only LoRA adapter against the frozen, full-document
SkillRet-Embedding-0.6B index, then trains a separate SkillRet reranker. Embedding
and added-reranker pipeline claims remain separate. The archived reference is
NDCG@10 **0.8109902409310652** over 4,392 official queries and 6,006 skills.
Recomputing its saved rankings reproduced the reported metrics exactly before
any new model inference. See `results/archived_baseline_verification.json`.

## Published checkpoint

The completed official query-only runs use the same 4,392 queries and 6,006
frozen document vectors. Paired 95% intervals use 10,000 query resamples
against the archived reference.

| Condition | NDCG@10 | Recall@10 | NDCG@10 change, percentage points (95% interval) |
| --- | ---: | ---: | ---: |
| Archived reference | 0.810990 | 0.876442 | — |
| Reconstructed starting point | 0.810978 | 0.876442 | −0.001 |
| Pure query LoRA, seed 42 | 0.817349 | 0.882362 | +0.636 [+0.485, +0.790] |
| Pure query LoRA, seed 43 | 0.817469 | 0.883045 | +0.648 [+0.494, +0.801] |
| Fixed query mixture, seed 42 | 0.816318 | 0.881451 | +0.533 [+0.398, +0.665] |
| Fixed query mixture, seed 43 | 0.815925 | 0.881641 | +0.493 [+0.361, +0.626] |

Both adapters also improved on the matched reconstruction. That one-time
reconstruction recovered every archived top-1 result, with 4,242 identical
top-10 orders. The archived results remain the primary reference.
[Reconstruction review](results/RECONSTRUCTION_REVIEW.json) ·
[Seed 42 audit](results/official_query_seed42_independent.json) ·
[Seed 43 audit](results/official_query_seed43_independent.json) ·
[Figure](figures/official-query-only-two-seeds.png)

Development selection and confirmation are complete, along with the adapter
checkpoints and their recorded input and score checks. On confirmation the
combined pipeline scored 0.929409 NDCG@10 against 0.903197 for frozen dense
retrieval. Neither adapter showed a positive interval against its matched
unchanged control. The locked official reranker, scaling, and end-to-end
latency measurements are in progress. This checkpoint contains completed
query-only and development evidence.

## CPU replay

Download Git LFS payloads for this checkpoint, then use Python 3.12 and NumPy
2.2.6 to reproduce the five official query-only rankings and metrics, the
completed development panels, and their score reuse checks.

```bash
git lfs pull --include="experiments/benchmarks/skillret-local-challenger-checkpoint-20261109/**"
cd experiments/benchmarks/skillret-local-challenger-checkpoint-20261109
python replay_checkpoint.py
```

The lossless archive contains all currently indexed training and official
document and query vectors, three adapted development query panels, exact IDs,
input hashes, and relevance labels. The [vector manifest](vectors/manifest.json)
records shapes and checksums. Replay requires only the checked-out files and
NumPy. The [verification receipt](results/checkpoint_replay.json) records
21,960 official and 7,680 development dense rankings and 10 development
reranker panels. The full campaign's replay also covers the official
reranker and scaling conditions.

`protocol.json` freezes the development campaign. Whole source-repository,
identity and normalized-body components determine train/selection/confirmation
membership. Both development panels contain 512 queries and search the complete
10,123-skill training-origin corpus. The loss and training miner exclude all
held-out development skills. Cross-partition multi-positive queries and exact
query/positive-set duplicates are excluded in a published manifest. These public
training examples may already have been used by the released checkpoints.
The historically exposed official test is not a fresh blind test.

Base model weights, full raw texts, optimizer recovery state and installation
caches stay outside Git. Skill text is untrusted input and is never executed.
The exact baseline query instruction, last-token pooling, FP32 normalization,
left padding, 1,000,000 RoPE theta, repeated-KV efficient SDPA and 32K document
limit are retained. The historical two truncated test documents will remain
truncated under that reference condition.

The original WSL document/query vector cache was absent from this server. The
user authorized reconstructing it once. New vectors must be checked against all
archived baseline rankings; mismatches must be reported rather than hidden.
BGE and Qwen8B vectors are never substituted for the 0.6B index.

## Local execution

Use Python 3.12 and a separate environment with the pinned dependencies in
`runtime-lock.txt`. The measured CUDA installation is reused through a read-only
package path; dependency changes are installed only in the new environment.
Set `SKILLRET_CHALLENGER_CACHE` to a persistent directory outside the checkout.
Optionally set `SKILLRET_TRAIN_CACHE` to an existing pinned train/test cache. Acquisition downloads and verifies missing files, named
`train-{skills,queries,qrels}.jsonl` and `{skills,queries,qrels}.jsonl`.
Acquisition uses only public downloads; inference and updates use local weights.

```bash
python src/prepare.py acquire
python src/prepare.py verify-saved
python src/prepare.py prepare
python -m pytest -q tests
python src/guard.py embedding-parity python src/parity.py
python src/guard.py index-train python src/cli.py index train
python src/guard.py query-pilot python src/cli.py train-query --lr 3e-6 --steps 100
```

Use a new guard attempt name after any failed/interrupted run; immutable vector
journals validate each row and resume without re-embedding completed inputs.
Training recovery directories hold adapters, optimizer/scaler/scheduler and RNG
state. Use `--resume PATH` with the original recipe and total step budget.
For example, if the 3e-6 continuation stopped after step 100 and its
`step-00100/recovery.pt` exists, resume the same trial:

```bash
python src/guard.py query-full-selected-lr-01-retry-2 python src/train_query.py \
  --name full-selected-lr --lr 3e-6 --steps 1200 \
  --resume "$SKILLRET_CHALLENGER_CACHE/checkpoints/query/full-selected-lr/step-00100"
```

Read the guard receipt before resuming. Use the actual saved step and original
time budget. Recovery archives later journal rows rather than mixing them into
the resumed history. Run the campaign controller again after the resumed stage
finishes so it can complete recipe selection.

Query adapters are deliberately asymmetric: enable them only for query encoding.
Never merge them into or re-encode the frozen document index. Mixtures with
weights strictly between zero and one require two query encodings.

The reranker uses the released yes/no scoring template and the difference
`logit(yes)-logit(no)`, with only the last hidden state and two frozen output-head
rows. A synthetic fixture compares both scores and adapter gradients with the
unoptimized full-vocabulary reference. Evidence selection is label blind and
identical for positives and negatives. Source offsets, omitted metadata and
selected windows are recorded; this is selected-evidence reranking, not complete
body coverage. A separately measured full-coverage development ablation scores
all windows sequentially with skill-level max aggregation.

No official challenger scoring is permitted until `FINAL_EVAL_LOCK.json` freezes
checkpoints, candidate pools, evidence policy, fusion and the primary contrast.
No official test result may trigger tuning in this campaign.

The historical scaling panel preserves all 20 query labels and the ordered
membership of the eight saved libraries. If the locked primary combines both
adapters, scaling includes that pipeline and the released reranker on the same
adapted-query candidates.

The offline combined comparisons can reuse a score only for the identical
checkpoint, exact token IDs, batch size and padded length. The development gate
requires bit-identical scores and a measured speedup for all three checkpoints
before enabling this. A fixed sample of official requests also gets direct
scoring, and CPU replay verifies each reused value. The separately reported
uncached latency sample always runs the original scoring path.
