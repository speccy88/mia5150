# Primary-source verification notes

Checked for this handoff on 2026-09-28. These notes distinguish source verification from execution: no CLM weights were loaded, no embeddings generated, no training performed, and no RunPod account accessed by the handoff author.

Codex must resolve and record immutable revisions at execution time; these URLs point to mutable upstream pages.

## Official CLM model and implementation

- Model: https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- Code: https://github.com/Contrastive-LM/CLM
- Heads: https://raw.githubusercontent.com/Contrastive-LM/CLM/main/src/clm/heads.py
- Embedder: https://raw.githubusercontent.com/Contrastive-LM/CLM/main/src/clm/embedder.py
- Inference: https://raw.githubusercontent.com/Contrastive-LM/CLM/main/src/clm/engine.py
- Training: https://raw.githubusercontent.com/Contrastive-LM/CLM/main/train/finetune.py
- Embedding preparation: https://raw.githubusercontent.com/Contrastive-LM/CLM/main/train/embed_utils.py
- Fine-tuning guide: https://raw.githubusercontent.com/Contrastive-LM/CLM/main/docs/FINETUNING.md

The model card identifies frozen Qwen3-8B last-token representations and two pretrained projection heads. The heads are not random. The implementation supports loading the existing head weights as initialization.

Important inspected details:

1. `Embedder._fetch` converts vectors to FP32 and applies L2 normalization before caching. `HeadPair._project` then applies the nonlinear head and normalizes its output. Omitting the input normalization changes the experiment.
2. `HeadPair` reconstructs architecture from checkpoint configuration, not a generic guessed MLP. The implementation currently lists a 4096-dimensional input and default 512-dimensional output; actual checkpoint metadata must still be checked.
3. `Engine` already includes a head-free `clm-raw` ablation. `rank` currently accepts `instructions`, not `question`.
4. The scalar scale is positive and is applied to cosine scores. Scaling may change softmax sharpness, but not exact rank order. Earlier explanations that scaling alone amplifies ranking changes were overstated.
5. The upstream trajectory loss masks other examples with the same task/step code, but it does not generally construct all SKILLRET cross-query shared-skill relevance relations. A many-to-many positive mask is necessary for the proposed adaptation.
6. The default within-task evaluation is a trajectory-oriented measure. It must not select our head checkpoint; the desired criterion is full-training-catalog validation retrieval.
7. The upstream fine-tuning guide is an autonomous experimentation instruction document, including an open-ended loop. It is not an instruction to override this run's finite budget or frozen design.

## Dataset

- Card and load schema: https://huggingface.co/datasets/ThakiCloud/SKILLRET
- Raw card: https://huggingface.co/datasets/ThakiCloud/SKILLRET/raw/main/README.md

The current card documents three configurations (`skills`, `queries`, `qrels`) with train/test splits. Expected counts: 10,123/6,006 skills, 63,259/4,392 queries, and 127,190/7,187 qrels. Queries include multiple relevant skill IDs. The experiment must use the real data and count it, rather than forcing counts to match a previous revision.

The description-only representation is our deliberate controlled benchmark choice. It is not the published full-SKILL.md evaluation protocol. The dataset uses synthetic queries; provenance and possible incompleteness/overlap in judgments belong in the report.

## BGE

- https://huggingface.co/BAAI/bge-m3

BGE-M3 supports several retrieval modes. The agreed row uses only its dense vectors with the model's documented normalization/pooling. Do not silently enable sparse or multi-vector hybrid scoring. 'Best BGE' was a conversational shorthand; this benchmark does not establish that BGE-M3 is universally the best BGE checkpoint.

## RunPod lifecycle and billing

- https://www.runpod.io/pricing
- https://docs.runpod.io/accounts-billing/billing
- https://docs.runpod.io/pods/manage-pods

Live MCP quotes are authoritative for the job, not a price copied here. Budget includes setup, idle time and storage. Stopping an inference process is not stopping the Pod. Persistent storage can continue billing after compute stops. Low-balance shutdown can destroy data in some storage configurations; proactive local backup and explicit cleanup are required.

## Verification performed for the handoff

`python -m pytest` ran 29 CPU checks of the original helper code in `starter/reference_checks.py`. All passed. Coverage includes analytical retrieval metrics, invalid-input rejection, deterministic tie breaking, shared-skill positive masks, symmetric/multi-positive loss equivalence and gradients, nested catalogs, and budget arithmetic.

This does NOT validate upstream runtime compatibility, real model accuracy, actual spending controls, a trained checkpoint, or transfer to the user's computer. Codex must perform those on the real execution path.
