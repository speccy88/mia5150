# MetaEncoder v1 on SkillRet: RunPod study

This study evaluates `facebook/meta-encoder-v1` using only its released BF16 model for new neural inference. It does not train model weights or rerun historical baseline models.

**Execution status:** primary retrieval completed on all 6,006 skills / 4,392 queries; secondary experiments and final delivery reconciliation remain in progress. Read [REPORT.md](REPORT.md) for the measured scope and limitations; the existence of an implementation does not establish that an experiment completed. All model inference, vector operations, evaluation, statistical calculations and figure generation must execute on RunPod. Authoring and artifact transfers may occur elsewhere.

## Study

- Model revision: `ed04ebd4defb8e837b5eaea164143177272696ac`.
- Dataset: `ThakiCloud/SKILLRET@6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`, official **test split only**.
- Primary scope: 6,006 skills, 4,392 queries, 7,187 positive qrels.
- Canonical document: `name.strip() + " | " + description.strip() + " | " + skill_md.strip()`.
- Historical reference: SkillRet-Embedding-0.6B full text, NDCG@10 **0.810990**, Recall@10 **0.876442**, MRR@10 **0.856349**. This is an exposed public-test comparison, not a blind test.

| Arm | Intended comparison | Interpretation |
|---|---|---|
| Encode | Full-catalog cosine retrieval | Primary zero-shot outcome |
| Native retrieve | Same cached corpus, re-encoded queries | API/numerical equivalence, not an independent model improvement |
| Dimensions | Prefix truncation and renormalization | Exploratory; no native configurable output dimension was found |
| Binary (`noul`) | Top-10/20/50 candidate relevance | Cannot recover missing candidates |
| Choice | Pick one from 5/10/20 candidates | No full ranking is inferred from one choice |
| Score | Expected ordinal usefulness on five levels | Binary qrels remain binary ground truth |
| Calibration | Temperature fit and disjoint held-out query subset | Secondary labeled study, separate from zero-shot retrieval |

## Reproduce

See [PROTOCOL.md](PROTOCOL.md) and [MODEL_API_AUDIT.md](MODEL_API_AUDIT.md). Provision an authorized 80-GB-or-larger RunPod GPU, clone the repository there, install the exact runtime versions in `manifests/pip-freeze.txt`, and use the commands in `runpod/README.md`. Installation configuration must be reconciled against the actual execution evidence before publication.

Keep checkpoints on durable storage and export them before terminating resources. The controller must enforce a spending cap and reserve time for validation/export. Never continue a dependent full evaluation when primary embeddings are incomplete.

## Artifacts and navigation

- `src/`: staged embedding, retrieval, decisions and evaluation implementation.
- `tests/`: tests to run on RunPod; unrun tests are not validation evidence.
- `results/`: aggregate/per-query measurements and rankings, with exact denominators.
- `figures/`: RunPod-generated figures and their CSV/JSON inputs.
- `manifests/`: input hashes, configurations, runtime, artifact checksums and completion status.
- [DRIVE_ARTIFACTS.md](DRIVE_ARTIFACTS.md): actual verified embedding delivery.

Large FP32 arrays and checkpoints belong on Drive, not in Git. Model weights, authentication material, caches and raw datasets are excluded. Source inspection found no explicit model license in the audited snapshot; pinned upstream downloads are used rather than redistributing weights or upstream wrapper source.
