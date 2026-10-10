---
pretty_name: SkillReason-Bench
license: other
task_categories:
  - text-retrieval
language:
  - en
tags:
  - skill-retrieval
  - information-retrieval
  - agent-skills
  - implicit-intent
size_categories:
  - 10K<n<100K
configs:
  - config_name: queries
    data_files:
      - split: test
        path: data/queries.jsonl
  - config_name: corpus
    data_files:
      - split: corpus
        path: data/skills.jsonl
citation: |
  @misc{jiang2026skillreasonreasoningenhancedagentskill,
    title        = {SkillReason: Reasoning-Enhanced Agent Skill Retrieval for Implicit User Requests},
    author       = {Donghong Jiang and Endian Lin and Luoping Cui and Hanqing Liu and Mingjie Liu and Fan Yang and Hong Wang and Zhao Yang and Chuang Zhu},
    year         = {2026},
    eprint       = {2608.08640},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url          = {https://arxiv.org/abs/2608.08640}
  }
---

# SkillReason-Bench

[![Official Code Repository](https://img.shields.io/badge/Official%20Code-SkillReason-2ea44f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/donghong1/SkillReason)
[![Retriever 0.6B](https://img.shields.io/badge/%F0%9F%A4%97%20Retriever-0.6B-FFD21E.svg)](https://huggingface.co/donghongjiang/SkillReason-embedding-0.6b)
[![Retriever 4B](https://img.shields.io/badge/%F0%9F%A4%97%20Retriever-4B-FFD21E.svg)](https://huggingface.co/donghongjiang/SkillReason-embedding-4b)

> **Official code and evaluation toolkit:** [github.com/donghong1/SkillReason](https://github.com/donghong1/SkillReason)
> Released inference adapter, benchmark evaluation scripts, and reproducibility
> instructions are available in the repository.

SkillReason-Bench evaluates agent skill retrieval for implicit user requests.
Each request describes a concrete task goal while leaving the required skill
or execution procedure unstated. A retriever must identify the most relevant
skill from a large, heterogeneous corpus.

## Dataset Summary

| Component | Count |
|---|---:|
| Evaluation queries | 3,729 |
| Candidate skills | 61,228 |
| Gold skills | 3,729 |
| Fine-grained categories | 122 |
| Reporting domains | 9 |

Each query has one gold skill in `positive_ids`. The benchmark is English-only,
and the average query length is 33.2 words.

## Domain Coverage

| Domain | Queries |
|---|---:|
| Software | 644 |
| Data Science | 637 |
| General | 416 |
| Media and Design | 405 |
| Legal | 353 |
| Finance | 340 |
| Biomedical | 314 |
| Engineering | 313 |
| Science | 307 |

For domain-level reporting, records labeled `other` and `writing_docs` in the
released JSONL are grouped into the General domain.

## Files and Schema

### Queries

`data/queries.jsonl` contains one evaluation request per line.

| Field | Type | Description |
|---|---|---|
| `query_id` | string | Stable query identifier |
| `query` | string | Natural-language user request |
| `positive_ids` | list[string] | Gold skill IDs |
| `category` | string | Fine-grained source category |
| `domain` | string | Released domain label |

### Skill Corpus

`data/skills.jsonl` contains one candidate skill per line.

| Field | Type | Description |
|---|---|---|
| `skill_id` | string | Stable skill identifier |
| `name` | string | Skill name |
| `description` | string | Short capability description |
| `body` | string | Full skill document |
| `author`, `repo_slug`, `path` | string | Source metadata |
| `github_url`, `skillsmp_url` | string | Source URLs when available |

## Loading

Queries and corpus use separate dataset configurations because their schemas
are intentionally different.

```python
from datasets import load_dataset

repo_id = "donghongjiang/skillreason-bench"
queries = load_dataset(repo_id, "queries", split="test")
skills = load_dataset(repo_id, "corpus", split="corpus")

print(queries[0]["query"], queries[0]["positive_ids"])
print(skills[0]["skill_id"], skills[0]["name"])
```

## Evaluation Protocol

Rank all 61,228 skill documents for each query. A prediction is correct when
its `skill_id` matches an ID in the query's `positive_ids`. The standard
metrics are Recall@K, NDCG@10, and MRR.

SkillReason models use the following instruction for queries; skill documents
are encoded without this prefix:

```text
Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task
Query: <USER_REQUEST>
```

Skill documents are rendered from their name, description, and body. The
released SkillReason-Bench protocol uses a maximum sequence length of 2048
tokens.

The official adapter downloads the pinned dataset revision, applies the
released protocol, caches corpus embeddings, and writes both predictions and
metrics:

```bash
git clone https://github.com/donghong1/SkillReason.git
cd SkillReason
pip install -e .

DOWNLOAD=1 \
MODEL_SIZE=0.6b \
BENCHMARK=skillreason \
DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/evaluate_benchmark.sh
```

## Quality Control

Benchmark construction uses rule-based validation, exact and near-duplicate
removal, query-to-skill consistency checks, domain balancing, independent LLM
review, and human adjudication of reviewer disagreements. Queries that are
malformed, overly simple, ambiguous, mismatched, or explicitly reveal the
target skill are removed.

## Related Resources

- [SkillReason-embedding-0.6b](https://huggingface.co/donghongjiang/SkillReason-embedding-0.6b)
- [SkillReason-embedding-4b](https://huggingface.co/donghongjiang/SkillReason-embedding-4b)
- [SkillReason-reranker-0.6b](https://huggingface.co/donghongjiang/SkillReason-reranker-0.6b)
- [SkillReason-reranker-4b](https://huggingface.co/donghongjiang/SkillReason-reranker-4b)
- [Inference and evaluation toolkit](https://github.com/donghong1/SkillReason)

## Code and Paper

- **Code repository:** [github.com/donghong1/SkillReason](https://github.com/donghong1/SkillReason)
- **Paper:** [SkillReason: Reasoning-Enhanced Agent Skill Retrieval for Implicit User Requests](https://arxiv.org/abs/2608.08640)

## Citation

If you find SkillReason helpful for your research, please consider citing our
paper:

```bibtex
@misc{jiang2026skillreasonreasoningenhancedagentskill,
  title        = {SkillReason: Reasoning-Enhanced Agent Skill Retrieval for Implicit User Requests},
  author       = {Donghong Jiang and Endian Lin and Luoping Cui and Hanqing Liu and Mingjie Liu and Fan Yang and Hong Wang and Zhao Yang and Chuang Zhu},
  year         = {2026},
  eprint       = {2608.08640},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2608.08640}
}
```

## Give Us a Star 👏

If you find SkillReason useful for your research or applications, please
consider giving the repository a [star ⭐](https://github.com/donghong1/SkillReason).
Thank you for your support!

## License and Intended Use

Benchmark queries and annotations are released for research use. Skill
documents retain their source metadata and URLs in `data/skills.jsonl`; users
must follow the license and terms of each upstream repository.
