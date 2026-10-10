---
language:
- en
license: apache-2.0
size_categories:
- 100K<n<1M
task_categories:
- sentence-similarity
- text-retrieval
pretty_name: SkillRet Benchmark
tags:
- retrieval
- benchmark
- agents
- tool-use
- synthetic-data
- croissant
configs:
- config_name: skills
  data_files:
  - split: train
    path: data/skills/train.jsonl
  - split: test
    path: data/skills/test.jsonl
- config_name: queries
  data_files:
  - split: train
    path: data/queries/train.jsonl
  - split: test
    path: data/queries/test.jsonl
- config_name: qrels
  data_files:
  - split: train
    path: data/qrels/train.jsonl
  - split: test
    path: data/qrels/test.jsonl
dataset_info:
- config_name: skills
  features:
  - name: id
    dtype: string
  - name: name
    dtype: string
  - name: namespace
    dtype: string
  - name: description
    dtype: string
  - name: author
    dtype: string
  - name: stars
    dtype: int64
  - name: installs
    dtype: int64
  - name: license
    dtype: string
  - name: repo
    dtype: string
  - name: source_url
    dtype: string
  - name: raw_url
    dtype: string
  - name: major
    dtype: string
  - name: sub
    dtype: string
  - name: primary_action
    dtype: string
  - name: primary_object
    dtype: string
  - name: domain
    dtype: string
  - name: body
    dtype: string
  - name: skill_md
    dtype: string
  splits:
  - name: train
    num_examples: 10123
  - name: test
    num_examples: 6006
- config_name: queries
  features:
  - name: id
    dtype: string
  - name: original_id
    dtype: string
  - name: query
    dtype: string
  - name: skill_ids
    sequence: string
  - name: skill_names
    sequence: string
  - name: k
    dtype: int64
  - name: generator_model
    dtype: string
  - name: original_query
    dtype: string
  splits:
  - name: train
    num_examples: 63259
  - name: test
    num_examples: 4392
- config_name: qrels
  features:
  - name: query_id
    dtype: string
  - name: skill_id
    dtype: string
  - name: relevance
    dtype: int64
  splits:
  - name: train
    num_examples: 127190
  - name: test
    num_examples: 7187
---

# SkillRet Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2605.05726-b31b1b.svg)](https://arxiv.org/abs/2605.05726)

📄 **Technical report**: [SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents (arXiv:2605.05726)](https://arxiv.org/abs/2605.05726)

## Dataset Overview

SkillRet is a retrieval benchmark for matching natural-language user requests to agent skills. It contains a curated library of public agent skills from GitHub with synthetic training and evaluation queries.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Records | 218,157 |
| Total File Size | 714 MB |
| Total Skills | 17,810 |
| Training Skills | 10,123 |
| Test Skills | 6,006 |
| Out-of-split Skills | 1,681 |
| Training Queries | 63,259 |
| Evaluation Queries | 4,392 |
| Training Qrels | 127,190 |
| Evaluation Qrels | 7,187 |

## Dataset Structure

The dataset is organized into three subsets with `train` and `test` splits:

| Subset | Split | Records | Description |
|--------|-------|---------|-------------|
| `skills` | train | 10,123 | Skills used by the training split |
| `skills` | test | 6,006 | Skills used by evaluation split |
| `queries` | train | 63,259 | Synthetic training queries (Qwen3.5-122B-A10B) |
| `queries` | test | 4,392 | Synthetic evaluation queries (Claude Opus 4.6) |
| `qrels` | train | 127,190 | Binary relevance labels for training queries |
| `qrels` | test | 7,187 | Binary relevance labels for evaluation queries |

## Loading the Dataset

```python
from datasets import load_dataset

# Load evaluation data
skills  = load_dataset("ThakiCloud/SKILLRET", "skills",  split="test")
queries = load_dataset("ThakiCloud/SKILLRET", "queries", split="test")
qrels   = load_dataset("ThakiCloud/SKILLRET", "qrels",   split="test")

# Load training data
train_skills  = load_dataset("ThakiCloud/SKILLRET", "skills",  split="train")
train_queries = load_dataset("ThakiCloud/SKILLRET", "queries", split="train")
train_qrels   = load_dataset("ThakiCloud/SKILLRET", "qrels",   split="train")
```

## Skill Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique skill identifier (36 chars) |
| `name` | string | Skill name (2-61 chars) |
| `namespace` | string | Public skill namespace from source (13-92 chars) |
| `description` | string | Short natural-language skill description (10-2.32k chars) |
| `author` | string | Public GitHub username (2-31 chars) |
| `stars` | integer | GitHub repository star count (0-120k) |
| `installs` | integer | Marketplace install count (0-2.85k) |
| `license` | string | MIT or Apache-2.0 |
| `repo` | string | Source GitHub repository (5-62 chars) |
| `source_url` | string | GitHub URL to skill directory (47-161 chars) |
| `raw_url` | string | Raw URL to SKILL.md (59-180 chars) |
| `major` | string | Major taxonomy category (6 values) |
| `sub` | string | Sub-category (18 values) |
| `primary_action` | string | LLM-derived action tag (13 values) |
| `primary_object` | string | LLM-derived object tag (15 values) |
| `domain` | string | LLM-derived domain tag (13 values) |
| `skill_md` | string | Full Markdown skill body (369-145k chars) |
| `body` | string | Duplicate of `skill_md` |

## Query Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique query identifier |
| `original_id` | string | Original generation-time identifier |
| `query` | string | Natural-language user request |
| `skill_ids` | list[string] | Relevant skill IDs |
| `skill_names` | list[string] | Names of relevant skills |
| `k` | integer | Number of relevant skills |
| `generator_model` | string | Model used to generate the query |
| `original_query` | string | Pre-revision text, present on 14,226 `train` queries that were rewritten during generation; `null` elsewhere |

## Relevance Labels (Qrels)

Binary relevance format:

```json
{"query_id": "q-00001", "skill_id": "example-skill-id", "relevance": 1}
```

All listed pairs have `relevance = 1`. Unlisted pairs are treated as non-relevant.

## Taxonomy

Two-level functional taxonomy with **6 Major categories** and **18 Sub-categories**.

**Major Categories:**
- Software Engineering
- AI Agents
- Data & ML
- Content Creation
- Business & Planning
- Information Retrieval

## Evaluation Protocol

- **Evaluation Queries**: `queries` subset, `test` split
- **Candidate Corpus**: `skills` subset, `test` split
- **Relevance Labels**: `qrels` subset, `test` split
- **Document Text**: `name | description | skill_md`
- **Relevance**: Binary (1 = relevant, unlisted = non-relevant)
- **Recommended Metrics**: NDCG@k, Recall@k, MAP@k, MRR@k, Completeness@k

## Data Construction

**Skill Corpus Source**: Public agent skills from GitHub via `claude-plugins.dev`.

**Filtering**:
- English-dominant (non-English-script character ratio ≤ 3%)
- MIT or Apache-2.0 licensed only
- Deterministic quality and deduplication steps
- Semantic filtering of the evaluation split: embedding-based nearest-neighbour candidate generation followed by LLM adjudication over full skill bodies, using models from a different family than the query generator

**Query Generation**:
- Training: Qwen3.5-122B-A10B
- Evaluation: Claude Opus 4.6
- Style seeds: GAIA validation questions
- Validation: Automatic leakage detection, LLM review, human expert validation

## Reference Models

Trained baselines on this benchmark:

- [ThakiCloud/SkillRet-Embedding-0.6B](https://huggingface.co/ThakiCloud/SkillRet-Embedding-0.6B) — Qwen3-Embedding-0.6B fine-tune
- [ThakiCloud/SkillRet-Embedding-8B](https://huggingface.co/ThakiCloud/SkillRet-Embedding-8B) — Qwen3-Embedding-8B fine-tune

## Intended Use

- Evaluating skill retrieval models
- Evaluating embedding and reranking models on long-document retrieval
- Studying multi-skill retrieval scenarios
- Training/adapting retrieval models
- Category-stratified analysis

## Not Intended Use

- Ranking or judging GitHub authors
- Inferring personal/demographic attributes
- Safety certification of agent systems
- Measuring fairness across demographic groups
- Modeling private user behavior
- Treating synthetic queries as a complete replacement for real user logs

## Responsible AI Notes

### Synthetic Data
Training queries are synthetic (generated by LLMs and filtered/reviewed). Skill documents are from public GitHub repositories.

### Personal or Sensitive Information
Includes public GitHub usernames and repository metadata only. Does not collect private user data, demographic attributes, health information, or political/religious beliefs.

### Limitations and Biases

- English-dominant (not fully multilingual)
- Limited to public GitHub-derived skills
- Restricted to MIT/Apache-2.0 licenses
- Skill distribution skewed toward Software Engineering
- Query language reflects generator model styles
- Synthetic queries may not capture all real user request properties
- Public skills may differ from private/enterprise repositories

### Social Impact

**Positive**: Better measurement of retrieval quality, clearer failure mode diagnosis, more reproducible model comparisons.

**Negative**: Potential overfitting to a synthetic benchmark, misuse of scores as a full agent safety proxy, profiling of individual authors.

## Metadata and Format

- **Formats**: JSON, JSONL, Parquet
- **Libraries**: Datasets, pandas, Polars
- **License**: Apache-2.0 (benchmark metadata, queries, taxonomy) plus the source skill licenses (MIT or Apache-2.0)
- **Tasks**: Text Retrieval, Sentence Similarity
- **Modalities**: Tabular, Text
- **Language**: English

## Version

**v1.1** (current): Adds a semantic filtering pass over the evaluation split — skills 6,660 → 6,006, evaluation queries 4,997 → 4,392, evaluation qrels 8,347 → 7,187. Every retained query keeps its complete gold set, so the number of skills a query asks for always matches its labels. The `train` split is byte-identical to v1.0. Evaluation scores are therefore not directly comparable across v1.0 and v1.1.

**v1.0**: Public release, accompanies the technical report. Still available at revision `v1.0`:

```python
skills = load_dataset("ThakiCloud/SKILLRET", "skills", split="test", revision="v1.0")
```

## Citation

If you use SkillRet, please cite:

```bibtex
@article{kang2026skillret,
  title   = {SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents},
  author  = {Kang, Ryangkyung and Cho, Hongcheol and Kim, Youngeun},
  journal = {arXiv preprint arXiv:2605.05726},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.05726}
}
```

**Paper**: [https://arxiv.org/abs/2605.05726](https://arxiv.org/abs/2605.05726)
