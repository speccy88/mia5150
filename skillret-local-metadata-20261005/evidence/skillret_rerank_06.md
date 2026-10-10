---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- reranker
- skill-retrieval
- llm-agents
- cross-encoder
base_model: Qwen/Qwen3-Reranker-0.6B
pipeline_tag: text-ranking
base_model_relation: finetune
datasets:
- ThakiCloud/SKILLRET
---

# SkillRet-Reranker-0.6B

[![arXiv](https://img.shields.io/badge/arXiv-2605.05726-b31b1b.svg)](https://arxiv.org/abs/2605.05726)

This is a **reranker** fine-tuned for **AI agent skill retrieval**. Given a natural-language user request and a candidate agent skill, it scores how relevant and useful the skill is for the request. It is designed as the second stage after a first-stage retriever such as [SkillRet-Embedding-0.6B](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B) or [SkillRet-Embedding-8B](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-8B).

The model is fine-tuned from [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) on the SkillRet benchmark training split with binary cross-entropy on the `yes`/`no` token probability. It keeps the scoring interface of Qwen3-Reranker.

📄 **Technical report**: [SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents (arXiv:2605.05726)](https://arxiv.org/abs/2605.05726)

## Usage

### Transformers

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "ThakiCloud/SKILLRET-Reranker-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()

token_yes = tokenizer.convert_tokens_to_ids("yes")
token_no = tokenizer.convert_tokens_to_ids("no")

instruction = (
    "Given a skill search query, judge whether the skill document "
    "is relevant and useful for the query"
)
prefix = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
max_length = 8192


def format_pair(query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


@torch.no_grad()
def score(query: str, docs: list[str]) -> list[float]:
    pairs = [format_pair(query, d) for d in docs]
    enc = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_ids) - len(suffix_ids),
    )
    enc["input_ids"] = [prefix_ids + ids + suffix_ids for ids in enc["input_ids"]]
    enc = tokenizer.pad(enc, padding=True, return_tensors="pt").to(model.device)
    logits = model(**enc).logits[:, -1, :]
    stacked = torch.stack([logits[:, token_no], logits[:, token_yes]], dim=1)
    return torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().tolist()


query = "Help me set up a CI/CD pipeline for my Python project"
skills = [
    "ci-cd-setup | Configure continuous integration and deployment pipelines ...",
    "python-debugging | Debug Python applications using pdb and logging ...",
]
print(score(query, skills))  # higher = more relevant
```

Each skill document is formatted as `name | description | SKILL.md body`, the same representation used by the SkillRet embedding models.

## Training Details

- **Base model**: Qwen3-Reranker-0.6B (0.6B parameters)
- **Training data**: SkillRet benchmark training split (63,259 queries and 10,123 skills)
- **Objective**: binary cross-entropy on P(`yes`) for each query–skill pair
- **Hard negatives**: mined from four retrievers (SkillRet-Embedding-0.6B, SkillRet-Embedding-8B, Qwen3-Embedding-8B, harrier-oss-v1-0.6b). Ranks 21–60 from each retriever are merged into one pool of non-relevant candidates, and 15 negatives are sampled per positive.
- **Hardware**: 4× NVIDIA B200 GPUs (DDP)
- **Effective batch size**: 384 (96 per device × 4 GPUs)
- **Max sequence length**: 8,192 tokens
- **Learning rate**: 2e-5, warmup ratio 0.1
- **Schedule**: one epoch
- **Precision**: BF16

## Evaluation Results

Evaluated on the [SkillRet benchmark](https://arxiv.org/abs/2605.05726) evaluation split (4,392 queries, 6,006 skills). The reranker rescores the top-20 candidates returned by SkillRet-Embedding-8B.

| Model | NDCG@5 | NDCG@10 | NDCG@15 |
|-------|--------|---------|---------|
| SkillRet-Embedding-8B, no reranking | 0.8458 | 0.8644 | 0.8695 |
| **+ SkillRet-Reranker-0.6B (this model)** | **0.8610** | **0.8774** | **0.8821** |

Full metrics for this model:

| Metric        | @5     | @10    | @15    |
|---------------|--------|--------|--------|
| NDCG          | 0.8610 | 0.8774 | 0.8821 |
| Recall        | 0.8928 | 0.9357 | 0.9510 |
| Completeness  | 0.8206 | 0.8896 | 0.9128 |

## Intended Use

This model is designed to rerank candidate agent skills for a natural-language user request. It is part of the [SkillRet benchmark](https://arxiv.org/abs/2605.05726) release for evaluating skill retrieval systems for AI agents.

## Limitations

- Optimized for English-language queries and agent skills.
- Reranking is substantially slower than first-stage retrieval. Rescoring 20 candidates takes about 0.8 s per query on one B200, so latency-sensitive deployments may use the embedding model alone.

## Citation

If you use this model or the SkillRet benchmark, please cite:

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
