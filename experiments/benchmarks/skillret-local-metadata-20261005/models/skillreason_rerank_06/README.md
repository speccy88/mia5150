---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-ranking
base_model: Qwen/Qwen3-Reranker-0.6B
datasets:
  - donghongjiang/skillreason-bench
tags:
  - text-ranking
  - reranking
  - retrieval
  - agent-skills
  - skill-retrieval
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

# SkillReason-reranker-0.6b

[![Official Code Repository](https://img.shields.io/badge/Official%20Code-SkillReason-2ea44f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/donghong1/SkillReason)
[![Retriever](https://img.shields.io/badge/%F0%9F%A4%97%20Retriever-0.6B-FFD21E.svg)](https://huggingface.co/donghongjiang/SkillReason-embedding-0.6b)
[![Benchmark](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SkillReason--Bench-FFD21E.svg)](https://huggingface.co/datasets/donghongjiang/skillreason-bench)

> **Official code and evaluation toolkit:** [github.com/donghong1/SkillReason](https://github.com/donghong1/SkillReason)
> Released inference adapter, benchmark evaluation scripts, and reproducibility
> instructions are available in the repository.

SkillReason-reranker-0.6b scores the relevance of a candidate skill document
to a natural-language task request. It is initialized from
[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) and is
intended for the second stage of a retrieve-then-rerank pipeline.

## Model Details

| Property | Value |
|---|---|
| Parameters | 0.6B |
| Primary use | Agent skill reranking |
| Input | Query and candidate skill document |
| Output | `logit("yes") - logit("no")` |
| Recommended dtype | BF16 on supported GPUs |
| Released pipeline depth | Top 20 candidates |

## Quick Start

Use the official toolkit to preserve the released prompt and long-document
truncation behavior:

```bash
git clone https://github.com/donghong1/SkillReason.git
cd SkillReason
pip install -e .

skillreason-download --artifact reranker-0.6b --output-dir artifacts

skillreason-rerank \
  --model artifacts/models/SkillReason-reranker-0.6b \
  --candidates outputs/retrieval/predictions.jsonl \
  --corpus examples/skills.jsonl \
  --output-dir outputs/reranked \
  --devices 0 \
  --top-n 20
```

The candidate file is the `predictions.jsonl` produced by
`skillreason-retrieve`. The output contains the reranked skill IDs and scores,
plus metadata describing the resolved scoring protocol.

## Transformers Usage

The following example scores one query-document pair. For batching and long
documents, use the official toolkit so the answer suffix is preserved during
truncation.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "donghongjiang/SkillReason-reranker-0.6b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

system = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
instruction = (
    "Given a task description, judge whether the skill document is relevant "
    "and useful for completing the task"
)
body = (
    f"<Instruct>: {instruction}\n"
    "<Query>: <YOUR_USER_REQUEST>\n"
    "<Document>: <SKILL_NAME> | <SKILL_DESCRIPTION> | <SKILL_DOCUMENT>"
)
prompt = (
    f"<|im_start|>system\n{system}<|im_end|>\n"
    f"<|im_start|>user\n{body}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
yes_id = tokenizer.encode("yes", add_special_tokens=False)[-1]
no_id = tokenizer.encode("no", add_special_tokens=False)[-1]

with torch.no_grad():
    logits = model(**inputs, use_cache=False).logits[:, -1, :]

score = logits[:, yes_id].float() - logits[:, no_id].float()
print(score.item())
```

## Evaluation

The released protocol reranks the top 20 candidates returned by a first-stage
retriever. The [SkillReason toolkit](https://github.com/donghong1/SkillReason)
provides evaluation adapters for SkillReason-Bench, SRA-Bench, and SkillRet,
and records predictions, metrics, prompts, precision, and batch settings for
each run.

## Related Resources

- [SkillReason-embedding-0.6b](https://huggingface.co/donghongjiang/SkillReason-embedding-0.6b)
- [SkillReason-reranker-4b](https://huggingface.co/donghongjiang/SkillReason-reranker-4b)
- [SkillReason-Bench](https://huggingface.co/datasets/donghongjiang/skillreason-bench)
- [Inference and evaluation toolkit](https://github.com/donghong1/SkillReason)

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

## License

The checkpoint is released under the Apache License 2.0. Users are responsible
for following the licenses and terms of the skill documents they rerank.
