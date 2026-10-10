---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: feature-extraction
base_model: Qwen/Qwen3-Embedding-0.6B
datasets:
  - donghongjiang/skillreason-bench
tags:
  - sentence-similarity
  - feature-extraction
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

# SkillReason-embedding-0.6b

[![Official Code Repository](https://img.shields.io/badge/Official%20Code-SkillReason-2ea44f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/donghong1/SkillReason)
[![Benchmark](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SkillReason--Bench-FFD21E.svg)](https://huggingface.co/datasets/donghongjiang/skillreason-bench)
[![Reranker](https://img.shields.io/badge/%F0%9F%A4%97%20Reranker-0.6B-FFD21E.svg)](https://huggingface.co/donghongjiang/SkillReason-reranker-0.6b)

> **Official code and evaluation toolkit:** [github.com/donghong1/SkillReason](https://github.com/donghong1/SkillReason)
> Released inference adapter, benchmark evaluation scripts, and reproducibility
> instructions are available in the repository.

SkillReason is a reasoning-enhanced dense retriever for selecting reusable
agent skills from natural-language requests. It is designed for implicit
requests that describe a task goal without explicitly naming the required
skill or execution procedure.

The model is initialized from
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
Capability reasoning is used as privileged supervision during training and is
further optimized with retrieval feedback. Normal retrieval remains
**query-only** and does not require autoregressive rationale generation.

## Model Details

| Property | Value |
|---|---|
| Parameters | 0.6B |
| Primary use | Agent skill retrieval |
| Pooling | Final non-padding token |
| Similarity | Cosine similarity over L2-normalized embeddings |
| Recommended dtype | BF16 on supported GPUs |
| Recommended maximum length | 4096 tokens |

## Quick Start

The official toolkit handles document rendering, multi-GPU encoding,
content-addressed corpus caches, exact search, and benchmark adapters:

```bash
git clone https://github.com/donghong1/SkillReason.git
cd SkillReason
pip install -e .

skillreason-download --artifact retriever-0.6b --output-dir artifacts

skillreason-retrieve \
  --model artifacts/models/SkillReason-embedding-0.6b \
  --backend hf_last_token \
  --corpus examples/skills.jsonl \
  --queries examples/queries.jsonl \
  --output-dir outputs/retrieval \
  --corpus-cache outputs/cache/skills.npy \
  --query-prefix official \
  --devices 0 \
  --max-length 4096 \
  --top-k 10
```

## Transformers Usage

Apply the retrieval instruction to queries only. Skill documents should be
rendered as `name | description | body` without the query instruction.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_id = "donghongjiang/SkillReason-embedding-0.6b"
query_instruction = (
    "Instruct: Given a task description, retrieve the most relevant skill "
    "document that would help an agent complete the task\nQuery: "
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def last_token_pool(hidden_states, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    final_positions = (attention_mask.long() * positions).max(dim=1).values
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, final_positions]


@torch.no_grad()
def encode(texts, max_length=4096):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)
    output = model(**batch, use_cache=False)
    embeddings = last_token_pool(output.last_hidden_state, batch["attention_mask"])
    # Match the released evaluation protocol: normalize in the model dtype,
    # then convert the normalized vectors to FP32 for exact cosine search.
    return F.normalize(embeddings, p=2, dim=1).float()


queries = [query_instruction + "<YOUR_USER_REQUEST>"]
skills = [
    "<SKILL_NAME_1> | <SKILL_DESCRIPTION_1> | <SKILL_DOCUMENT_1>",
    "<SKILL_NAME_2> | <SKILL_DESCRIPTION_2> | <SKILL_DOCUMENT_2>",
]

scores = encode(queries) @ encode(skills).T
print(scores)
```

## Evaluation

The [SkillReason toolkit](https://github.com/donghong1/SkillReason) provides
the released adapters and protocol settings for SkillReason-Bench, SRA-Bench,
SkillRet, and SkillBench Core. For example:

```bash
DOWNLOAD=1 \
MODEL_SIZE=0.6b \
BENCHMARK=skillreason \
DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/evaluate_benchmark.sh
```

Each run records its resolved model, precision, query prefix, sequence length,
batch geometry, data version, predictions, and metrics.

<details>
<summary>Optional capability-analysis generation</summary>

The checkpoint can also be loaded as a causal language model for qualitative
capability analysis. This generation step is optional and is not used by the
standard query-only retrieval path.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "donghongjiang/SkillReason-embedding-0.6b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

prompt = """Analyze the user query for skill retrieval. Write a concise query analysis that describes what kinds of relevant skill capabilities are needed, especially when multiple skills may be required.

User query:
<YOUR_USER_REQUEST>

Query analysis:
"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

</details>

## Related Resources

- [SkillReason-embedding-4b](https://huggingface.co/donghongjiang/SkillReason-embedding-4b)
- [SkillReason-reranker-0.6b](https://huggingface.co/donghongjiang/SkillReason-reranker-0.6b)
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
for following the licenses and terms of the skill documents they index.
