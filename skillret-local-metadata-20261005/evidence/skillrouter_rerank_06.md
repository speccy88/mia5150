---
language:
- en
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3-Reranker-0.6B
pipeline_tag: text-ranking
tags:
- reranking
- cross-encoder
- agent-tools
- skill-routing
- qwen3
- listwise-learning
---

# SR-Rank-0.6B

`SR-Rank-0.6B` is a fine-tuned cross-encoder reranker for skill routing. It is designed to score a small candidate set of retrieved skills against a task query and select the single most relevant skill for an LLM agent.

## Model Summary

- Base model: `Qwen/Qwen3-Reranker-0.6B`
- Architecture: causal-LM-style cross-encoder reranker
- Input: query + candidate skill text, formatted into the Qwen reranker prompt template
- Output: scalar relevance score computed as `logit(yes) - logit(no)`
- Intended use: rerank top-K candidates from a first-stage retriever such as `SR-Emb-0.6B`

This model should be used on a candidate list, not as a standalone retriever over an 80K-scale corpus.

## Intended Uses

Use `SR-Rank-0.6B` after a first-stage retriever has already narrowed a large corpus to a candidate set, for example:

1. Retrieve top-20 skills with `pipizhao/SkillRouter-Embedding-0.6B`.
2. Score each candidate with `pipizhao/SkillRouter-Reranker-0.6B`.
3. Sort by `yes - no` score.
4. Take the top-1 or top-N skills for downstream planning or execution.

This model is not intended for free-form generation, chat, or long-document retrieval without candidate pruning.

## How to Use

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "pipizhao/SkillRouter-Reranker-0.6B"


def format_rerank_prompt(name, desc, body, query_text, desc_max=500, body_max=2000):
    instruction = (
        "Given a task description, judge whether the skill document "
        "is relevant and useful for completing the task"
    )
    doc_text = f"{name} | {desc[:desc_max]} | {body[:body_max]}"
    return (
        f"<Instruct>: {instruction}\n\n"
        f"<Query>: {query_text}\n\n"
        f"<Document>: {doc_text}"
    )


def build_qwen_reranker_inputs(tokenizer, prompt, max_length=4096):
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements '
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    tokens = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        return_attention_mask=False,
    )["input_ids"]
    input_ids = prefix_tokens + tokens + suffix_tokens
    return input_ids


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

token_yes = tokenizer.convert_tokens_to_ids("yes")
token_no = tokenizer.convert_tokens_to_ids("no")

query = "Implement a feature branch workflow with PR checks."
candidates = [
    {
        "name": "moai-foundation-git",
        "desc": "Git workflow conventions",
        "body": "# Git Workflow ...",
    },
    {
        "name": "concurrency-control",
        "desc": "Mutex patterns for CI",
        "body": "# Concurrency Control ...",
    },
]

scores = []
for cand in candidates:
    prompt = format_rerank_prompt(cand["name"], cand["desc"], cand["body"], query)
    input_ids = build_qwen_reranker_inputs(tokenizer, prompt)
    input_ids = torch.tensor([input_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
    score = (logits[:, token_yes] - logits[:, token_no]).item()
    scores.append(score)

best_idx = max(range(len(scores)), key=lambda i: scores[i])
print(best_idx, scores)
```

## Citation

If you use this model, please cite the SkillRouter paper once the preprint is public.

```bibtex
@misc{zheng2026skillrouterskillroutingllm,
      title={SkillRouter: Skill Routing for LLM Agents at Scale}, 
      author={YanZhao Zheng and ZhenTao Zhang and Chao Ma and YuanQiang Yu and JiHuai Zhu and Yong Wu and Tianze Xu and Baohua Dong and Hangcheng Zhu and Ruohui Huang and Gang Yu},
      year={2026},
      eprint={2603.22455},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.22455}, 
}
```
