---
language:
- en
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3-Embedding-0.6B
pipeline_tag: feature-extraction
tags:
- embedding
- retrieval
- agent-tools
- skill-routing
- qwen3
- contrastive-learning
---

# SR-Emb-0.6B

`SR-Emb-0.6B` is a compact embedding model for skill retrieval in large LLM-agent skill registries. It is the encoder used in the SkillRouter retrieve-and-rerank pipeline and is fine-tuned from `Qwen/Qwen3-Embedding-0.6B` for full-text skill routing over approximately 80K skills.

## Model Summary

- Base model: `Qwen/Qwen3-Embedding-0.6B`
- Architecture: bi-encoder / dual-encoder retrieval model
- Input:
  - Query: task description with an instruction prefix
  - Document: full skill text formatted as `name | description | body`
- Output: L2-normalized dense embeddings for cosine similarity search
- Intended use: first-stage retrieval in a two-stage skill routing system

On the SkillRouter benchmark, the released checkpoint improves over the untuned 0.6B base model and outperforms larger embedding baselines in top-1 routing accuracy.

## Intended Uses

Use this model when you need to retrieve a small candidate set of relevant skills, tools, or plugin documents from a large registry before downstream reranking or execution.

Typical workflow:

1. Format user tasks as retrieval queries.
2. Encode all skills offline.
3. Encode the incoming query online.
4. Compute cosine similarity.
5. Return top-K candidates, typically `K=20` or `K=50`.

This model is not intended for text generation or direct answer synthesis.

## How to Use

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "pipizhao/SkillRouter-Embedding-0.6B"
QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch, device=last_hidden_states.device), seq_lens]


def encode(texts, tokenizer, model, max_length=4096):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        embs = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embs = F.normalize(embs, p=2, dim=1)
    return embs


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, padding_side="left")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

query = QUERY_INSTRUCTION + "Implement a feature branch workflow with PR checks."
skills = [
    "moai-foundation-git | Git workflow conventions | # Git Workflow ...",
    "concurrency-control | Mutex patterns for CI | # Concurrency Control ...",
]

query_emb = encode([query], tokenizer, model)
skill_embs = encode(skills, tokenizer, model)
scores = (query_emb @ skill_embs.T).squeeze(0)
ranked = torch.argsort(scores, descending=True)
print(ranked.tolist(), scores[ranked].tolist())
```

## Citation

If you use this model, please cite the SkillRouter paper once the public preprint is available.

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
