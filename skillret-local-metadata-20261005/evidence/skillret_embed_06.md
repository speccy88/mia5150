---
language:
- en
license: apache-2.0
library_name: sentence-transformers
tags:
- sentence-transformers
- embeddings
- skill-retrieval
- llm-agents
- contrastive-learning
base_model: Qwen/Qwen3-Embedding-0.6B
pipeline_tag: sentence-similarity
base_model_relation: finetune
datasets:
- ThakiCloud/SKILLRET
---

# SkillRet-Embedding-0.6B

[![arXiv](https://img.shields.io/badge/arXiv-2605.05726-b31b1b.svg)](https://arxiv.org/abs/2605.05726)

This is a **sentence-transformers** model fine-tuned for **AI agent skill retrieval**. Given a natural-language user request, the model retrieves relevant agent skills from a large skill library.

The model is fine-tuned from [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) on the SkillRet benchmark training split using contrastive learning (MultipleNegativesRankingLoss).

📄 **Technical report**: [SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents (arXiv:2605.05726)](https://arxiv.org/abs/2605.05726)

## Usage

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("ThakiCloud/SkillRet-Embedding-0.6B", trust_remote_code=True)

query_prompt = "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: "

queries = [
    query_prompt + "Help me set up a CI/CD pipeline for my Python project"
]
skills = [
    "ci-cd-setup | Configure continuous integration and deployment pipelines ...",
    "python-debugging | Debug Python applications using pdb and logging ...",
]

q_emb = model.encode(queries, normalize_embeddings=True)
s_emb = model.encode(skills, normalize_embeddings=True)

similarities = q_emb @ s_emb.T
print(similarities)
```

## Training Details

- **Base model**: Qwen3-Embedding-0.6B (0.6B parameters)
- **Training data**: SkillRet benchmark training split (127,190 query–skill pairs from 63,259 queries and 10,123 skills)
- **Loss**: MultipleNegativesRankingLoss (InfoNCE) with cross-GPU negative sharing
- **Hardware**: 4× NVIDIA B200 GPUs (DDP)
- **Effective batch size**: 384 (96 per device × 4 GPUs)
- **Max sequence length**: 8,192 tokens
- **Learning rate**: 2e-5
- **Epochs**: 1
- **Training time**: ~6 hours
- **Precision**: BF16

## Evaluation Results

Evaluated on the [SkillRet benchmark](https://arxiv.org/abs/2605.05726) test split (4,392 queries, 6,006 skills).

| Metric | @5 | @10 | @15 |
|--------|------|------|------|
| NDCG | 0.7902 | 0.8112 | 0.8187 |
| Recall | 0.8238 | 0.8774 | 0.9016 |
| Completeness | 0.7083 | 0.7894 | 0.8265 |
| MAP | 0.7409 | 0.7538 | 0.7570 |
| Precision | 0.2591 | 0.1401 | 0.0962 |

## Intended Use

This model is designed for retrieving agent skills given natural-language user requests. It is part of the [SkillRet benchmark](https://arxiv.org/abs/2605.05726) submission for evaluating skill retrieval systems for AI agents.

## Limitations

- Optimized for English-language queries and agent skills.
- Performance may vary on domains outside the SkillRet benchmark distribution.
- The model retrieves skills but does not execute them.

## Framework Versions

- Python: 3.10.12
- Sentence Transformers: 5.4.1
- Transformers: 5.5.4
- PyTorch: 2.7.1+cu128

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

Paper (measurements behind the quantized variants and the student frontier):
[Where Post-Training Quantization Breaks Text Embedders: A Measured Map Across Four Embedder Families, arXiv:2609.16391](https://arxiv.org/abs/2609.16391)

```bibtex
@article{han2026ptqembedders,
  title   = {Where Post-Training Quantization Breaks Text Embedders: A Measured Map Across Four Embedder Families},
  author  = {Han, Hyojung},
  journal = {arXiv preprint arXiv:2609.16391},
  year    = {2026}
}
```
