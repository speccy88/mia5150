---
license: apache-2.0
base_model: Qwen/Qwen3-Reranker-0.6B
pipeline_tag: text-ranking
tags:
- sentence-transformers
- cross-encoder
- agent-skill-retrieval
---

# R3-Rerank-0.6B

The latest agent skill reranking model at the 0.6B scale. **R3-Reranker** is the cross-encoder
(rerank) stage of R3-Skill's two-stage retriever for query-conditional agent skill retrieval. It
scores each (query, skill) pair jointly, paired with
[R3-Embedding-0.6B](https://huggingface.co/tencent/R3-embedding-0.6b) for recall.

- 📄 Paper: [Skill Is Not Document: A Query-Conditional Benchmark and Two-Stage Retriever for LLM Agent Skill Routing](https://arxiv.org/abs/2606.03565)
- 💻 Code: [Tencent/R3-Skill](https://github.com/Tencent/R3-Skill)

## Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("tencent/R3-rerank-0.6b")
query = "I need to compose music"
skills = [  # The format is "name | description | skill_md"
    "music-composer | Composes original music | Creates music for various media formats ...",
    "music-lyricist | Writes lyrics for songs | Creates lyrics for various music genres ...",
    "music-editor | Edits and mixes music tracks | Provides audio editing and mixing services ...",
]
pairs = [(query, skill) for skill in skills]
scores = model.predict(pairs)
print(scores)
# [ 0.34937477 -1.7738094  -1.6604462 ]
```

## Citation

```bibtex
@inproceedings{r3skill2026,
  title  = {Skill Is Not Document: A Query-Conditional Benchmark and Two-Stage Retriever for LLM Agent Skill Routing},
  author = {Wang, Zifei and Wen, Wei and Ji, Qiang and Qiao, Ruizhi and Sun, Xing},
  year   = {2026},
  url    = {https://arxiv.org/abs/2606.03565},
}
```
