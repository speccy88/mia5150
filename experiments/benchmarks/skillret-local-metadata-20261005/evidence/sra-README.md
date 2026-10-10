---
license: mit
language:
  - en
task_categories:
  - question-answering
  - text-generation
pretty_name: SRA-Bench
size_categories:
  - 10K<n<100K
tags:
  - skill-retrieval
  - skill-retrieval-augmentation
  - llm-agents
  - agentic-ai
  - agent-skills
  - tool-use
  - retrieval-augmented-generation
  - rag
  - benchmark
  - reasoning
  - agent-memory
  - skill-learning
  - arxiv:2604.24594
---

# SRA-Bench: Skill Retrieval Augmentation for Agentic AI

[![arXiv](https://img.shields.io/badge/arXiv-2604.24594-b31b1b.svg)](https://arxiv.org/abs/2604.24594)
[![GitHub](https://img.shields.io/badge/GitHub-SR--Agents-181717.svg?logo=github)](https://github.com/oneal2000/SR-Agents)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/oneal2000/SR-Agents/blob/main/LICENSE)

**SRA-Bench** is a benchmark for studying **Skill Retrieval Augmentation (SRA)**: how LLM agents retrieve, select, and use external **skills** to solve capability-intensive tasks.

> **RAG retrieves knowledge. SRA retrieves capabilities.**

Modern LLM agents increasingly rely on external resources such as tools, APIs, memory, workflows, executable code, and reusable skills.  
SRA-Bench provides a large-scale benchmark for evaluating whether agents can retrieve the right skill, decide whether it is useful, and use it to improve final task performance.

---

## News

- **2026-04**: We released **SRA-Bench** and the official **SR-Agents** toolkit.
- Paper: [Skill Retrieval Augmentation for Agentic AI](https://arxiv.org/abs/2604.24594)
- Code: [github.com/oneal2000/SR-Agents](https://github.com/oneal2000/SR-Agents)

---

## Overview

SRA-Bench contains:

| Component | Number |
|---|---:|
| Test instances | **5,400** |
| Manually constructed gold skills | **636** |
| Total skills in retrieval corpus | **26,262** |
| Web-collected distractor skills | **25,626** |
| Source task families | **6** |

The benchmark covers both **single-skill** and **multi-skill** scenarios, including theorem application, logical reasoning, tool-use workflows, medical calculators, mathematical concepts, and software-library usage.

---

## Benchmark Composition

| Dataset | Capability Type | # Instances | # Skills | Evaluation |
|---|---|---:|---:|---|
| TheoremQA | Theorem application | 747 | 320 | Rule-based |
| LogicBench | Logical reasoning patterns | 760 | 19 | Rule-based |
| ToolQA | Tool-use workflows | 1,430 | 14 | Rule-based |
| MedCalc-Bench | Medical calculators | 1,100 | 55 | Rule-based |
| CHAMP | Mathematical concepts | 223 | 89 | Rule-based |
| BigCodeBench | Software libraries | 1,140 | 139 | Execution-based |

---

## What Is a Skill?

In SRA-Bench, a **skill** is a reusable capability artifact that can help an agent solve a class of tasks.

A skill may contain:

- task-solving instructions;
- formulas, rules, or reasoning procedures;
- tool-use workflows;
- executable code snippets;
- API or library usage guidance;
- structured domain-specific procedures.

A standard RAG document usually provides **information**.  
A skill provides **actionable capability**.

---

## Evaluation Dimensions

SRA-Bench supports decomposed evaluation of the full Skill Retrieval Augmentation pipeline:

### 1. Skill Retrieval

Can the agent retrieve the gold skill from a large noisy skill corpus?

### 2. Skill Incorporation

Can the agent decide whether and how to use the retrieved skill?

### 3. End-Task Execution

Does skill augmentation improve final task performance over the base model?

This design makes SRA-Bench useful for studying not only retrieval quality, but also whether LLM agents can make **need-aware augmentation decisions**.

---

## Files

```text
corpus/
  corpus.json

instances/
  theoremqa.json
  logicbench.json
  toolqa.json
  medcalc.json
  champ.json
  bigcodebench.json
```

- `corpus/corpus.json`: the full skill corpus with 26,262 skills.
- `instances/`: task instances with gold skill annotations and evaluation metadata.

---

## Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="WeihangSu/SRA-Bench",
    repo_type="dataset",
    local_dir="data/bench"
)
```

Or use the Hugging Face CLI:

```bash
huggingface-cli download WeihangSu/SRA-Bench \
  --repo-type dataset \
  --local-dir data/bench
```

---

## Official Toolkit

The official toolkit is available at:

```text
https://github.com/oneal2000/SR-Agents
```

Install:

```bash
git clone https://github.com/oneal2000/SR-Agents.git
cd SR-Agents
pip install -e .
```

The toolkit supports the complete pipeline:

```text
skill retrieval → skill-augmented inference → task evaluation
```

It also includes retrieval baselines, skill-use methods, evaluation scripts, and OpenAI-compatible inference backends.

---

## Recommended Use Cases

SRA-Bench can be used to study:

- skill retrieval for LLM agents;
- capability retrieval beyond knowledge retrieval;
- tool-use and workflow retrieval;
- agent memory and reusable skill libraries;
- need-aware augmentation;
- skill reranking and selection;
- robustness under noisy retrieved skills;
- comparison between full-skill injection, LLM selection, progressive disclosure, and oracle skill settings.

---

## Citation

If SRA-Bench is useful for your research, or if the findings in our paper help your work, we would be very grateful if you could cite our paper and star the GitHub repository.

Your support helps more researchers discover this resource and helps establish **Skill Retrieval Augmentation** as a shared research problem for scalable agent capabilities.

```bibtex
@article{su2026skill,
  title={Skill Retrieval Augmentation for Agentic AI},
  author={Su, Weihang and Long, Jianming and Ai, Qingyao and Tang, Yichen and Wang, Changyue and Tu, Yiteng and Liu, Yiqun},
  journal={arXiv preprint arXiv:2604.24594},
  year={2026}
}
```

---

## Links

- Paper: https://arxiv.org/abs/2604.24594
- Code: https://github.com/oneal2000/SR-Agents
- Dataset: https://huggingface.co/datasets/WeihangSu/SRA-Bench

---

## Contact

For questions, suggestions, or bug reports, please open an issue on GitHub:

https://github.com/oneal2000/SR-Agents/issues

We welcome feedback, new baselines, and extensions of SRA-Bench.