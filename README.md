# MIA 5150 Course Project Repository Template

Start with the [experiment index](EXPERIMENTS.md) for all preserved SkillRet studies,
their separate code/results folders, evaluation scope, and publication audit.

## Proposal PDF

Edit [project_proposal.tex](project_proposal.tex) and push to `main` to generate a
new PDF automatically. [Build LaTeX PDFs](https://github.com/speccy88/mia5150/actions/workflows/latex-pdf.yml)
compiles the custom proposal and the supplied NeurIPS example, including its
checklist. Download either PDF from the latest successful run's **Artifacts**.
See the [editing and build guide](docs/LATEX.md) for local builds and manual reruns.

## Executed skill-discovery pilot

The [SkillRet pilot](skillret-pilot/README.md) contains the runnable code, two
notebooks, tests, private local setup instructions, and recorded OpenRouter/Ollama
evidence. Start with its [short PDF report](skillret-pilot/reference_run/MIA5150_SkillRet_Short_Report.pdf)
and [workflow](skillret-pilot/ARCHITECTURE.md). The course template below is preserved.

The separate [full-catalog RunPod benchmark](skillret-full-catalog-runpod/README.md)
records five retrieval methods, the trained CLM head, per-query results, tests,
and a [benchmark PDF](skillret-full-catalog-runpod/REPORT.pdf).

The separate [JevK5 LoRA RunPod pilot](skillret-jevk5-runpod/README.md)
records the 2 October 2026 adaptation test, its initial timeout and completed
100-update resume, selected adapter, and offline-verifiable evidence. Recall@5
improved from 47.33% to 50.98% on 256 validation queries; this was a small
validation pilot, with no official test evaluation or skill execution.

The separate [CPU-only SkillRet benchmark](skillret-cpu-benchmark/README.md)
contains retrieval and binary-relevance adapter code, offline tests, and the
recorded 32-query comparison of 14 pipelines across four catalog sizes. It
includes compact result tables and a [CPU report](skillret-cpu-benchmark/reference_run/report/SkillRet_CPU_Benchmark.pdf),
with the sampling, timing, and historical-evidence limits explained in its README.

The [SkillRet reranker study](skillret-reranker-study/README.md) adds Jev and CLM
code plus their saved 200-query results. It also compares them with the pilot's
local and OpenRouter LLM results, with the differing protocols called out in its
README. Its recorded Jev and CLM Recall@5 scores are 70.92% and 38.17% on the
same BM25 top-20 candidate sets.

> **Course**: MIA 5150 — Generative AI and Large Language Models (Fall 2026)  
> **Course Weight**: 60% of Final Grade  
> **Submission Format**: NeurIPS 2026 LaTeX Template (PDF) + GitHub Repository Link  

---

## 📌 Project Overview

- **Project Title**: `[Insert Project Title]`
- **Project Category**: `[Select One: Benchmark and Empirical Study | Application and Capability Map | Algorithmic Innovation | Systematization of Knowledge (SoK)]`
- **Team Members**:
  - `Anonymous` (`[Student Number]`) — `[Role/Focus]`
  - `Anonymous` (`[Student Number]`) — `[Role/Focus]`
  - `Anonymous` (`[Student Number]`) — `[Role/Focus]`

### 🎯 The One-Sentence Claim
> *"We claim that **`[Claim]`**, and we will show it by measuring **`[Primary Metric]`** against **`[Baseline/Control]`** on **`[Dataset/Benchmark]`**."*

---

## 📋 Course Project Requirements & Checklist

### 1. Project Categories & Evidence Requirements
Each project must fit one of the four official categories:
- [ ] **Benchmark and Empirical Study**: Controlled comparison on an established task with a defined metric, published baseline, controlled variables, ablations, variance across seeds, and reported compute cost.
- [ ] **Application and Capability Map**: Evaluation on a specific domain/task where off-the-shelf models break, including a custom evaluation set or taxonomy of failure modes.
- [ ] **Algorithmic Innovation**: Modification to loss, architecture, sampling, or decoding strategy addressing a specific limitation, compared directly against the unmodified method under identical conditions.
- [ ] **Systematization of Knowledge (SoK)**: Structuring a modern Generative AI topic into a new framework/taxonomy, including trace/literature analysis and reproductions.

---

### 2. Timeline & Deliverable Schedule

| Week | Date | Deliverable / Milestone | Grade Weight | Key Requirements & Constraints |
| :--- | :--- | :--- | :---: | :--- |
| **Week 2** | 18 Sep | Team Registration | — | Register team of 2–3 on Brightspace. (Solo requires prior instructor approval). |
| **Week 3** | Pre-Proposal | 2-Day Trial Run | — | Download data, run minimal baseline end-to-end, and print **1 metric number**. Confirm memory, API cost, & package compatibility. |
| **Week 4** | **Fri 2 Oct** | **Project Proposal** | **15%** | **2–3 pages PDF** via NeurIPS 2026 LaTeX template. Must include team details, one-sentence claim, 2-page problem statement, trial run result, timeline, and backup plan. |
| **Weeks 5–7**| Oct | End-to-End Baseline | — | Establish working baseline pipeline. *(If no baseline by Week 7, project scope is too large).* |
| **Week 8** | 30 Oct | Reading Week | — | Draft Related Work section. |
| **Week 9** | **Fri 6 Nov** | **Progress Check-in** | **0%** | **0.5 page on Brightspace**. Non-graded status report: working metric, blockers, timeline/backup status. |
| **Weeks 9–11**| Nov | Main Experiments & Ablations | — | Conduct core experiments, ablations, and seed variations. Save checkpoints to external storage. |
| **Weeks 12–13**| 27 Nov / 4 Dec | **Project Presentation** | **20%** | **10-minute presentation + 5-minute Q&A**. All team members must speak. Focus on claim (first 2 min), approach, results, and limitations. |
| **Week 13** | **Fri 4 Dec** | **Final Report & Code Link** | **25%** | **Max 6 pages PDF** (NeurIPS 2026 LaTeX template) + repository link with full reproduction instructions. |

---

### 3. Hardware & Compute Constraints

Projects must strictly adhere to the resource budget (~15 GB GPU memory per job):
- **Allowed Hardware**: Google Colab (Free T4), Kaggle Notebooks (~30h/week quota), or Course JupyterHub shared GPUs.
- **Workable Workloads**:
  - LoRA / QLoRA fine-tuning on 1B to 7B parameter models (e.g., Llama/Qwen/Mistral).
  - Inference up to ~8B models in 4-bit quantization.
  - Training small LMs (<50M parameters) from scratch for specific ablation studies.
  - CLIP linear probes or prompt studies.
  - Vector retrieval over corpora up to ~10 GB.
- **Out of Scope (Prohibited due to compute limits)**:
  - Full pre-training of language models.
  - Full fine-tuning of models >1B parameters.
  - Running 30B+ parameter models locally.
  - High-resolution diffusion training from scratch.

---

### 4. Technical Submission Standards & LaTeX Requirements

- **LaTeX Template Required**: All written reports (Proposal & Final Report) **must** be compiled with the [NeurIPS 2026 LaTeX Template](https://overleaf.com/latex/templates/formatting-instructions-for-neurips-2026). (Word or Google Docs submissions will not be marked).
- **Report Structure (6 Pages Max)**:
  1. **Abstract**: Problem, approach, and headline numerical result.
  2. **Introduction**: Motivation, research question, and explicit list of contributions.
  3. **Related Work**: Organized by concept/idea (not chronological list).
  4. **Method**: Sufficient detail for complete independent reproduction.
  5. **Experiments & Results**: Setup (models, data, hyperparameters, seeds, hardware), tables, figures, ablations.
  6. **Analysis & Discussion**: Critical evaluation of model behavior and failure modes.
  7. **Limitations**: Honest assessment of constraints and unverified conditions.
  8. **Conclusion & Future Work**
  9. **References (BibTeX)**
  10. **Appendix**: Mandatory team contribution paragraph per member.

---

## 🛠️ Repository Structure & Reproduction Guide

### Directory Layout

```text
.
├── README.md                 # Project README and reproduction guide
├── requirements.txt          # Python environment dependencies
├── configs/                  # Experiment configuration files (YAML/JSON)
├── data/                     # Data processing scripts (or download instructions)
│   └── README.md             # Dataset acquisition & processing guide
├── src/                      # Core source code
│   ├── models/               # Model architectures / wrappers / adapters
│   ├── data_loader.py        # Dataset loading & preprocessing pipelines
│   ├── trainer.py            # Training / fine-tuning execution loops
│   └── evaluate.py           # Metric calculation & evaluation scripts
├── notebooks/                # Exploratory analysis & demo notebooks
├── scripts/                  # Shell scripts for running full experiment pipelines
│   ├── run_trial.sh          # 2-Day trial run script
│   ├── run_baseline.sh       # Baseline model evaluation
│   └── run_experiments.sh    # Main experimental matrix & ablations
└── results/                  # Saved figures, tables, and metric logs
```

---

## 🚀 Quickstart & Reproduction Steps

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install exact dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
```bash
# Download and preprocess dataset (or run data preparation script)
python src/data_loader.py --prepare_data
```
*Note: If data cannot be distributed directly, follow instructions in `data/README.md`.*

### 3. Reproducing the Baseline (Trial Run)
```bash
# Run baseline pipeline on 1% subsample to verify pipeline
python src/evaluate.py --config configs/baseline.yaml --subsample 0.01
```

### 4. Reproducing Main Results & Figures
```bash
# Run core experiment suite across fixed seeds
python src/trainer.py --config configs/main_experiment.yaml --seeds 42,123,456

# Generate paper figures and summary tables
python src/evaluate.py --generate_plots --output_dir results/
```

---

## 🧪 Experimental Protocol & Random Seeds

To ensure complete scientific reproducibility:
- **Fixed Seeds**: All stochastic operations (data splits, model initialization, sampling) use random seeds: `[e.g., 42, 123, 456]`.
- **Compute Accounting**: Each training and evaluation run records GPU wall-clock time and memory usage.
- **Checkpoints**: Intermediate checkpoints are automatically synchronized to persistent storage (Google Drive / S3).

---

## 🛡️ Fallback Backup Plan (Risk Management)

- **Primary Risk 1**: `[e.g., GPU Out-Of-Memory during fine-tuning / High API latency]`
  - *Mitigation*: Fallback to 1% dataset subsample or switch from 7B QLoRA to 1B model with 4-bit quantization.
- **Primary Risk 2**: `[e.g., Dataset access restriction / API rate limits]`
  - *Mitigation*: Switch to open-source backup dataset `[Dataset Name]`.
- **Reduced Backup Project**: If main plan fails by Week 9 check-in, the team will pivot to: `[Description of scaled-down backup project]`.

---

## 👥 Team Contribution Matrix & AI Usage

### Member Contributions
| Member Name | Specific Contributions & Responsibilities |
| :--- | :--- |
| **`[Member 1]`** | `[e.g., Dataset preparation, baseline implementation, evaluation pipeline]` |
| **`[Member 2]`** | `[e.g., Model fine-tuning script, QLoRA integration, loss function modifications]` |
| **`[Member 3]`** | `[e.g., Experiment execution, hyperparameter sweep, ablation plots, LaTeX drafting]` |

### AI Tools Declaration
In accordance with course policies:
- AI coding assistants (e.g., GitHub Copilot, Claude, ChatGPT) were used for code debugging, syntax assistance, and paper reading.
- All core experimental design, hypothesis formulation, code logic verification, and paper drafting were conducted and thoroughly understood by team members.

---

## 📜 Evaluation Rubrics Summary

### Final Report Grading Matrix (25%)
- **Technical Quality (30%)**: Sound methodology, competent implementation, correct model handling.
- **Evaluation & Rigour (25%)**: Suitable metrics, controlled comparisons, ablations, seed variance, contamination checks.
- **Original Contribution (25%)**: Depth of insight, clear takeaway messages, novel findings/analysis.
- **Communication (20%)**: Paper structure, figure clarity, concise prose, correct citations.

### Presentation Grading Matrix (20%)
- Quality of Evidence (30%), Clarity of Claim (25%), Technical Depth (20%), Delivery & Timing (15%), Q&A Response (10%).

## Public repository privacy

This repository uses an anonymous publication snapshot with a single `main` branch. Personal team information and credential-shaped examples have been redacted; recorded benchmark results remain available. See [publication privacy and safe updates](docs/PUBLICATION-PRIVACY.md) and the [16-study benchmark catalog](docs/SkillRet-Benchmark-Catalog.md).

Project contributors: `deepujoy88`, `speccy88`, and `vsimard`.
