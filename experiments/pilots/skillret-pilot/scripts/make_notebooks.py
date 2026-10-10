"""Regenerate exactly two thin notebooks without cached output or live opt-ins."""

from pathlib import Path
import nbformat as n

ROOT = Path(__file__).resolve().parents[1]


def create(name, cells):
    book = n.v4.new_notebook(
        cells=[
            n.v4.new_markdown_cell(text) if kind == "md" else n.v4.new_code_cell(text)
            for kind, text in cells
        ]
    )
    book.metadata.kernelspec = {
        "display_name": "Python 3 (project .venv)",
        "language": "python",
        "name": "python3",
    }
    book.metadata.language_info = {"name": "python", "version": "3.12"}
    n.validate(book)
    n.write(book, ROOT / name)


create(
    "01_data_and_retrieval.ipynb",
    [
        (
            "md",
            "# 1. Data and retrieval\n\n## tl;dr\nThe executed full-document BM25 baseline reached Recall@5 **0.516667** over **200 queries / 6,006 skills**. This is retrieval, not skill execution. Saved evidence is reused with checksum validation; shared functions own computation.",
        ),
        (
            "md",
            "## Context & Methods\n\nFixed seed 42 and immutable SkillRet revision. Full-document BM25 follows the pinned CPU source path; capped-metadata BM25/BGE form a separate matched comparison.\n\n### Key assumptions\nSynthetic queries, public skills, incomplete production validity. Metadata cap: 1,024 characters then 128 BGE tokens; seven vector queries exceed 512 encoder tokens. No prompts are tuned on the reserved final IDs.",
        ),
        (
            "code",
            "from pathlib import Path\nfrom pilot import ROOT, load_data, inspect_data, baseline, read_json, rows\nfrom trial import vectors\nimport pandas as pd",
        ),
        ("md", "## Data\n### 1. Verify real inputs and disjoint reservations"),
        (
            "code",
            "summary = inspect_data()\nassert set(read_json('manifests/pilot_query_ids.json')).isdisjoint(read_json('manifests/final_query_ids.json'))\n{'revision': summary['revision'], 'counts': summary['counts'], 'integrity': summary['checks']}",
        ),
        ("md", "### 2. Inspect consumed pilot examples only"),
        (
            "code",
            "examples = read_json('results/pilot_examples.json')\npd.DataFrame([{'query_id': x['query']['id'], 'canonical_query': x['query']['query'], 'positives_evaluator_only': len(x['query']['skill_ids'])} for x in examples])",
        ),
        ("md", "## Results\n### 3. Real BM25 number, with saved raw rankings"),
        (
            "code",
            "b = baseline(200)\nraw = [r for r in rows('results/baseline_rows.jsonl') if r['run_id'] == b['run_id']]\nindependent = sum(len({s for s, _ in r['ranking'][:5]} & set(r['evaluator_only_gold'])) / len(r['evaluator_only_gold']) for r in raw) / len(raw)\nassert abs(independent - b['metrics']['recall_at_5']) < 1e-12\n{'queries': len(raw), 'corpus_size': b['corpus_size'], 'recall_at_5': independent, 'recall_at_10': b['metrics']['recall_at_10']}",
        ),
        ("md", "### 4. Matched capped-metadata retrieval"),
        (
            "code",
            "embedding = vectors()\npd.DataFrame([{'method': name, 'queries': r['query_count'], **r['metrics']} for name, r in [('BM25 metadata', rows('results/metadata_bm25_summary.jsonl')[-1]), ('BGE-small metadata', rows('results/vector_baseline_summary.jsonl')[-1])]])",
        ),
        (
            "md",
            "## Takeaways\nThe pinned data and CPU retrieval path work. Matched-metadata BM25 Recall@5 is 0.605833 versus BGE-small 0.492500 in this exploratory sample. This does not establish a general winner. Preserve the 4,192 reserved final IDs. See reports/TRIAL_REPORT.md for limitations and distinct protocols.",
        ),
    ],
)

create(
    "02_deepagents_trial.ipynb",
    [
        (
            "md",
            "# 2. Native Deep Agents trial\n\n## tl;dr\nNative metadata loading, isolated selection and actual bounded SKILL.md reads are tested. Saved OpenRouter and local Ollama results are reported separately. Model calls require an explicit notebook opt-in; opening/executing this notebook normally spends no API credits.\n\nFresh setup: follow README.md through `pilot.py vectors` first. Missing live stages remain explicitly unrun. Recorded results are separate under reference_run/.",
        ),
        (
            "md",
            "## Context & Methods\nUse the real create_deep_agent / SkillsMiddleware / StateBackend. Main mode offers only select_skills; activation separately permits bounded native read_file.\n\n### Key assumptions\nPer-query nested gold-preserving libraries are a conditional distractor study. The same selector, prompts and metadata apply to all methods. Gold remains outside model state.",
        ),
        (
            "code",
            'import os\nfrom agent_runner import offline_check, Blocked\nfrom trial import smoke, run_trial, activation_smoke\nfrom pilot import ROOT, doctor, rows, read_json\nfrom handoff import report\nimport pandas as pd\nPAID_NOTEBOOK_RUNS = os.getenv("ALLOW_PAID_NOTEBOOK_RUNS") == "1"',
        ),
        ("md", "## Data\n### 1. Check local prerequisites without displaying secrets"),
        (
            "code",
            "status = doctor()\n{k: status[k] for k in ['python', 'env_file_present', 'key_present', 'paid_permission']}",
        ),
        (
            "md",
            "## Results\n### 2. Exercise the real native graph with scripted model control",
        ),
        ("code", "offline_check()"),
        ("md", "### 3. Native SKILL.md disclosure, explicitly offline"),
        ("code", "activation_smoke(live=False)"),
        (
            "md",
            "### 4. Explicitly opt-in paid stages\nA fresh notebook never enables spending. ALLOW_PAID_NOTEBOOK_RUNS=1 is additional to ALLOW_PAID_RUNS=1, a local key and the persistent guard.",
        ),
        (
            "code",
            "if not PAID_NOTEBOOK_RUNS:\n    paid_status = {'status': 'SKIPPED', 'reason': 'Paid notebook opt-in is disabled; live canary/trial not run in this notebook.'}\nelse:\n    try:\n        canary = smoke(live=True)\n        if not canary.get('canary_correct'):\n            raise Blocked('canary_failed')\n        comparison = run_trial(quick=True, live=True)\n        disclosure = activation_smoke(live=True)\n        paid_status = {'status': 'COMPLETED', 'comparison': comparison, 'disclosure': disclosure}\n    except Blocked as exc:\n        paid_status = {'status': 'BLOCKED', 'reason': str(exc)}\npaid_status",
        ),
        ("md", "### 5. Read report tables from persisted results"),
        (
            "code",
            "# The original single-model report requires a recorded trial catalogue.\nif (ROOT / 'results/trial.jsonl').exists():\n    report_summary = report()\n    single_model_table = pd.read_csv(ROOT / 'reports/scaling.csv')\nelse:\n    report_summary = {'status': 'NOT RUN', 'reason': 'No single-model trial in this workspace.'}\n    single_model_table = pd.DataFrame()\nsingle_model_table if not single_model_table.empty else report_summary",
        ),
        (
            "code",
            "from IPython.display import Image, display\nfigure = ROOT / 'reports/figures/candidate_recall.png'\nif figure.exists():\n    display(Image(filename=str(figure)))\nelse:\n    print('No local trial figure yet. Historical figures are in reference_run/.')",
        ),
        (
            "md",
            "### 6. Local Ollama alternative\nBy default this section reads saved results only. Set ALLOW_LOCAL_NOTEBOOK_RUNS=1 explicitly to invoke the bounded local runner. It uses installed models and no paid API; completed matching cases are reused.",
        ),
        (
            "code",
            'from local_runner import run as run_local, report as local_report\nif os.getenv("ALLOW_LOCAL_NOTEBOOK_RUNS") == "1":\n    local_run_status = run_local()\nelse:\n    local_run_status = {"status": "SKIPPED", "reason": "Local inference opt-in disabled; reading saved evidence only."}\nlocal_run_status',
        ),
        (
            "code",
            "if rows('results/ollama/canaries.jsonl'):\n    local_summary = local_report()\nelse:\n    local_summary = {'selection': [], 'activation': [], 'failures': [],\n                     'http_requests': 0, 'api_charge_usd': 0, 'status': 'NOT RUN'}\npd.DataFrame(local_summary['selection'])",
        ),
        (
            "code",
            '{"activation": local_summary["activation"], "failures": local_summary["failures"], "http_requests": local_summary["http_requests"], "api_charge_usd": local_summary["api_charge_usd"]}',
        ),
        (
            "md",
            "### 7. Five-model OpenRouter follow-up\nSame five pilot queries, N=50, listing/BM25/vector/domain-vector. Read saved canary outcomes, failed attempts, exact-match denominators, separate activation and the shared spending ledger. This cell makes no API calls.",
        ),
        (
            "code",
            "from scripts.report_openrouter import report as paid_report\nif any((ROOT / 'results/openrouter' / label / 'trial.jsonl').exists()\n       for label in ['qwen38-27b', 'qwen38-flash', 'deepseek41-flash', 'mimo-v2-6-pro', 'glm53-flash']):\n    paid_summary = paid_report()\nelse:\n    paid_summary = {'selection': [], 'models': [], 'budget': {},\n                    'estimated_cad_at_1_50': None, 'status': 'NOT RUN'}\npd.DataFrame(paid_summary['selection']).reindex(\n    columns=['model', 'method', 'attempted', 'valid', 'exact', 'candidate_recall', 'mean_logical_tokens'])",
        ),
        (
            "code",
            '{"models": paid_summary["models"], "budget": paid_summary["budget"], "estimated_cad": paid_summary["estimated_cad_at_1_50"]}',
        ),
        (
            "md",
            "### 8. Three-query scaling benchmark\nFive fixed OpenRouter models at N=50, 250 and 1,000. This bounded expansion reuses matching N=50 evidence and retains invalid outputs and provider exclusions. The following cells regenerate the report from saved data only.",
        ),
        (
            "code",
            "from scripts.report_scaling import report as scaling_report\nif (ROOT / 'manifests/openrouter_scaling.json').exists():\n    scaling_summary = scaling_report()\nelse:\n    scaling_summary = {'status': 'NOT RUN', 'summary': [], 'planned': 180,\n                       'saved': 0, 'reused_cases': 0, 'attempted': 0, 'valid': 0,\n                       'exact': None, 'new_attempts': 0, 'incremental_actual_usd': None,\n                       'budget': {}, 'failures': {}}\npd.DataFrame(scaling_summary['summary']).reindex(\n    columns=['model', 'N', 'method', 'attempted', 'valid', 'exact', 'candidate_recall'])",
        ),
        (
            "code",
            '{k: scaling_summary[k] for k in ["planned", "saved", "reused_cases", "attempted", "valid", "exact", "new_attempts", "incremental_actual_usd", "budget", "failures"]}',
        ),
        (
            "md",
            "## Takeaways\nA fresh offline run checks native middleware and bounded reads without claiming live accuracy. Live selection and local inference remain unrun until explicitly authorized. Compare any new results with reference_run/ only after checking protocol and denominator differences. No skill execution is measured.",
        ),
    ],
)
