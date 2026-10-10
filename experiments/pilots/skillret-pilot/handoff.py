"""Generate small reports/figures from saved evidence and an allowlisted ZIP."""

from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import zipfile
from pilot import ROOT, read_json, save_json, rows
from agent_runner import Budget


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    with (ROOT / path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)


def mean(values):
    return sum(values) / len(values) if values else None


def report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    local_note = ""
    paid_note = ""
    if (ROOT / "scripts/report_openrouter.py").exists() and (
        ROOT / "manifests/openrouter/z-ai-glm-5.3-flash.json"
    ).exists():
        from scripts.report_openrouter import report as paid_report

        paid = paid_report()
        paid_note = "**Current follow-up:** See [five-model OpenRouter report](OPENROUTER_MODELS.md) for live model tests and costs. The original configuration below is preserved as historical offline evidence. Any spending ledger total below is project-wide, including later model tests."
    if (ROOT / "manifests/openrouter_scaling.json").exists():
        from scripts.report_scaling import report as scaling_report

        scaling_report()
        paid_note = "**Latest benchmark:** See [OpenRouter scaling results](OPENROUTER_SCALING.md) for N=50/250/1000 across five models and the original [N=50 pilot](OPENROUTER_MODELS.md). The original configuration below is preserved as historical evidence; its spending ledger is project-wide."
    if (ROOT / "results/ollama/canaries.jsonl").exists():
        from local_runner import report as local_report

        local = local_report()
        local_note = f"User-authorized local Ollama follow-up: **{sum(r['valid'] for r in local['selection'])} valid real selector decisions** and **{sum(r['native_reads'] for r in local['activation'])} native model-requested reads** are recorded separately in [OLLAMA_TRIAL_REPORT](OLLAMA_TRIAL_REPORT.md). The OpenRouter-only status and costs below retain their original scope."
    data = read_json("results/dataset_summary.json")
    baseline = rows("results/baseline_summary.jsonl")[-1]
    vb = rows("results/vector_baseline_summary.jsonl")[-1]
    mb = rows("results/metadata_bm25_summary.jsonl")[-1]
    embedding = read_json("manifests/embedding.json")
    raw = rows("results/trial.jsonl")
    # Last configuration is the current executed trial; never mix configurations.
    run_id = raw[-1]["run_id"] if raw else None
    distinct = {}
    for r in raw:
        if r["run_id"] != run_id:
            continue
        if (
            r["case_key"] not in distinct
            or r["mode"] == "live"
            or distinct[r["case_key"]]["mode"] != "live"
        ):
            distinct[r["case_key"]] = r
    trial = list(distinct.values())
    live = [r for r in trial if r["mode"] == "live"]
    summary = []
    strata = []
    prompt_rows = []
    for n in [50, 250, 1000]:
        for strategy in ["listing", "bm25", "vector", "domain_vector"]:
            group = [r for r in trial if r["N"] == n and r["strategy"] == strategy]
            valid = [r for r in group if r["status"] == "valid" and r["mode"] == "live"]
            attempted = [
                r
                for r in group
                if r.get("attempt_count", 0) > 0 and r["mode"] == "live"
            ]
            first = [r["candidate_recall"] for r in group if "candidate_recall" in r]
            row = {
                "N": n,
                "strategy": strategy,
                "planned_cases": 20,
                "saved_cases": len(group),
                "retrieval_cases": len(first),
                "candidate_recall": mean(first),
                "candidate_completeness": mean(
                    [
                        r["candidate_completeness"]
                        for r in group
                        if "candidate_completeness" in r
                    ]
                ),
                "live_attempted": len(attempted),
                "valid_live": len(valid),
                "conditional_exact_match": mean(
                    [r["selection"]["exact_set_match"] for r in valid]
                ),
                "conditional_f1": mean([r["selection"]["f1"] for r in valid]),
                "service_reliability": len(valid) / len(attempted)
                if attempted
                else None,
                "planned_live_coverage": len(valid) / 20,
                "mean_logical_tokens": mean(
                    [
                        r["logical_usage"]["input_tokens"]
                        + r["logical_usage"]["output_tokens"]
                        for r in valid
                    ]
                ),
                "conservative_context_exclusions": sum(
                    not r["effective_prompt_preflight"]["conservative_context_fits"]
                    for r in group
                    if "effective_prompt_preflight" in r
                ),
            }
            summary.append(row)
            for kind in ["single", "multiple"]:
                g = [
                    r
                    for r in group
                    if (len(r["evaluator_only_gold"]) == 1) == (kind == "single")
                ]
                strata.append(
                    {
                        "N": n,
                        "strategy": strategy,
                        "gold_group": kind,
                        "queries": len(g),
                        "candidate_recall": mean(
                            [
                                r["candidate_recall"]
                                for r in g
                                if "candidate_recall" in r
                            ]
                        ),
                        "valid_live": sum(
                            r["mode"] == "live" and r["status"] == "valid" for r in g
                        ),
                    }
                )
            estimates = [
                r["effective_prompt_preflight"]
                for r in group
                if "effective_prompt_preflight" in r
            ]
            prompt_rows.append(
                {
                    "N": n,
                    "strategy": strategy,
                    "native_prompt_checks": len(estimates),
                    "mean_input_tokens_estimated": mean(
                        [
                            e["input_tokens_estimated"]
                            for e in estimates
                            if "input_tokens_estimated" in e
                        ]
                    ),
                    "max_input_tokens_estimated": max(
                        [
                            e["input_tokens_estimated"]
                            for e in estimates
                            if "input_tokens_estimated" in e
                        ],
                        default=None,
                    ),
                    "mean_byte_token_upper_bound": mean(
                        [e["input_token_upper_bound"] for e in estimates]
                    ),
                    "context_exclusions_by_conservative_bound": row[
                        "conservative_context_exclusions"
                    ],
                }
            )
    write_csv("reports/scaling.csv", summary)
    write_csv("reports/single_multi.csv", strata)
    write_csv("reports/prompt_estimates.csv", prompt_rows)
    base_rows = [
        {
            "method": name,
            "representation": rep,
            "queries": r["query_count"],
            "corpus_size": r["corpus_size"],
            **r["metrics"],
        }
        for name, rep, r in [
            ("BM25", "full name | description | skill_md", baseline),
            ("BM25", "common capped metadata", mb),
            ("BGE-small", "common capped metadata; 512-token query limit", vb),
        ]
    ]
    write_csv("reports/retrieval_baselines.csv", base_rows)
    (ROOT / "reports/figures").mkdir(exist_ok=True)
    charts = [
        (
            "selection_exact_match",
            "conditional_exact_match",
            "Selection exact-set match (valid live decisions)",
        ),
        (
            "logical_tokens",
            "mean_logical_tokens",
            "Mean total logical LLM tokens per workflow",
        ),
        (
            "candidate_recall",
            "candidate_recall",
            "Mean candidate recall (retrieval only)",
        ),
    ]
    for filename, metric, title in charts:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.89)
        points = 0
        for strategy in ["listing", "bm25", "vector", "domain_vector"]:
            g = [
                r
                for r in summary
                if r["strategy"] == strategy and r[metric] is not None
            ]
            if g:
                ax.plot(
                    [r["N"] for r in g],
                    [r[metric] for r in g],
                    marker="o",
                    label=strategy,
                )
                points += len(g)
        ax.set(
            xlabel="Skills in each per-query nested library (N)",
            title=title,
            xticks=[50, 250, 1000],
        )
        ax.set_xlim(0, 1080)
        if filename != "logical_tokens":
            ax.set(ylim=(0, 1.05), ylabel="Fraction")
        else:
            ax.set(ylabel="Provider-reported tokens")
        if not points:
            ax.text(
                0.5,
                0.53,
                "NO LIVE OBSERVATIONS\n0 / 240 planned selector decisions\nLocal key and paid permission missing",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_yticks([])
        else:
            ax.legend()
        note = (
            "Domain-vector omitted: paid router unrun. n=20 queries per point."
            if filename == "candidate_recall"
            else "Missing points are unrun, not zero accuracy or zero tokens."
        )
        fig.text(0.5, 0.035, note, ha="center", fontsize=9)
        fig.savefig(ROOT / f"reports/figures/{filename}.png", dpi=150)
        plt.close(fig)
    budget = Budget()
    budget.path.touch(exist_ok=True)
    costs = budget.totals()
    fx = float(
        os.getenv(
            "CAD_PER_USD",
            str(read_json("pilot.defaults.json")["budget"]["cad_per_usd"]),
        )
    )
    tests = sorted((ROOT / "evidence").glob("pytest-*.log"))
    test_status = tests[-1].read_text().strip().split("\n")[-1] if tests else "not run"
    offline_reads = rows("evidence/activation_offline.jsonl")
    offline_unique = {r["query_id"]: r for r in offline_reads}
    live_reads = rows("results/activation.jsonl")
    native = read_json("evidence/native_offline_check.json")
    verified = len(native["metadata"]["calls"]) == 1
    run_spec = read_json(f"manifests/runs/{run_id}.json") if run_id else {}
    context_total = sum(r["conservative_context_exclusions"] for r in summary)
    valid_count = sum(r["valid_live"] for r in summary)
    attempted_count = sum(r["live_attempted"] for r in summary)
    summary_data = {
        "dataset_revision": data["revision"],
        "trial_run_id": run_id,
        "full_document_bm25": baseline["metrics"],
        "planned_selector_cases": 240,
        "saved_cases": len(trial),
        "valid_live": valid_count,
        "attempted_live": attempted_count,
        "conservative_context_exclusions_offline": context_total,
        "budget": costs,
        "cad_per_usd": fx,
        "estimated_cad_actual": costs["actual_usd"] * fx,
        "offline_native_read_cases": len(offline_unique),
        "live_activation_cases": len(live_reads),
        "tests": test_status,
        "assessment": (
            "GO: small live feasibility demonstrated"
            if valid_count and live_reads
            else "CONDITIONAL GO: offline feasibility established; live acceptance incomplete"
        ),
    }
    save_json("reports/summary.json", summary_data)
    table = "\n".join(
        f"| {r['method']} | {r['representation'].replace('|', '&#124;')} | {r['recall_at_5']:.4f} | {r['recall_at_10']:.4f} | {r['completeness_at_5']:.4f} |"
        for r in base_rows
    )
    scaling = "\n".join(
        f"| {r['N']} | {r['strategy']} | {r['retrieval_cases']} | {r['candidate_recall']:.4f} |"
        for r in summary
        if r["candidate_recall"] is not None
    )
    estimates = "\n".join(
        f"| {r['N']} | {r['strategy']} | {r['mean_input_tokens_estimated']:.0f} | {r['context_exclusions_by_conservative_bound']}/{r['native_prompt_checks']} |"
        for r in prompt_rows
        if r["mean_input_tokens_estimated"] is not None
    )
    lead = (
        "Real retrieval and native Deep Agents discovery work locally. The local key and ALLOW_PAID_RUNS=1 were absent, so the live stages remain blocked. This is a blocked pilot, not the successful live pilot defined in CODEX_TASK.md."
        if not attempted_count
        else "The saved result tables below show completed live cases, failures, context exclusions and their denominators. Native disclosure is a separate integration check, never task execution success."
    )
    report_text = f"""# MIA5150 SkillRet trial — saved experimental evidence

{paid_note}

{local_note}

**Original OpenRouter protocol: {summary_data['assessment']}.** {lead}

## First result and data

The first experiment, before any API spending, obtained **BM25 Recall@5 = {baseline['metrics']['recall_at_5']:.6f} (51.67%)**, a macro average over **200 fixed queries against all 6,006 skills**. Recall@10 = {baseline['metrics']['recall_at_10']:.6f}; Completeness@5 = {baseline['metrics']['completeness_at_5']:.3f} (75/200 queries). All positive labels count, including multi-skill queries. This is neither selection accuracy nor skill-execution success.

SkillRet revision `{data['revision']}` has {data['counts']['skills']:,} skills, {data['counts']['queries']:,} queries and {data['counts']['qrels']:,} qrels in test. Referential integrity, duplicate-ID and query/qrel disagreement checks passed. There are {data['duplicate_names']} duplicate readable names, handled with unique s-UUID frontmatter. No missing name/description/body/query text was observed. Observed source licenses: MIT 5,270; Apache-2.0 736. Dataset metadata and queries are Apache-2.0; retain original source attribution. See the [pinned card](../evidence/dataset_card.md), [inspection](../results/dataset_summary.json) and [manifest](../manifests/dataset.json).

The 200-query sample uses seed 42; 20 interleaved single/multiple-positive cases support the scaling pilot. The remaining 4,192 IDs are disjoint and reserved. Their schemas/labels were audited for integrity but their texts were not used for prompts, examples, embeddings or tuning. Public GitHub skills and synthetic requests do not establish production behavior; the corpus is heavily skewed toward Software Engineering.

## Retrieval protocols and results

All rows below use the same 200 queries / 6,006 documents. The full-document and capped-metadata rows are different representations and must not be presented as a method-only comparison.

| Retriever | Representation | Recall@5 | Recall@10 | Completeness@5 |
|---|---|---:|---:|---:|
{table}

Full-document BM25 faithfully ports the CPU path in `skillret/eval.py:build_skill_text, eval_bm25`, Git commit `0cc508e6bfcfff1fe6352d90ff5957ba38a64bba`: full name/description/body, bm25s English stopwords, lowercase/default two-character regex, no stemming, Lucene BM25 k1=1.5/b=0.75, positive-score hits. Installed bm25s is 0.3.11; upstream lock has 0.3.3. Local metrics match the pinned evaluator on three known-answer controls (including empty and multi-positive rankings). MRR is a local addition. This is a **partial-protocol subset run, not reproduction of a published full benchmark score**. Source snapshot and parity evidence are included.

Matched metadata is original name + description, capped once to 1,024 characters then 128 BGE tokens (including special tokens), used identically by listing/BM25/vector/native metadata. {embedding['skills_character_capped']} skills hit the character cap; {embedding['skills_token_capped_after_char_cap']} hit the subsequent token cap. BGE-small-en-v1.5 revision `{embedding['revision']}` runs on CPU, normalized 384-dimensional vectors, exact dot products, fixed documented query prefix. **{len(embedding['queries_truncated_at_512_tokens'])}/200 query embeddings were truncated at 512 tokens**, with IDs saved in the embedding manifest. LLM query text is never truncated. Full-document baseline took {baseline['wall_seconds']:.3f} seconds; the recorded embedding stage took {embedding['wall_seconds']:.3f} seconds, including setup/download and encoding on this machine; neither is a forecast.

## Conditional distractor scaling

Current run: `{run_id}`. Each query has nested gold-preserving libraries N=50/250/1000, random distractors and fixed ID-derived display order. This is a conditional scaling study, not a shared production catalogue or official full-corpus benchmark. Gold is used only by the sampler/evaluator. Retrieval uses only common metadata; the native StateBackend holds only candidates. Domain routing must see only canonical query text and the fixed six-major taxonomy, never query k/names/labels.

| N | Method | Real retrieval cases | Candidate recall |
|---:|---|---:|---:|
{scaling}

Listing coverage is 1 by construction. Domain-vector results require a real LLM router; absent rows are unrun, never a gold-domain substitute or global fallback. See [single/multiple diagnostics](single_multi.csv). There are **{valid_count}/240 valid live selector cases, {attempted_count} attempted live cases**. Conditional semantic accuracy and service reliability use their explicit per-method denominators in scaling.csv; they are not estimable with zero attempts. Missing selector results are not zeros. Offline scripted replies are excluded from semantic metrics.

## Native integration, disclosure and context

Real `create_deep_agent` and `SkillsMiddleware` from Deep Agents 0.7.19 loaded canonical SKILL.md frontmatter from an isolated StateBackend. The metadata-only graph removes filesystem tools via supported profile exclusions, disables the general-purpose subagent and summarization, and checks the actual effective schemas at every model boundary. Native loading, unique names, absence of body-only/noncandidate sentinels and terminal one-call handling passed. No host filesystem is mounted; skills are never executed.

Actual native read_file disclosure ran on **{len(offline_unique)} cases (two real SkillRet, three hand-authored fixtures)** using a scripted model. These verify native reading only. At most 100 lines / 12,000 returned characters per read, at most two reads and four model calls; truncation is explicit. Live activation has {len(live_reads)} recorded cases; zero means unrun/blocked. See [offline disclosure traces](../evidence/activation_offline.jsonl).

Effective native prompts for the real scaling candidates were inspected using a scripted empty terminal response. This is prompt preflight evidence only. Qwen's pinned tokenizer estimates below are not billed provider counts. Cost reserves use a conservative UTF-8 byte bound plus framing; context checks use the corresponding tokenizer estimate plus 10% and 1,024 tokens of framing margin (byte fallback for other models). That context estimate excludes {context_total} cases; this is not proof the provider would reject all of them. No catalogue is truncated, summarized or silently reduced.

| N | Method | Mean estimated input tokens | Conservative context exclusions |
|---:|---|---:|---:|
{estimates}

## OpenRouter status, costs and failures

Public metadata currently permits pinning `{run_spec.get('llm',{}).get('model','unavailable')}` to `{run_spec.get('llm',{}).get('provider_tag','unavailable')}`, fallback disabled. It advertises tools and auto tool choice, not forced terminal function selection. Temperature=0; maximum selector output=512, router output=96. Reasoning effort none is requested; actual behavior requires a live response. Wall time is workflow latency, never pure model generation throughput. Actual usage and reliability require completed live cases; absent values remain unavailable.

The native ChatOpenRouter adapter required small typed-SDK compatibility handling: provider max-price fields must be decimal strings, reasoning uses effort=none, and raw sanitized HTTP response capture preserves the provider field dropped by the SDK response schema. HTTP MockTransport tests inspect actual serialized requests, usage and completion IDs; they do not establish live compatibility. Explicit SDK retries=None plus max_retries=0 yields one transport attempt; transient failures are retained without automatic retry. Timeout is 90 seconds (90,000 ms at the SDK). A live canary remains required before any comparison.

Actual paid requests: **{costs['requests']}**. Actual returned cost: **USD {costs['actual_usd']:.6f}**; unresolved reserves: USD {costs['unresolved_reserved_usd']:.6f}. Estimated CAD equivalent of actual cost: **CAD {costs['actual_usd']*fx:.6f}**, using {fx:.2f} CAD/USD as a planning assumption, not a verified exchange rate. Persistent file locking, request caps, pre-request byte/output reserves, both USD/CAD ceilings and retained ambiguous-failure reserves protect the local runner. They do not control unrelated account activity or bank fees. No future total is quoted as measured spend.

[Known failures](../evidence/KNOWN_FAILURES.md) preserve initial parsing, middleware, tokenizer and adapter-test failures. Earlier run configurations are retained but excluded from current aggregates. Latest test log: **{test_status}**. Both notebooks must show executed outputs and a SKIPPED paid cell by default; notebook validation is in `evidence/notebook-execution.json`.

## Proposal decision and resume

The team has evidence for a narrow feasibility proposal. Live comparison claims require completed cases above; zero live cases means that milestone is incomplete. No result here supports a general superiority claim at scale.

Run `python3 setup_local_key.py` privately in this project, enter the key there and authorize the bounded trial. Then run `bash scripts/resume.sh`. The script runs guarded canary → five-query N=50 trial → bounded expansion → separate activation → tests/notebooks/report/package, stopping on nonzero status. Reopening notebooks never opts into paid calls; that requires a separate ALLOW_PAID_NOTEBOOK_RUNS=1 plus the normal key/permission/budget checks. A handoff needs Python 3.12, installed pinned dependencies, data/model downloads, a key and API credits for live stages; it is not an offline self-contained live benchmark.
"""
    (ROOT / "reports/TRIAL_REPORT.md").write_text(report_text)
    proposal = f"""# Proposal notes — exploratory evidence, not a completed paper

{local_note}

**Candidate claim:** Compare accuracy and cost of native metadata discovery with growing per-query skill libraries, and test whether single-domain routing improves efficiency or loses relevant skills. No method must win. No training, fine-tuning, skill execution, knowledge graph or agent-chain platform is involved.

**Category:** Controlled retrieval-plus-selection systems feasibility trial on public skills and synthetic queries. Offline retrieval and native middleware/tool isolation are established; current completed live selector cases: {valid_count}. Zero means the API-backed comparison remains blocked.

**Actual trial number:** Full-document BM25 macro Recall@5 {baseline['metrics']['recall_at_5']:.6f} on 200 reserved pilot queries / 6,006 skills, SkillRet `{data['revision']}`. Native full-corpus source port, local bm25s 0.3.11 rather than upstream 0.3.3; no published-score reproduction claim. Matched capped-metadata BM25 Recall@5 {mb['metrics']['recall_at_5']:.6f}, BGE-small {vb['metrics']['recall_at_5']:.6f}. See TRIAL_REPORT for input caps and embedding-query truncations.

**Call arithmetic:** At Q queries, three sizes and four strategies: at most 12Q selector requests + Q cached physical domain-route requests, before exclusions; a fresh standalone domain workflow instead incurs its router for every query/size. For Q=20 that is 240+20=260 planned requests, plus one canary and at most 20 activation model calls (five cases × four calls), maximum 281 under the 300-call cap. For Q=100 the analogous 1,301–1,321 request planning range exceeds this pilot's cap and requires a separately authorized design. These are arithmetic bounds, not observed calls. Observed calls = {costs['requests']}; when this is zero, observed average paid cost/latency and empirical larger-study dollar forecasts are unavailable. Otherwise use the per-stage saved logical usage before extrapolating. Context exclusions will reduce the runnable set. Do not use cached routing to make logical domain-vector tokens free.

**Limitations:** 20 exploratory conditional libraries cannot support p-values, production generalization or scaling superiority. Gold preservation simplifies retrieval coverage; random distractors may be easier than hard negatives. The corpus is English-dominant and Software Engineering-heavy; queries are synthetic; there are incomplete relevance/external-validity questions. 200 consumed query IDs are excluded from the 4,192 reserved final IDs. No prompts were tuned on the final set. Seven vector query embeddings truncate at the encoder limit. Public data may overlap model pretraining; this trial does not measure contamination.

**Small backup:** Keep full-corpus BM25 vs local BGE on fixed, matched metadata, then a smaller N=50 listing/selection study once the guarded canary works. Retain failures, unknown costs and context skips.

**Team roles (assign names):** dataset/provenance — TBD; integration/budget — TBD; evaluation/reproducibility — TBD; proposal/report — TBD.

**Verified sources:** [SkillRet dataset](https://huggingface.co/datasets/ThakiCloud/SKILLRET/tree/{data['revision']}), [pinned evaluator](https://github.com/ThakiCloud/SKILLRET/blob/0cc508e6bfcfff1fe6352d90ff5957ba38a64bba/skillret/eval.py), [Deep Agents skills](https://docs.langchain.com/oss/python/deepagents/skills), [BGE-small card](https://huggingface.co/BAAI/bge-small-en-v1.5), and installed adapter source/transport evidence. Suggested Graph of Skills, Li scaling paper, Skill Shadowing and SRA are **unreviewed**; no novelty or performance claims are adopted from them.
"""
    (ROOT / "reports/PROPOSAL_NOTES.md").write_text(proposal)
    return summary_data


ROOT_FILES = {
    "README.md",
    "ARCHITECTURE.md",
    "VALIDATION.md",
    "requirements-report.txt",
    "START_HERE.md",
    "AGENTS.md",
    "CODEX_TASK.md",
    "SOURCE_NOTES.md",
    "ACCEPTANCE_CHECKLIST.md",
    "PLAN.md",
    "pilot.defaults.json",
    "requirements.in",
    "requirements.lock.txt",
    "pilot.py",
    "agent_runner.py",
    "trial.py",
    "handoff.py",
    "reference_metrics.py",
    "setup_local_key.py",
    "local_runner.py",
    ".gitignore",
    ".gitattributes",
    "01_data_and_retrieval.ipynb",
    "02_deepagents_trial.ipynb",
}
DIRECTORIES = {
    "tests",
    "scripts",
    "prompts",
    "results",
    "manifests",
    "reports",
    "evidence",
    "reference_run",
    "validation",
}


def export_files():
    candidates = [ROOT / p for p in ROOT_FILES if (ROOT / p).is_file()]
    for directory in DIRECTORIES:
        for p in (ROOT / directory).rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            if any(
                x.startswith(".") or x == "__pycache__"
                for x in p.relative_to(ROOT).parts
            ):
                continue
            if p.suffix in {".pyc", ".lock"}:
                continue
            # Keep the one immutable reviewed reference archive, never arbitrary ZIPs.
            if (
                p.suffix == ".zip"
                and p.relative_to(ROOT).as_posix()
                != "reference_run/MIA5150_skillret_handoff.zip"
            ):
                continue
            if p.suffix == ".zip" and hashlib.sha256(p.read_bytes()).hexdigest() != (
                "871dba2b651449ef0315874c8489b5520b3d85889fb62cdad7bfe3336d4abd58"
            ):
                raise ValueError("Reference archive changed; review it before export")
            candidates.append(p)
    return sorted(set(candidates))


def package():
    from dotenv import load_dotenv

    load_dotenv(
        ROOT / ".env", override=False
    )  # Private read for redaction/secret scan; never exported.
    report()
    paths = export_files()
    key = os.getenv("OPENROUTER_API_KEY")
    manifest = []
    for p in paths:
        if p.is_symlink():
            raise ValueError("Symlinks forbidden in export")
        content = p.read_bytes()
        if key and key.encode() in content:
            raise ValueError("Secret scan blocked export: " + str(p.relative_to(ROOT)))
        if re.search(rb"sk-or-v1-[A-Za-z0-9_-]{12,}", content):
            raise ValueError(
                "Token pattern blocked export: " + str(p.relative_to(ROOT))
            )
        if re.search(
            rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=]{16,}",
            content,
        ):
            raise ValueError(
                "Private key block blocked export: " + str(p.relative_to(ROOT))
            )
        manifest.append(
            {
                "path": str(p.relative_to(ROOT)),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    # Archive audit is generated in memory to avoid self-referential checksums.
    audit = {
        "export_policy": "explicit allowlist; no symlinks/hidden cache/bytecode; only the SHA-verified reference ZIP is included",
        "secret_scan": "passed: current credential, OpenRouter token patterns and PEM private keys",
        "files": manifest,
        "notebooks": sum(p.suffix == ".ipynb" for p in paths),
    }
    if audit["notebooks"] != 2:
        raise ValueError("Export must contain exactly two notebooks")
    out = ROOT / "dist/MIA5150_skillret_handoff.zip"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        # Preserve earlier packages if manually rebuilding later.
        out.rename(
            out.with_name("MIA5150_skillret_handoff-" + str(time.time_ns()) + ".zip")
        )
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in paths:
            archive.write(p, p.relative_to(ROOT))
        archive.writestr("EXPORT_MANIFEST.json", json.dumps(audit, indent=2))
    with zipfile.ZipFile(out) as archive:
        assert archive.testzip() is None
        for item in manifest:
            assert (
                hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
            )
    result = {
        "zip": str(out),
        "files": len(manifest) + 1,
        "bytes": out.stat().st_size,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "secret_scan": "passed",
        "notebooks": 2,
    }
    save_json("evidence/package-verification.json", result)
    return result
