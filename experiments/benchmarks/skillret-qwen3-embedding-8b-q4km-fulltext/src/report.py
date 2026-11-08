"""Build tables, measured-performance summaries, PNG/SVG figures and report."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import STUDY, SIZES, atomic_json, bootstrap, file_hash, jsonl, latency_summary

R = STUDY / "results"
F = STUDY / "figures"
COLOR = "#2364AA"


def load(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def build():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    F.mkdir(exist_ok=True)
    audit = load(R / "input_audit.json", {})
    model = load(R / "model.json", {})
    fit = load(R / "feasibility.json", {})
    progress = load(R / "progress.json", {"stage": "not_started"})
    calls = jsonl(R / "embedding_calls.jsonl") if (R / "embedding_calls.jsonl").exists() else []
    telemetry = jsonl(R / "telemetry.jsonl") if (R / "telemetry.jsonl").exists() else []
    primary_telemetry = [r for r in telemetry if r.get("protocol_sha256") == file_hash(STUDY / "protocol.json")]
    scaling = [load(R / f"scaling-n{n}.json") for n in SIZES if (R / f"scaling-n{n}.json").exists()]
    full = load(R / "full_evaluation.json")
    full_ci = load(R / "full_confidence.json", {})
    cached_replay = load(R / "cache_replay.json")
    full_rows = jsonl(R / "full_rankings.jsonl") if full else []
    full_uncached = [r for r in full_rows if not r["cache_hit"]]
    full_cached = [r for r in full_rows if r["cache_hit"]]
    rows = [r for s in scaling for r in s["rows"]]
    write_csv(R / "scaling_per_query.csv", [{k: v for k, v in row.items() if k not in ("ranking", "scores")} for row in rows])
    document_calls = [r for r in calls if r["kind"] == "document"]
    doc_seconds = sum(r["latency_s"] for r in document_calls)
    doc_tokens = sum(r["prompt_eval_count"] for r in document_calls)
    first_document_start = (datetime.fromisoformat(document_calls[0]["utc"]) - timedelta(seconds=document_calls[0]["latency_s"])) if document_calls else None
    indexing_and_query_telemetry = [r for r in primary_telemetry if first_document_start and datetime.fromisoformat(r["utc"]) >= first_document_start]
    index_history = jsonl(R / "indexing_history.jsonl") if (R / "indexing_history.jsonl").exists() else []
    index_wall = sum(r["stage_wall_s"] for r in index_history)
    indexed_complete_stages = index_history[-1]["cumulative_skills"] if index_history else 0
    interruptions = [r for r in index_history if r.get("status") == "interrupted"]
    native_events = [r for r in calls if r.get("native_tokenizer_verified_count") is not None]
    previous_invocations = [load(p)["wall_seconds"] for p in (R / "attempts").glob("*/failure.json") if load(p).get("stage") == "full"]
    performance = {"document_uncached_calls": len(document_calls), "document_uncached_api_seconds": doc_seconds if document_calls else None,
                   "document_actual_runtime_tokens": doc_tokens if document_calls else None,
                   "index_wall_seconds_completed_stages": index_wall if index_history else None,
                   "unique_skills_in_completed_index_stages": indexed_complete_stages,
                   "skills_per_second_completed_stages": indexed_complete_stages / index_wall if index_wall else None,
                   "interrupted_index_wall_brackets": interruptions,
                   "native_tokenizer_verification_events": len(native_events),
                   "primary_total_invocation_wall_s": (progress.get("wall_seconds_this_invocation", 0) + sum(previous_invocations)) if progress.get("stage") == "completed" else None,
                   "chunk_embeddings_per_second_active": len(document_calls) / doc_seconds if doc_seconds else None,
                   "tokens_per_second_active": doc_tokens / doc_seconds if doc_seconds else None,
                   "document_indexing_wall_boundary": "indexing_stages.csv / indexing_history.jsonl; API time excludes tokenization, cache validation and persistence",
                   "cold_start_model_load_s": fit.get("cold_start_load_s"),
                   "cold_start_first_embedding_s": fit.get("cold_start_first_embedding_s"),
                   "primary_gpu_peak_sampled_mib": max((r["gpu_memory_mib"] for r in primary_telemetry), default=None),
                   "primary_indexing_query_gpu_peak_sampled_mib": max((r["gpu_memory_mib"] for r in indexing_and_query_telemetry), default=None),
                   "gpu_peak_sampled_mib": max((r["gpu_memory_mib"] for r in telemetry), default=None),
                   "system_peak_sampled_used_gib": max((r["system_used_bytes"] / 1024**3 for r in telemetry), default=None),
                   "system_min_available_gib": min((r["system_available_bytes"] / 1024**3 for r in telemetry), default=None),
                   "gpu_utilization_mean_percent": float(np.mean([r["gpu_utilization_percent"] for r in telemetry])) if telemetry else None,
                   "gpu_peak_temperature_c": max((r["gpu_temperature_c"] for r in telemetry), default=None),
                   "telemetry_sample_interval_s": 1, "paid_services_used": False,
                   "electricity_cost": None,
                   "query_embedding_uncached": latency_summary([r["query_embedding_uncached_s"] for r in rows]),
                   "similarity_search": latency_summary([r["similarity_search_s"] for r in rows]),
                   "total_retrieval_uncached": latency_summary([r["total_retrieval_s"] for r in rows]),
                   "cached_query_search": latency_summary([r["cached_query_search_s"] for r in rows]),
                   "verified_cache_replay": cached_replay,
                   "full_query_count": len(full_rows) if full else None,
                   "full_uncached_query_count": len(full_uncached) if full else None,
                   "full_cached_query_count": len(full_cached) if full else None,
                   "full_uncached_query_embedding": latency_summary([r["query_embedding_uncached_s"] for r in full_uncached]),
                   "full_similarity_search": latency_summary([r["similarity_search_s"] for r in full_rows]),
                   "full_uncached_retrieval_including_cache_persistence": latency_summary([r["total_retrieval_s_this_execution"] for r in full_uncached]),
                   "full_cached_retrieval": latency_summary([r["total_retrieval_s_this_execution"] for r in full_cached]),
                   "wall_seconds_last_completed_invocation": progress.get("wall_seconds_this_invocation")}
    atomic_json(R / "performance.json", performance)
    latency_rows = []
    for s in scaling:
        n = s["library_size"]
        for field in ("query_embedding_uncached_s", "similarity_search_s", "total_retrieval_s", "cached_query_search_s"):
            latency_rows.append({"model": "qwen3-embedding:8b-q4_K_M", "library_size": n,
                                 "measurement": field, **latency_summary([r[field] for r in s["rows"]])})
    write_csv(R / "latency_by_size.csv", latency_rows)
    baselines = []
    base_path = STUDY.parent / "skillret-embedding-jev-fusion/results/baseline_skillret_06b.json"
    base = load(base_path)
    baselines.append({"method": "SkillRet Embedding 0.6B full text", "queries": 4392,
                      "document_view": "stripped full text; 32768 tokens; two documents truncated",
                      "NDCG@10": base["evaluation"]["aggregate"]["NDCG@10"],
                      "Recall@10": base["evaluation"]["aggregate"]["Recall@10"],
                      "MRR@10": base["evaluation"]["aggregate"]["MRR@10"],
                      "comparison": "context only: chunking, context, whitespace, precision and runtime differ",
                      "source": str(base_path.relative_to(STUDY.parent.parent.parent)), "source_sha256": file_hash(base_path)})
    for key, label in (("public-full-bm25", "BM25 header"), ("public-full-qwen_embed_06", "Qwen Embedding 0.6B header"),
                       ("public-full-skillret_embed_06", "SkillRet Embedding 0.6B header"), ("public-full-rrf", "RRF60 header hybrid")):
        p = STUDY.parent / f"skillret-local-metadata-20261005/results/{key}.json"
        data = load(p)["metrics"]
        baselines.append({"method": label, "queries": data["query_count"], "document_view": "metadata/header; neural limit 1024",
                          "NDCG@10": data["aggregate"]["NDCG@10"], "Recall@10": data["aggregate"]["Recall@10"],
                          "MRR@10": None, "comparison": "context only: different document view; source reports MRR@5",
                          "source": str(p.relative_to(STUDY.parent.parent.parent)), "source_sha256": file_hash(p)})
    atomic_json(R / "contextual_baselines.json", baselines)
    write_csv(R / "contextual_baselines.csv", baselines)
    listing_context = []
    if full:
        listing_source = STUDY.parent / "skill-discovery-oauth-v1-20261103/results/summary_by_size.csv"
        for old in csv.DictReader(listing_source.open()):
            if int(old["N"]) in (50, 1000, 2000) and int(old["recorded"]) == int(old["valid"]) == 20:
                listing_context.append({"model": old["model"], "library_size": int(old["N"]),
                                        "questions": 20, "correct_top1_names": int(old["correct_top1"]),
                                        "top1_name_accuracy": int(old["correct_top1"]) / 20,
                                        "document_view": "name + description[:300] listing",
                                        "comparison": "context only; generation and name scoring differ from full-text ID retrieval",
                                        "source_sha256": file_hash(listing_source)})
        write_csv(R / "contextual_listing_baselines.csv", listing_context)
    figure_manifest = []

    def save(fig, name, sources):
        for suffix in ("png", "svg"):
            target = F / f"{name}.{suffix}"
            fig.savefig(target, dpi=300, bbox_inches="tight", metadata={"Creator": "Research Team"})
            if suffix == "svg":
                target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
            figure_manifest.append({"file": target.name, "sha256": file_hash(target), "source": sources,
                                    "source_sha256": {source: file_hash(STUDY / source) if (STUDY / source).exists() else None
                                                      for source in sources}})
        plt.close(fig)

    def axes(title, ylabel):
        fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
        ax.set_title(title, loc="left", weight="bold", pad=15)
        ax.set_xlabel("Library size (skills; logarithmic scale)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_xticks(SIZES, [f"{n:,}" for n in SIZES], fontsize=9)
        ax.set_xlim(40, 7600)
        ax.minorticks_off()
        ax.grid(axis="y", alpha=.2)
        fig.supxlabel("Qwen3 Embedding 8B Q4_K_M · 4,096 dimensions · 4,096-token context · 20 repeated questions per size", fontsize=10)
        return fig, ax

    for metric, name, title in (("Top1_ID_accuracy", "01_top1_accuracy", "Exact skill-ID Top-1 accuracy"),
                                ("Recall@10", "02_recall10", "Recall@10"),
                                ("NDCG@10", "03_ndcg10", "NDCG@10"), ("MRR@10", "04_mrr10", "MRR@10")):
        fig, ax = axes(title, "Query mean (0–1)")
        if metric in ("Top1_ID_accuracy", "Recall@10"):
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel("Accuracy" if metric == "Top1_ID_accuracy" else "Recall")
        values = {s["library_size"]: s for s in scaling}
        means = [values[n]["evaluation"]["aggregate"][metric] if n in values else np.nan for n in SIZES]
        ax.plot(SIZES, means, "o-", color=COLOR)
        for n, s in values.items():
            ci = bootstrap([r[metric] for r in s["rows"]])["per_column_95ci"][0]
            mean = s["evaluation"]["aggregate"][metric]
            ax.errorbar([n], [mean], yerr=[[mean - ci[0]], [ci[1] - mean]], capsize=4, color=COLOR)
        ax.set_ylim(-.025, 1.05)
        ax.text(.01, .02, "95% question-bootstrap intervals; absent points are unmeasured", transform=ax.transAxes, fontsize=9)
        save(fig, name, [f"results/scaling-n{n}.json" for n in values])
    fig, ax = axes("Measured query latency", "Seconds per query")
    for field, label, style, color in (("total_retrieval_s", "Uncached embed + search: median", "o-", COLOR),
                                       ("cached_query_search_s", "Cached query vector: search median", "s--", "#9A6700")):
        medians, p95s = [], []
        for n in SIZES:
            selected = [r[field] for r in rows if r["library_size"] == n]
            medians.append(float(np.median(selected)) if selected else np.nan)
            p95s.append(float(np.quantile(selected, .95)) if selected else np.nan)
        ax.plot(SIZES, medians, style, label=label, color=color)
        if field == "total_retrieval_s":
            ax.plot(SIZES, p95s, ":", color=color, label="Uncached p95 (observation spread)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    save(fig, "05_query_latency", ["results/latency_by_size.csv"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    if doc_seconds:
        ax[0].bar(["Chunk embeddings"], [len(document_calls) / doc_seconds], color=COLOR)
        ax[1].bar(["Active document API time"], [doc_seconds / 60], color=COLOR)
        ax[0].set_ylabel("Chunk embeddings / second")
        ax[1].set_ylabel("Minutes")
    else:
        for a in ax:
            a.text(.5, .5, "Indexing not measured", ha="center", transform=a.transAxes)
    fig.suptitle("Qwen3 8B Q4_K_M full-text indexing · API boundaries; cache and audit time separate")
    fig.supxlabel(f"{len(document_calls):,} uncached chunk calls · {doc_tokens:,} runtime tokens · context 4,096 · dimensions 4,096 · input batch 1", fontsize=9)
    save(fig, "06_indexing_throughput_time", ["results/embedding_calls.jsonl", "results/indexing_stages.csv"])
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    if telemetry:
        first = datetime.fromisoformat(telemetry[0]["utc"])
        elapsed = [(datetime.fromisoformat(r["utc"]) - first).total_seconds() / 60 for r in telemetry]
        plot_times, plot_memory = [], []
        for i, (minute, row) in enumerate(zip(elapsed, telemetry)):
            if i and (minute - elapsed[i - 1]) * 60 > 2.5:
                plot_times.append(np.nan)
                plot_memory.append(np.nan)
            plot_times.append(minute)
            plot_memory.append(row["gpu_memory_mib"])
        ax.plot(plot_times, plot_memory, color=COLOR, linewidth=1)
        ax.axhline(8192, color="#555555", linestyle="--", label="Physical VRAM: 8,192 MiB")
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "No GPU telemetry yet", ha="center", transform=ax.transAxes)
    ax.set_xlabel("Elapsed wall time from first sample (minutes; gaps = no telemetry)")
    ax.set_ylabel("Whole-device VRAM (MiB)")
    ax.set_ylim(bottom=0)
    ax.set_title("Measured GPU memory · RTX 2060 SUPER · Qwen3 8B Q4_K_M · one-second samples", loc="left")
    save(fig, "07_gpu_memory", ["results/telemetry.jsonl"])
    # There is no prior method with identical unstripped text and max-chunk protocol.
    # Separate protocol groups make the contextual character explicit.
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    labels = [b["method"] for b in baselines]
    values = [b["NDCG@10"] for b in baselines]
    if full:
        labels.insert(0, "Qwen3 8B Q4_K_M — complete chunks")
        values.insert(0, full["aggregate"]["NDCG@10"])
    ax.barh(labels[::-1], values[::-1], color=["#9A9A9A"] * (len(values) - bool(full)) + ([COLOR] if full else []))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recorded NDCG@10 · 4,392 queries / 6,006 skills")
    ax.set_title("Contextual results: document protocols differ", loc="left", weight="bold")
    fig.supxlabel("Not a matched model comparison: chunking, context, text view, precision and runtimes differ", fontsize=10)
    save(fig, "08_contextual_baselines", ["results/contextual_baselines.json", "results/full_evaluation.json"])
    atomic_json(F / "manifest.json", figure_manifest)
    summary_rows = [f"| {s['library_size']:,} | {s['evaluation']['aggregate']['Top1_ID_accuracy']:.3f} | {s['evaluation']['aggregate']['Recall@10']:.3f} | {s['evaluation']['aggregate']['NDCG@10']:.3f} | {s['evaluation']['aggregate']['MRR@10']:.3f} | {s['evaluation']['incorrect_count']} |" for s in scaling]
    pending = [n for n in SIZES if n not in [s["library_size"] for s in scaling]]
    report = ["# Qwen3 Embedding 8B Q4_K_M full-text SkillRet benchmark", "",
              f"Execution status: **{progress['stage']}**. {len(scaling)}/8 scaling conditions have saved results. "
              + ("The full 4,392-query evaluation is complete." if full else "The full 4,392-query evaluation has no completed aggregate yet."), "",
              "This study embeds the entire original skill text. It preserves historical results and uses exact skill IDs, including duplicate names. "
              "The optional metadata ablation is unattempted. Lossless document and query vectors are packaged for model-free retrieval replay.", "",
              "## Scaling results", "", "Each condition contains the same 20 exposed, single-positive questions. Missing conditions have no measured score.", "",
              "| Skills | Top-1 ID accuracy | Recall@10 | NDCG@10 | MRR@10 | Incorrect Top-1 |",
              "|---:|---:|---:|---:|---:|---:|", *summary_rows, "",
              f"Unmeasured library sizes: {', '.join(map(str, pending)) or 'none'}.", "",
              "![Top-1 accuracy](figures/01_top1_accuracy.png)", "![Recall@10](figures/02_recall10.png)",
              "![NDCG@10](figures/03_ndcg10.png)", "![MRR@10](figures/04_mrr10.png)", "",
              "## Official full-catalog evaluation", ""]
    if full:
        report += ["| Metric | Query mean | 95% question-bootstrap interval |", "|---|---:|---|",
                   *[f"| {k} | {full['aggregate'][k]:.6f} | [{full_ci[k]['per_column_95ci'][0][0]:.6f}, {full_ci[k]['per_column_95ci'][0][1]:.6f}] |" for k in ("NDCG@10", "Recall@10", "MRR@10")], "",
                   "All 4,392 queries and 7,187 binary qrels contribute to the denominator. Additional NDCG/MAP/Recall/Precision at 5/10/15 and coverage at 20/50 are in [full_evaluation.json](results/full_evaluation.json)."]
    else:
        report += ["No completed full-query metrics are available. Saved partial rankings, if any, are checkpoints and do not establish a full-test score."]
    report += ["", "## Corpus, chunking and reproducibility", "",
               f"The input audit covers {audit.get('skills', 0):,} skills, {audit.get('original_tokens_total', 0):,} original tokens "
               f"(mean {audit.get('original_tokens_mean', 0):.2f}; median {audit.get('original_tokens_median', 0):.1f}; maximum {audit.get('original_tokens_max', 0):,}). "
               f"There are {audit.get('skills_requiring_chunking', 0):,} chunked skills and {audit.get('chunks', 0):,} chunks, totaling "
               f"{audit.get('chunk_tokens_including_overlap', 0):,} tokenizer tokens including overlap. No body is missing. "
               "The audited chunk plan covers 100% of original text; completed embedding coverage is reported separately in content_coverage.json after every skill has been encoded.", "",
               "Documents are exactly `name + ' | ' + description + ' | ' + skill_md`, preserving original whitespace. Documents up to 4,094 tokenizer tokens use one vector. "
               "Longer documents use exact Unicode source slices, 128-token overlap and preferred Markdown section boundaries within the last 256 tokens. "
               "Each substring is retokenized and hashed. `/api/embed` uses `truncate:false`, 4,096 dimensions, a 4,096-token context and one input per call. "
               "Float32 L2 normalization precedes exact dot-product search. Chunk scores aggregate by maximum; unique skills occupy the top 50. "
               "Source text, model digest, chunking/configuration and runtime identify persistent caches outside the repository.", "",
               "The first maximum-length feasibility probe was safely rejected with HTTP 400 at 4,095 content tokens. "
               "The pinned [Ollama admission code](https://github.com/ollama/ollama/blob/v0.40.15/server/routes.go) reserves two admission slots although this GGUF adds one EOS. "
               "The final 4,094-token content budget respects that measured boundary without truncation. "
               "The retained [failed attempt](results/attempts/01-admission-boundary/failure.json) is separate from completed benchmark outcomes. "
               "The [token-batch feasibility measurements](results/feasibility_token_batches.json) test 512/1,024/2,048 token microbatches with one input per call. "
               "Their warm maximum-length times were similar, so the primary keeps the lower-memory 512-token microbatch. Input batch size remains one.", "",
               "A later strict token-count check stopped at 1,680 indexed skills. For that exact full input, the HF audit tokenizer counted 3,411 content tokens, "
               "while the native GGUF tokenizer with special-token parsing counted 3,409 plus EOS, exactly matching the 3,410 evaluated tokens. "
               "The source contained a control-token spelling that the tokenizers group differently; no source text was truncated. "
               "The [retained parity audit](results/attempts/02-tokenizer-parity/native_tokenizer_audit.json) records both parsing modes. "
               "The resumed validator checks every nonstandard count against the same model's complete-input native tokenizer before checkpointing. "
               "Original-length audit statistics use the pinned HF tokenizer; actual processing totals use Ollama's prompt_eval_count. "
               "Model configuration, input text, chunk spans and completed rankings did not change; existing verified vectors were reused.", "",
               "The strict host swap guard later stopped another invocation after a single additional 4 KiB swap-out, while 4.47 GiB of RAM remained available and no OOM was recorded. "
               "The [resource event](results/attempts/03-swap-guard/resource_event.json) preserves that whole-host observation without assigning it to a process. "
               "After unloading the benchmark model, the next invocation reused verified caches with unchanged inference settings and unchanged guards. "
               "The small swap event remains in telemetry; resetting the invocation baseline does not erase it from the record.", "",
               f"Model digest: `{model.get('model_digest', 'not yet measured')}`. Runtime: `{model.get('runtime', {})}`. "
               "The exact model details, software, GPU/CPU residency and runtime options are retained in model.json, hardware.json, feasibility.json and protocol.json.", "",
               "The seven historical libraries retain every saved ID and its exact order. The 6,006-skill library uses a separately identified fresh `Random(42).sample` over source order. "
               "The historical libraries are not fully nested: completing the 2,000- and 5,000-skill conditions requires cumulative unions of 2,075 and 5,018 distinct skills. "
               "Changes across sizes therefore reflect both library size and candidate membership. All conditions restrict the same accumulating corpus index to their exact eligible IDs. "
               "All gold IDs are present. Historical gold positions are 1–20; some queries contain names or close lexical hints. "
               "Embeddings and scores receive only document/query text, with no gold labels, IDs or position features. Exact score ties use ascending full-text SHA-256; all 6,006 content hashes are unique. "
               "These gold-conditioned, exposed pools are not a blinded or unbiased sample of skill discovery.", "",
               "## Metrics and uncertainty", "",
               "The tested metric implementation is reused from the official full-text reproduction. NDCG uses linear qrel gains; positive relevance defines hits. "
               "AP@K divides by all relevant skills; precision divides by K, including short/missing results. Every expected question remains in the denominator. "
               "Scaling uses one exact gold ID, so Hit@K equals Recall@K and MAP@K equals MRR@K. The official test has one to three positives, where these equalities do not hold. "
               "Top-1 in the official multi-positive test means the first ID is any relevant ID, not exact-set recovery. "
               "Scaling confidence resamples 20 question clusters with replacement and preserves the same sampled question indices across sizes, with 10,000 resamples and seed 42. "
               "The 160 repeated cases are never treated as 160 independent questions. Full-test intervals resample 4,392 questions. "
               "These intervals describe variation across the exposed question panel; they do not repair selection bias or certify generalization. Perfect observed hit rates can produce degenerate bootstrap intervals. "
               "The independent validator recomputes metrics from saved rankings without calling the production metric helper.", "",
               "## Measured performance", "",
               (f"Active document API time: {doc_seconds:.3f} seconds across {len(document_calls):,} uncached chunk calls and {doc_tokens:,} runtime tokens. " if document_calls else "Document indexing has no measurements yet. ")
               + f"Measured active throughput: {performance['tokens_per_second_active']} tokens/s; {performance['chunk_embeddings_per_second_active']} chunks/s. "
               f"Model cold load: {performance['cold_start_model_load_s']} s (unloaded model, without an OS page-cache eviction). "
               f"Whole-device sampled peak VRAM: {performance['gpu_peak_sampled_mib']} MiB; peak system used RAM: {performance['system_peak_sampled_used_gib']} GiB. "
               f"Peak GPU temperature: {performance['gpu_peak_temperature_c']} °C. One-second telemetry can miss shorter peaks.", "",
               f"The selected 512-token microbatch's indexing/query phase has sampled peak VRAM {performance['primary_indexing_query_gpu_peak_sampled_mib']} MiB. "
               "The broader monitored startup includes the earlier 2,048-token microbatch residency; its 5,614 MiB measurement is kept separate from steady indexing. "
               "The memory figure leaves gaps across unmonitored restart/recovery intervals.", "",
               f"Recorded index stages cover {indexed_complete_stages:,} unique skills in {index_wall:.3f} wall seconds, "
               f"or {performance['skills_per_second_completed_stages']} skills/s. Finished stages use elapsed stopwatches; the interrupted stage uses an explicitly marked "
               "UTC bracket from the previous scaling artifact's write time to the recorded stop, including boundary serialization overhead. "
               "In an unfinished run these totals exclude the active stage; per-call API totals include every completed uncached chunk call. "
               "Active API/token totals use successfully journaled responses; requests stopped before checkpointing remain in interrupted wall time, without inferred token counts. "
               "The total primary invocation time sums the interrupted and resumed run receipts; recovery pauses and model acquisition are separate.", "",
               "[performance.json](results/performance.json) and [latency_by_size.csv](results/latency_by_size.csv) retain mean, median, p50, p95 and maximum latencies. "
               "Scaling makes a fresh query embedding request at every size and measures total embedding plus search wall time. Cached-query search is a separate repeated search, "
               "and is never compared as if it included embedding or generation. Full-test latency records cache hits and component sums separately. "
               "Index history and per-call journals distinguish active inference from cache validation, persistence and resumed wall time. "
               "Local elapsed compute time is the execution-cost measure; electricity charges are unmeasured and no paid inference is used.", "",
               "![Latency](figures/05_query_latency.png)", "![Indexing](figures/06_indexing_throughput_time.png)", "![GPU memory](figures/07_gpu_memory.png)", "",
               "## Prior evidence and comparison limits", "",
               "The recorded SkillRet 0.6B full-text baseline is verified from its saved machine-readable result: "
               f"NDCG@10 **{baselines[0]['NDCG@10']:.6f}**, Recall@10 **{baselines[0]['Recall@10']:.6f}**, MRR@10 **{baselines[0]['MRR@10']:.6f}**. "
               "It uses the same pinned corpus, 4,392 queries and official qrels, but strips field whitespace and uses one vector with a 32,768-token cap; its token audit records two truncated documents. "
               "The new complete-text max-chunk protocol, quantization and runtime differ, so this is contextual evidence and not exact methodological equivalence or an isolated model-size effect.", "",
               "The [contextual baseline table](results/contextual_baselines.csv) also reads the saved 4,392-query BM25, Qwen 0.6B, SkillRet header and RRF60 outputs. "
               "Those use metadata/header text and 1,024-token neural limits; they are not directly matched full-text controls. "
               "The separate 20-query seed-5150 scaling panel differs in questions, nested libraries and relevance, and its scores cannot be substituted for this panel. "
               "OAuth/local listing studies use the same current 20 questions and historical pools but descriptions are capped at 300 characters, outputs are name-based generation, "
               "and runtime settings differ. Their latency includes generation. They provide context, not a controlled vector-versus-selector model comparison.", "",
               "![Contextual baselines](figures/08_contextual_baselines.png)", "",
               "## Validation, limitations and status", "",
               "See [validation.json](results/validation.json) for the exact completed scope. Offline tests check Unicode coverage, boundary behavior, finite unit embeddings, cosine scoring, "
               "max-chunk deduplication, out-of-pool exclusions, exact-ID labels, multi-positive denominators and paired bootstrap behavior. "
               "Graphs are generated directly from measured JSON/journals, with missing conditions left unmeasured; figures/manifest.json binds chart bytes to sources. "
               "Model weights, raw corpus text, original per-text runtime caches, personal paths and credentials stay outside this repository. "
               "The requested lossless document/query vector snapshot is included through Git LFS. All raw historical artifacts are preserved. "
               "Official qrels define relevance for scoring; a semantically useful but unjudged skill can still count as an error. "
               "No blinded-generalization or skill-execution claim follows from retrieval scores. The optional full-text-versus-metadata ablation remains unattempted.", ""]
    if (R / "failure.json").exists():
        report += ["A failed attempt is preserved in [failure.json](results/failure.json); it must not be described as a completed condition.", ""]
    if cached_replay:
        at = report.index("![Latency](figures/05_query_latency.png)")
        report[at:at] = [f"The completed cache-only replay loaded and verified {cached_replay['chunks_loaded']:,} document vectors in "
                         f"{cached_replay['index_cache_validation_and_assembly_s']:.3f} seconds, then reproduced all 4,392 saved rankings with zero model requests. "
                         f"Cached query read-plus-search median/p95: {cached_replay['query_total_cached']['p50_s']:.6f} / "
                         f"{cached_replay['query_total_cached']['p95_s']:.6f} seconds. This includes cache-file integrity checks, "
                         "and remains separate from the primary uncached embedding measurements. The operating-system file cache was not evicted, "
                         "so this does not measure cold-storage reads.", ""]
    if full_uncached:
        at = report.index("![Latency](figures/05_query_latency.png)")
        timing = performance["full_uncached_retrieval_including_cache_persistence"]
        report[at:at] = [f"The official test records {len(full_uncached):,} uncached and {len(full_cached):,} cached query-vector executions. "
                         f"Uncached total retrieval mean/median/p95/max: {timing['mean_s']:.6f} / {timing['p50_s']:.6f} / "
                         f"{timing['p95_s']:.6f} / {timing['maximum_s']:.6f} seconds. These totals include query cache persistence and "
                         f"search. The summed primary run invocations took {performance['primary_total_invocation_wall_s']:.3f} seconds, "
                         "including scaling and the preserved interrupted invocation, with recovery pauses and setup outside that boundary.", ""]
    published = load(R / "published_vector_replay.json")
    if published:
        snapshot = load(STUDY / "vectors/manifest.json")
        report += ["## Published vectors and model-free replay", "",
                   f"The lossless [Git LFS snapshot](vectors/README.md) contains all 6,913 document-chunk vectors, "
                   f"all 4,392 official query vectors and all 20 scaling query vectors, plus skill/query IDs, chunk ownership and 7,187 public qrels. "
                   f"Every vector retains its original 4,096 float32 values. The archive is {snapshot['bytes']:,} bytes; "
                   f"[manifest.json](vectors/manifest.json) records archive and uncompressed-array SHA-256 checksums, shapes and dtypes.", "",
                   f"The [validated snapshot replay](results/published_vector_replay.json) reproduced all 160 scaling and 4,392 official top-50 rankings "
                   f"and every metric identically, with zero model/API/network requests and zero reads from the original runtime cache. "
                   f"Archive loading and validation took {published['archive_loading_validation_s']:.3f} seconds; the complete replay took "
                   f"{published['total_wall_s']:.3f} seconds on the measured CPU. Run `OPENBLAS_NUM_THREADS=1 python src/vector_snapshot.py replay` "
                   "after `git lfs pull` to repeat retrieval without embedding the corpus or queries again. "
                   "This verifies the retrieval evidence; it does not rerun embedding inference or its GPU timing.", ""]
    if full and scaling:
        first, last = scaling[0], scaling[-1]
        report += ["## Conclusions", "",
                   f"Qwen3 8B Q4_K_M completed full-source retrieval at all eight library sizes and all 4,392 official questions. "
                   f"Its official NDCG@10 is {full['aggregate']['NDCG@10']:.6f}, Recall@10 {full['aggregate']['Recall@10']:.6f}, "
                   f"and MRR@10 {full['aggregate']['MRR@10']:.6f}. "
                   f"Its recorded NDCG@10 is below the contextual SkillRet 0.6B full-text result of {baselines[0]['NDCG@10']:.6f}; "
                   "this run does not establish a quality benefit from the larger quantized embedding model or from full-source max-chunk retrieval. "
                   f"Within the exposed 20-question panel, exact Top-1 is {first['evaluation']['aggregate']['Top1_ID_accuracy']:.0%} "
                   f"at {first['library_size']} skills and {last['evaluation']['aggregate']['Top1_ID_accuracy']:.0%} at {last['library_size']:,} skills.", "",
                   "The measured configuration fits this 8 GB GPU with reported full layer offload and documented host buffers. "
                   "Persistent vector caching supports verified replay without re-embedding the corpus. "
                   "Max-chunk aggregation gives longer skills more opportunities to score; its effect needs a separate ablation. "
                   "The missing metadata-only ablation also leaves the effect of adding full skill text unmeasured. "
                   "Prior baseline scores remain contextual because text handling, context, quantization and runtime differ. "
                   "These observations support reproducibility and measured retrieval on the pinned public data, with the stated limits on generalization.", ""]
        report += ["## Historical 20-question listing context", "",
                   "The saved OAuth listing outputs below use the same questions and historical eligible pools, with 300-character descriptions and name-level scoring. "
                   "They are contextual generation results, with a different document view and scoring unit. No controlled model-quality or latency difference is inferred.", "",
                   "| Historical model | Skills | Correct first skill name / 20 |", "|---|---:|---:|",
                   *[f"| {r['model']} | {r['library_size']:,} | {r['correct_top1_names']} |" for r in listing_context], "",
                   "The [machine-readable context](results/contextual_listing_baselines.csv) preserves source hashes. "
                   "The [local native listing study](../skill-discovery-local-native-20261103/COMPARISON.md) additionally has parser/context/KV differences and retains its blocked conditions; "
                   "its generation timings must remain separate from cached or uncached vector search.", ""]
    (STUDY / "REPORT.md").write_text("\n".join(report))
    print("report updated:", progress["stage"], len(scaling), "scaling conditions")


if __name__ == "__main__":
    build()
