"""Build the measured report and final archive on RunPod."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import statistics
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak

from data_stage import read_jsonl, write_json

NAMES = ["bm25_full", "bge_m3_dense_full", "qwen3_raw_full",
         "clm_pretrained_full", "clm_skillret_full"]


def bootstrap(root: Path):
    import random
    queries = read_jsonl(root / "data/test_queries.jsonl")
    by_group = defaultdict(list)
    for i,q in enumerate(queries): by_group[q["group"]].append(i)
    groups = sorted(by_group)
    values = {m: np.array([r["recall_at_5"] for r in read_jsonl(root/"results"/f"{m}_per_query.jsonl")])
              for m in NAMES}
    assert all(len(x)==len(queries) for x in values.values())
    rng = random.Random(42)
    samples = {m:[] for m in NAMES}
    deltas = {"raw_to_pretrained":[],"pretrained_to_skillret":[]}
    for _ in range(1000):
        sampled = [i for g in rng.choices(groups,k=len(groups)) for i in by_group[g]]
        means = {m:float(values[m][sampled].mean()) for m in NAMES}
        for m in NAMES: samples[m].append(means[m])
        deltas["raw_to_pretrained"].append(means["clm_pretrained_full"]-means["qwen3_raw_full"])
        deltas["pretrained_to_skillret"].append(means["clm_skillret_full"]-means["clm_pretrained_full"])
    output={"unit":"test query/original-ID group","seed":42,"replicates":1000,
            "recall_at_5_ci":{m:np.quantile(samples[m],[.025,.975]).tolist() for m in NAMES},
            "paired_deltas":{k:{"mean":float(np.mean(v)),"ci95":np.quantile(v,[.025,.975]).tolist()}
                             for k,v in deltas.items()}}
    write_json(root/"results/bootstrap.json",output)
    return output


def audit_and_examples(root: Path):
    """Independently recompute a deterministic metric subset from saved ranks."""
    index=json.loads((root/"results/score_index.json").read_text())
    qids,sids=index["query_ids"],index["skill_ids"]
    gold=defaultdict(set)
    for r in read_jsonl(root/"data/test_qrels.jsonl"):
        gold[r["query_id"]].add(r["skill_id"])
    checked=0
    for method in NAMES:
        ranks=np.load(root/"results"/f"{method}_rankings.npy",mmap_mode="r")
        scores=np.load(root/"results"/f"{method}_scores.npy",mmap_mode="r")
        per=read_jsonl(root/"results"/f"{method}_per_query.jsonl")
        assert ranks.shape==scores.shape==(len(qids),len(sids)) and len(per)==len(qids)
        for i in range(min(20,len(qids))):
            ordered=ranks[i].tolist()
            assert len(set(ordered))==len(sids) and min(ordered)==0 and max(ordered)==len(sids)-1
            assert ordered==np.lexsort((np.asarray(sids,dtype=str),-scores[i])).tolist()
            relevant=[j for j,pos in enumerate(ordered) if sids[pos] in gold[qids[i]]]
            recall=sum(j<5 for j in relevant)/len(gold[qids[i]])
            rr=1/(relevant[0]+1) if relevant else 0
            assert abs(recall-per[i]["recall_at_5"])<1e-12
            assert abs(rr-per[i]["reciprocal_rank"])<1e-12
            checked+=1
    write_json(root/"evidence/evaluator_crosscheck.json",
               {"method_query_rows_checked":checked,"ranking_permutation":True,
                "score_order_ties":True,"independent_recall_at_5_and_mrr":True})
    qtext={r["id"]:r["text"] for r in read_jsonl(root/"data/test_queries.jsonl")}
    stext={r["id"]:r["text"] for r in read_jsonl(root/"data/test_skills.jsonl")}
    per=read_jsonl(root/"results/clm_skillret_full_per_query.jsonl")
    failures=[]
    for r in sorted(per,key=lambda r:(r["recall_at_5"],r["reciprocal_rank"],r["query_id"])):
        if r["recall_at_5"]>=1: continue
        qid=r["query_id"]
        failures.append({"query_id":qid,"query":qtext[qid],
                         "gold":[{"id":s,"text":stext[s]} for s in sorted(gold[qid])],
                         "predicted_top_10":[{"id":s,"text":stext[s]} for s in r["top_10"]],
                         "recall_at_5":r["recall_at_5"],"reciprocal_rank":r["reciprocal_rank"]})
        if len(failures)==10: break
    write_json(root/"results/error_examples.json",failures)
    return failures


def make_figure(root, agg, boot):
    names=["BM25","BGE-M3","Qwen3 raw","CLM pretrained","CLM SkillRet"]
    y=[agg[m]["recall_at_5"] for m in NAMES]
    lo=[boot["recall_at_5_ci"][m][0] for m in NAMES]
    hi=[boot["recall_at_5_ci"][m][1] for m in NAMES]
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.bar(names,y,color=["#5b6770","#3484b8","#7768a8","#a66c28","#2b8a62"])
    ax.errorbar(range(5),y,yerr=[np.maximum(0,np.array(y)-lo),
                                 np.maximum(0,np.array(hi)-y)],fmt="none",ecolor="black",capsize=3)
    ax.set_ylim(0,max(.1,min(1,max(hi)*1.2)));ax.set_ylabel("Macro Recall@5")
    ax.tick_params(axis="x",labelrotation=18)
    fig.tight_layout();p=root/"figures/recall_at_5.png";fig.savefig(p,dpi=160);plt.close(fig)
    hist=json.loads((root/"results/validation_history.json").read_text())
    fig,ax=plt.subplots(figsize=(8,3))
    ax.plot([r["epoch"] for r in hist],[r["recall_at_5"] for r in hist],marker="o")
    ax.set(xlabel="Epoch",ylabel="Train-catalog validation Recall@5")
    fig.tight_layout();p2=root/"figures/validation_curve.png";fig.savefig(p2,dpi=160);plt.close(fig)


def pdf(root: Path, agg, boot, failures):
    styles=getSampleStyleSheet(); normal=styles["BodyText"]; title=styles["Title"]
    story=[]
    def p(s,style=normal): story.append(Paragraph(s,style));story.append(Spacer(1,8))
    def table(rows,widths=None):
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dbe8ef")),
                               ("GRID",(0,0),(-1,-1),.3,colors.grey),
                               ("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("FONTSIZE",(0,0),(-1,-1),8),
                               ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,10))
    audit=json.loads((root/"data/dataset_audit.json").read_text())
    hist=json.loads((root/"results/validation_history.json").read_text())
    cohort=json.loads((root/"data/training_cohort.json").read_text())
    best=json.loads((root/"checkpoints/clm_skillret/best_selection.json").read_text())
    p("SkillRet full-catalog retrieval benchmark",title)
    p("Five fixed methods, description-only skill text, one official test catalog. Generated on RunPod " +
      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + ".")
    rows=[["Method","Recall@5","95% group CI","MRR","Hit@1"]]
    labels=["BM25","BGE-M3 dense","Qwen3-8B pooled","CLM pretrained","CLM SkillRet adapted"]
    for label,m in zip(labels,NAMES):
        a=agg[m];ci=boot["recall_at_5_ci"][m]
        rows.append([label,f'{a["recall_at_5"]:.4f}',f'{ci[0]:.4f}–{ci[1]:.4f}',
                     f'{a["reciprocal_rank"]:.4f}',f'{a["hit_at_1"]:.4f}'])
    table(rows,[120,75,115,75,75])
    delta=agg["clm_skillret_full"]["recall_at_5"]-agg["clm_pretrained_full"]["recall_at_5"]
    p(f"Observed adapted-minus-pretrained Recall@5: {delta:+.4f}. This is one training seed; "
      "the confidence interval resamples test query groups, not training randomness.")
    story.append(Image(str(root/"figures/recall_at_5.png"),width=6.9*inch,height=3.0*inch))
    p("All five rows rank every test skill for every test query. No reranking shortlist or generative selector was used.")
    story.append(PageBreak())
    p("Protocol and data",title)
    table([["Partition","Skills","Queries","Qrels"],
           *[[s,*[str(audit["counts"][s][k]) for k in ("skills","queries","qrels")]] for s in ("train","test")]])
    p("SkillRet dataset revision: " + audit["revision"] + ". Query input is the canonical query string. "
      "Skill input is name, colon, space, description. The SkillRet query set is synthetic. "
      "Only train labels enter adaptation; test labels enter final evaluation after checkpoint selection.")
    p("BM25 uses rank-bm25 BM25Okapi with lowercased Unicode word tokens, k1=1.5 and b=0.75. "
      "BGE-M3 uses dense normalized embeddings. Qwen3-8B BF16 uses last-token pooling and "
      "L2 normalization before CLM heads. The pretrained and adapted CLM rows use the same frozen Qwen vectors.")
    p("Train and test skill IDs are disjoint at this dataset revision. ID disjointness does not establish "
      "semantic independence or completeness of relevance judgments. Duplicate texts retain distinct skill IDs.")
    story.append(PageBreak())
    p("Quality and paired comparisons",title)
    table([["Method","Recall@1","Recall@10","NDCG@5","Complete@5"],
           *[[lab,*[f'{agg[m][k]:.4f}' for k in ("recall_at_1","recall_at_10","ndcg_at_5","completeness_at_5")]]
             for lab,m in zip(labels,NAMES)]],[120,85,85,85,85])
    for key,description in [("raw_to_pretrained","Pretrained CLM minus raw Qwen"),
                            ("pretrained_to_skillret","SkillRet CLM minus pretrained CLM")]:
        d=boot["paired_deltas"][key]
        p(f'{description}: {d["mean"]:+.4f} Recall@5; paired 95% group interval '
          f'[{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}].')
    p("Hit@1 and Recall@1 differ when a query has multiple relevant skills. MRR uses the full ranking, "
      "not only top five. Exact saved score matrices and stable skill-ID tie breaking permit replay.")
    story.append(PageBreak())
    p("Head adaptation and validation",title)
    p(f'Training cohort: {cohort["selected_training_groups"]} train query groups, with a separate '
      f'{len(cohort["validation_query_ids"])}-query validation split. The cohort was selected from '
      "train groups before test scores were opened, using the measured Qwen pilot and budget guard.")
    p(f'Original epoch-0 validation Recall@5: {hist[0]["recall_at_5"]:.4f}. '
      f'Best actually trained epoch: {best["epoch"]}, validation Recall@5 {best["recall_at_5"]:.4f}. '
      "Checkpoint choice used only train-catalog validation Recall@5, then MRR.")
    story.append(Image(str(root/"figures/validation_curve.png"),width=6.7*inch,height=2.5*inch))
    p("Only the released state/action projection heads and learned scale were updated. "
      "Each batch included all judged positives for its unique queries; the many-to-many loss "
      "also marked positives shared across queries. Unjudged pairs served as negatives, which is a limitation.")
    story.append(PageBreak())
    p("Timing and resource observations",title)
    latency=[]
    for model in ("bm25","bge_m3","qwen3"):
        path=root/"results"/f"{model}_online_latency.json"
        if path.exists():
            a=json.loads(path.read_text());v=[r["text_to_top1_seconds"] for r in a]
            latency.append([model,str(len(v)),f"{np.median(v):.3f}",f"{np.quantile(v,.95):.3f}"])
    table([["Model","Fresh queries","p50 seconds","p95 seconds"],*latency])
    p("The latency sample includes fresh query encoding and full-catalog top-1 scoring with the model warm. "
      "It is a small sample. The BGE scoring path used CPU NumPy; encoder inference used the GPU. "
      "Qwen and both CLM rows share encoder work; the saved matrices are offline replay data and "
      "are not independent end-to-end latency measurements for the CLM rows.")
    for kind in ("skills","queries"):
        path=root/"evidence"/f"bge_{kind}_timing.json"
        if path.exists():
            t=json.loads(path.read_text());p(f'BGE test {kind}: {t["records"]} rows, {t["tokens"]} tokenizer '
                f'tokens, {t["encode_seconds"]:.1f} seconds observed preprocessing.')
    timing_path=root/"evidence/evaluation_timing.json"
    if timing_path.exists():
        tm=json.loads(timing_path.read_text())
        p(f'Full test scoring took {tm["bm25_build_and_full_score_seconds"]:.1f}s for BM25, '
          f'{tm["bge_full_score_seconds"]:.1f}s for BGE vectors and '
          f'{tm["qwen_raw_full_score_seconds"]:.1f}s for raw Qwen vectors. '
          'These matrix times exclude encoder loading and text encoding.')
        p(f'CLM projection of the full test queries and skills took '
          f'{tm["clm_pretrained_full_project_test_seconds"]:.3f}s for pretrained heads and '
          f'{tm["clm_skillret_full_project_test_seconds"]:.3f}s for adapted heads.')
    samples=root/"evidence/qwen_resource_samples.jsonl"
    if samples.exists():
        vals=[];rss=[]
        for line in samples.read_text().splitlines():
            sample=json.loads(line)
            qwen_rows=[x for x in sample["processes"] if x["gpu_used_mib"].isdigit()
                       and int(x["gpu_used_mib"])>10000]
            if not qwen_rows: continue
            try: vals.append(int(sample["gpu_memory_used_total_util_percent"].split(",")[0]))
            except (ValueError,IndexError): pass
            rss += [x["cpu_rss_kib"] for x in qwen_rows if x["cpu_rss_kib"] is not None]
        if vals:
            p(f'Qwen process monitoring sampled {len(vals)} points: maximum total GPU memory '
              f'{max(vals):,} MiB; maximum attributed process RSS '
              f'{(max(rss)/1024):,.0f} MiB.' if rss else
              f'Qwen process monitoring sampled {len(vals)} points: maximum total GPU memory {max(vals):,} MiB.')
    story.append(PageBreak())
    p("Interpretation, limits and reuse",title)
    p("This comparison supports statements about the pinned SkillRet synthetic queries, description-only "
      "skills, frozen encoders and one head-training seed. It does not establish universal retrieval quality "
      "or successful skill execution. BGE and BM25 differ from Qwen in architecture and pretraining, "
      "so only the Qwen → CLM → adapted CLM chain isolates head effects.")
    p("The archive includes every generated vector, ordered ID list, score matrix, checkpoint, split manifest, "
      "source pin, test evidence and scripts. The extracted best head requires the exact pinned Qwen3-8B "
      "encoder and input recipe to handle new text; the 8B backbone itself is intentionally not bundled.")
    p("Sources: ThakiCloud/SKILLRET dataset; BAAI/bge-m3; Qwen/Qwen3-8B; "
      "Contrastive-LM/CLM-v0.1-8B and pinned CLM source. Revisions and hashes are in the archive manifest.")
    p("RunPod billing is measured separately. A local cleanup receipt is appended to the final archive after "
      "the remote backup has been checked and the pod terminated.")
    ledger=root/"evidence/budget_ledger.json"
    if ledger.exists():
        b=json.loads(ledger.read_text())
        p(f'RunPod credit at start: ${b["starting_balance"]:.4f}; latest observed credit '
          f'({escape(b["latest_utc"])}): ${b["latest_balance"]:.4f}. '
          'These are billing observations before final pod cleanup, not an end-of-run invoice.')
    story.append(PageBreak())
    p("Observed retrieval misses",title)
    p("The examples below were chosen only after final evaluation. They are actual test queries and "
      "saved predictions; labels may omit semantically valid alternatives.")
    printable=[r for r in failures if all(ord(c)<128 for c in
               r["query"]+"".join(x["text"] for x in r["gold"][:2]+r["predicted_top_10"][:2]))]
    p("Shown examples are the first ASCII-renderable misses in the saved worst-case list.")
    for r in printable[:2]:
        p(f'Query {escape(r["query_id"])}: {escape(r["query"][:240])}')
        p('Gold: '+escape('; '.join(x["text"][:100] for x in r["gold"][:2])))
        p('Top predictions: '+escape('; '.join(x["text"][:100] for x in r["predicted_top_10"][:2])))
    p("Reproduce the scores from the archived vectors with src/reproduce_from_vectors.py on a GPU. "
      "The released and adapted CLM heads are saved separately. The trained head can be loaded with "
      "src/load_clm_head.py using the pinned Qwen pre-head recipe in MODEL_CARD.md.")
    out=root/"REPORT.pdf"
    SimpleDocTemplate(str(out),pagesize=letter,rightMargin=48,leftMargin=48,
                      topMargin=40,bottomMargin=40).build(story)
    import fitz
    doc=fitz.open(out)
    qa=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.2,1.2))
        target=root/"evidence"/f"report_page_{i+1:02d}.png";pix.save(target)
        qa.append({"page":i+1,"text_chars":len(page.get_text()),"render_bytes":target.stat().st_size})
    assert len(qa)>=7 and all(x["text_chars"]>100 for x in qa)
    write_json(root/"evidence/pdf_qa.json",qa)


def notebooks(root: Path):
    import nbformat as nbf
    specs=[("01_prepare_embeddings_and_train.ipynb","Preparation, embeddings and head training",
            "See data/dataset_audit.json, data/training_cohort.json, vectors/, results/validation_history.json and checkpoints/."),
           ("02_full_catalog_benchmark_and_report.ipynb","Full-catalog evaluation and report",
            "See results/aggregate.json, results/bootstrap.json, results/*_scores.npy and REPORT.pdf.")]
    for filename,title,detail in specs:
        nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(f"# {title}\n\nExecuted remotely on RunPod. {detail}"),
            nbf.v4.new_code_cell("from pathlib import Path\nimport json\nroot = Path.cwd()\nprint(json.loads((root / 'data/dataset_audit.json').read_text())['counts'])\nprint(json.loads((root / 'evidence/completion.json').read_text())['status'])"),
            nbf.v4.new_markdown_cell("The reusable CLI is src/remote_run.py. Results are already saved; rerunning a model stage requires a RunPod GPU and a new budget guard.")])
        nbf.write(nb,root/filename)


def archive(root: Path):
    excluded={"model-cache","piptmp","incremental","export",".venv","__pycache__",".git","control"}
    files=sorted(p for p in root.rglob("*") if p.is_file() and not any(x in excluded for x in p.parts)
                 and p.name not in {"FINAL.zip","MANIFEST.sha256"} and not p.is_symlink())
    lines=[f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}" for p in files]
    (root/"MANIFEST.sha256").write_text("\n".join(lines)+"\n")
    zpath=root/"FINAL.zip"
    with ZipFile(zpath,"w",allowZip64=True) as z:
        for p in files+[root/"MANIFEST.sha256"]:
            z.write(p,p.relative_to(root),compress_type=ZIP_STORED if p.suffix in {".npy",".pt",".png",".pdf"} else ZIP_DEFLATED)
    write_json(root/"evidence/remote_archive_receipt.json",{"sha256":hashlib.sha256(zpath.read_bytes()).hexdigest(),
               "bytes":zpath.stat().st_size,"entries":len(files)+1})


def main(root: Path):
    agg=json.loads((root/"results/aggregate.json").read_text())
    assert list(agg)==NAMES
    failures=audit_and_examples(root)
    boot=bootstrap(root);make_figure(root,agg,boot);pdf(root,agg,boot,failures);notebooks(root)
    from shutil import copy2
    copy2(root/"src/reproduce_from_vectors.py",root/"reproduce_from_vectors.py")
    lines=["# Completed SkillRet five-method RunPod benchmark", "", "All five official test queries-versus-skills catalogs were scored on RunPod.",
           "The report is REPORT.pdf; machine-readable metrics are results/aggregate.json.", "",
           "| Method | Macro Recall@5 |", "|---|---:|"]
    for method in NAMES: lines.append(f'| {method} | {agg[method]["recall_at_5"]:.4f} |')
    lines += ["", "The best adapted head is checkpoints/clm_skillret/best_head.pt.",
              "Its config is checkpoints/clm_skillret/training_config.json; the loader is src/load_clm_head.py.",
              "MODEL_CARD.md gives the pinned encoder recipe. `reproduce_from_vectors.py` reruns ranking from saved vectors.",
              "This ZIP contains scores, rankings, vectors, checkpoints, tests, source, and PDF QA.",
              "The local post-cleanup receipt records final cost and resource deletion."]
    (root/"START_HERE.md").write_text("\n".join(lines)+"\n")
    write_json(root/"evidence/completion.json",{"status":"COMPLETE","stages":["data","bge","qwen","train","evaluate","report"]})
    archive(root)
