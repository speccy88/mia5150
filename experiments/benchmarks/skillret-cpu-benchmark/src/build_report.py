#!/usr/bin/env python3
"""Source-backed SkillRet recovery reports using only the packaged HTML renderer.

No inference, installation, raw-row reconstruction, or historical-data invention.
Retained aggregates remain distinct from fresh measurements at every stage.
"""
from __future__ import annotations
import argparse,csv,glob,hashlib,json,math,os,sqlite3,statistics,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
SIZES=(10,100,1000,6006)
METRICS=('mrr5','recall1','recall5','hit1','hit5','ndcg5','complete5','candidate_recall','candidate_hit')
EMBEDDINGS=('bge-small','e5-small','embeddinggemma-int8')
DECISIONS=('gliner2.5-small','verdict-151m-onnx','laya-multilingual','julia-1')
LABELS={'bm25':'BM25','bge-small':'BGE-small-en-v1.5','e5-small':'multilingual-e5-small','embeddinggemma-int8':'EmbeddingGemma INT8','gliner2.5-small':'GLiNER2.5-small','verdict-151m-onnx':'Verdict ONNX','laya-multilingual':'Laya multilingual','julia-1':'Julia-1'}
HISTORY_SQL='SELECT model, method, catalog_size, queries, mrr5, recall5, hit1, p50_ms, p95_ms FROM retained_summary ORDER BY mrr5 DESC;'
QUALITY_SQL='SELECT model, method, catalog_size, COUNT(*) AS queries, '+', '.join('AVG('+m+') AS '+m for m in METRICS)+', AVG(shortlist_mrr5) AS shortlist_mrr5 FROM benchmark_per_query GROUP BY model, method, catalog_size ORDER BY model, method, catalog_size;'

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def relpath(path,root): return str(Path(path).resolve().relative_to(Path(root).resolve()))
def number(value,field,context,optional=False):
 if value in ('',None):
  if optional:return None
  raise ValueError(f'{context}: missing {field}')
 result=float(value)
 if not math.isfinite(result):raise ValueError(f'{context}: nonfinite {field}')
 return result
def percentile(values,p):
 if not values:return None
 x=sorted(values);q=(len(x)-1)*p;lo=math.floor(q);hi=math.ceil(q)
 return x[lo]+(x[hi]-x[lo])*(q-lo)
def cohort_hash(ids):return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
def completed_csv_is_current(root,csv_path,run_path):
 csv_path=Path(csv_path);run_path=Path(run_path)
 if not csv_path.exists() or not run_path.exists():return False
 if csv_path.stat().st_mtime_ns<=run_path.stat().st_mtime_ns:return True
 manifest=Path(root)/'rank_restore_manifest.json'
 if not manifest.exists():return False
 try:
  entries=[e for e in read_json(manifest)['files'] if e['original_file']==relpath(csv_path,root)]
  return len(entries)==1 and entries[0].get('original_bytes',csv_path.stat().st_size)==csv_path.stat().st_size and sha256(csv_path)==entries[0]['original_sha256']
 except (ValueError,KeyError,TypeError,OSError):return False

def aggregate_sql(rows):
 db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
 fields=['model','method','catalog_size',*METRICS,'shortlist_mrr5']
 db.execute('CREATE TABLE benchmark_per_query (model TEXT, method TEXT, catalog_size INTEGER, '+', '.join(m+' REAL' for m in [*METRICS,'shortlist_mrr5'])+')')
 db.executemany('INSERT INTO benchmark_per_query VALUES ('+','.join('?' for _ in fields)+')',[tuple(r.get(k) for k in fields) for r in rows])
 result=[dict(r) for r in db.execute(QUALITY_SQL)];db.close();return result

def historical_rows(retained):
 # SQL is a report transformation of retained aggregate observations, not a
 # reconstruction or re-execution of their unavailable original measurements.
 fields=['model','method','catalog_size','queries','mrr5','recall5','hit1','p50_ms','p95_ms']
 db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
 db.execute('CREATE TABLE retained_summary (model TEXT,method TEXT,catalog_size INTEGER,queries INTEGER,mrr5 REAL,recall5 REAL,hit1 REAL,p50_ms REAL,p95_ms REAL)')
 db.executemany('INSERT INTO retained_summary VALUES (?,?,?,?,?,?,?,?,?)',[tuple(r.get(k) for k in fields) for r in retained.get('full_catalog_aggregates',[])])
 rows=[{**dict(r),'label':LABELS.get(r['model'],r['model'])+' · '+r['method'],'evidence_status':'Historical retained aggregate; original raw unavailable'} for r in db.execute(HISTORY_SQL)]
 db.close();return rows

def source(sid,label,path,description,sql=None):
 result={'id':sid,'label':label,'path':path,'query':{'description':description}}
 if sql:result['query'].update(engine='SQLite',language='sql',sql=sql)
 return result

def markdown(sid,title,body,source_id=None,level=2):
 out={'id':sid,'type':'markdown','body':'#'*level+' '+title+'\n\n'+body}
 if source_id:out['sourceId']=source_id
 return out

def table(sid,title,dataset,fields,source_id,sort,subtitle=None):
 out={'id':sid,'title':title,'dataset':dataset,'sourceId':source_id,'layout':'full','defaultSort':{'field':sort[0],'direction':sort[1]},'columns':[{'field':f,'label':label,**({'format':fmt} if fmt else {})} for f,label,fmt in fields]}
 if subtitle:out['subtitle']=subtitle
 return out

def chart(sid,title,dataset,x,y,source_id,subtitle):
 return {'id':sid,'title':title,'subtitle':subtitle,'type':'horizontalBar','dataset':dataset,'sourceId':source_id,'layout':'full','intent':'comparison','encodings':{'x':{'field':x,'type':'nominal'},'y':{'field':y,'type':'quantitative'},'tooltip':[{'field':'queries','label':'Query denominator','type':'quantitative'},{'field':'recall5','label':'Recall@5','type':'quantitative','format':'percent'}]},'valueFormat':'number','palette':{'kind':'sequential','name':'blue'},'labels':{'values':'all'},'settings':{'sort':'descending'},'emptyState':'No reviewed observations available; missing values are not zero.'}

def make_artifact(root,fresh=None):
 now=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
 retained=read_json(root/'docs/retained_observations.json');history=historical_rows(retained)
 fast=[{**r,'label':LABELS.get(r['model'],r['model'])} for r in retained.get('fast_exploratory_common8',[])]
 title='SkillRet CPU benchmark: fresh recovery and retained observations'
 sources=[source('historical','Retained historical aggregate observations','docs/retained_observations.json','Recorded observations retained after the original workspace became unavailable. Original raw rankings, timing rows, vectors, and per-pair files are not present; this is not a raw-data reconstruction.',HISTORY_SQL),source('fresh_transform','Fresh recovery metric aggregation','src/report_aggregation.sql','Actual macro aggregation of completed, reviewed fresh per-query records; historical aggregates are never imported into this relation.',QUALITY_SQL)]
 datasets={'historical':history,'historical_fast8':fast}
 blocks=[markdown('title',title,'CPU-only · version-pinned short-header adaptation · read-only checkpoint',level=1),
  markdown('summary','What this checkpoint can establish','The original raw benchmark workspace became unavailable. Fresh measurements are being collected under the remaining approved runtime budget. Until a common fresh cohort completes, the earlier 4,392-query values below are retained historical observations only; they cannot be rescored, paired to new query subsets, or treated as fresh recovery results.'),
  markdown('recovery_status','Evidence loss is explicit, and the recovery run is separate','This report does not reconstruct missing original rows or model outputs. The fresh target is a common 64-query evaluation at catalog sizes 10, 100, 1,000, and 6,006, with preregistered all-model common48 and common32 fallbacks. Full-catalog request latency targets the same 32 query IDs and two repeats. Inference stops at 11:52 UTC and delivery is bounded by 12:12 UTC on 2 October 2026. Both Qwen entries remain excluded.'),
  markdown('definitions','Metrics and denominators come before comparisons','MRR@5 is the mean reciprocal rank of the first judged-relevant skill in the first five positions, or zero when none appears. Recall@5 averages the fraction of all judged-positive skills recovered; Hit@1/5 measures any-relevant success. Complete@5 requires the entire gold set. Candidate recall and candidate-any-hit describe shortlist coverage, not final ranking quality. For a fixed shortlist, candidate-any-hit bounds achievable MRR. All rates are fractional; no missing value is imputed as zero. Missing qrels are scored non-relevant under the released labels; they are not proof that another skill could not help.'),
  markdown('scope','The sampled catalogs define a controlled short-header task','SkillRet v1.1 contains 6,006 held-out skills and 4,392 test queries. The scoreable view is skill name plus newline plus the first 1,024 description characters, rather than official full skill_md content. Smaller catalogs are deterministic, per-query, nested and gold-preserving. Membership order is re-sorted before scoring; gold labels are never scoring features. The domain_vector ablation selects the top-1 normalized centroid over the six published skill major categories, then ranks vectors only inside that category. It requires existing skill metadata; it is not a 13-domain-tag classifier and never routes from gold query labels. Retrieval ties use canonical skill-ID order; native score ties preserve frozen retriever order. These conditions do not reproduce the official full-text leaderboard or a single globally shared small index.','historical'),
  markdown('fresh_pending','Fresh common-cohort evidence is not yet complete','Only completed fresh CSVs with validated run receipts, exact query populations, verified rankings, and distinct provenance will enter the primary report. All models must share the same selected cohort; unequal partial results will never be pooled into a model ranking.'),
  markdown('methodology','The comparison keeps model roles and timing boundaries explicit','BGE-small-en-v1.5, multilingual-e5-small and EmbeddingGemma INT8 are embedding models. GLiNER2.5-small, Verdict ONNX, Laya multilingual and Julia-1 are evaluated through documented binary-relevance adapters on a common BGE hybrid shortlist. The tested native pipeline scores query–candidate pairs for binary relevance over a fixed top-20 shortlist. Native listwise choice routing over short skill labels is a different pipeline and was not tested; twenty full discovery headers exceed some native head budgets. Poor pairwise-adapter results therefore do not establish that native choice routing is generally poor, including for a Jev-like router. Full-catalog retrieval and shortlist reranking are different operations. Preserve model revisions, runtime/precision, prompts, pooling, token limits, calibration, input truncation and abstention treatment. CPU configuration and process peak RSS come from fresh run receipts; historical hardware is not assumed to describe the recovery process.'),
  markdown('latency_scope','Latency requires fresh requests and identical observations','The primary timing comparison will use identical 32-query × 2-repeat observations at the full catalog. It includes fresh BGE query encoding, BM25/vector scoring, RRF retrieval and the audited native reference adapter where applicable. Native time includes token preflight, query fitting and scoring; score caches are not reused for timing. Corpus/index preparation is reported separately. BGE runs inside each native dependency environment: Transformers 5.0.0 for Julia, 4.57.6 for GLiNER, and 5.18 for the others. Fresh shortlist equality with the frozen reference is asserted. These are reference-pipeline timings, not isolated model-forward times. Smaller-catalog decision quality may have no timing measurement. Serial QPS is reciprocal mean latency, not concurrent throughput. Process peak RSS is a high-water mark, not model-resident memory. Per-stage measurements preserve retrieval versus native-adapter cost; a sum of stage medians need not equal the median total.'),
  markdown('historical_scope','Historical full-catalog aggregates survive, but their original rows do not','The following values were recorded from completed earlier readouts and independent audits. Their original per-query and per-timing files are currently unavailable. The 4,392-query denominator and older 64-query × 3-repeat timing design are retained as historical context only. No historical mean is mixed into a fresh common32/common64 comparison, confidence interval, or paired test.','historical')]
 charts=[chart('historical_mrr','Retained full-catalog MRR@5 observations','historical','label','mrr5','historical','Historical 4,392-query aggregate observations; original raw files unavailable. Not the fresh recovery cohort.')]
 tables=[table('historical_exact','Retained historical values (raw unavailable)','historical',[
  ('label','System',None),('queries','Accuracy queries','number'),('mrr5','MRR@5','number'),('recall5','Recall@5','percent'),('hit1','Hit@1','percent'),('p50_ms','Historical median ms','number'),('p95_ms','Historical p95 ms','number')],'historical',('mrr5','desc'))]
 blocks += [{'id':'historical_mrr_visual','type':'chart','chartId':'historical_mrr'},{'id':'historical_exact_visual','type':'table','tableId':'historical_exact'},
  markdown('historical_fast_scope','Appendix: the earlier eight-query test had no MRR-improvement headroom','Every model used the same eight queries and BGE hybrid top-10 shortlist. Initial MRR@5 and the candidate-any-hit ceiling were both 0.75, leaving no possibility of MRR improvement. This sample tested preservation or degradation only. One timing observation per query does not support a reliable p95. Initial ONNX defaults differed from later spinning-off settings, so those timing regimes are never pooled. Laya latency was recovered from a flushed log at 0.1 ms precision. These are retained observations; original raw fast files are unavailable.','historical')]
 tables.append(table('historical_fast_exact','Retained exploratory eight-query observations','historical_fast8',[
  ('label','Model',None),('queries','Queries','number'),('mrr5','MRR@5','number'),('median_ms_reported_rounded2','Recorded median ms','number')],'historical',('label','asc')))
 blocks += [{'id':'historical_fast_visual','type':'table','tableId':'historical_fast_exact'},
  markdown('limitations','Interpretation is limited by representation, sample size, and evidence availability','Historical aggregates have no available raw-data audit trail in this recovery workspace. Fresh results must establish their own provenance and common query denominators. Gold-preserving sampled catalogs, short text, finite-choice adapter design, truncation, shortlist misses and shared-host contention may affect outcomes. Native-reranker conclusions apply to the fixed binary prompt, short-header representation, frozen shortlist and released relevance judgments. Missing qrels do not prove an alternative skill is useless. Rankings are descriptive, not causal or universal model superiority claims. No uncertainty estimate is invented from unavailable original rows.'),
  markdown('next_steps','Use only what finishes inside the approved recovery budget','Complete and preserve each major fresh stage without extending the inference cutoff. Prefer common64 only when every required model/condition completes; otherwise use the preregistered shared common48 or common32 fallback and disclose missing coverage or timing. Extension cohort size is chosen from measured runtime budget, before inspecting extension accuracy; the first48 prefix was approved before any extension results. Keep the historical evidence separate. Do not resume work after the approved stop without new authorization.'),
  markdown('questions','Questions remaining after a CPU benchmark','Would the measured result transfer to full skill text, a user-specific catalog, different CPU hardware, or a different deployment latency budget? Which multi-skill misses or abstentions matter most to the actual workflow?')]
 artifact={'surface':'report','manifest':{'version':1,'surface':'report','title':title,'description':'Fresh CPU recovery benchmark with historical retained observations explicitly separated.','generatedAt':now,'cards':[],'charts':charts,'tables':tables,'sources':sources,'blocks':blocks},'snapshot':{'version':1,'generatedAt':now,'status':'partial','datasets':datasets,'accessIssues':[{'id':'fresh_results_pending','dataset':'historical','message':'Fresh recovery evaluation is incomplete. Historical aggregates are not a replacement for missing original raw data.'}]},'sources':[]}
 if fresh:
  from report_fresh import augment_report
  artifact=augment_report(root,artifact,fresh)
 return artifact

def find_plugin(explicit=None):
 candidates=[Path(explicit)] if explicit else [Path(p) for p in glob.glob('/home/user/.codex/plugins/cache/openai-curated-remote/data-analytics/*')]
 for p in sorted(candidates,reverse=True):
  if (p/'skills/build-report/scripts/deliver_portable_artifact.mjs').is_file():return p
 raise RuntimeError('Installed canonical report renderer unavailable; no custom HTML fallback is permitted')

def deliver(root,artifact,out,plugin=None,artifact_only=False):
 out.mkdir(parents=True,exist_ok=True);path=out/'artifact.json';write_json(path,artifact);plugin=find_plugin(plugin)
 if artifact_only:
  code="const s=require(process.argv[1]);const fs=require('fs');const r=s.callTool('validate_artifact',JSON.parse(fs.readFileSync(process.argv[2],'utf8')));delete r.artifact_payload;console.log(JSON.stringify(r));"
  command=['node','-e',code,str(plugin/'mcp/server.cjs'),str(path)]
 else:command=['npm','run','report:deliver','--','--input',str(path),'--output',str(out/'report.html')]
 result=subprocess.run(command,cwd=plugin,text=True,capture_output=True)
 lines=[x for x in result.stdout.splitlines() if x.startswith('{')]
 receipt=json.loads(lines[-1]) if lines else {'ok':False,'stderr':result.stderr}
 receipt.update(returncode=result.returncode,artifact_sha256=sha256(path),snapshot_status=artifact['snapshot']['status'])
 write_json(out/'report_build_receipt.json',receipt)
 print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
 return result.returncode if result.returncode else (0 if receipt.get('ok') else 1)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output-dir',type=Path);p.add_argument('--checkpoint',action='store_true');p.add_argument('--input-audit',action='store_true',help='Read final stable input audits only after measured phases stop');p.add_argument('--artifact-only',action='store_true');p.add_argument('--plugin-root',type=Path);a=p.parse_args()
 root=a.root.resolve();out=(a.output_dir or root/'artifacts').resolve()
 fresh=None
 if (root/'src/report_fresh.py').exists():
  from report_fresh import load_fresh_evidence
  fresh=load_fresh_evidence(root,audit_inputs=a.input_audit)
 artifact=make_artifact(root,fresh)
 if a.checkpoint:
  artifact['snapshot']['status']='partial'
  artifact['snapshot'].setdefault('accessIssues',[]).append({'id':'checkpoint','dataset':'fresh_quality' if fresh else 'historical','message':'Checkpoint of currently completed evidence while the approved recovery is in progress.'})
 if fresh:out.mkdir(parents=True,exist_ok=True);write_json(out/'report_analysis.json',fresh)
 (root/'src/report_aggregation.sql').write_text(QUALITY_SQL+'\n',encoding='utf-8')
 return deliver(root,artifact,out,a.plugin_root,a.artifact_only)

if __name__=='__main__':
 try:raise SystemExit(main())
 except (ValueError,RuntimeError,FileNotFoundError,KeyError) as e:print('Report blocked: '+str(e),file=sys.stderr);raise SystemExit(2)
