"""Fresh recovery evidence, common-cohort gates, and report augmentation.

Reads only completed recovery outputs. Historical aggregate observations are
never considered a source of per-query data or fresh metrics.
"""
from __future__ import annotations
import csv,json,math,statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone
import build_report as br


COHORT_PRIORITY=(64,48,32)
BOOTSTRAP_SQL='SELECT * FROM verified_paired_bootstrap ORDER BY catalog_size, model, method;'
ALIASES={'julia':'julia-1','gliner':'gliner2.5-small','verdict':'verdict-151m-onnx','laya':'laya-multilingual'}
def canonical_model(value):return ALIASES.get(value,value)
def model_file(root,directory,model,suffix):
    names=[model]+[alias for alias,canonical in ALIASES.items() if canonical==model]
    paths=[root/'results'/directory/(name+suffix) for name in names]
    existing=[p for p in paths if p.exists()]
    if len(existing)>1:
        if len({br.sha256(p) for p in existing})>1:raise ValueError(f'{directory}/{model}: ambiguous differently-valued alias files')
    return existing[0] if existing else paths[0]

def rank_metrics(ranked,gold):
 gold=set(gold);rel=[int(x in gold) for x in ranked[:5]];first=next((i+1 for i,v in enumerate(rel) if v),None)
 ideal=sum(1/math.log2(i+2) for i in range(min(5,len(gold))))
 return {'mrr5':1/first if first else 0.,'recall1':(rel[0] if rel else 0)/len(gold),'recall5':sum(rel)/len(gold),
  'hit1':float(bool(rel and rel[0])),'hit5':float(any(rel)),'ndcg5':sum(v/math.log2(i+2) for i,v in enumerate(rel))/ideal,
  'complete5':float(sum(rel)==len(gold))}


def read_completed_quality(root,directory,model,queries,skills):
 path=model_file(root,directory,model,'_per_query.csv');run_path=path.with_name(path.name.replace('_per_query.csv','_run.json'))
 if not br.completed_csv_is_current(root,path,run_path):return None
 run=br.read_json(run_path)
 if run.get('pilot') or run.get('partial') or run.get('status','completed')!='completed':return None
 rows=[];seen=set()
 with path.open(newline='',encoding='utf-8') as f:
  reader=csv.DictReader(f)
  required={'model','method','catalog_size','query_id','gold_count','ranked_skill_ids',*br.METRICS}
  if not required<=set(reader.fieldnames or []):raise ValueError(f'{directory}/{model}: missing required result columns')
  for line,raw in enumerate(reader,2):
   context=f'{directory}/{path.name}:{line}';size=int(raw['catalog_size']);qid=raw['query_id'];key=(raw['method'],size,qid)
   if canonical_model(raw['model'])!=model or size not in br.SIZES or qid not in queries or key in seen:raise ValueError(f'{context}: unexpected or duplicate result key')
   seen.add(key);gold=queries[qid]['gold_skill_ids'];group='multi' if len(gold)>1 else 'single'
   if int(raw['gold_count'])!=len(gold) or raw.get('query_group',group)!=group:raise ValueError(f'{context}: gold cardinality mismatch')
   ranked=json.loads(raw['ranked_skill_ids'])
   if not isinstance(ranked,list) or len(set(ranked))!=len(ranked) or not set(ranked)<=skills:raise ValueError(f'{context}: invalid ranking identifiers')
   row={**raw,'model':model,'catalog_size':size,'gold_count':len(gold),'query_group':group,'ranked':ranked}
   for m in br.METRICS:
    row[m]=br.number(raw[m],m,context)
    if not 0<=row[m]<=1:raise ValueError(f'{context}: metric outside [0,1]')
   for m,value in rank_metrics(ranked,gold).items():
    if not math.isclose(row[m],value,rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f'{context}: {m} does not match ranking/qrels')
   row['shortlist_mrr5']=br.number(raw.get('shortlist_mrr5'),'shortlist_mrr5',context,optional=True)
   row['query_truncated_pairs']=int(raw.get('query_truncated_pairs') or raw.get('query_truncated_pair_incidence')) if raw.get('query_truncated_pairs') not in (None,'') or raw.get('query_truncated_pair_incidence') not in (None,'') else None
   rows.append(row)
 return {'rows':rows,'run':run,'files':[path,run_path],'directory':directory}


def exact_grid(run,models_methods,expected_ids):
 if run is None:return False
 groups=defaultdict(set)
 for row in run['rows']:groups[(row['method'],row['catalog_size'])].add(row['query_id'])
 return all(groups[(m,size)]==set(expected_ids) for m in models_methods for size in br.SIZES)


def load_latency(root,directory,model,expected_methods,expected_ids,quality_run=None):
 path=model_file(root,directory,model,'_latency.csv');run_path=path.with_name(path.name.replace('_latency.csv','_run.json'))
 if not path.exists() or not run_path.exists() or path.stat().st_mtime_ns>run_path.stat().st_mtime_ns:return [],[],None
 meta=br.read_json(run_path)
 if meta.get('pilot') or meta.get('skip_timing') or meta.get('status','completed')!='completed':return [],[],None
 if quality_run:
  qc=quality_run.get('configuration',quality_run.get('config'));tc=meta.get('configuration',meta.get('config'))
  if qc is not None and tc is not None and qc!=tc:raise ValueError(f'{directory}/{model}: configuration differs from selected quality run')
  if quality_run.get('score_cache_fingerprint')!=meta.get('score_cache_fingerprint'):raise ValueError(f'{directory}/{model}: score fingerprint differs from selected quality run')
 groups=defaultdict(list);seen=set()
 with path.open(newline='',encoding='utf-8') as f:
  for line,row in enumerate(csv.DictReader(f),2):
   method=row['method'];size=int(row['catalog_size']);qid=row['query_id'];repeat=int(row['repeat'])
   if canonical_model(row['model'])!=model or size!=6006 or method not in expected_methods:raise ValueError(f'{path.name}:{line}: unexpected timing condition')
   if qid not in expected_ids or repeat not in (0,1):raise ValueError(f'{path.name}:{line}: observation outside matched32×2 timing scope')
   key=(method,qid,repeat)
   if key in seen:raise ValueError(f'{path.name}:{line}: duplicate timing observation')
   seen.add(key);field='full_pipeline_ms' if row.get('full_pipeline_ms') not in (None,'') else 'total_ms'
   value=br.number(row.get(field),field,path.name)
   if value<0:raise ValueError(f'{path.name}:{line}: negative latency')
   groups[method].append({'query_id':qid,'repeat':repeat,'total_ms':value,
    'rerank_ms':br.number(row.get('rerank_ms'),'rerank_ms',path.name,optional=True),
    'stage1_ms':br.number(row.get('stage1_ms'),'stage1_ms',path.name,optional=True)})
 result=[];expected={(q,r) for q in expected_ids for r in (0,1)}
 for method in expected_methods:
  rows=groups[method]
  if {(r['query_id'],r['repeat']) for r in rows}!=expected:continue
  values=[r['total_ms'] for r in rows]
  result.append({'model':model,'method':method,'label':br.LABELS[model]+' · '+method,'catalog_size':6006,
    'queries':len(expected_ids),'repeats':2,'observations':len(rows),'p50_ms':br.percentile(values,.5),'p95_ms':br.percentile(values,.95),
    'min_ms':min(values),'max_ms':max(values),'mean_ms':statistics.fmean(values),'serial_qps':1000/statistics.fmean(values),
    'rerank_p50_ms':br.percentile([r['rerank_ms'] for r in rows if r['rerank_ms'] is not None],.5),
    'stage1_p50_ms':br.percentile([r['stage1_ms'] for r in rows if r['stage1_ms'] is not None],.5)})
 return result,[path,run_path],meta


def summary_rows(rows,ids,cohort):
 subset=[r for r in rows if r['query_id'] in ids]
 result=br.aggregate_sql(subset);bykey=defaultdict(list)
 for row in subset:bykey[(row['model'],row['method'],row['catalog_size'])].append(row)
 for row in result:
  actual=bykey[(row['model'],row['method'],row['catalog_size'])]
  if {r['query_id'] for r in actual}!=set(ids):raise ValueError('Aggregate is not the requested exact cohort')
  row.update(label=br.LABELS[row['model']]+' · '+row['method'],cohort=cohort,cohort_hash=br.cohort_hash(ids),
    query_truncated_pairs=sum(r['query_truncated_pairs'] for r in actual if r['query_truncated_pairs'] is not None) if any(r['query_truncated_pairs'] is not None for r in actual) else None)
  row['mrr5_delta']=row['mrr5']-row['shortlist_mrr5'] if row['shortlist_mrr5'] is not None else None
  row['mrr5_headroom']=row['candidate_hit']-row['shortlist_mrr5'] if row['shortlist_mrr5'] is not None else None
 return result


def choose_common_size(completed_by_size,retrieval_complete=True):
 if not retrieval_complete:return None
 return next((n for n in COHORT_PRIORITY if all(completed_by_size.get(n,{}).get(m,False) for m in br.DECISIONS)),None)

def input_audits(root,ids,retrieval,decision):
 result=[];paths=[]
 for model,run in retrieval.items():
  if model=='bm25' or run is None:continue
  cfg=run['run'].get('model_configuration',{});docs=cfg.get('documents',{});queries=cfg.get('queries',{})
  da=docs.get('token_audit',{});qa=queries.get('token_audit',{})
  result.append({'model':model,'kind':'embedding token limit','query_denominator':queries.get('count'),
   'queries_shortened':qa.get('truncated_count'),'pair_denominator':None,'pairs_shortened':None,
   'document_denominator':docs.get('count'),'documents_shortened':da.get('truncated_count'),
   'candidate_cut_pairs':None,'residual_truncation_pairs':None,
   'scope_note':'Counts cover the full 64 encoded recovery queries; not relabeled as the smaller primary cohort.'})
 for model,run in decision.items():
  path=model_file(root,'pair_scores',model,'.jsonl')
  meta_path=path.with_name(path.stem+'_metadata.json')
  if not path.exists() or not meta_path.exists():raise ValueError(f'{model}: required native pair audit unavailable')
  meta=br.read_json(meta_path)
  if meta.get('fingerprint')!=run['run'].get('score_cache_fingerprint'):raise ValueError(f'{model}: pair audit fingerprint differs from selected quality run')
  allowed={(r['query_id'],sid) for r in run['rows'] if r['query_id'] in ids for sid in r['ranked']}
  before=path.stat();records={}
  with path.open(encoding='utf-8') as f:
   for line in f:
    row=json.loads(line);pair=(row['query_id'],row['skill_id'])
    if pair not in allowed:continue
    if pair in records:raise ValueError(f'{model}: duplicate unique-pair audit')
    records[pair]=row.get('audit',{})
  after=path.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input audit changed while reading; retry only after the inference stage stops')
  if set(records)!=allowed:raise ValueError(f'{model}: incomplete native pair audit coverage')
  shortened={qid for (qid,sid),a in records.items() if a.get('query_truncated')}
  limit=run['run'].get('configuration',run['run'].get('config',{})).get('max_length',512 if model=='verdict-151m-onnx' else 1024)
  result.append({'model':model,'kind':'native query-only pair fitting','query_denominator':len(ids),'queries_shortened':len(shortened),
   'pair_denominator':len(records),'pairs_shortened':sum(bool(a.get('query_truncated')) for a in records.values()),
   'document_denominator':None,'documents_shortened':None,
   'candidate_cut_pairs':sum(a.get('candidate_preserved') is False for a in records.values()),
   'residual_truncation_pairs':sum(bool(a.get('stateTruncated') or a.get('optionsTruncated') or a.get('tokens',0)>limit) for a in records.values()),
   'scope_note':'Unique query-candidate pairs across all selected catalogs; duplicate catalog incidences are not summed.'})
  paths += [path,meta_path]
 return result,paths


def load_fresh_evidence(root,*,audit_inputs=False):
 query_path=root/'data/processed/queries.jsonl'
 if not query_path.exists():return {'status':'blocked','reason':'Fresh frozen query file is not yet available','primary_count':None,'summary':[],'timing':[],'coverage':[],'input_files':[]}
 query_rows=[json.loads(x) for x in query_path.read_text().splitlines() if x.strip()]
 queries={r['query_id']:r for r in query_rows};order=[r['query_id'] for r in query_rows]
 expected_order=br.read_json(root/'data/processed/selected_query_ids.json') if (root/'data/processed/selected_query_ids.json').exists() else order
 if order!=expected_order or len(order)!=64 or len(set(order))!=64:raise ValueError('Recovery queries must be the exact frozen, ordered common64 population')
 skills=set(br.read_json(root/'data/processed/full_catalog.json'));files=[query_path,root/'data/processed/full_catalog.json']
 extension_target=64;choice_path=root/'protocol/extension_choice.json'
 finalization_path=root/'protocol/report_finalization.json';finalization=br.read_json(finalization_path) if finalization_path.exists() else None
 if finalization:files.append(finalization_path)
 if choice_path.exists():
  choice=br.read_json(choice_path);extension_target=choice['query_count']
  if extension_target not in COHORT_PRIORITY or choice['query_ids']!=order[:extension_target]:raise ValueError('Approved extension prefix differs from frozen query order')
  files.append(choice_path)
 retrieval={};coverage=[];all_runs=[]
 for model in ('bm25',*br.EMBEDDINGS):
  methods=('bm25',) if model=='bm25' else ('vector','hybrid','domain_vector')
  run=read_completed_quality(root,'retrieval64',model,queries,skills)
  ok=exact_grid(run,methods,order)
  retrieval[model]=run if ok else None
  coverage.append({'model':model,'stage':'retrieval64','queries':64,'complete':ok,'detail':'Exact required method × four-catalog grid' if ok else 'Incomplete or unavailable'})
  if run:files+=run['files'];all_runs.append({'model':model,'stage':'retrieval64','metadata':run['run']})
 available_retrieval=[]
 for run in retrieval.values():
  if run:available_retrieval+=summary_rows(run['rows'],order,'fresh-retrieval-64')
 decision={};chosen=None;available_native=[];decisions_by_size={}
 for count in COHORT_PRIORITY:
  candidates={}
  if count>extension_target:
   for model in br.DECISIONS:coverage.append({'model':model,'stage':'quality'+str(count),'queries':count,'complete':None,'detail':'Not scheduled: approved runtime-only target is '+str(extension_target)})
   continue
  for model in br.DECISIONS:
   run=read_completed_quality(root,'quality'+str(count),model,queries,skills)
   ok=exact_grid(run,('hybrid_rerank',),order[:count]);candidates[model]=run if ok else None
   coverage.append({'model':model,'stage':'quality'+str(count),'queries':count,'complete':ok,'detail':'Exact four-catalog shared cohort' if ok else 'Incomplete or unavailable'})
   if run:files+=run['files'];all_runs.append({'model':model,'stage':'quality'+str(count),'metadata':run['run']})
   if ok:available_native+=summary_rows(run['rows'],order[:count],'fresh-native-'+str(count))
  decisions_by_size[count]=candidates
 completed_map={n:{m:bool(r) for m,r in runs.items()} for n,runs in decisions_by_size.items()}
 if finalization:
  required_final=finalization['primary_quality_queries']
  completed_map={n:v for n,v in completed_map.items() if n==required_final}
 chosen=choose_common_size(completed_map,all(retrieval.values()))
 decision=decisions_by_size[chosen] if chosen else {}
 ids=order[:chosen] if chosen else [];summary=[];segments=[];paired=[]
 if chosen:
  primary_rows=[r for run in retrieval.values() for r in run['rows'] if r['query_id'] in ids]+[r for run in decision.values() for r in run['rows']]
  baseline={(r['catalog_size'],r['query_id']):r for r in retrieval['bge-small']['rows'] if r['method']=='hybrid'}
  for run in decision.values():
   for row in run['rows']:
    base=baseline[(row['catalog_size'],row['query_id'])]
    if row['shortlist_mrr5'] is None or not math.isclose(row['shortlist_mrr5'],base['mrr5'],abs_tol=1e-10):raise ValueError('Decision shortlist MRR differs from independently measured BGE hybrid baseline')
    if set(row['ranked'])!=set(base['ranked']):raise ValueError('Decision ranking does not preserve the frozen BGE shortlist membership')
    for field in ('candidate_recall','candidate_hit'):
     if not math.isclose(row[field],base[field],abs_tol=1e-10):raise ValueError('Decision candidate coverage differs from frozen BGE baseline')
  summary=summary_rows(primary_rows,ids,'fresh-common'+str(chosen))
  grouped=defaultdict(list)
  for row in primary_rows:grouped[(row['model'],row['method'],row['catalog_size'])].append(row)
  for key,rows in grouped.items():
   for group in ('single','multi'):
    subset=[r for r in rows if r['query_group']==group]
    if subset:segments.append({'model':key[0],'method':key[1],'catalog_size':key[2],'query_group':group,'queries':len(subset),**{m:statistics.fmean(r[m] for r in subset) for m in br.METRICS}})
   if key[1]=='hybrid_rerank':
    delta=[r['mrr5']-r['shortlist_mrr5'] for r in rows]
    paired.append({'model':key[0],'method':key[1],'catalog_size':key[2],'queries':len(rows),'cohort_hash':br.cohort_hash(ids),
      'baseline':'BGE hybrid, same-query frozen shortlist','mrr5_delta':statistics.fmean(delta),'wins':sum(x>0 for x in delta),'ties':sum(x==0 for x in delta),'losses':sum(x<0 for x in delta)})
 timing=[];timing_meta=[]
 for model in ('bm25',*br.EMBEDDINGS,*br.DECISIONS):
  native=model in br.DECISIONS;methods=('hybrid_rerank',) if native else (('bm25',) if model=='bm25' else ('vector','hybrid','domain_vector'))
  run=decision.get(model) if native else retrieval.get(model)
  rows,paths,meta=load_latency(root,'timing32' if native else 'retrieval64',model,methods,order[:32],run['run'] if run else None)
  timing+=rows;files+=paths
  if meta:timing_meta.append({'model':model,'metadata':meta})
 required_timing={('bm25','bm25'),*((m,'hybrid') for m in br.EMBEDDINGS),*((m,'hybrid_rerank') for m in br.DECISIONS)}
 completed_timing={(r['model'],r['method']) for r in timing};missing_timing=sorted(required_timing-completed_timing)
 state={'state':'running_or_not_recorded'}
 for p in (root/'protocol/terminal_status.json',root/'protocol/controller_status.json',root/'results/recovery_status.json',root/'results/expanded_status.json'):
  if p.exists():state=br.read_json(p);state.setdefault('state',state.get('status','unknown'));files.append(p);break
 audit_rows=[]
 if audit_inputs and chosen:
  audit_rows,audit_paths=input_audits(root,ids,retrieval,decision);files+=audit_paths
 intervals=[];bootstrap_path=root/'docs/notebook_paired_bootstrap.json'
 if chosen and bootstrap_path.exists():
  bootstrap=br.read_json(bootstrap_path)
  if bootstrap.get('schema_version')!=2 or bootstrap.get('selected_decision_query_count')!=chosen:raise ValueError('Bootstrap cohort does not match final primary cohort')
  for source_entry in bootstrap.get('source_sha256',[]):
   source_path=(root/source_entry['path']).resolve()
   if not source_path.is_relative_to(root.resolve()) or not source_path.exists() or br.sha256(source_path)!=source_entry['sha256']:raise ValueError('Bootstrap source fingerprint mismatch: '+source_entry['path'])
  summary_map={(r['model'],r['method'],r['catalog_size']):r for r in summary}
  for row in bootstrap['comparisons']:
   if row['queries']!=chosen or row['cohort_hash']!=br.cohort_hash(ids):raise ValueError('Bootstrap comparison has a mismatched query population')
   model=canonical_model(row['model']);baseline_model,baseline_method=row['baseline_id'].split('/')
   measured=summary_map[(model,row['method'],row['catalog_size'])]['mrr5']-summary_map[(canonical_model(baseline_model),baseline_method,row['catalog_size'])]['mrr5']
   if not math.isclose(row['mrr5_delta'],measured,abs_tol=1e-10) or row['ci95_low']>row['ci95_high']:raise ValueError('Bootstrap interval disagrees with independently aggregated report values')
   if row['wins']+row['ties']+row['losses']!=chosen:raise ValueError('Bootstrap paired counts do not add to cohort size')
   intervals.append({**row,'model':model})
  import sqlite3
  db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
  columns=['model','method','catalog_size','cohort','cohort_hash','baseline_id','queries','mrr5_delta','ci95_low','ci95_high','wins','ties','losses','seed','bootstrap_replicates','candidate_depth']
  integer={'catalog_size','queries','wins','ties','losses','seed','bootstrap_replicates','candidate_depth'}
  real={'mrr5_delta','ci95_low','ci95_high'}
  db.execute('CREATE TABLE verified_paired_bootstrap ('+', '.join(c+' '+('INTEGER' if c in integer else 'REAL' if c in real else 'TEXT') for c in columns)+')')
  db.executemany('INSERT INTO verified_paired_bootstrap VALUES ('+','.join('?' for c in columns)+')',[tuple(r.get(c) for c in columns) for r in intervals])
  intervals=[dict(r) for r in db.execute(BOOTSTRAP_SQL)];db.close()
  files.append(bootstrap_path)
 file_info=[{'path':br.relpath(p,root),'sha256':br.sha256(p),'bytes':p.stat().st_size} for p in sorted(set(files))]
 for row in coverage:
  if row['stage']=='retrieval64' and retrieval.get(row['model']):row['methods']=len(('bm25',) if row['model']=='bm25' else ('vector','hybrid','domain_vector'))
 in_progress=not bool(finalization and finalization.get('inference_stopped')) and not (str(state.get('state','')).startswith('completed') or state.get('state') in ('stopped','deadline_reached')) and datetime.now(timezone.utc)<datetime.fromisoformat('2026-10-02T11:52:00+00:00')
 return {'finalization':finalization,'status':'ready' if chosen and not missing_timing and not in_progress else 'partial','in_progress':in_progress,'primary_count':chosen,'extension_target':extension_target,'primary_ids':ids,'cohort_hash':br.cohort_hash(ids) if ids else None,
  'intervals':intervals,'input_audits':audit_rows,'summary':summary,'available_retrieval':available_retrieval,'available_native':available_native,'segments':segments,'paired':paired,'timing':timing,'coverage':coverage,'runs':all_runs,'timing_runs':timing_meta,
  'missing_timing':missing_timing,'controller':state,'input_files':file_info,
  'readiness':'Common64 when all models finish, else preregistered common48, else common32; exact eight-pipeline full-catalog32×2 timing required.'}


def augment_report(root,artifact,evidence):
 manifest=artifact['manifest'];snapshot=artifact['snapshot'];blocks=manifest['blocks'];count=evidence.get('primary_count')
 datasets=snapshot['datasets'];datasets.update(fresh_quality=evidence.get('summary',[]),fresh_segments=evidence.get('segments',[]),fresh_timing=evidence.get('timing',[]),fresh_coverage=evidence.get('coverage',[]),fresh_paired=evidence.get('paired',[]))
 for i,f in enumerate(evidence.get('input_files',[])):
  manifest['sources'].append(br.source('fresh_file_'+str(i),f['path'],f['path'],'Fresh recovery evidence; SHA256 '+f['sha256']))
 manifest['sources'].append(br.source('fresh_dataset','Pinned fresh recovery dataset','data/processed/manifest.json','Pinned source checksums, scoreable text, frozen hash-order common64 and nested catalogs.'))
 state=evidence.get('controller',{}).get('state','not recorded')
 coverage_text=f'Fresh primary cohort: {count if count else "not complete"}. The measured recovery evidence is reported below. Retrieval must complete its exact 64-query method × four-catalog grid; all four native models must share common64, the preregistered common48 prefix, or the same common32 fallback. Timings require complete identical 32×2 observations. Stage completion and historical observations remain separate.'
 if evidence.get('finalization'):
  title_block=next(b for b in blocks if b['id']=='title');title_block['body']=title_block['body'].replace('read-only checkpoint','final measured common32 result')
  for row in evidence.get('coverage',[]):
   if row['stage']=='quality48':row['detail']='Completed supplement only; excluded from common32 primary' if row['complete'] else 'Extension stopped; no completed result and no retry'
   if row['stage']=='quality64':row['detail']='Not scheduled after the runtime-only48 choice; final primary is32'
  recovery_block=next(b for b in blocks if b['id']=='recovery_status')
  recovery_block['body']='## The recovery run is complete on the common32 scope\n\nThe original raw workspace became unavailable; no missing raw rows were reconstructed from aggregate means. Fresh common32 quality and matched32-query ×2-repeat timings are complete. The attempted48-query extension ended after a safety restriction involving Microsoft telemetry blocked Verdict48; no inference retry occurred. Completed Julia48 and GLiNER48 measurements are supplements only, and native64 was not scheduled. Historical4,392-query aggregates remain genuinely observed evidence with unavailable original raw files. Both Qwen entries remain excluded.'
 pending=next(b for b in blocks if b['id']=='fresh_pending');pending['body']='## Fresh run coverage\n\n'+coverage_text
 if evidence.get('finalization'):
  pending['body']+=' Final primary scope is the complete common'+str(count)+' result. '+evidence['finalization']['reason']+' Completed larger-prefix results are supplements only.'
 else:pending['body']+=' Approved runtime-only extension target: '+str(evidence.get('extension_target',64))+'. Larger native cohorts are not scheduled.'
 fields=[('model','Model',None),('stage','Stage',None),('queries','Planned queries','number'),('complete','Complete',None),('detail','Check',None)]
 manifest['tables'].append(br.table('fresh_coverage_table','Fresh stage completion','fresh_coverage',fields,'fresh_transform',('model','asc')))
 insert_at=blocks.index(pending)+1;blocks.insert(insert_at,{'id':'fresh_coverage_visual','type':'table','tableId':'fresh_coverage_table'})
 if not count:
  extra=[]
  fields=[('label','System',None),('cohort','Cohort',None),('catalog_size','Catalog','number'),('queries','Queries','number'),('mrr5','MRR@5','number'),('recall5','Recall@5','percent'),('shortlist_mrr5','Initial shortlist MRR','number')]
  for name,label in (('available_retrieval','Completed fresh retrieval checkpoints'),('available_native','Completed native quality checkpoints')):
   rows=evidence.get(name,[])
   if rows:
    datasets[name]=rows
    extra.append(br.markdown(name+'_section',label,'These are completed fresh measurements with the denominator shown in every row. The all-model primary comparison is not complete; missing systems and different cohort sizes are not silently pooled or ranked as an overall result.','fresh_transform'))
    manifest['tables'].append(br.table(name+'_table',label,name,fields,'fresh_transform',('label','asc')))
    extra.append({'id':name+'_visual','type':'table','tableId':name+'_table'})
  full=[r for r in evidence.get('available_retrieval',[]) if r['catalog_size']==6006]
  if full:
   best=max(full,key=lambda r:r['mrr5'])
   summary_block=next(b for b in blocks if b['id']=='summary')
   summary_block['body']=f"## What the completed fresh retrieval stage establishes\n\nFresh retrieval has {len(evidence['available_retrieval'])} completed model/method/catalog conditions on the same 64 queries. Among the available full-catalog retrieval conditions, {best['label']} has the highest observed MRR@5 ({best['mrr5']:.4f}). The native decision-model panel is still incomplete, so this is a retrieval-stage checkpoint, not the final eight-pipeline comparison. Earlier 4,392-query aggregates remain genuinely observed historical evidence with unavailable original raw files."
   datasets['checkpoint_fresh_retrieval']=full
   manifest['charts'].append(br.chart('checkpoint_fresh_retrieval_chart','Fresh full-catalog retrieval MRR@5','checkpoint_fresh_retrieval','label','mrr5','fresh_transform','Same 64 fresh queries; 6,006 skills. Native decision comparisons are still pending.'))
   extra.insert(1,{'id':'checkpoint_fresh_retrieval_visual','type':'chart','chartId':'checkpoint_fresh_retrieval_chart'})
  pos=next(i for i,b in enumerate(blocks) if b['id']=='methodology');blocks[pos:pos]=extra
  snapshot['status']='partial';return artifact
 primary=evidence['summary'];cross=[r for r in primary if r['catalog_size']==6006 and r['method'] in ('bm25','hybrid','hybrid_rerank')]
 if len(cross)!=8:raise ValueError('Matched full-catalog comparison must contain exactly eight pipelines')
 best=max(cross,key=lambda r:r['mrr5']);baseline=next(r for r in cross if r['model']=='bge-small' and r['method']=='hybrid')
 title=next(b for b in blocks if b['id']=='summary')
 title['body']=f"## What the fresh common-cohort evidence establishes\n\nOn the same {count} fresh queries and full 6,006-skill catalog, the highest observed MRR@5 is {best['mrr5']:.4f} for {best['label']}. The BGE hybrid initial shortlist reaches {baseline['mrr5']:.4f}; its candidate-any-hit ceiling is {baseline['candidate_hit']:.4f}. This is a matched, descriptive comparison, not a universal model ranking or a significance claim. Historical 4,392-query aggregates remain in a separate section because their original raw files are unavailable."
 new=[br.markdown('fresh_comparison','Eight pipelines use the same fresh query IDs',f'Every row below uses the exact same {count} query IDs. BM25 and the three embedding hybrids search the full catalog. Each native decision pipeline can reorder only the common BGE hybrid shortlist (up to 20 candidates; 10 for the smallest catalog). Candidate-any-hit limits MRR improvement for native pipelines. These native rows are binary pairwise relevance adapters, not native listwise selection over short skill labels; that different choice-routing pipeline remains untested. The separate 32×2 timing sample is identical across included systems.','fresh_transform')]
 datasets['fresh_cross']=cross
 fields=[('label','Pipeline',None),('queries','Queries','number'),('mrr5','MRR@5','number'),('recall5','Recall@5','percent'),('hit1','Hit@1','percent'),('complete5','Complete@5','percent'),('shortlist_mrr5','Native initial MRR','number'),('candidate_hit','Candidate-any-hit','percent')]
 manifest['tables'].append(br.table('fresh_cross_table',f'Full-catalog quality on identical {count} queries','fresh_cross',fields,'fresh_transform',('mrr5','desc')))
 new.append({'id':'fresh_cross_visual','type':'table','tableId':'fresh_cross_table'})
 for size in br.SIZES:
  name='fresh_quality_'+str(size);rows=[r for r in primary if r['catalog_size']==size];datasets[name]=rows;winner=max(rows,key=lambda r:r['mrr5'])
  new.append(br.markdown(name+'_section',f'Observed quality at catalog size {size:,}',f"The highest observed MRR@5 is {winner['mrr5']:.4f} for {winner['label']}. All bars use the same {count} query IDs. Smaller catalogs are controlled gold-preserving tasks; their scores do not forecast deployment scaling. Metadata-assisted domain routing is labeled separately from unrestricted vector search.",'fresh_transform'))
  manifest['charts'].append(br.chart(name,f'MRR@5 at catalog size {size:,}',name,'label','mrr5','fresh_transform',f'Fresh common {count}; exact shared query IDs; higher is better.'))
  new.append({'id':name+'_visual','type':'chart','chartId':name})
 # Keep definitions before the main numerical evidence, then retain methodology
 # and the historical observations as later sections.
 pos=next(i for i,b in enumerate(blocks) if b['id']=='methodology');blocks[pos:pos]=new
 fields=[('label','Pipeline',None),('queries','Queries','number'),('repeats','Repeats','number'),('p50_ms','Median total ms','number'),('stage1_p50_ms','Median retrieval ms','number'),('rerank_p50_ms','Median native adapter ms','number'),('p95_ms','Observed p95 ms','number'),('min_ms','Observed min ms','number'),('max_ms','Observed max ms','number'),('serial_qps','Serial QPS','number')]
 manifest['tables'].append(br.table('fresh_timing_table','Matched full-catalog request timings','fresh_timing',fields,'fresh_transform',('p50_ms','asc'),'Exactly the same first 32 query IDs × 2 repetitions; smaller-catalog native timing is intentionally absent.'))
 pos=next(i for i,b in enumerate(blocks) if b['id']=='latency_scope')+1;blocks.insert(pos,{'id':'fresh_timing_visual','type':'table','tableId':'fresh_timing_table'})
 if evidence.get('paired'):
  fields=[('model','Model',None),('catalog_size','Catalog','number'),('queries','Paired queries','number'),('mrr5_delta','MRR change vs initial shortlist','number'),('wins','Wins','number'),('ties','Ties','number'),('losses','Losses','number')]
  manifest['tables'].append(br.table('fresh_paired_table','Native reranker changes on identical queries','fresh_paired',fields,'fresh_transform',('catalog_size','desc')))
  pos=next(i for i,b in enumerate(blocks) if b['id']=='historical_scope')
  blocks[pos:pos]=[br.markdown('fresh_paired_section','Pairing separates reranking changes from retrieval coverage','Every change subtracts the measured BGE hybrid initial ranking on the same query and catalog. No historical mean enters this comparison. This evaluates the fixed binary pairwise adapter, not native listwise choice routing. Wins, ties and losses use query-level MRR; differences remain descriptive until independently checked uncertainty estimates are available.','fresh_transform'),{'id':'fresh_paired_visual','type':'table','tableId':'fresh_paired_table'}]
 fields=[('model','Model',None),('method','Method',None),('catalog_size','Catalog','number'),('query_group','Gold group',None),('queries','Queries','number'),('recall5','Recall@5','percent'),('complete5','Complete@5','percent')]
 manifest['tables'].append(br.table('fresh_segment_table','Single- and multi-positive query coverage','fresh_segments',fields,'fresh_transform',('catalog_size','desc')))
 pos=next(i for i,b in enumerate(blocks) if b['id']=='historical_scope');blocks[pos:pos]=[br.markdown('fresh_segment_section','Finding one skill does not establish complete multi-skill coverage','Recall@5 and Complete@5 keep all released gold labels in the denominator. Compare single- and multi-positive query groups with their exact sample counts; do not substitute Hit@5 for recall.','fresh_transform'),{'id':'fresh_segment_visual','type':'table','tableId':'fresh_segment_table'}]
 snapshot['status']=evidence['status'];snapshot.pop('accessIssues',None)
 if evidence.get('in_progress'):snapshot['accessIssues']=[{'id':'recovery_in_progress','dataset':'fresh_quality','message':'The fresh recovery benchmark is still running. This is a checkpoint of completed evidence.'}]
 if evidence.get('missing_timing'):snapshot.setdefault('accessIssues',[]).extend([{'id':'fresh_timing_missing','dataset':'fresh_timing','message':'Required matched timing incomplete for '+', '.join(m+'/'+method for m,method in evidence['missing_timing'])+'. Small-catalog decision timing is intentionally excluded.'}])
 if evidence.get('intervals'):
  datasets['fresh_intervals']=evidence['intervals']
  manifest['sources'].append(br.source('fresh_bootstrap','Independent fresh paired-query bootstrap','docs/notebook_paired_bootstrap.json','The independent notebook computed 2,000 paired query-resampling replicates, seed20261002, percentile95% intervals. This SQL imports and projects its stored interval rows, not a SQL recomputation of bootstrap sampling. Exact cohort, all source fingerprints and point differences checked against the canonical report. Descriptive query uncertainty only; no multiplicity adjustment, hardware uncertainty or causal guarantee.',BOOTSTRAP_SQL))
  fields=[('model','Model',None),('method','Method',None),('catalog_size','Catalog','number'),('baseline_id','Baseline',None),('queries','Queries','number'),('mrr5_delta','MRR difference','number'),('ci95_low','95% lower','number'),('ci95_high','95% upper','number')]
  manifest['tables'].append(br.table('fresh_intervals_table','Descriptive paired-query uncertainty','fresh_intervals',fields,'fresh_bootstrap',('catalog_size','desc')))
  pos=next(i for i,b in enumerate(blocks) if b['id']=='historical_scope')
  blocks[pos:pos]=[br.markdown('fresh_intervals_section','Uncertainty remains conditional on this small, fixed benchmark','The independent notebook resamples paired query-level differences2,000 times with seed20261002 and reports2.5th/97.5th percentile intervals. Query IDs, source fingerprints and mean differences match this report. The intervals describe query sampling within this protocol; they do not include training or hardware uncertainty, multiplicity correction, causality or guarantees outside this catalog. Native differences are relative to the shared BGE hybrid initial shortlist; other methods are relative to BM25.','fresh_bootstrap'),{'id':'fresh_intervals_visual','type':'table','tableId':'fresh_intervals_table'}]
  for block in blocks:
   if block.get('id')=='fresh_paired_section':block['body']=block['body'].replace('differences remain descriptive until independently checked uncertainty estimates are available.','point differences remain descriptive; independently checked paired-query intervals are reported below.')
 if evidence.get('finalization'):
  stopped=[r for r in evidence.get('available_native',[]) if r['queries']>count]
  if stopped:
   datasets['partial_extension_supplements']=stopped
   fields=[('label','System',None),('catalog_size','Catalog','number'),('queries','Queries','number'),('mrr5','MRR@5','number'),('shortlist_mrr5','Initial shortlist MRR','number'),('recall5','Recall@5','percent')]
   manifest['tables'].append(br.table('partial_extension_table','Completed larger-prefix supplements','partial_extension_supplements',fields,'fresh_transform',('label','asc')))
   pos=next(i for i,b in enumerate(blocks) if b['id']=='historical_scope')
   blocks[pos:pos]=[br.markdown('partial_extension_section','Appendix: Julia48 and GLiNER48 do not form an all-model48 comparison','Only Julia and GLiNER completed the 48-query extension. Verdict48 execution was blocked by a safety restriction involving a Microsoft telemetry connection; no inference retry occurred and Laya48 was not completed. These measured supplements are preserved, but they never replace the common32 primary comparison or imply a complete48-query result. A prospective privacy change is separate from the executed scorer and recorded measurements.','fresh_transform'),{'id':'partial_extension_visual','type':'table','tableId':'partial_extension_table'}]
 if evidence.get('available_retrieval'):
  datasets['fresh_retrieval64']=evidence['available_retrieval']
  fields=[('label','System',None),('catalog_size','Catalog','number'),('queries','Queries','number'),('mrr5','MRR@5','number'),('recall5','Recall@5','percent'),('hit1','Hit@1','percent'),('complete5','Complete@5','percent')]
  manifest['tables'].append(br.table('fresh_retrieval64_table','Separate fresh retrieval-only64 results','fresh_retrieval64',fields,'fresh_transform',('catalog_size','desc')))
  pos=next(i for i,b in enumerate(blocks) if b['id']=='historical_scope')
  blocks[pos:pos]=[br.markdown('fresh_retrieval64_section','Appendix: complete fresh retrieval64 measurements','Retrieval-only runs measured all64 frozen queries. Their complete64-query means are retained here for audit, separate from the primary matched native/retrieval cohort. The primary comparison filters these raw retrieval rows to the same chosen query IDs; it never contrasts a64-query mean with a48- or32-query mean. This fresh raw evidence is also distinct from historical4,392-query aggregates whose original rows are unavailable.','fresh_transform'),{'id':'fresh_retrieval64_visual','type':'table','tableId':'fresh_retrieval64_table'}]
 next_steps=next(b for b in blocks if b['id']=='next_steps')
 target=evidence.get('extension_target',64)
 next_steps['body']='## Use the approved bounded result\n\n'+('The recovery run is still in progress. Complete only the approved '+str(target)+'-query native target inside the existing cutoff. ' if evidence.get('in_progress') else 'The bounded recovery run has ended. No larger native cohort should be started without a new user request. ')+f'The current primary comparison uses{count} identical query IDs. Full retrieval-only64 and historical4,392-query observations remain separate. Choose a tested hybrid according to the measured quality/CPU trade-off; evaluating native listwise short-label routing would be a different future experiment, not a conclusion established here.'
 if evidence.get('input_audits'):
  datasets['fresh_input_audits']=evidence['input_audits']
  fields=[('model','Model',None),('kind','Audit unit',None),('query_denominator','Audited queries','number'),('queries_shortened','Queries shortened','number'),('pair_denominator','Unique pairs','number'),('pairs_shortened','Pairs shortened','number'),('document_denominator','Documents','number'),('documents_shortened','Documents shortened','number'),('candidate_cut_pairs','Candidate-cut pairs','number')]
  manifest['tables'].append(br.table('fresh_input_audit_table','Measured input truncation','fresh_input_audits',fields,'fresh_transform',('model','asc')))
  pos=next(i for i,b in enumerate(blocks) if b['id']=='historical_scope')
  blocks[pos:pos]=[br.markdown('fresh_input_audit_section','Query lengths differ across model and candidate budgets','Native counts deduplicate query-candidate pairs across all four catalogs and count affected queries once. Query-only fitting preserves the full candidate and label options; the retained prefix can differ by candidate. Embedding counts cover all 64 encoded recovery queries and all 6,006 documents, even when the primary paired cohort is smaller. Missing counts remain unavailable rather than zero. These audits do not isolate the causal effect of truncation on quality.','fresh_transform'),{'id':'fresh_input_audit_visual','type':'table','tableId':'fresh_input_audit_table'}]
 hardware=[]
 for item in evidence.get('runs',[]):
  run=item['metadata'];hw=run.get('hardware',{});modelmeta=run.get('model_configuration',run.get('model',{}))
  if not isinstance(modelmeta,dict):modelmeta={}
  config=run.get('configuration',run.get('config',{}));emb=run.get('embeddings',{})
  hardware.append({'model':item['model'],'stage':item['stage'],'cpu':hw.get('cpu',hw.get('model')), 'threads':hw.get('threads'),
   'runtime':modelmeta.get('runtime'),'precision':modelmeta.get('precision'),'max_tokens':modelmeta.get('max_length',config.get('max_length')),
   'peak_rss_mib':run.get('peak_rss_mib'),'decision_only_peak_rss_mib':run.get('decision_only_peak_rss_mib'),
   'load_seconds':run.get('load_seconds',modelmeta.get('load_seconds',emb.get('load_seconds'))),'document_encoding_s':modelmeta.get('documents',{}).get('encoding_seconds',emb.get('doc_encoding_seconds')),
   'new_pairs_encoded':run.get('new_pair_count',run.get('new_pairs_encoded')),'batch_inference_seconds':run.get('new_pair_seconds',run.get('batch_inference_seconds'))})
 if hardware:
  datasets['fresh_hardware']=hardware
  fields=[('model','Model',None),('stage','Stage',None),('cpu','CPU',None),('threads','Threads','number'),('runtime','Runtime',None),('precision','Precision',None),('max_tokens','Token limit','number'),('peak_rss_mib','Peak process RSS MiB','number'),('decision_only_peak_rss_mib','Decision-only peak MiB','number'),('load_seconds','Load seconds','number'),('document_encoding_s','Corpus encoding seconds','number'),('new_pairs_encoded','New encoded pairs','number')]
  manifest['tables'].append(br.table('fresh_hardware_table','Fresh run configuration and setup costs','fresh_hardware',fields,'fresh_transform',('model','asc'),'Blank values are unavailable, not zero. Cached timing-only passes do not overwrite quality-stage memory evidence.'))
  pos=next(i for i,b in enumerate(blocks) if b['id']=='historical_scope');blocks[pos:pos]=[br.markdown('fresh_hardware_section','Memory and setup costs come from actual recovery runs','Each row preserves its measured stage. Retrieval-process peaks include building and batch-16 encoding of the 6,006-skill corpus as well as retrieval. Native rows here are the explicitly named quality-stage processes; separate native timing processes also hold BGE plus the classifier with prebuilt document vectors. These mixed-stage peaks are not a uniform serving-memory benchmark or a ranking of model weights. Process peak RSS includes the process environment and any co-resident components; it is not a model-resident allocation. Initial load, corpus embedding and accuracy-pair preparation are excluded from warm request latency.','fresh_transform'),{'id':'fresh_hardware_visual','type':'table','tableId':'fresh_hardware_table'}]
 return artifact
