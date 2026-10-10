#!/usr/bin/env python3
"""Create an explicitly abridged, four-page PDF brief from validated final metrics.

This is a separately authored compact companion, not a faithful HTML export.
Do not run during measured inference/timing. Render every page and inspect the
page images before delivery.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re
from pathlib import Path
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
MODEL_LABEL={'bm25':'BM25','bge-small':'BGE-small','e5-small':'E5-small','embeddinggemma-int8':'Gemma INT8','gliner2.5-small':'GLiNER2.5','verdict-151m-onnx':'Verdict','laya-multilingual':'Laya multilingual','julia-1':'Julia-1'}
ORDER=[('bm25','bm25'),('bge-small','hybrid'),('e5-small','hybrid'),('embeddinggemma-int8','hybrid'),('gliner2.5-small','hybrid_rerank'),('verdict-151m-onnx','hybrid_rerank'),('laya-multilingual','hybrid_rerank'),('julia-1','hybrid_rerank')]
ALIASES={'julia':'julia-1','gliner':'gliner2.5-small','verdict':'verdict-151m-onnx','laya':'laya-multilingual'}

def label(row):
 model=ALIASES.get(row['model'],row['model']);name=MODEL_LABEL.get(model,model)
 return ('BGE shortlist + '+name) if row['method']=='hybrid_rerank' else (name+' hybrid' if row['method']=='hybrid' else name)

def key(row):return (ALIASES.get(row['model'],row['model']),row['method'])
def num(x,d=3):return 'n/a' if x is None else f'{float(x):,.{d}f}'
def pct(x):return 'n/a' if x is None else f'{100*float(x):.1f}%'
def clean(text):
 return str(text).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('\u2212','-')

def read_sources(root,artifact_path):
 raw=artifact_path.read_bytes();a=json.loads(raw);d=a['snapshot']['datasets']
 cross=d.get('fresh_cross',[])
 if len(cross)!=8 or {key(r) for r in cross}!=set(ORDER):raise ValueError('A complete matched eight-pipeline primary table is required')
 counts={r['queries'] for r in cross}
 if len(counts)!=1 or not counts<={32,48,64}:raise ValueError('Primary cohort must be exact common32,48 or64')
 n=next(iter(counts));quality=d.get('fresh_quality',[])
 for pair in ORDER:
  rows=[r for r in quality if key(r)==pair]
  if {r['catalog_size'] for r in rows}!={10,100,1000,6006} or any(r['queries']!=n for r in rows):raise ValueError('Incomplete primary size grid')
 timing={key(r):r for r in d.get('fresh_timing',[]) if key(r) in ORDER}
 for row in timing.values():
  if (row['queries'],row['repeats'],row['observations'])!=(32,2,64):raise ValueError('Latency is not the matched32×2 cohort')
 return a,d,n,{key(r):r for r in cross},timing,hashlib.sha256(raw).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--artifact',type=Path);p.add_argument('--output',type=Path)
 args=p.parse_args();root=args.root.resolve();artifact=args.artifact or root/'artifacts/artifact.json';out=args.output or root/'output/pdf/skillret_cpu_compact_brief.pdf'
 a,data,n,cross,timing,artifact_hash=read_sources(root,artifact)
 # Heavy imports are deliberately deferred until explicit generation.
 from reportlab.lib import colors
 from reportlab.lib.pagesizes import A4
 from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
 from reportlab.pdfbase import pdfmetrics
 from reportlab.pdfbase.ttfonts import TTFont
 from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
 fontdir=Path('/usr/share/fonts/truetype/dejavu')
 for name,file in [('Body','DejaVuSans.ttf'),('BodyBold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
 pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
 INK=colors.HexColor('#162A46');BLUE=colors.HexColor('#205AA7');MUTED=colors.HexColor('#526071');LIGHT=colors.HexColor('#EEF3F8');RULE=colors.HexColor('#D7DFE8')
 styles={
  'title':ParagraphStyle('title',fontName='BodyBold',fontSize=24,leading=29,textColor=INK,spaceAfter=10),
  'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=12,leading=16,textColor=INK,spaceBefore=10,spaceAfter=6),
  'body':ParagraphStyle('body',fontName='Body',fontSize=9.3,leading=13,textColor=INK,spaceAfter=7),
  'small':ParagraphStyle('small',fontName='Body',fontSize=8.1,leading=11,textColor=MUTED,spaceAfter=6),
  'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.5,leading=11,textColor=INK),
  'head':ParagraphStyle('head',fontName='BodyBold',fontSize=8.1,leading=10.5,textColor=colors.white),
  'tag':ParagraphStyle('tag',fontName='BodyBold',fontSize=8,leading=11,textColor=BLUE,spaceAfter=7)}
 def P(text,style='body',markup=False):return Paragraph(clean(text) if markup else escape(clean(text)),styles[style])
 width=A4[0]-100
 def grid(headers,rows,widths,padding=6):
  cells=[[P(h,'head') for h in headers]]+[[P(v,'cell') for v in row] for row in rows]
  t=Table(cells,colWidths=[width*x for x in widths],repeatRows=1,hAlign='LEFT')
  commands=[('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding),('LINEBELOW',(0,0),(-1,0),.6,INK),('LINEBELOW',(0,-1),(-1,-1),.5,RULE)]
  for i in range(1,len(cells)):
   if i%2==0:commands.append(('BACKGROUND',(0,i),(-1,i),LIGHT))
  t.setStyle(TableStyle(commands));return t
 story=[]
 def heading(title,subtitle=None):
  story.extend([P('COMPACT BENCHMARK BRIEF  /  FRESH RECOVERY','tag'),P(title,'title')])
  if subtitle:story.append(P(subtitle,'small'))
 def body(text):story.append(P(text))
 def sub(title):story.append(P(title,'h2'))
 def endpage():story.append(PageBreak())
 def foot(canvas,doc):
  canvas.saveState();canvas.setStrokeColor(RULE);canvas.setLineWidth(.5);canvas.line(50,39,A4[0]-50,39)
  canvas.setFont('Body',7.5);canvas.setFillColor(MUTED);canvas.drawString(50,26,f'SkillRet CPU brief | Fresh common {n} | 2 October 2026');canvas.drawRightString(A4[0]-50,26,f'{doc.page} / 4');canvas.restoreState()
 best=max(cross.values(),key=lambda r:r['mrr5']);hybrids=[r for pair,r in cross.items() if pair[1]=='hybrid' and pair in timing]
 fastest=min(hybrids,key=lambda r:timing[key(r)]['p50_ms']) if hybrids else None
 baseline=cross[('bge-small','hybrid')]
 native=[r for pair,r in cross.items() if pair[1]=='hybrid_rerank']
 # PAGE 1: result and primary matched quality.
 heading('SkillRet on CPU',f'A separately authored, abridged report from the validated benchmark metrics. Primary quality: the same {n} queries, all 6,006 skills. Warm latency: a separate shared 32 queries x 2 repetitions.')
 story.append(P('MRR@5 averages 1/rank for the first relevant result within five, or zero if absent. Recall@5 measures the share of all judged-relevant skills found; Hit@1 tests the first result; Complete@5 requires the full gold set.','small'))
 body(f'Best observed full-catalog quality: {label(best)}, with MRR@5 {num(best["mrr5"])} and Recall@5 {pct(best["recall5"])}. This is a descriptive result on this cohort, not a general model ranking.')
 if fastest:body(f'Lower-latency hybrid option: {label(fastest)} recorded median {num(timing[key(fastest)]["p50_ms"],1)} ms, versus {num(timing.get(key(best),{}).get("p50_ms"),1)} ms for the highest-MRR pipeline. Choose according to the deployment quality/latency budget; these CPU measurements are not service-level guarantees.')
 e5=cross.get(('e5-small','hybrid'));e5_time=timing.get(('e5-small','hybrid'))
 if e5 and e5_time and key(best)!=('e5-small','hybrid'):
  body(f"E5-small hybrid is a useful quality/latency compromise in this sample: MRR@5 {num(e5['mrr5'])} at {num(e5_time['p50_ms'],1)} ms. Its point estimate is only {num(best['mrr5']-e5['mrr5'])} below the highest-MRR pipeline; that small gap should not be treated as broad superiority.")
 sub('Matched full-catalog quality')
 story.append(grid(['Pipeline','MRR@5','Recall@5','Hit@1','Complete@5'],[[label(cross[k]),num(cross[k]['mrr5']),pct(cross[k]['recall5']),pct(cross[k]['hit1']),pct(cross[k]['complete5'])] for k in ORDER],[.36,.15,.17,.15,.17]))
 story.append(Spacer(1,8))
 body(f'The native rows all reorder the same BGE+BM25 hybrid shortlist. Its initial MRR@5 is {num(baseline["mrr5"])} and candidate-any-hit ceiling is {num(baseline["candidate_hit"])}. Missing first-stage gold skills cannot be recovered by any second-stage ranker.')
 if all(r['mrr5']<=baseline['mrr5']+1e-12 for r in native):body('None of the four binary pairwise adapters improves MRR@5 over that initial ranking in this tested setup. This does not establish that native short-label choice routing is generally poor.')
 body('For a Jev-like router, treat native listwise choice over short skill labels as a different, untested pipeline. Twenty complete discovery headers exceed some native head budgets; the tested pairwise prompt is not a substitute for that design.')
 endpage()
 # PAGE 2: all requested catalog sizes, same denominator, multi-positive coverage.
 heading('Catalogs and coverage',f'All values here use the same fresh {n}-query population. Small catalogs are per-query, deterministic, nested and gold-preserving; they are not a single global small index. Retrieval-only 64-query measurements are preserved separately; the primary comparison filters them to the same selected native-query IDs.')
 quality={(key(r),r['catalog_size']):r for r in data['fresh_quality']}
 sub('MRR@5 across the four controlled catalogs')
 story.append(grid(['Pipeline','10 skills','100 skills','1,000 skills','6,006 skills'],[[label(cross[k])]+[num(quality[(k,size)]['mrr5']) for size in (10,100,1000,6006)] for k in ORDER],[.36,.16,.16,.16,.16],padding=5))
 story.append(Spacer(1,8))
 body('Rerankers receive up to 20 BGE hybrid candidates, or all 10 candidates in the smallest condition. Higher scores on smaller, gold-preserving catalogs should not be read as a production scaling forecast.')
 multi={key(r):r for r in data.get('fresh_segments',[]) if r['catalog_size']==6006 and r['query_group']=='multi' and key(r) in ORDER}
 if multi:
  nq=next(iter(multi.values()))['queries'];sub(f'Multi-positive queries: {nq} of {n}')
  story.append(grid(['Pipeline','Recall@5','Complete@5'],[[label(cross[k]),pct(multi.get(k,{}).get('recall5')),pct(multi.get(k,{}).get('complete5'))] for k in ORDER],[.58,.21,.21],padding=5))
  story.append(Spacer(1,8));body('Recall retains every judged-positive skill in the denominator. Complete@5 requires all gold skills, so finding one relevant skill can still be an incomplete answer.')
 endpage()
 # PAGE 3: actual timing boundary and measured resource context.
 heading('Warm CPU request cost','Identical first 32 frozen query IDs, two fresh observations each, full 6,006-skill catalog. Lower latency is better; p95 describes this measured sample only.')
 sub('End-to-end latency and native adapter component')
 story.append(grid(['Pipeline','Median total ms','Observed p95 ms','Median native ms'],[[label(cross[k]),num(timing.get(k,{}).get('p50_ms'),1),num(timing.get(k,{}).get('p95_ms'),1),num(timing.get(k,{}).get('rerank_p50_ms'),1)] for k in ORDER],[.40,.20,.20,.20],padding=5))
 story.append(Spacer(1,8))
 body('Native totals include fresh BGE/BM25/RRF retrieval plus reference-adapter token preflight, query fitting and scoring. Timed requests do not reuse score caches; document vectors/indexes are prebuilt. These are pipeline times, not isolated model-forward times.')
 body('BGE uses each native environment: Transformers 5.0.0 (Julia), 4.57.6 (GLiNER), 5.18 (others). Fresh shortlist equality is asserted. Stage medians need not sum to the total median.')
 hardware=data.get('fresh_hardware',[]);selected={}
 for r in hardware:
  model=ALIASES.get(r['model'],r['model'])
  if r['stage']=='retrieval64' or r['stage']==f'quality{n}':selected[model]=r
 if selected:
  sub('Recorded process peaks, with stage context')
  rows=[]
  for k in ORDER:
   r=selected.get(k[0],{});rows.append([MODEL_LABEL[k[0]],'Corpus build + retrieval' if r.get('stage')=='retrieval64' else f'{n}-query pair scoring',num(r.get('peak_rss_mib'),0),str(r.get('threads') or 'n/a')])
  story.append(grid(['Model','Recorded stage','Peak process MiB','CPU threads'],rows,[.29,.33,.24,.14],padding=5))
  cpu=next((r.get('cpu') for r in selected.values() if r.get('cpu')),None)
  story.append(Spacer(1,7));body(('Recorded CPU: '+cpu+'. ' if cpu else '')+'These are mixed-stage process peaks, not model-weight or serving-only memory. Retrieval includes batch-16 corpus encoding; native rows are quality-stage pair scoring. Native timing processes separately hold BGE plus the classifier. Setup is excluded from warm latency.')
 endpage()
 # PAGE 4: uncertainty, method interpretation and independently retained history.
 heading('Limits and methods','The companion archive retains full tables, fresh raw data, source revisions and the executed notebook.')
 intervals=[]
 ci_path=root/'docs/notebook_paired_bootstrap.json'
 if data.get('fresh_intervals'):
  ci={'comparisons':data['fresh_intervals']};expected_hash=cross[ORDER[0]].get('cohort_hash')
  for r in ci.get('comparisons',[]):
   model=ALIASES.get(r.get('model'),r.get('model'))
   if r.get('catalog_size')==6006 and r.get('queries')==n and r.get('cohort_hash')==expected_hash and model in MODEL_LABEL and r.get('method')=='hybrid_rerank':
    pair=(model,'hybrid_rerank');expected=cross[pair]['mrr5']-cross[pair]['shortlist_mrr5']
    if not math.isclose(float(r['mrr5_delta']),expected,abs_tol=1e-10):raise ValueError('Companion interval differs from the validated paired delta')
    intervals.append((model,r))
 if intervals:
  sub('Native MRR change versus the same-query initial shortlist')
  story.append(grid(['Adapter','Paired MRR change','Descriptive 95% interval'],[[MODEL_LABEL[m],f"{float(r['mrr5_delta']):+.3f}",f"[{float(r['ci95_low']):+.3f}, {float(r['ci95_high']):+.3f}]"] for m,r in intervals],[.38,.25,.37]))
  story.append(Spacer(1,7));body('Paired query bootstrap: 2,000 resamples, seed 20261002; conditional on this protocol and its judgments. These pointwise intervals are not multiplicity-adjusted, causal or out-of-domain guarantees.')
  bge_ci=next((r for r in data.get('fresh_intervals',[]) if r['model']=='bge-small' and r['method']=='hybrid' and r['catalog_size']==6006),None)
  if bge_ci:body(f"For context, BGE hybrid versus BM25 has estimated MRR change {float(bge_ci['mrr5_delta']):+.3f}, interval [{float(bge_ci['ci95_low']):+.3f}, {float(bge_ci['ci95_high']):+.3f}]. This illustrates the limited precision of a 32-query comparison.")
 else:body('No independently matched uncertainty intervals are included in this brief. Point estimates and model ordering remain descriptive.')
 audits=data.get('fresh_input_audits',[])
 if audits:
  native_a=[r for r in audits if r['kind']=='native query-only pair fitting'];emb_a=[r for r in audits if r['kind']=='embedding token limit']
  text='Distinct queries shortened: '+', '.join(MODEL_LABEL[r['model']]+' '+str(r['queries_shortened'])+'/'+str(r['query_denominator']) for r in native_a)+'. '
  text+='Embedding counts: '+', '.join(MODEL_LABEL[r['model']]+' '+str(r['queries_shortened'])+'/'+str(r['query_denominator']) for r in emb_a)+' (all 64 encoded queries). '
  text+='Candidate cuts: '+str(sum(r['candidate_cut_pairs'] or 0 for r in native_a))+' in '+str(sum(r['pair_denominator'] or 0 for r in native_a))+' unique model/query/candidate evaluations. Embedding document truncations: '+str(sum(r['documents_shortened'] or 0 for r in emb_a))+' across three runs of 6,006 documents. Native query prefixes may differ by candidate.'
  body(text)
 sub('What was measured')
 for text in [
  f'Primary quality: the same {n} outcome-independently selected queries. Runtime, not accuracy, determined the attempted extension.',
  'SkillRet v1.1 short headers: name plus the first 1,024 description characters, not full skill_md. No official full-text leaderboard reproduction.',
  'domain_vector uses the top-1 centroid of six skill major categories, then vectors within it. Existing skill metadata only; no gold-query routing or 13-tag classifier.',
  'Fixed pairwise binary prompts test reranking; native short-label listwise choice routing remains untested.',
  'Missing qrels mean non-relevant under released judgments, not that another skill cannot help. Coverage and truncation constrain results.',
  'Four CPU threads on one host. Recovery spinning-off ONNX timings are not pooled with historical default-ORT timings.'
 ]:body('- '+text)
 sub('Historical evidence and the recovery boundary')
 body('Final scope is the common 32-query cohort. Only Julia/GLiNER completed 48; those results are supplements. A safety restriction involving a Microsoft telemetry connection blocked the 48-query Verdict stage; no inference retry occurred. Prospective privacy changes did not alter the measurements. Earlier 4,392-query values were genuinely measured and audited, but their original raw files became unavailable. They remain historical only. The earlier eight-query test had zero MRR headroom; neither historical set enters fresh means or intervals.')
 story.append(Spacer(1,4))
 story.append(P('<link href="https://huggingface.co/datasets/ThakiCloud/SKILLRET/tree/6583d7d2ed07644d0fb8938ed8178f3a7dc42a12" color="#205AA7">Pinned ThakiCloud/SKILLRET v1.1 dataset</link>. Exact model revisions, fresh CSVs and full audit details are in the archive.','small',markup=True))
 story.append(P('Analysis source SHA256: '+artifact_hash,'small'))
 out.parent.mkdir(parents=True,exist_ok=True)
 doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=48,bottomMargin=50,title='SkillRet CPU - compact benchmark brief',author='dot',pageCompression=1)
 doc.build(story,onFirstPage=foot,onLaterPages=foot)
 from pypdf import PdfReader
 reader=PdfReader(str(out));text='\n'.join(page.extract_text() or '' for page in reader.pages)
 receipt={'pdf':str(out),'pages':len(reader.pages),'artifact_sha256':artifact_hash,'primary_queries':n,'timing_pipelines':len(timing),'interval_rows':len(intervals),'selectable_text_characters':len(text),'visual_qa':'pending; render every page to PNG and inspect before delivery'}
 out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');out.with_suffix('.txt').write_text(text)
 print(json.dumps(receipt))
 if len(reader.pages)!=4:raise SystemExit('PDF needs layout revision: expected exactly four readable pages')
if __name__=='__main__':main()
