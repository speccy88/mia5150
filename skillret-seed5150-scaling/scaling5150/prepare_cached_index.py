"""Reuse matching cached embeddings; re-encode only common-view changes."""
import argparse,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150';sys.path.insert(0,str(ROOT))
from harness import load,view,digest
from neural import Embed
import numpy as np
def main():
 a=argparse.ArgumentParser();a.add_argument('role',choices=['skillret_embed_06','qwen_embed_06','skillrouter_embed_06','skillreason_embed_06']);args=a.parse_args();role=args.role
 out=HERE/('index-'+role);out.mkdir(exist_ok=False);manifest=json.loads((HERE/'manifest.json').read_text());texts=json.loads((HERE/'documents.json').read_text());skills=load('skills','test');ids=[s['id'] for s in skills];old_queries=json.loads((ROOT/'manifests/scorer-queries.json').read_text());old=json.loads((ROOT/f'results/{role}-header-0.json').read_text());assert old['corpus_hash']==digest([view(s,'header') for s in skills]) and old['queries_hash']==digest(old_queries)
 dpath=ROOT/f'cache/{role}-header-0-docs.npy';qpath=ROOT/f'cache/{role}-header-0-queries.npy';dv=np.load(dpath);qv=np.load(qpath);assert dv.shape==(6006,1024) and qv.shape==(128,1024)
 changed=[i for i,s in enumerate(skills) if texts[s['id']]!=view(s,'header')];assert len(changed)==13
 start=time.perf_counter();model=Embed(role,1024);startup=time.perf_counter()-start;model.encode(['warmup'],batch=1)
 new,stats=model.encode([texts[ids[i]] for i in changed],batch=1);assert stats['truncated_count']==0
 for i,v in zip(changed,new):dv[i]=v
 positions={q['id']:i for i,q in enumerate(old_queries)};qindices=[positions[q['id']] for q in manifest['queries']]
 assert np.isfinite(dv).all() and np.isfinite(qv[qindices]).all();np.save(out/'docs.npy',dv);np.save(out/'queries.npy',qv[qindices])
 result={'role':role,'document_ids':ids,'query_ids':[q['id'] for q in manifest['queries']],'view_sha256':manifest['text_sha256'],'new_document_forwards':13,'new_query_forwards':0,'reused_document_vectors':5993,'reused_query_vectors':20,'changed_document_ids':[ids[i] for i in changed],'changed_stats':stats,'startup_seconds':startup,'old_index_forward_seconds':sum(old['document_stats']['batch_seconds']),'historical_query_seconds':[old['query_stats']['batch_seconds'][i] for i in qindices],'query_raw_tokens':[old['query_stats']['raw_token_counts'][i] for i in qindices],'query_token_limit':old['query_stats']['max_length'],'source_cache_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [dpath,qpath]},'source_model_manifest':old['model_manifest'],'cache_timing_note':'Historical full index/query timing retained, not fresh inference timing. Only13changed document forwards measured now; warmup excluded.'}
 (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({'role':role,'changed_docs':13,'startup_seconds':startup,'new_forward_seconds':sum(stats['batch_seconds']),'query_ids':result['query_ids']}))
if __name__=='__main__':main()
