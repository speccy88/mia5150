"""Lossless A/B/C vectors and shared query vectors; model-free public replay."""
import argparse
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from common import (STUDY,CONFIGS,SIZES,atomic_json,digest,file_hash,jsonl,panel,
                    rank_skills,evaluate,independent,compare_evaluation)

V=STUDY/'vectors'


def write_archive(path,arrays):
    temporary=path.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,array in sorted(arrays.items()):
            assert array.dtype.kind in 'fiuU' and not array.dtype.hasobject
            buffer=io.BytesIO();np.save(buffer,array,allow_pickle=False)
            info=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,buffer.getvalue(),compresslevel=6)
    temporary.replace(path)


def load_archive(path,manifest):
    assert file_hash(path)==manifest['sha256'],'Missing Git LFS payload or corrupt archive'
    with np.load(path,allow_pickle=False) as archive:
        assert set(archive.files)==set(manifest['arrays'])
        arrays={key:archive[key] for key in archive.files}
    for key,array in arrays.items():
        meta=manifest['arrays'][key]
        assert list(array.shape)==meta['shape'] and str(array.dtype)==meta['dtype']
        assert digest(np.ascontiguousarray(array).tobytes())==meta['sha256_contiguous_bytes']
        if key.startswith('documents_') or key.endswith('_query_vectors'):
            assert array.dtype==np.float32 and array.shape[1]==1024 and np.isfinite(array).all()
            assert np.max(np.abs(np.linalg.norm(array,axis=1)-1))<2e-5
    return arrays


def export(cache):
    r=STUDY/'results';assert json.loads((r/'progress.json').read_text())['stage']=='completed'
    identity=json.loads((r/'model.json').read_text())['cache_identity'];root=cache/'vectors'/digest(identity)
    def read(text_hash,kind):
        key=digest({'text_sha256':text_hash,'identity':identity,'kind':kind})
        path=root/(key+'.npy');meta=json.loads((root/(key+'.json')).read_text())
        assert meta['text_sha256']==text_hash and meta['vector_sha256']==file_hash(path)
        vector=np.load(path,allow_pickle=False)
        assert vector.dtype==np.float32 and vector.shape==(1024,) and np.isfinite(vector).all()
        assert abs(float(np.linalg.norm(vector))-1)<2e-5
        return vector
    docs0=jsonl(r/'documents-B_matched.jsonl');ids=[d['skill_id'] for d in docs0];index={sid:i for i,sid in enumerate(ids)}
    pools=json.loads((r/'candidate_pools.json').read_text());seen=set();order=[]
    for n in SIZES:
        for sid in pools[str(n)]['ids']:
            if sid not in seen:seen.add(sid);order.append(index[sid])
    arrays={'skill_ids':np.asarray(ids),'skill_source_sha256':np.asarray([d['source_full_sha256'] for d in docs0])}
    for config in CONFIGS:
        docs=jsonl(r/('documents-'+config+'.jsonl'));vectors=[];owners=[];hashes=[]
        for i in order:
            for c in docs[i]['chunks']:
                vectors.append(read(c['text_sha256'],'document'));owners.append(i);hashes.append(c['text_sha256'])
        arrays['documents_'+config]=np.stack(vectors);arrays['owners_'+config]=np.asarray(owners,dtype=np.int32)
        arrays['chunk_hashes_'+config]=np.asarray(hashes)
    queries=jsonl(cache/'data/queries.jsonl');qids=[q['id'] for q in queries];qindex={qid:i for i,qid in enumerate(qids)}
    questions=panel();qrels=jsonl(cache/'data/qrels.jsonl')
    arrays.update({'official_query_ids':np.asarray(qids),'official_query_sha256':np.asarray([digest(q['query']) for q in queries]),
        'official_query_vectors':np.stack([read(digest(q['query']),'query') for q in queries]),
        'panel_query_ids':np.asarray([f'panel-{i:02d}' for i in range(20)]),
        'panel_query_vectors':np.stack([read(digest(q['query']),'query') for q in questions]),
        'panel_gold_ids':np.asarray([q['gold_id'] for q in questions]),
        'qrel_query_indices':np.asarray([qindex[q['query_id']] for q in qrels],dtype=np.int32),
        'qrel_skill_indices':np.asarray([index[q['skill_id']] for q in qrels],dtype=np.int32),
        'qrel_relevance':np.asarray([q['relevance'] for q in qrels],dtype=np.uint8)})
    V.mkdir(exist_ok=True);path=V/'dense-index-v1.npz';write_archive(path,arrays)
    manifest={'file':path.name,'sha256':file_hash(path),'bytes':path.stat().st_size,
        'protocol_sha256':file_hash(STUDY/'protocol.json'),'model_identity':identity,
        'format':'Lossless ZIP of numeric/Unicode NPY; allow_pickle=False; unchanged float32 values',
        'skills':6006,'official_queries':4392,'scaling_queries':20,'qrels':7187,
        'chunks':{c:len(arrays['documents_'+c]) for c in CONFIGS},'raw_text_included':False,'model_weights_included':False,
        'arrays':{key:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256_contiguous_bytes':digest(np.ascontiguousarray(a).tobytes())} for key,a in arrays.items()}}
    atomic_json(V/'manifest.json',manifest);print('EXPORTED',path.name,path.stat().st_size,flush=True)


def replay():
    started=time.perf_counter();manifest=json.loads((V/'manifest.json').read_text())
    assert file_hash(STUDY/'protocol.json')==manifest['protocol_sha256']
    arrays=load_archive(V/manifest['file'],manifest);load_s=time.perf_counter()-started
    ids=arrays['skill_ids'].tolist();index={sid:i for i,sid in enumerate(ids)};qids=arrays['official_query_ids'].tolist()
    assert len(ids)==len(set(ids))==6006 and len(qids)==len(set(qids))==4392
    labels={qid:{} for qid in qids}
    assert len(arrays['qrel_relevance'])==7187 and np.all(arrays['qrel_relevance']==1)
    for qi,si in zip(arrays['qrel_query_indices'],arrays['qrel_skill_indices']):
        assert ids[si] not in labels[qids[qi]];labels[qids[qi]][ids[si]]=1
    pools=json.loads((STUDY/'results/candidate_pools.json').read_text());ties=arrays['skill_source_sha256'];receipts={}
    for config in CONFIGS:
        matrix=arrays['documents_'+config];owners=arrays['owners_'+config]
        assert set(owners.tolist())==set(range(6006))
        folder=STUDY/'results'/config;count=0;active=set()
        for n in SIZES:
            active.update(index[sid] for sid in pools[str(n)]['ids']);mask=np.isin(owners,list(active))
            restricted,restricted_owners=matrix[mask],owners[mask]
            saved=json.loads((folder/f'scaling-n{n}.json').read_text());rankings={};gold={}
            for qi,qid in enumerate(arrays['panel_query_ids'].tolist()):
                selected,scores=rank_skills(restricted,restricted_owners,arrays['panel_query_vectors'][qi],[index[sid] for sid in pools[str(n)]['ids']],ties)
                rankings[qid]=[ids[i] for i in selected];gold[qid]={str(arrays['panel_gold_ids'][qi]):1}
                assert rankings[qid]==saved['rows'][qi]['ranking']
                assert np.allclose(scores,saved['rows'][qi]['scores'],atol=1e-6,rtol=0)
                count+=1
            compare_evaluation(saved['evaluation'],{qid:independent(g,rankings[qid]) for qid,g in gold.items()},gold,rankings)
        saved={r['query_id']:r for r in jsonl(folder/'full_rankings.jsonl')};rankings={}
        for qi,qid in enumerate(qids):
            selected,scores=rank_skills(matrix,owners,arrays['official_query_vectors'][qi],range(6006),ties)
            rankings[qid]=[ids[i] for i in selected]
            assert rankings[qid]==saved[qid]['ranking']
            assert np.allclose(scores,saved[qid]['scores'],atol=1e-6,rtol=0)
            if (qi+1)%1000==0:print('REPLAY',config,qi+1,flush=True)
        result=evaluate(labels,rankings);original=json.loads((folder/'full_evaluation.json').read_text())
        assert result==original
        compare_evaluation(original,{qid:independent(g,rankings[qid]) for qid,g in labels.items()},labels,rankings)
        receipts[config]={'scaling_cases':count,'official_queries':4392,'all_rankings_identical':True,'all_metrics_identical':True}
    receipt={'status':'completed','configurations':receipts,'model_api_requests':0,'network_requests':0,
        'original_cache_files_used':0,'archive_load_validation_s':load_s,'total_wall_s':time.perf_counter()-started,
        'archive_sha256':manifest['sha256'],'score_tolerance':1e-6}
    atomic_json(STUDY/'results/published_vector_replay.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('export','replay'))
    parser.add_argument('--cache',type=Path,default=Path.home()/'.cache/mia-bgem3');args=parser.parse_args()
    export(args.cache) if args.action=='export' else replay()
