"""Independent saved-ranking arithmetic, source coverage and identity audit."""
import argparse
import json
import math
from pathlib import Path
from common import (STUDY,REPO,QWEN,CONFIGS,SIZES,atomic_json,digest,file_hash,full_text,
                    jsonl,panel,independent,compare_evaluation)


def check_rows(rows, labels, eligible):
    expected_ids = set(eligible)
    assert len({r['query_id'] for r in rows}) == len(rows)
    rankings = {}
    for row in rows:
        qid = row['query_id']
        assert qid in labels
        ranked,scores = row['ranking'],row['scores']
        assert len(ranked) == len(set(ranked)) == len(scores) == min(50,len(eligible))
        assert set(ranked) <= expected_ids
        assert all(math.isfinite(s) and -1.00001 <= s <= 1.00001 for s in scores)
        assert all(a>=b for a,b in zip(scores,scores[1:]))
        assert row['query_cache_persistence_included'] is True
        assert row['total_retrieval_uncached_s'] >= row['query_embedding_uncached_s'] + row['similarity_search_s']
        rankings[qid] = ranked
    return rankings


def validate(cache):
    r=STUDY/'results';protocol=json.loads((STUDY/'protocol.json').read_text())
    assert file_hash(QWEN/'protocol.json')==protocol['qwen_protocol_sha256']
    assert file_hash(REPO/'experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/test_queries.json')==protocol['panel_sha256']
    assert file_hash(r/'candidate_pools.json')==protocol['candidate_pools_sha256']
    for path,sha in protocol['shared_helper_sha256'].items():assert file_hash(REPO/path)==sha
    for name,sha in protocol['data_sha256'].items(): assert file_hash(cache/'data'/(name+'.jsonl'))==sha
    skills=jsonl(cache/'data/skills.jsonl');queries=jsonl(cache/'data/queries.jsonl');raw=jsonl(cache/'data/qrels.jsonl')
    assert len(skills)==len({s['id'] for s in skills})==6006
    assert len(queries)==len({q['id'] for q in queries})==4392 and len(raw)==7187
    ids={s['id'] for s in skills};qtexts={q['id']:q['query'] for q in queries};labels={}
    for row in raw:
        assert row['skill_id'] in ids and row['relevance']==1
        assert row['skill_id'] not in labels.get(row['query_id'],{})
        labels.setdefault(row['query_id'],{})[row['skill_id']]=1
    assert set(labels)==set(qtexts)
    assert all(set(q['skill_ids'])==set(labels[q['id']]) for q in queries)
    assert (r/'candidate_pools.json').read_bytes()==(QWEN/'results/candidate_pools.json').read_bytes()
    pools=json.loads((r/'candidate_pools.json').read_text());questions=panel()
    for n in SIZES:
        pool=pools[str(n)]['ids']
        assert len(pool)==len(set(pool))==n and set(pool)<=ids
        assert digest(pool)==pools[str(n)]['ordered_ids_sha256']
        assert all(q['gold_id'] in pool for q in questions)
    scopes={};old=jsonl(QWEN/'results/documents.jsonl')
    for config in CONFIGS:
        docs=jsonl(r/('documents-'+config+'.jsonl'))
        assert file_hash(r/('documents-'+config+'.jsonl'))==protocol['document_manifest_sha256'][config]
        assert len(docs)==6006
        for skill,doc,historical in zip(skills,docs,old):
            text=full_text(skill);view=skill['name']+' | '+skill['description'] if config=='A_metadata' else text
            assert doc['skill_id']==skill['id']==historical['skill_id']
            assert doc['source_full_sha256']==digest(text)==historical['source_sha256']
            assert doc['view_sha256']==digest(view) and doc['source_characters']==len(view)
            end=0
            for c in doc['chunks']:
                assert 0<=c['start']<=end<c['end']<=len(view)
                assert digest(view[c['start']:c['end']])==c['text_sha256']
                assert c['tokens']<=protocol['content_token_budget']
                end=c['end']
            assert end==len(view)
        folder=r/config;done=[]
        for n in SIZES:
            path=folder/f'scaling-n{n}.json'
            if not path.exists():continue
            data=json.loads(path.read_text());rows=data['rows']
            assert len(rows)==20 and [x['query_id'] for x in rows]==[f'panel-{i:02d}' for i in range(20)]
            gold={f'panel-{i:02d}':{q['gold_id']:1} for i,q in enumerate(questions)}
            for i,row in enumerate(rows):
                assert row['gold_id']==questions[i]['gold_id'] and row['query_sha256']==digest(questions[i]['query'])
            ranked=check_rows(rows,gold,pools[str(n)]['ids'])
            independent_rows={qid:independent(g,ranked[qid]) for qid,g in gold.items()}
            compare_evaluation(data['evaluation'],independent_rows,gold,ranked)
            done.append(n)
        full_rows=jsonl(folder/'full_rankings.jsonl') if (folder/'full_rankings.jsonl').exists() else []
        ranked=check_rows(full_rows,labels,ids)
        for row in full_rows:assert row['query_sha256']==digest(qtexts[row['query_id']])
        if (folder/'full_evaluation.json').exists():
            assert len(full_rows)==4392 and set(ranked)==set(labels)
            evaluation=json.loads((folder/'full_evaluation.json').read_text())
            compare_evaluation(evaluation,{qid:independent(g,ranked[qid]) for qid,g in labels.items()},labels,ranked)
        if (folder/'content_coverage.json').exists():
            coverage=json.loads((folder/'content_coverage.json').read_text())
            assert coverage['skills']==6006 and coverage['chunks']==sum(len(d['chunks']) for d in docs)
            assert coverage['coverage_percent']==100 and coverage['missing_or_failed_documents']==0
        scopes[config]={'scaling_sizes_independently_checked':done,'official_queries_independently_checked':len(full_rows),
                        'official_aggregate_independently_checked':(folder/'full_evaluation.json').exists() and len(full_rows)==4392}
    complete=all(scope['scaling_sizes_independently_checked']==list(SIZES) and scope['official_queries_independently_checked']==4392 and scope['official_aggregate_independently_checked'] for scope in scopes.values())
    result={'status':'passed_with_scope_recorded','completed_requested_scope':complete,'configurations':scopes,
        'sources_hashed_and_complete':True,'candidate_ids_and_order_identical_to_qwen':True,
        'metrics_independent_from_saved_rankings':True,'official_query_denominator':4392,'official_qrel_pairs':7187,
        'coverage_and_multirelevant_denominators_checked':True}
    atomic_json(r/'validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=Path.home()/'.cache/mia-bgem3')
    validate(parser.parse_args().cache)
