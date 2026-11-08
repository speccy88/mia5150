from pathlib import Path
import json
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import (STUDY, QWEN, SIZES, digest, jsonl, file_hash, rank_skills, unit,
                    evaluate, independent, compare_evaluation, bootstrap)
from engine import Encoder


def test_exact_candidate_pools_and_gold_denominators():
    path = STUDY / 'results/candidate_pools.json'
    assert path.read_bytes() == (QWEN / 'results/candidate_pools.json').read_bytes()
    pools = json.loads(path.read_text())
    for n in SIZES:
        row = pools[str(n)]
        assert len(row['ids']) == len(set(row['ids'])) == n
        assert row['ordered_ids_sha256'] == digest(row['ids']) and row['absent_gold'] == 0


def test_primary_b_preserves_every_qwen_chunk_span_and_source_hash():
    docs = jsonl(STUDY / 'results/documents-B_matched.jsonl')
    old = jsonl(QWEN / 'results/documents.jsonl')
    assert len(docs) == len(old) == 6006
    for current, historical in zip(docs, old):
        assert current['skill_id'] == historical['skill_id']
        assert current['source_full_sha256'] == historical['source_sha256']
        assert [(c['start'],c['end'],c['text_sha256']) for c in current['chunks']] == [(c['start'],c['end'],c['text_sha256']) for c in historical['chunks']]


def test_all_three_view_plans_have_gapless_hashed_coverage():
    for config in ('A_metadata','B_matched','C_native8k'):
        docs = jsonl(STUDY / ('results/documents-' + config + '.jsonl'))
        assert len({d['skill_id'] for d in docs}) == 6006
        for doc in docs:
            end = 0
            for c in doc['chunks']:
                assert 0 <= c['start'] <= end < c['end'] <= doc['source_characters']
                assert len(c['text_sha256']) == 64 and c['tokens'] <= 8190
                end = c['end']
            assert end == doc['source_characters']
        if config == 'A_metadata': assert all(len(d['chunks']) == 1 for d in docs)


def test_unit_cosine_max_chunk_deduplication_and_pool_exclusion():
    v = unit([3,4], dimensions=2)
    assert v.dtype == np.float32 and float(v @ unit([0,1],dimensions=2)) == pytest.approx(.8)
    vectors = np.array([[1,0],[1,0],[1,0],[0,1]],dtype=np.float32)
    owners = np.array([0,0,1,2])
    a,scores = rank_skills(vectors,owners,np.array([1,0],dtype=np.float32),[0,1],['b','a','c'])
    b,_ = rank_skills(vectors,owners,np.array([1,0],dtype=np.float32),[1,0],['b','a','c'])
    assert a.tolist() == b.tolist() == [1,0] and scores.tolist()==[1,1]
    assert 2 not in a


def test_independent_multipositive_metrics_include_missing_questions():
    labels = {'q':{'a':1,'b':1},'missing':{'c':1}}
    rankings = {'q':['x','a','b']}
    values = evaluate(labels,rankings)
    rows = {qid:independent(gold,rankings.get(qid,[])) for qid,gold in labels.items()}
    compare_evaluation(values,rows,labels,rankings)
    assert values['aggregate']['Recall@10']==.5
    assert values['per_query']['q']['MAP@10']==pytest.approx((.5+2/3)/2)
    assert values['candidate_coverage']['50']['relevant_pair_micro']==pytest.approx(2/3)
    assert values['missing_count']==1


def test_paired_cluster_bootstrap_keeps_twenty_questions_linked_across_sizes():
    deltas = np.array([-1]*8+[0]*8+[1]*4,dtype=float)
    a=bootstrap(deltas,count=1000)
    repeated=bootstrap(np.repeat(deltas[:,None],8,axis=1),count=1000)
    assert repeated['question_clusters']==20
    assert a['across_columns_95ci']==repeated['across_columns_95ci']
    assert repeated['across_columns_mean']==pytest.approx(-.2)


def test_cache_resume_validates_float32_norm_and_rejects_corruption(tmp_path):
    encoder=Encoder.__new__(Encoder)
    encoder.identity={'model':'fixture'};encoder.root=tmp_path
    text='fixture';key=digest({'text_sha256':digest(text),'identity':encoder.identity,'kind':'document'})
    path=tmp_path/(key+'.npy');meta=tmp_path/(key+'.json')
    vector=unit(np.ones(1024),dimensions=1024)
    np.save(path,vector,allow_pickle=False)
    meta.write_text(json.dumps({'text_sha256':digest(text),'vector_sha256':file_hash(path)}))
    output,receipt=encoder.encode(text)
    assert np.array_equal(output,vector) and receipt['cache_hit']
    path.write_bytes(b'corrupt')
    with pytest.raises(AssertionError):encoder.encode(text)
