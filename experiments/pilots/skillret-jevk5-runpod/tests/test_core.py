import math
import pytest
import numpy as np
from skillret_decider.core import *
from scripts.build import dense_rank

def fixture():
    skills={f's{i}':{'id':f's{i}','name':f'skill{i}','description':f'Function {i}'} for i in range(20)}
    query={'id':'q1','query':'Please analyze the data','skill_ids':['s2','s3'],'k':2,'skill_names':['secret label']}
    return skills,query,list(skills)

def test_multi_positive_targets():
    skills,q,rank=fixture();e=make_example(q,rank,skills,['s2','s3'])
    assert len(e['question']['criteria'])==16
    assert sorted(e['gold_probs'].values())==[0.]*14+[.5,.5]
    assert e['state']==q['query'] and 'secret label' not in e['state']
    for k,s in e['option_skill_ids'].items(): assert e['gold_probs'][k]==(.5 if s in ['s2','s3'] else 0.)

def test_none_target_on_miss_without_gold_injection():
    skills,q,rank=fixture();e=make_example(q,rank,skills,['s19'])
    key=next(k for k,s in e['option_skill_ids'].items() if s==NONE)
    assert e['gold_probs'][key]==1 and 's19' not in e['retrieval_order'] and not e['gold_injected']

def test_positive_outside_shortlist_stays_in_metric_denominator():
    assert query_metrics(['a'],{'a','b'})['recall@1']==.5
    assert query_metrics(['a'],{'a','b'})['hit@1']==1

def test_training_augmentation_excludes_all_positives_from_negatives():
    skills,q,rank=fixture();e=make_example(q,rank,skills,['s1','s19'],augmented=True)
    assert {'s1','s19'} <= set(e['retrieval_order']) and len(set(e['retrieval_order']))==15

@pytest.mark.parametrize('split',['validation','test'])
def test_eval_gold_injection_forbidden(split):
    skills,q,rank=fixture()
    with pytest.raises(ValueError): make_example(q,rank,skills,['s19'],augmented=True,split=split)

def test_unlabelled_inference_view():
    skills,q,rank=fixture();e=make_example(q,rank,skills,None,split='test')
    assert 'gold_probs' not in e and 'expected' not in e

def test_shuffling_is_deterministic_and_not_fixed():
    skills,q,rank=fixture();a=make_example(q,rank,skills,['s2'])
    assert a==make_example(q,rank,skills,['s2'])
    positions=set()
    for i in range(30):
        e=make_example({**q,'id':f'q{i}'},rank,skills,['s2'])
        positions.add(next(k for k,s in e['option_skill_ids'].items() if s=='s2'))
    assert len(positions)>10

def test_schema_export_preserves_soft_targets():
    skills,q,rank=fixture();e=upstream_row(make_example(q,rank,skills,['s1','s2']))
    assert set(e)=={'id','source','state','question','expected','gold_probs'}
    assert sum(e['gold_probs'].values())==1

def test_qrels_duplicates_and_missing_fail():
    with pytest.raises(ValueError): load_qrels([],{'q'},{'s'})
    row={'query_id':'q','skill_id':'s','relevance':1}
    with pytest.raises(ValueError): load_qrels([row,row],{'q'},{'s'})
    with pytest.raises(ValueError): load_qrels([row],{'other'},{'s'})

def test_grouping_joins_original_id_and_revised_text():
    qs=[{'id':'a','query':'First','original_id':'same'}, {'id':'b','query':'Second','original_id':'same'},
        {'id':'c','query':'third','original_query':' SECOND '},{'id':'d','query':'unrelated'}]
    fam=query_families(qs)
    assert fam['a']==fam['b']==fam['c'] and fam['a']!=fam['d']
    with pytest.raises(ValueError): split_queries(qs,prior_manifest={'train_query_ids':['a','d'],'validation_query_ids':['b','c']})
    sp=split_queries(qs,prior_manifest={'train_query_ids':['a','b','c'],'validation_query_ids':['d']})
    assert sp['method'].startswith('reused_prior')

def test_default_split_reproducible():
    qs=[{'id':str(i),'query':f'unique-{i}'} for i in range(200)]
    a=split_queries(qs);assert a==split_queries(list(reversed(qs)))
    assert set(a['train_query_ids']).isdisjoint(a['validation_query_ids'])

def test_bad_shortlists_fail():
    skills,q,rank=fixture()
    for bad in ([rank[0]]*15,rank[:2],['bad']+rank[:14]):
        with pytest.raises(ValueError): make_example(q,bad,skills,['s2'])

def test_logits_mapping_accepts_either_positive():
    skills,q,rank=fixture();e=make_example(q,rank,skills,['s2','s3'])
    for positive in ['s2','s3']:
        logits=[10 if e['option_skill_ids'][k]==positive else 0 for k in e['question']['criteria']]
        p=prediction_from_logits(e,logits)
        assert p['ranked_skill_ids'][0]==positive
        assert evaluate_predictions([p],{'q1':['s2','s3']})['mean']['hit@1']==1

def test_metric_ceiling_and_all_negative_controls():
    p={'query_id':'q','candidate_skill_ids':['a','x'],'ranked_skill_ids':['a','x']}
    m=evaluate_predictions([p],{'q':['a','b','c']})['mean']
    assert m['recall@5']==m['oracle_recall@5']==m['candidate_recall@15']==pytest.approx(1/3)
    wrong={'query_id':'q','candidate_skill_ids':['x','y'],'ranked_skill_ids':['y','x']}
    n=evaluate_predictions([wrong],{'q':['a','b']})['mean']
    assert n['recall@5']==n['ndcg@5']==n['mrr@15']==0

def test_partial_or_altered_prediction_fails():
    labels={'q':['a']}
    with pytest.raises(ValueError): evaluate_predictions([],labels)
    with pytest.raises(ValueError): evaluate_predictions([{'query_id':'q','candidate_skill_ids':['a'],'ranked_skill_ids':['b']}],labels)
    with pytest.raises(ValueError): query_metrics(['a','a'],{'a'})

def test_ndcg_and_mrr_known_values():
    m=query_metrics(['x','a','y'],{'a'})
    assert m['ndcg@5']==pytest.approx(1/math.log2(3)) and m['mrr@15']==.5

def test_bootstrap_pairs():
    a={'per_query':[{'query_id':str(i),'recall@5':.25} for i in range(10)]}
    b={'per_query':[{'query_id':str(i),'recall@5':.5} for i in range(10)]}
    c=paired_bootstrap(a,b,repeats=100)
    assert c['mean_difference']==.25 and c['bootstrap_95_ci']==[.25,.25]

def test_dense_ranking_stable_ties():
    rank=dense_rank(np.array([[1.,0.]],dtype=np.float32),np.array([[1.,0.],[1.,0.],[0.,1.]],dtype=np.float32),['a','b','c'])
    assert rank==[['a','b','c']]

def test_exclusive_writes(tmp_path):
    write_json(tmp_path/'x.json',{'a':1})
    with pytest.raises(FileExistsError): write_json(tmp_path/'x.json',{'a':2})

def test_abstention_not_confused_with_forced_ranking_hit():
    p={'query_id':'q','candidate_skill_ids':['a','b'],'ranked_skill_ids':['a','b'],'top_decision_skill_id':NONE}
    m=evaluate_predictions([p],{'q':['a']})['mean']
    assert m['hit@1']==1 and m['decision_correct']==0 and m['false_abstention']==1
