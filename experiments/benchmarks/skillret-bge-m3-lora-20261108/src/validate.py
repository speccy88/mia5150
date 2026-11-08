"""Independent original formulas plus immutable base/adapted input identity."""
from common import *

def validate_checkpoint_selection():
    r=STUDY/'results';membership=read(STUDY/'manifests/split_membership.json')
    examples=[e for e in jsonl(STUDY/'manifests/positive_passage_selection.jsonl') if e['split']=='validation']
    gold={e['query_id']:{sid:1 for sid in e['all_positive_ids']} for e in examples}
    candidates=set(membership['validation_skill_ids']);assert len(gold)==128
    checked=[]
    for label,folder in [('selected_recipe',r),('discarded_pilot',r/'attempts/01-lexical-positive-pilot')]:
        if label=='discarded_pilot' and not folder.exists():continue
        curve=jsonl(folder/'validation_curve.jsonl')
        for row in curve:
            data=read(folder/f"validation-step{row['optimizer_step']:05d}.json");rankings=data['rankings']
            assert set(rankings)==set(gold)
            for qid,ranked in rankings.items():
                assert len(ranked)==len(set(ranked))==50 and set(ranked)<=candidates
                assert set(gold[qid])<=candidates
            compare_evaluation(data['evaluation'],{qid:independent(g,rankings[qid]) for qid,g in gold.items()},gold,rankings)
            assert data['evaluation']['aggregate']==row['metrics']
        trained=[row for row in curve if row['optimizer_step']>0]
        winner=max(trained,key=lambda row:(row['metrics']['NDCG@10'],-row['optimizer_step']))
        summary=read(folder/'training_summary.json')
        assert summary['best_step']==winner['optimizer_step']
        assert abs(summary['best_validation_ndcg10']-winner['metrics']['NDCG@10'])<1e-12
        pilot=read(folder/'pilot_gate.json');pilot_best=max(row['metrics']['NDCG@10'] for row in trained if row['optimizer_step']<=200)
        assert abs(pilot_best-pilot['best_trained_validation_ndcg10'])<1e-12
        assert pilot['proceed_to_full']==(pilot_best>=pilot['frozen_validation_ndcg10']-.01)
        checked.append({'recipe':label,'validation_checkpoints':len(curve),'queries_per_checkpoint':128,
            'independent_metrics_passed':True,'selected_step':winner['optimizer_step'],'selection_rule_passed':True,
            'pilot_continuation_rule_passed':True})
    assert read(r/'selected_adapter.json')['optimizer_step']==read(r/'training_summary.json')['best_step']
    atomic_json(r/'checkpoint_selection_validation.json',{'status':'passed','recipes':checked,'official_test_scores_used':False})

def validate():
    verify_baseline()
    reference=module('immutable_bge_validator',BASE/'src/validate.py')
    reference.validate(CACHE)
    validate_checkpoint_selection()
    for config in CONFIGS:
        name='documents-'+config+'.jsonl'
        assert file_hash(STUDY/'results'/name)==file_hash(BASE/'results'/name)
        adapted=read(STUDY/'results'/config/'full_evaluation.json')
        original=read(BASE/'results'/config/'full_evaluation.json')
        assert set(adapted['per_query'])==set(original['per_query'])
        assert adapted['missing_count']==0
        for size in SIZES:
            a=read(STUDY/'results'/config/f'scaling-n{size}.json')
            b=read(BASE/'results'/config/f'scaling-n{size}.json')
            assert [(x['query_id'],x['query_sha256'],x['gold_id']) for x in a['rows']]==[(x['query_id'],x['query_sha256'],x['gold_id']) for x in b['rows']]
    assert file_hash(STUDY/'results/candidate_pools.json')==file_hash(BASE/'results/candidate_pools.json')
    atomic_json(STUDY/'results/paired_input_validation.json',{'status':'passed',
        'all_baseline_artifact_hashes_unchanged':True,'candidate_bytes_identical':True,
        'all_three_document_manifest_bytes_identical':True,'scaling_queries_golds_and_order_identical':True,
        'official_query_ids_identical':True,'all_adapted_queries_completed':True})

if __name__=='__main__':validate()
