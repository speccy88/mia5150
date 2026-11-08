"""Independent original formulas plus immutable base/adapted input identity."""
from common import *

def validate():
    verify_baseline()
    reference=module('immutable_bge_validator',BASE/'src/validate.py')
    reference.validate(CACHE)
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
