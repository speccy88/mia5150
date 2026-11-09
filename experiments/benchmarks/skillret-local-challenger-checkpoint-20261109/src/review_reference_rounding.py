"""Accept only preregistered harmless rounding differences in a single reconstruction."""
from common import CACHE, OLD, STUDY, read, rows, save, sha


def main():
    results = STUDY/'results'
    parity = results/'reconstruction_parity.json'
    diagnostic = results/'reconstruction_diagnostics.json'
    difference_file = results/'reconstruction_ranking_differences.jsonl'
    accepted = results/'RECONSTRUCTION_REVIEW.json'
    assert not accepted.exists()
    d = read(diagnostic)
    assert read(parity)['status'] == 'runtime_difference_investigate'
    assert d['query_count'] == 4392 and d['identical_top50_count'] < 4392
    differences = rows(difference_file)
    assert d['identical_top50_count'] == 4392-len(differences)
    eps = 1e-5
    margin_fields = ('max_old_score_margin_for_reordered_common_pair',
                     'max_rebuilt_score_margin_for_reordered_common_pair',
                     'max_archived_score_margin_of_lost_id_above_cutoff',
                     'max_rebuilt_score_margin_of_gained_id_above_lost_id')
    old = read(OLD/'results/baseline_skillret_06b.json')['evaluation']['aggregate']
    new = read(results/'final/reconstructed-base.metrics.json')['aggregate']
    keys = [k for k in old.keys() & new.keys() if k.endswith(('@5','@10'))]
    assert 'NDCG@10' in keys and 'Recall@10' in keys and 'MRR@10' in keys
    checks = {
        'all_top10_orders_identical': d['identical_top10_count']==4392,
        'all_common_metrics_at_5_and_10_identical': all(abs(new[k]-old[k])<=1e-12 for k in keys),
        'all_219600_aligned_scores_within_1e-5': d['max_aligned_archived_score_error']<=eps,
        'every_disagreeing_rank_margin_within_2e-5': all(0<=row[k]<=2*eps for row in differences for k in margin_fields),
    }
    if not all(checks.values()):
        print('Reference differences exceed preregistered rounding-only acceptance; keep the single reconstruction for review.', checks, flush=True)
        return
    vectors={name:sha(CACHE/'vectors'/name/'vectors.npy') for name in ('test-documents','test-base-queries')}
    save(accepted, {
        'decision':'retain_single_reconstruction_and_both_controls',
        'explanation':'All top-10 rankings and shared metrics at cutoffs 5 and 10 are identical. Every aligned score and changed tail ordering margin meets the locked rounding-only bounds. Tail mismatches remain documented and the archived reference is still the primary contrast.',
        'evaluation_lock_sha256':sha(STUDY/'FINAL_EVAL_LOCK.json'),
        'strict_parity_receipt_sha256':sha(parity),
        'diagnostics_sha256':sha(diagnostic),
        'reconstructed_rankings_sha256':sha(results/'final/reconstructed-base.jsonl'),
        'retained_array_sha256':vectors,
        'automatic_rounding_checks':checks,
    })
    print('Accepted only the locked rounding-only case using the original reconstructed bytes.', flush=True)


if __name__ == '__main__':
    main()
