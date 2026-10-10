"""Audit frozen offline plans against unchanged dense pools; no network or secrets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import run_jev_rerank as prep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--base', type=Path, default=Path('/home/user/code/mia5150-local'))
    args = parser.parse_args()
    root = args.experiment_dir
    results = root / 'results'
    qrels = prep.read_jsonl(args.base / 'data/qrels/test.jsonl')
    gold = {(r['query_id'], r['skill_id']) for r in qrels if r['relevance'] > 0}
    dense = prep.index_rows(prep.read_jsonl(results / 'dense_top50.jsonl'), 'query_id')
    output = {'status': 'offline_preparation_audited_not_executed', 'network_calls': 0,
              'credentials_accessed': False, 'scopes': {}, 'previous_plans': {},
              'selection_unchanged': True, 'fixed_top20_unchanged': True,
              'caveat': 'Truncation is required by the frozen conservative byte estimator, not proven native Jev overflow. No provider acceptance or accuracy effect is measured.'}
    all_pairs = []
    manifests = {}
    for scope, expected in (('smoke', 10), ('pilot', 200)):
        folder = results / ('jev-plan-' + scope + '-context-v1')
        manifest = prep.read_json(folder / 'preparation_manifest.json')
        expected_hash = manifest['plan_sha256']
        assert prep.digest({k: v for k, v in manifest.items() if k != 'plan_sha256'}) == expected_hash
        old = prep.read_json(results / ('jev-plan-' + scope) / 'preparation_manifest.json')
        assert [r['query_id'] for r in manifest['rows']] == [r['query_id'] for r in old['rows']]
        assert len(manifest['rows']) == expected
        original_bytes = retained_bytes = requests = full_pairs = 0
        truncated = []
        max_total = max_pair = 0
        for row in manifest['rows']:
            qid = row['query_id']
            assert row['status'] == 'prepared_not_run'
            pool = prep.dense_candidates(dense[qid])[:20]
            assert row['dense_top20'] == pool
            assert [s for batch in row['batches'] for s in batch['candidate_ids']] == [r['skill_id'] for r in pool]
            assert len(row['candidate_text_audit']) == 20
            for batch in row['batches']:
                request = prep.read_json(folder / batch['request_file'])
                assert prep.digest(request) == batch['request_sha256']
                assert request['state'] == {'task': row['query']}
                estimates = prep.estimate_request_context(request, manifest['context']['overhead_tokens'])
                total = estimates['estimated_input_token_upper_bound'] + manifest['context']['reserved_output_tokens']
                pair = estimates['estimated_longest_pair_token_upper_bound'] + manifest['context']['reserved_output_tokens']
                assert total <= 64000 and pair <= 32000
                max_total = max(max_total, total)
                max_pair = max(max_pair, pair)
                requests += 1
            for audit in row['candidate_text_audit']:
                original_bytes += audit['original_skill_md_utf8_bytes']
                retained_bytes += audit['prepared_skill_md_utf8_bytes']
                pair = {'scope': scope, 'query_id': qid, 'gold_relevant_analysis_only': int((qid, audit['skill_id']) in gold), **audit}
                all_pairs.append(pair)
                if audit['truncated']:
                    truncated.append(pair)
                else:
                    full_pairs += 1
                    assert audit['original_document_sha256'] == audit['prepared_document_sha256']
        assert full_pairs + len(truncated) == expected * 20
        fractions = [r['retained_skill_md_utf8_byte_fraction'] for r in truncated]
        stats = {'selected': expected, 'prepared': len(manifest['rows']), 'blocked': 0,
                 'pair_count': expected * 20, 'truncated_pairs': len(truncated), 'full_text_pairs': full_pairs,
                 'queries_with_truncation': len({r['query_id'] for r in truncated}),
                 'unique_truncated_skills': len({r['skill_id'] for r in truncated}),
                 'original_skill_md_utf8_bytes': original_bytes, 'prepared_skill_md_utf8_bytes': retained_bytes,
                 'removed_skill_md_utf8_bytes': original_bytes - retained_bytes,
                 'retained_skill_md_utf8_fraction': retained_bytes / original_bytes,
                 'truncated_pair_retained_fraction_min': min(fractions) if fractions else None,
                 'truncated_pair_retained_fraction_median': statistics.median(fractions) if fractions else None,
                 'gold_relevant_truncated_pairs_analysis_only': sum(r['gold_relevant_analysis_only'] for r in truncated),
                 'request_count': requests, 'max_total_estimate_with_output': max_total,
                 'max_pair_estimate_with_output': max_pair,
                 'manifest_path': folder.relative_to(results).as_posix() + '/preparation_manifest.json',
                 'plan_sha256': expected_hash}
        output['scopes'][scope] = stats
        output['previous_plans'][scope] = {'selected': old['counts']['selected'], 'prepared': old['counts']['prepared'],
                                         'blocked': old['counts']['blocked'], 'plan_sha256': old['plan_sha256']}
        manifests[scope] = manifest
    pilot_rows = {r['query_id']: r for r in manifests['pilot']['rows']}
    assert all(row == pilot_rows[row['query_id']] for row in manifests['smoke']['rows'])
    output['smoke_exact_subset_of_pilot'] = True
    output['policy'] = manifests['pilot']['context']
    output['source_counts'] = {'queries': 4392, 'skills': 6006, 'qrels': 7187}
    output['qrels_usage'] = 'post-preparation impact counting only; never used to select queries, cuts, batches, or provider inputs'
    (results / 'context_preparation_audit.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    with (results / 'context_candidate_audit.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = list(all_pairs[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in all_pairs:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in pair.items()})
    lines = ['# Frozen Jev context preparation', '', output['caveat'], '',
             'The original blocked plans remain unchanged. Every selected query retains its original dense top20 pool. Smoke requests are an exact subset of the pilot requests, permitting cache reuse.', '',
             '| Scope | Queries prepared | Full-text pairs | Truncated pairs | Queries with truncation | Requests | Body bytes retained |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, s in output['scopes'].items():
        lines.append(f"| {name} | {s['prepared']}/{s['selected']} | {s['full_text_pairs']} | {s['truncated_pairs']} | {s['queries_with_truncation']} | {s['request_count']} | {s['retained_skill_md_utf8_fraction']:.2%} |")
    lines += ['', 'Batching is greedy in frozen dense-rank order. Each original candidate is tested alone first. Full text is unchanged when admitted. The optional policy otherwise retains the longest Unicode-codepoint prefix of skill_md that passes both canonical-JSON byte checks. Query, name, description, rubric, IDs and ranks are unchanged. No relevance label chooses a cut.', '',
              'The study interprets documented 64k/32k as conservative local 64,000/32,000 thresholds. It adds 2,048 framing units and reserves 2,048 output units. These are study choices. Exact native Jev token counts and boundary behavior remain unknown. See [primary-source audit](../CONTEXT_SOURCE_AUDIT.md).', '',
              'Per-pair source/prepared hashes, counts, retained fractions and reasons are saved in context_candidate_audit.csv and the frozen manifests. Gold-relevance flags appear only in this after-the-fact diagnostic CSV; they are absent from requests and preparation decisions.', '',
              'Any later quality result must distinguish full-text and text-reduced query subsets while keeping all selected queries in the overall denominator. No Jev accuracy, latency, usage, or cost has been measured. No credential or endpoint was accessed.']
    (results / 'context_preparation_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: output[k] for k in ('status', 'scopes', 'smoke_exact_subset_of_pilot')}, sort_keys=True))


if __name__ == '__main__':
    main()
