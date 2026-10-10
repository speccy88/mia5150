"""Prepare bounded SearchJev pilot pairs using only stdlib and local SkillRet JSONL.

Run with the parent task's fresh resource guard. Never imports model/source code,
downloads, calls APIs, or writes the dataset. Writes pairs.json and preparation-
audit.json beside this script unless --output-dir is supplied.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

REVISION = '6583d7d2ed07644d0fb8938ed8178f3a7dc42a12'
PINNED = {
    'data/skills/train.jsonl': (10123, 'b4978863f6c36c1d42af4a779f86b8e5809b4709c353aa698ec494e3ed663a2b'),
    'data/skills/test.jsonl': (6006, '1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81'),
    'data/queries/train.jsonl': (63259, 'e1189e402c74d11ba57af3bbc60308cc4a9b51f29f2de265e541d0599067624c'),
    'data/queries/test.jsonl': (4392, '4d07a35f40e597f8468c6dab480c95a1f9b40ecbde80274081ba2e7c643d0f6d'),
    'data/qrels/train.jsonl': (127190, '7480dbf936eeee9905c752a681b029bae8e13f67dcacdf4b716684ca21da83e1'),
    'data/qrels/test.jsonl': (7187, 'edbfae0b8109de460d54e27e6235b1b47316f34956744db0571ad1b4ddb860af'),
}

def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()

def text_hash(value):
    return sha_bytes(value.strip().lower().encode('utf-8'))

def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=Path('/home/user/code/mia5150-local'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    started = time.time()
    root = args.source_root
    source_receipts = {}

    def rows(rel, pass_label=None):
        digest = hashlib.sha256()
        count = 0
        with (root / rel).open('rb') as stream:
            for line in stream:
                digest.update(line)
                count += 1
                yield json.loads(line)
        expected_count, expected_sha = PINNED[rel]
        assert count == expected_count, (rel, 'row count changed', count)
        assert digest.hexdigest() == expected_sha, (rel, 'pinned source hash changed')
        source_receipts[pass_label or rel] = {
            'path': str(root / rel), 'rows': count, 'sha256': digest.hexdigest(),
            'bytes': (root / rel).stat().st_size, 'verified': True,
        }

    manifest_path = root / 'manifests/training-partitions.json'
    manifest_bytes = manifest_path.read_bytes()
    partition = json.loads(manifest_bytes)
    data_manifest_path = root / 'evidence/data-manifest.json'
    assert json.loads(data_manifest_path.read_bytes())['revision'] == REVISION
    assert len(partition['train']) == 512 and len(partition['dev']) == 128
    assert not set(partition['train']) & set(partition['dev'])
    allowed = set(partition['train_allowed_skill_ids'])

    skills = {}
    split_skill_ids = {}
    for split in ('train', 'test'):
        split_skill_ids[split] = set()
        for row in rows(f'data/skills/{split}.jsonl'):
            sid = row['id']
            assert sid not in skills, ('duplicate/cross-split skill ID', sid)
            skills[sid] = {'name': row['name'], 'repo': row.get('repo')}
            split_skill_ids[split].add(sid)
    assert allowed <= split_skill_ids['train']

    parent = {sid: sid for sid in skills}
    def family(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    groups = {}
    for sid, row in skills.items():
        for key in [('repo', row.get('repo')), ('name', row['name'].strip().lower())]:
            if key[1]:
                if key in groups:
                    parent[family(sid)] = family(groups[key])
                else:
                    groups[key] = sid

    qrels = {}
    for split in ('train', 'test'):
        qrels[split] = defaultdict(set)
        for row in rows(f'data/qrels/{split}.jsonl'):
            qid, sid = row['query_id'], row['skill_id']
            assert row['relevance'] == 1
            assert sid in split_skill_ids[split]
            assert sid not in qrels[split][qid], ('duplicate positive edge', qid, sid)
            qrels[split][qid].add(sid)

    wanted_query_ids = set(partition['train']) | set(partition['dev'])
    query_rows = {}
    public_query_hashes = set()
    all_query_ids = {}
    for split in ('train', 'test'):
        all_query_ids[split] = set()
        for row in rows(f'data/queries/{split}.jsonl'):
            qid = row['id']
            assert qid not in all_query_ids[split], ('duplicate query ID', qid)
            all_query_ids[split].add(qid)
            assert set(row['skill_ids']) == qrels[split][qid]
            assert len(row['skill_ids']) == len(set(row['skill_ids'])) == row['k']
            if split == 'train' and qid in wanted_query_ids:
                query_rows[qid] = {'query': row['query'], 'text_sha256': text_hash(row['query'])}
            if split == 'test':
                public_query_hashes.add(text_hash(row['query']))
        assert all_query_ids[split] == set(qrels[split])
    assert set(query_rows) == wanted_query_ids

    # Historical exposure includes the entire public test, not merely the128 panel.
    public_gold_families = {family(s) for gold in qrels['test'].values() for s in gold}
    previous_dev_families = {family(s) for q in partition['dev'] for s in qrels['train'][q]}
    forbidden_train_families = public_gold_families | previous_dev_families
    forbidden_query_hashes = public_query_hashes | {query_rows[q]['text_sha256'] for q in partition['dev']}
    eval_qids = partition['dev'][:16]
    train_qids = []
    skipped = []
    selected_query_hashes = set()
    for qid in partition['train']:
        gold = qrels['train'][qid]
        reasons = []
        if not gold <= allowed:
            reasons.append('positive outside existing train_allowed_skill_ids')
        if {family(s) for s in gold} & forbidden_train_families:
            reasons.append('gold family intersects prior development/public-test positive family')
        if query_rows[qid]['text_sha256'] in forbidden_query_hashes | selected_query_hashes:
            reasons.append('normalized query text already in excluded/selected set')
        if reasons:
            skipped.append({'query_id': qid, 'reasons': reasons})
            continue
        train_qids.append(qid)
        selected_query_hashes.add(query_rows[qid]['text_sha256'])
        if len(train_qids) == 32:
            break
    assert len(train_qids) == 32, 'Insufficient eligible train queries; no relaxed fallback'

    # All evaluation candidate families are forbidden for training. This applies
    # to negative records as well as positives, stronger than disjoint gold only.
    train_negative_pool = sorted(s for s in allowed if family(s) not in forbidden_train_families)
    eval_negative_pool = sorted(s for s in split_skill_ids['train'] if family(s) in forbidden_train_families)
    pair_specs = {'train': [], 'eval': []}
    for split, qids, negative_pool in [('train', train_qids, train_negative_pool), ('eval', eval_qids, eval_negative_pool)]:
        for qid in qids:
            gold = qrels['train'][qid]
            negatives = sorted(
                (sid for sid in negative_pool if sid not in gold),
                key=lambda sid: (sha_bytes(f'searchjev-skillret-pilot-v1|{split}|{qid}|{sid}'.encode()), sid),
            )[:len(gold)]
            assert len(negatives) == len(gold) and not set(negatives) & gold
            pair_specs[split].extend((qid, sid, 1) for sid in sorted(gold))
            pair_specs[split].extend((qid, sid, 0) for sid in negatives)
    random.Random(271828).shuffle(pair_specs['train'])

    train_families = {family(s) for _, s, _ in pair_specs['train']}
    eval_families = {family(s) for _, s, _ in pair_specs['eval']}
    assert not train_families & eval_families, 'Actual train/eval candidate family overlap'
    assert not train_families & forbidden_train_families
    assert not set(train_qids) & set(eval_qids)
    assert not {query_rows[q]['text_sha256'] for q in train_qids} & {query_rows[q]['text_sha256'] for q in eval_qids}

    selected_skills = {s for records in pair_specs.values() for _, s, _ in records}
    text_rows = {}
    truncation = Counter()
    for row in rows('data/skills/train.jsonl', 'data/skills/train.jsonl:selected_text_pass'):
        if row['id'] in selected_skills:
            text_rows[row['id']] = {}
            for output_key, source_key in [('skill_name', 'name'), ('skill_description', 'description'), ('skill_body', 'body')]:
                value = row.get(source_key) or ''
                truncation[output_key] += len(value) > 4000
                text_rows[row['id']][output_key] = value[:4000]
    assert set(text_rows) == selected_skills
    records = {}
    coverage = {}
    for split, specs, qids in [('train', pair_specs['train'], train_qids), ('eval', pair_specs['eval'], eval_qids)]:
        records[split] = [
            {'query_id': qid, 'skill_id': sid, 'query': query_rows[qid]['query'], **text_rows[sid], 'y': y}
            for qid, sid, y in specs
        ]
        required = {(q, s) for q in qids for s in qrels['train'][q]}
        retained = {(q, s) for q, s, y in specs if y == 1}
        assert required == retained
        assert all((s in qrels['train'][q]) == bool(y) for q, s, y in specs)
        coverage[split] = {'queries': len(qids), 'positive_edges_required': len(required), 'positive_edges_retained': len(retained), 'negative_pairs': sum(y == 0 for _, _, y in specs), 'all_positives_preserved': True, 'negative_gold_collisions': 0}

    audit = {
        'dataset_revision': REVISION,
        'source_root': str(root),
        'source_files': source_receipts,
        'partition_manifest': {'path': str(manifest_path), 'sha256': sha_bytes(manifest_bytes)},
        'data_manifest': {'path': str(data_manifest_path), 'sha256': sha_bytes(data_manifest_path.read_bytes())},
        'script_sha256': sha_bytes(Path(__file__).read_bytes()),
        'seed': 271828,
        'selection': 'First32 eligible in existing train512 order after exclusions; first16 existing dev128. No score/label-count/outcome selection.',
        'train_query_ids': train_qids, 'eval_query_ids': eval_qids,
        'skipped_train_query_ids_with_reasons': skipped,
        'family_definition': 'Connected components over combined train/test skill catalogs using exact repository OR stripped lowercase exact name.',
        'excluded_train_family_sources': ['All128 prior local adaptation dev positive families', 'All4392 historically exposed public-test query positive families'],
        'train_negative_pool_count': len(train_negative_pool),
        'eval_negative_pool_count': len(eval_negative_pool),
        'negative_method': 'One unique deterministic SHA256-ranked non-gold negative per positive; all official gold excluded. Train negatives additionally within original train_allowed_skill_ids and outside every forbidden family. Eval negatives from forbidden families to guarantee every actual candidate family is disjoint.',
        'actual_train_candidate_families': sorted(train_families),
        'actual_eval_candidate_families': sorted(eval_families),
        'actual_train_eval_family_overlap_count': 0,
        'actual_train_exposed_positive_family_overlap_count': 0,
        'coverage': coverage,
        'rendered_text': 'Selected skill name, description, body fields each retain first4000 characters; query text unchanged. Training tokenizer applies its own separately reported token bound.',
        'truncated_selected_skill_fields': dict(truncation),
        'selected_unique_skills': len(selected_skills),
        'exposure': 'Exposed development diagnostic; not blind, not a pristine final test, and not proof of pretrained-model generalization. Historic256 dev IDs at a different dataset revision are unavailable in the examined exposure ledger; no absence-of-exposure claim.',
        'eval_design': 'Gold-conditioned balanced binary query-skill relevance pairs; not a retrieval-ranked shortlist. Evaluate fixed identical pairs before/after. CE+Brier, score separation and threshold accuracy are diagnostics; pair balancing and explicit gold insertion invalidate full-corpus retrieval claims.',
        'canonical_train_records_sha256': sha_bytes(json_bytes(records['train'])),
        'canonical_eval_records_sha256': sha_bytes(json_bytes(records['eval'])),
        'elapsed_seconds': time.time() - started,
    }
    output = {'schema_version': 'searchjev-skillret-pairs-v1', 'train': records['train'], 'eval': records['eval'], 'audit': audit}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / 'pairs.json'
    output_bytes = json.dumps(output, ensure_ascii=False, indent=2).encode('utf-8')
    output_path.write_bytes(output_bytes)
    receipt = {**audit, 'pairs_path': str(output_path), 'pairs_sha256': sha_bytes(output_bytes), 'pairs_bytes': len(output_bytes)}
    (args.output_dir / 'preparation-audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'pairs_path': str(output_path), 'pairs_sha256': sha_bytes(output_bytes), 'coverage': coverage, 'family_overlap': 0, 'skipped_train_queries': len(skipped), 'seconds': time.time() - started}, indent=2))

if __name__ == '__main__':
    main()
