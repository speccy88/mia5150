"""Read-only, standard-library SkillRet data audit. No model imports or writes."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import time

root = Path('/home/user/code/mia5150-local')
started = time.time()
expected = json.loads((root / 'evidence/data-manifest.json').read_text())
report = {'dataset_revision': expected['revision'], 'files': {}, 'schemas': {}}
skills = {}
queries = {}
qrels = {}
for split in ['train', 'test']:
    skills[split] = {}
    queries[split] = {}
    qrels[split] = defaultdict(set)
    for kind in ['skills', 'queries', 'qrels']:
        rel = f'data/{kind}/{split}.jsonl'
        digest = hashlib.sha256()
        rows = duplicates = 0
        fields = Counter()
        relevance = Counter()
        with (root / rel).open('rb') as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                rows += 1
                fields.update(row.keys())
                if kind == 'skills':
                    duplicates += row['id'] in skills[split]
                    skills[split][row['id']] = {k: row.get(k) for k in ['id', 'name', 'repo']}
                elif kind == 'queries':
                    duplicates += row['id'] in queries[split]
                    queries[split][row['id']] = {'gold': set(row['skill_ids']), 'text_sha256': hashlib.sha256(row['query'].strip().lower().encode()).hexdigest(), 'k': row.get('k')}
                else:
                    relevance[str(row['relevance'])] += 1
                    duplicates += row['skill_id'] in qrels[split][row['query_id']]
                    qrels[split][row['query_id']].add(row['skill_id'])
        spec = expected['files'][rel]
        report['files'][rel] = {'rows': rows, 'bytes': (root / rel).stat().st_size, 'sha256': digest.hexdigest(), 'matches_recorded_manifest': rows == spec['rows'] and digest.hexdigest() == spec['sha256'], 'duplicate_ids_or_edges': duplicates}
        if kind == 'qrels':
            report['files'][rel]['relevance_values'] = dict(relevance)
        report['schemas'][rel] = dict(fields)
    report[split] = {
        'query_gold_histogram': dict(sorted(Counter(len(q['gold']) for q in queries[split].values()).items())),
        'qrels_query_count': len(qrels[split]),
        'unique_positive_edges': sum(map(len, qrels[split].values())),
        'query_gold_qrels_mismatches': [qid for qid, q in queries[split].items() if q['gold'] != qrels[split][qid]],
        'k_mismatches': sum(q['k'] != len(q['gold']) for q in queries[split].values()),
        'missing_positive_skill_ids': sorted({d for gold in qrels[split].values() for d in gold} - set(skills[split])),
        'qrels_ids_without_query': sorted(set(qrels[split]) - set(queries[split])),
    }

allskills = {**skills['train'], **skills['test']}
parent = {sid: sid for sid in allskills}
def find(sid):
    while parent[sid] != sid:
        parent[sid] = parent[parent[sid]]
        sid = parent[sid]
    return sid
def union(a, b):
    parent[find(a)] = find(b)
seen = {}
for sid, row in allskills.items():
    for key in [('repo', row.get('repo')), ('name', (row.get('name') or '').strip().lower())]:
        if key[1]:
            if key in seen:
                union(sid, seen[key])
            else:
                seen[key] = sid

partition = json.loads((root / 'manifests/training-partitions.json').read_text())
lora = json.loads((root / 'manifests/lora-hard.json').read_text())
def gold_ids(split, qids):
    return {d for qid in qids for d in qrels[split][qid]}
def families(ids):
    return {find(d) for d in ids}
train_gold = gold_ids('train', partition['train'])
dev_gold = gold_ids('train', partition['dev'])
train_family = families(train_gold)
dev_family = families(dev_gold)
public_gold = gold_ids('test', queries['test'])
public_gold_family = families(public_gold)
public_catalog_family = families(skills['test'])
allowed = set(partition['train_allowed_skill_ids'])
report['existing_partition'] = {
    'source': 'manifests/training-partitions.json',
    'train_queries': len(partition['train']), 'dev_queries': len(partition['dev']),
    'train_positive_edges': sum(len(qrels['train'][q]) for q in partition['train']),
    'dev_positive_edges': sum(len(qrels['train'][q]) for q in partition['dev']),
    'allowed_train_skills': len(allowed),
    'all_train_gold_in_allowed': train_gold <= allowed,
    'dev_gold_in_allowed': len(dev_gold & allowed),
    'train_dev_query_id_overlap': len(set(partition['train']) & set(partition['dev'])),
    'train_dev_normalized_query_text_overlap': len({queries['train'][q]['text_sha256'] for q in partition['train']} & {queries['train'][q]['text_sha256'] for q in partition['dev']}),
    'prior_neural_lora_train_queries': len(lora['train_query_ids']),
    'prior_neural_lora_dev_queries': len(lora['dev_query_ids']),
    'prior_neural_lora_train_positive_edges': sum(len(qrels['train'][q]) for q in lora['train_query_ids']),
    'prior_neural_lora_dev_positive_edges': sum(len(qrels['train'][q]) for q in lora['dev_query_ids']),
}
report['family_exposure'] = {
    'definition': 'Connected components over union of official train/test skills, exact repository string OR stripped lowercase exact skill name; diagnostic computation, not a new split.',
    'skill_id_train_test_overlap': len(set(skills['train']) & set(skills['test'])),
    'union_component_count': len(families(allskills)),
    'existing_train_gold_families': len(train_family),
    'existing_dev_gold_families': len(dev_family),
    'existing_train_dev_family_overlap_after_union': len(train_family & dev_family),
    'existing_dev_gold_families_in_exposed_public_test_gold': len(dev_family & public_gold_family),
    'existing_dev_gold_families_in_exposed_public_test_catalog': len(dev_family & public_catalog_family),
    'existing_train_gold_families_in_exposed_public_test_gold': len(train_family & public_gold_family),
    'all_test_query_labels_historically_exposed': True,
    'historic_256_dev_ids_available_in_examined_manifest': False,
    'conclusion': 'No certified blind or family-unexposed holdout. Existing grouped train/dev supports a disclosed within-experiment diagnostic only. Historical 256-query dev belongs to different revision and its exact IDs were not supplied by the examined ledger.',
}
report['recommendation'] = {
    'data': 'Reuse existing train512/dev128 grouping; select bounded training query prefix and preserve every qrel edge for every selected query. Include each query only if every gold skill is retained.',
    'negatives': 'Sample only train_allowed_skill_ids minus the complete per-query official gold set. Never truncate a mixed examples list after constructing positives.',
    'tiny_eval': 'Freeze first 4 or 8 existing dev128 IDs before scoring. Include all positives plus a fixed small nonpositive set per query, label as gold-conditioned binary relevance diagnostic; or use first4 existing lora-hard dev_pools unmodified for honest retrieval reranking diagnostic. Evaluate exactly same pairs/pools baseline and adapted. No blind/test/generalization claims.',
    'metrics': 'Pair-level CE+Brier, positive vs negative probability separation, paired score differences; for unmodified frozen pools include candidate recall ceiling and macro MRR/NDCG/Recall with all official gold denominators.',
}
report['elapsed_seconds'] = time.time() - started
print(json.dumps(report, indent=2))
