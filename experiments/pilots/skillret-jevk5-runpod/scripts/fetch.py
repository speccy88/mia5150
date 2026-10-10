"""Download one official split at a pinned revision. Test download is explicit."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from skillret_decider.core import read_jsonl, index_unique, load_qrels, write_json, sha256_file

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--revision', default='a050ad2', help='Resolve the documented short SHA to a full immutable SHA')
    p.add_argument('--allow-test-download', action='store_true')
    args = p.parse_args()
    if args.split == 'test' and not args.allow_test_download:
        p.error('Do not download the final test split until the procedure is frozen; use --allow-test-download.')
    if args.out.exists(): raise FileExistsError(args.out)
    from huggingface_hub import HfApi, hf_hub_download
    repo = 'ThakiCloud/SKILLRET'
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    args.out.mkdir(parents=True)
    files = {}
    for part in ('skills', 'queries', 'qrels'):
        name = f'data/{part}/{args.split}.jsonl'
        source = hf_hub_download(repo, name, repo_type='dataset', revision=revision)
        destination = args.out / f'{part}.jsonl'
        shutil.copyfile(source, destination)
        files[part] = {'upstream_path': name, 'sha256': sha256_file(destination)}
    skills, queries, qrels = [read_jsonl(args.out / f'{part}.jsonl') for part in ('skills', 'queries', 'qrels')]
    si, qi = index_unique(skills), index_unique(queries)
    load_qrels(qrels, set(qi), set(si))
    expected = {'train': (10123,63259,127190), 'test': (6006,4392,7187)}[args.split]
    observed = (len(skills),len(queries),len(qrels))
    if observed != expected:
        raise RuntimeError(f'Expected v1.1 counts {expected}, found {observed}. Stop and inspect the revision; do not silently accept a changed dataset.')
    write_json(args.out / 'manifest.json', {'repository':repo, 'revision':revision, 'requested_revision':args.revision,
        'split':args.split, 'counts':dict(zip(('skills','queries','qrels'),observed)), 'files':files})
    print(json.dumps({'out':str(args.out), 'revision':revision, 'counts':observed}, indent=2))

if __name__ == '__main__': main()
