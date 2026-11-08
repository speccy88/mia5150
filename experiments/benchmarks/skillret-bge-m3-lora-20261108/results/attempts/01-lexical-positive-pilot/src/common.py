"""Separate LoRA artifacts; immutable baseline helpers are imported read-only."""
import json
import os
from pathlib import Path
import importlib.util

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
BASE = STUDY.parent / 'skillret-bge-m3-fulltext-20261108'
BASE_REV = '0c7a0cfe868aac8e5122680b53dc3afab2ae72f9'
CACHE = Path(os.environ.get('SKILLRET_LORA_CACHE', str(Path.home()/'.cache/mia-bge-lora')))
BASE_CACHE = Path.home()/'.cache/mia-bgem3'
DATA_REV = '6583d7d2ed07644d0fb8938ed8178f3a7dc42a12'

def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

shared=module('immutable_bge_common',BASE/'src/common.py')
for name in ('QWEN','QWEN_REV','MODEL_REV','MODEL_SHA','CONFIGS','SIZES','atomic_json','digest',
             'file_hash','full_text','jsonl','unit','chunks','panel','rank_skills','evaluate',
             'bootstrap','latency_summary','independent','compare_evaluation'):
    globals()[name]=getattr(shared,name)

def read(path):return json.loads(Path(path).read_text())

def verify_baseline():
    validation=read(BASE/'results/validation.json')
    assert validation['completed_requested_scope']
    assert read(BASE/'results/progress.json')['stage']=='completed'
    for item in read(BASE/'FILE_MANIFEST.json')['files']:
        path=BASE/item['path']
        with path.open('rb') as stream:prefix=stream.read(160)
        if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
            assert ('oid sha256:'+item['sha256']).encode() in prefix
        else:assert file_hash(path)==item['sha256'],item['path']
    return {'commit':BASE_REV,'all_artifact_hashes_verified':True,'baseline_rerun':False,
            'completed_configurations':list(CONFIGS),'manifest_sha256':file_hash(BASE/'FILE_MANIFEST.json')}
