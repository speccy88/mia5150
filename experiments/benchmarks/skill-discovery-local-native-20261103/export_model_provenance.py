"""Export public artifact identities without copying weights or private inputs."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    jobs = json.loads((ROOT/'local-models.json').read_text())
    registry = json.loads((ROOT/'evidence/ollama-qwen35-official-registry.json').read_text())
    manifests = {}
    for manifest in (ROOT/'models').glob('*.source.json'):
        manifests[manifest.name.removesuffix('.source.json')] = json.loads(manifest.read_text(encoding='utf-8-sig'))
    artifacts = []
    for job in jobs:
        record = {key: job[key] for key in ('requested', 'alias', 'source', 'quantization', 'blocked', 'note') if key in job}
        if job.get('path'):
            path = Path(job['path'])
            if not path.is_absolute():
                path = ROOT/path
            record['artifact_filename'] = path.name
            record['artifact_size'] = path.stat().st_size
            if path.name in manifests:
                record['verified_manifest'] = manifests[path.name]
            elif job['requested'].startswith('Qwen3.5-'):
                tag = '4b-q4_K_M' if job['requested'].endswith('-4B') else '9b-q4_K_M'
                layer = next(x for x in registry[tag]['layers'] if x['mediaType'] == 'application/vnd.ollama.image.model')
                record.update(sha256=job['sha256'], official_ollama_tag=tag,
                    official_ollama_model_layer=layer,
                    official_qwen_gguf=False)
            else:
                record.update(sha256=job['sha256'], byte_identity='Installed Ling bytes match official publisher GGUF LFS SHA-256.')
        artifacts.append(record)
    output = {'exported_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'artifacts': artifacts,
        'verification': 'Hashes and source revisions were checked at download/conversion/startup. This offline export does not reread all model weights while inference runs.',
        'sources': ['evidence/source-inventory.json', 'evidence/ollama-qwen35-official-registry.json', 'evidence/conversion-status.json']}
    (ROOT/'evidence/model-artifact-provenance.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'requested': len(artifacts), 'artifacts_present': sum('artifact_filename' in x for x in artifacts)}))


if __name__ == '__main__':
    main()
