"""Summarize read-only samples; shared usage alone does not identify CUDA spill."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    grouped = collections.defaultdict(list)
    path = ROOT/'evidence/wddm-memory-telemetry.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            sample = json.loads(line)
            grouped[sample['model']].append(sample)
    models = []
    for model, samples in grouped.items():
        values = collections.defaultdict(list)
        for sample in samples:
            for counter in sample['samples']:
                if counter['status'] == 0:
                    values[counter['counter']].append(counter['bytes']/1024**2)
        models.append({'model': model, 'sample_count': len(samples),
            'first_utc': samples[0]['utc'], 'last_utc': samples[-1]['utc'],
            'mib_ranges': {k: {'min': min(v), 'max': max(v)} for k, v in values.items()}})
    output = {'models': models,
        'scope': 'Sampling began during Ornith, not for earlier models or startup-rejected models. PID/start/executable verified before each sample.',
        'interpretation': 'Dedicated/shared/non-local/local/committed WDDM process counters are reported directly. Shared memory can include host allocations and driver staging; nonzero shared usage alone neither proves transformer CPU offload nor identifies which CUDA buffer was paged. Sampling is not a complete residency trace.'}
    (ROOT/'evidence/wddm-memory-summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'sampled_models': len(models)}))


if __name__ == '__main__':
    main()
