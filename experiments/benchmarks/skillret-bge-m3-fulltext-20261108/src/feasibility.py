"""CUDA admission, repeat consistency and context-memory tests before quality."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from common import STUDY, QWEN, atomic_json, full_text, jsonl
from engine import Encoder, Monitor, public


def main(cache):
    monitor = Monitor('feasibility-loading')
    try:
        encoder = Encoder(cache, monitor, frozen=False)
        source_docs = jsonl(QWEN / 'results/documents.jsonl')
        longest = max(source_docs, key=lambda d: d['original_tokens'])
        skills = jsonl(cache / 'data/skills.jsonl')
        text = full_text(skills[longest['source_index']])
        tokenized = encoder.audit_tokenizer.encode(text, add_special_tokens=False)
        probes, selected = [], None
        for target in (512, 1024, 2048, 4096, 6144, 8192):
            encoder.monitor.phase = f'feasibility-context-{target}'
            end = tokenized.offsets[target - 2][0]
            while len(encoder.audit_tokenizer.encode(text[:end], add_special_tokens=False).ids) > target - 2:
                end -= 1
            probe = text[:end]
            encoder.context = target
            encoder.torch.cuda.empty_cache()
            started = time.perf_counter()
            try:
                vector, first = encoder.encode(probe, 'feasibility', use_cache=False)
                again, second = encoder.encode(probe, 'feasibility-repeat', use_cache=False)
                difference = float(np.max(np.abs(vector - again)))
                assert difference < 1e-5
                row = {'target_context': target, 'status': 'passed', 'input_tokens': first['input_tokens'],
                       'repeat_max_abs_difference': difference, 'vector_dimensions': len(vector),
                       'vector_dtype': str(vector.dtype), 'norm': float(np.linalg.norm(vector)),
                       'first_encoding_s': first['encoding_s'], 'warm_encoding_s': second['encoding_s'],
                       'cuda_peak_allocated_bytes': max(first['cuda_peak_allocated_bytes'], second['cuda_peak_allocated_bytes']),
                       'cuda_peak_reserved_bytes': max(first['cuda_peak_reserved_bytes'], second['cuda_peak_reserved_bytes']),
                       'resource_sample': monitor.sample()}
                selected = target
            except encoder.torch.cuda.OutOfMemoryError:
                encoder.torch.cuda.empty_cache()
                row = {'target_context': target, 'status': 'CUDA_OOM', 'cpu_fallback': False,
                       'message': 'Stopped increasing context after first OOM; previous completed context remains eligible'}
                probes.append(row)
                break
            probes.append(row)
            atomic_json(STUDY / 'results/feasibility_progress.json', {'probes': probes, 'selected_context': selected})
            print('CONTEXT', target, row['status'], row.get('input_tokens'), row.get('cuda_peak_reserved_bytes'), flush=True)
        assert selected is not None and selected >= 4096
        atomic_json(STUDY / 'results/feasibility.json', {'status': 'passed', 'selected_context': selected,
            'probes': probes, 'batch_size': 1, 'model_loading_s': encoder.load_s,
            'all_parameters_cuda_fp16': True, 'attention': encoder.identity['attention'],
            'source_probe_policy': 'Complete submitted prefixes of the longest source skill; feasibility only, no retrieval quality',
            'precision': encoder.identity['precision'], 'model_identity': encoder.identity,
            'resource_guards': {'available_ram_min_mib': 1024, 'vram_max_mib': 7168, 'temperature_max_c': 80, 'swap_traffic_max_mib': 64}})
    except Exception as error:
        atomic_json(STUDY / 'results/feasibility_failure.json', {'exception': type(error).__name__, 'message': public(str(error))})
        raise
    finally:
        monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/mia-bgem3')
    main(parser.parse_args().cache)
