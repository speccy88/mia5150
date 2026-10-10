"""Build factual final SearchJev evidence/report from saved local artifacts only.

Standard library only. No subprocesses, model imports, external calls, or writes
to active scripts. Parent must run after bounded checkpoint and PC release.
Usage: python final_summary.py [--root WORKSPACE] [--release PATH]
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        return {'artifact_read_error': repr(exc), 'path': str(path)}


def read_lines(path):
    if not path.is_file():
        return
    with path.open(encoding='utf-8-sig', errors='replace') as stream:
        for number, line in enumerate(stream, 1):
            try:
                item = json.loads(line)
            except ValueError:
                yield {'_parse_error_line': number}
                continue
            yield item


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def health(path, host_only=False):
    values = {k: [] for k in ['host_available_bytes', 'guest_available_bytes', 'gpu_used_mib', 'gpu_temp_c', 'gpu_utilization', 'swap_used_bytes', 'host_cpu_percent', 'host_age_seconds']}
    count = errors = swap_changes = high = maxhigh = 0
    last_host_stamp = None
    first_swap = last_swap = None
    first_utc = last_utc = None
    for row in read_lines(path):
        if '_parse_error_line' in row:
            errors += 1
            continue
        count += 1
        utc = row.get('utc')
        if number(utc):
            first_utc = utc if first_utc is None else min(first_utc, utc)
            last_utc = utc if last_utc is None else max(last_utc, utc)
        host = row if host_only else row.get('host', {})
        samples = {'host_available_bytes': host.get('available_bytes'), 'host_cpu_percent': host.get('cpu_percent')}
        if not host_only:
            samples.update({k: row.get(k) for k in ['guest_available_bytes', 'gpu_used_mib', 'gpu_temp_c', 'gpu_utilization', 'swap_used_bytes']})
            if number(utc) and number(host.get('utc')):
                samples['host_age_seconds'] = utc - host['utc']
            swap = row.get('swap')
            if isinstance(swap, dict):
                if last_swap is not None and swap != last_swap:
                    swap_changes += 1
                first_swap = swap if first_swap is None else first_swap
                last_swap = swap
        for key, value in samples.items():
            if number(value):
                values[key].append(value)
        if host.get('utc') != last_host_stamp and number(host.get('cpu_percent')):
            high = high + 1 if host['cpu_percent'] > 80 else 0
            maxhigh = max(maxhigh, high)
            last_host_stamp = host.get('utc')
    output = {
        'path': str(path), 'samples': count, 'malformed_lines': errors,
        'first_utc': first_utc, 'last_utc': last_utc,
        'observed_seconds': last_utc - first_utc if first_utc is not None and last_utc is not None else None,
        'max_consecutive_unique_host_samples_above_80_percent': maxhigh,
        'guest_swap_counter_change_events': swap_changes if not host_only else None,
        'guest_swap_first': first_swap, 'guest_swap_last': last_swap,
    }
    for key, sample in values.items():
        output[key] = {'min': min(sample), 'max': max(sample), 'samples': len(sample)} if sample else None
    output['guest_swap_counter_deltas'] = {k: last_swap.get(k, 0) - first_swap.get(k, 0) for k in set(first_swap) | set(last_swap)} if first_swap and last_swap else None
    return output


def tail(path, byte_limit=16000):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - byte_limit))
        return stream.read().decode('utf-8', errors='replace')


def run_summary(run):
    updates = load(run / 'updates.json', [])
    valid_list = isinstance(updates, list)
    if not valid_list:
        updates = []
    ids = [u.get('update') for u in updates if isinstance(u, dict)]
    consecutive = ids == list(range(1, len(ids) + 1)) and len(ids) == len(updates)
    all_finite = bool(updates) and all(u.get('gradients_finite') is True and number(u.get('gradient_norm_before_clip')) and u['gradient_norm_before_clip'] > 0 for u in updates)
    proof = load(run / 'two-update-proof.json')
    result = load(run / 'result.json')
    baseline = load(run / 'baseline.json')
    adapted = load(run / 'adapted-update20.json')
    guard = load(run / 'guard-stop.json')
    exit_state = load(run / 'exit.json')
    command = load(run / 'command.json', [])
    log = tail(run / 'worker.log')
    runtime_probe = any('runtime_check.py' in str(v) for v in command) if isinstance(command, list) else False
    metadata_failure = runtime_probe and 'AttributeError:' in log and ('has no attribute' in log) and ('hub_kernels' in log or 'modeling_qwen3_5' in log)
    preload_assertion_failure = runtime_probe and 'AssertionError' in log
    independent_run_equivalence_failure = 'AssertionError: two-update checkpoint differs from independently reloaded smoke checkpoint' in log
    explicit_checkpoint_reload_failure = any(marker in log for marker in ('AssertionError: save reload adapter mismatch', 'AssertionError: reload logits differ', 'AssertionError: reloaded base mismatch'))
    missing_or_nonfinite_gradient_failure = 'AssertionError: missing/nonfinite adapter gradient' in log
    bounded_numerical_failure = any(marker in log for marker in ('RuntimeError: Nonfinite gradients persisted through bounded GradScaler retries', 'RuntimeError: GradScaler fell below the bounded minimum scale'))
    overflow_events = load(run / 'overflow-events.json', [])
    if not isinstance(overflow_events, list):
        overflow_events = []
    claimed_updates = result.get('completed_optimizer_updates') if isinstance(result, dict) else None
    checkpoint_name = f'checkpoint-update{claimed_updates}' if isinstance(claimed_updates, int) else None
    checkpoint = run / checkpoint_name if checkpoint_name else None
    checkpoint_exists = bool(checkpoint and (checkpoint / 'adapter_model.safetensors').is_file() and (checkpoint / 'adapter_config.json').is_file())
    final_supported = bool(
        isinstance(result, dict) and isinstance(claimed_updates, int) and claimed_updates > 0
        and consecutive and all_finite and len(updates) == claimed_updates
        and isinstance(proof, dict) and proof.get('passed') is True
        and proof.get('frozen_base_identical') is True
        and proof.get('save_reload_adapter_identical') is True
        and number(proof.get('save_reload_max_abs_logit_diff'))
        and proof['save_reload_max_abs_logit_diff'] <= 1e-4
        and result.get('frozen_base_unchanged') is True
        and checkpoint_exists and isinstance(baseline, dict) and 'metrics' in baseline
        and isinstance(adapted, dict) and 'metrics' in adapted
    )
    if final_supported:
        status = 'completed_pilot_supported_by_optimizer_updates_and_checkpoint'
    elif independent_run_equivalence_failure:
        status = 'independent_training_bitwise_reproducibility_assertion_failed_after_finite_updates'
    elif explicit_checkpoint_reload_failure:
        status = 'explicit_checkpoint_reload_verification_failed'
    elif missing_or_nonfinite_gradient_failure:
        status = 'missing_or_nonfinite_gradient_assertion_stopped_before_next_optimizer_update'
    elif bounded_numerical_failure:
        status = 'bounded_grad_scaler_numerical_retries_exhausted'
    elif updates:
        status = 'partial_or_unfinalized_training_with_saved_optimizer_update_records'
    elif guard:
        status = 'guard_stopped_without_saved_optimizer_updates'
    elif metadata_failure:
        status = 'runtime_probe_metadata_attribute_failure_not_model_failure'
    elif preload_assertion_failure:
        status = 'runtime_probe_preload_assertion_failure_not_model_failure'
    elif isinstance(exit_state, dict) and exit_state.get('returncode') not in (None, 0):
        status = 'worker_failed_without_saved_optimizer_updates'
    elif isinstance(baseline, dict) and 'metrics' in baseline:
        status = 'baseline_only_no_saved_optimizer_updates'
    elif isinstance(exit_state, dict) and exit_state.get('returncode') == 0:
        status = 'auxiliary_run_exited_successfully_no_training_claim'
    else:
        status = 'no_final_outcome_recorded'
    update_seconds = [u['seconds'] for u in updates if number(u.get('seconds'))]
    evidence = {
        'name': run.name, 'path': str(run), 'status': status,
        'command': command, 'exit': exit_state, 'guard_stop': guard,
        'recorded_optimizer_updates': len(updates), 'update_ids': ids,
        'updates_consecutive_from_one': consecutive, 'logged_gradients_all_finite_and_nonzero': all_finite,
        'final_result_claim_supported': final_supported, 'result': result,
        'two_update_proof': proof, 'runtime': load(run / 'runtime.json'),
        'source_hashes_at_run_start': load(run / 'source-hashes.json'),
        'health': health(run / 'telemetry.jsonl'),
        'baseline_metrics': baseline.get('metrics') if isinstance(baseline, dict) else None,
        'adapted_metrics': adapted.get('metrics') if isinstance(adapted, dict) else None,
        'training_update_seconds': sum(update_seconds) if update_seconds else None,
        'mean_update_seconds': sum(update_seconds) / len(update_seconds) if update_seconds else None,
        'linear_200_update_compute_seconds_estimate': 200 * sum(update_seconds) / len(update_seconds) if update_seconds else None,
        'runtime_probe_metadata_failure': metadata_failure,
        'runtime_probe_preload_assertion_failure': preload_assertion_failure,
        'independent_training_bitwise_equivalence_failure': independent_run_equivalence_failure,
        'explicit_checkpoint_reload_failure': explicit_checkpoint_reload_failure,
        'missing_or_nonfinite_gradient_failure': missing_or_nonfinite_gradient_failure,
        'bounded_numerical_failure': bounded_numerical_failure,
        'overflow_events': overflow_events,
        'overflow_summary': {
            'events': len(overflow_events),
            'all_unsafe_optimizer_steps_skipped': all(e.get('optimizer_update_skipped') is True for e in overflow_events) if overflow_events else None,
            'all_rejected_steps_verified_adapter_unchanged': all(e.get('adapter_unchanged') is True for e in overflow_events) if overflow_events else None,
            'all_recorded_scale_changes_halved': all(number(e.get('old_scale')) and number(e.get('new_scale')) and e['new_scale'] == .5 * e['old_scale'] for e in overflow_events) if overflow_events else None,
            'updates_retried': sorted({e['update'] for e in overflow_events if isinstance(e.get('update'), int)}),
            'minimum_recorded_retry_scale': min((e['new_scale'] for e in overflow_events if number(e.get('new_scale'))), default=None),
            'not_counted_as_completed_optimizer_updates': True,
        },
        'last_safe_checkpoint_receipt': load(run / 'last-safe.json'),
        'failure_log_tail': log if (guard or metadata_failure or (isinstance(exit_state, dict) and exit_state.get('returncode') not in (None, 0))) else None,
        'checkpoint_files': [],
        'latest_event': None,
    }
    events = list(read_lines(run / 'events.jsonl'))
    if events:
        evidence['latest_event'] = events[-1]
    # Only small adapter checkpoints are hashed; the 1.7GB frozen model is not
    # reread. Its original guarded-download receipt remains the pin evidence.
    for directory in sorted(run.glob('checkpoint-*')):
        if directory.is_dir():
            for item in sorted(directory.iterdir()):
                if item.is_file() and item.name in ('adapter_config.json', 'adapter_model.safetensors'):
                    evidence['checkpoint_files'].append({'path': str(item), 'bytes': item.stat().st_size, 'sha256': digest(item)})
    for key in ('updates.json', 'result.json', 'two-update-proof.json', 'configuration.json', 'runtime.json', 'baseline.json', 'adapted-update20.json', 'overflow-events.json', 'last-safe.json'):
        item = run / key
        if item.is_file():
            evidence.setdefault('artifact_hashes', {})[key] = digest(item)
    before = load(run / 'base-before.json')
    after2 = load(run / 'base-after2.json')
    evidence['saved_base_hashes_equal_at_update2'] = before == after2 if before and after2 else None
    if isinstance(baseline, dict) and isinstance(adapted, dict):
        a, b = baseline.get('predictions', []), adapted.get('predictions', [])
        same = bool(a and b) and [(r.get('query_id'), r.get('skill_id'), r.get('y')) for r in a] == [(r.get('query_id'), r.get('skill_id'), r.get('y')) for r in b]
        evidence['matched_before_after_pair_order_and_labels'] = same
        evidence['recomputed_metric_delta'] = {key: adapted['metrics'][key] - baseline['metrics'][key] for key in ('loss', 'ce', 'brier', 'accuracy', 'auc') if number(baseline.get('metrics', {}).get(key)) and number(adapted.get('metrics', {}).get(key))}
        if same:
            evidence['mean_absolute_relevant_probability_change'] = sum(abs(x['prob_relevant'] - y['prob_relevant']) for x, y in zip(a, b)) / len(a)
        if not same:
            evidence['final_result_claim_supported'] = False
            evidence['status'] = 'result_present_but_before_after_pair_alignment_not_verified'
    evidence['configuration'] = load(run / 'configuration.json')
    return evidence


def fmt(value, digits=4):
    return f'{value:.{digits}f}' if number(value) else 'unavailable'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    runs = [run_summary(p) for p in sorted((root / 'runs').iterdir()) if p.is_dir()] if (root / 'runs').is_dir() else []
    candidates = [r for r in runs if r['recorded_optimizer_updates'] or any(Path(str(v)).name == 'train_pilot.py' or Path(str(v)).name.startswith('pilot20') for v in (r.get('command') or []))]
    completed = [r for r in candidates if r['final_result_claim_supported']]
    select = max(completed, key=lambda r: (r['recorded_optimizer_updates'], r['health'].get('last_utc') or 0)) if completed else (max(candidates, key=lambda r: r['health'].get('last_utc') or 0) if candidates else None)
    prep = load(root / 'data-audit/preparation-audit.json')
    data = load(root / 'data-audit/skillret-audit.json')
    official = load(root / 'official-review/audit-summary.json', {})
    manifest = load(root / 'models/manifest.json', [])
    release_path = args.release
    if release_path is None:
        release_path = next((root / name for name in ('pc-release.json', 'release.json', 'PC-release.json') if (root / name).is_file()), None)
    release = load(release_path) if release_path else None
    release_confirmed = isinstance(release, dict) and release.get('pc_released') is True
    evidence = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'generator_sha256': digest(Path(__file__)),
        'root': str(root), 'selected_training_run': select['name'] if select else None,
        'selected_outcome': select['status'] if select else 'no_training_run_found',
        'runs': runs, 'host_monitor_health': health(root / 'host.jsonl', host_only=True),
        'pins': official.get('pins') if isinstance(official, dict) else None,
        'official_adapter_configuration': official.get('adapter') if isinstance(official, dict) else None,
        'recorded_model_download_manifest': manifest,
        'model_hash_verification_scope': 'Carries guarded-download recorded SHA256/LFS identities; final generator does not reread frozen base weights. Saved adapter checkpoints are independently hashed by this generator.',
        'data_preparation': prep, 'data_source_audit': data,
        'pc_release': {'path': str(release_path) if release_path else None, 'explicit_pc_released_true': release_confirmed, 'evidence': release},
        'guard_observation_limitations': [
            'Guard performs assertions before appending a failing snapshot. Inspect guard-stop.json sample/previous fields when available; the first pilot guard-stop predates those extra fields, so its violating snapshot is absent from telemetry.',
            'CPU streaks count unique host timestamps, not duplicate polling snapshots.',
            'Guest pswpin/pswpout changes measure active guest swap; host paging counters were not present in host monitor schema.',
            'Recorded process exit alone does not certify full PC release. Generator requires explicit pc_released:true for that claim.',
        ],
    }
    observed_health = [r['health'] for r in runs]
    aggregate_health = {
        'guarded_runs': len(runs),
        'logged_samples': sum(h['samples'] for h in observed_health),
        'max_consecutive_unique_host_samples_above_80_percent_within_any_run': max((h['max_consecutive_unique_host_samples_above_80_percent'] for h in observed_health), default=0),
        'guard_stops': [{'run': r['name'], 'stop': r['guard_stop']} for r in runs if r['guard_stop']],
        'guest_swap_counter_change_events': sum(h['guest_swap_counter_change_events'] for h in observed_health),
    }
    for key in ('host_available_bytes', 'guest_available_bytes', 'gpu_used_mib', 'gpu_temp_c', 'gpu_utilization', 'swap_used_bytes', 'host_cpu_percent', 'host_age_seconds'):
        items = [h[key] for h in observed_health if h.get(key)]
        aggregate_health[key] = {'min': min(i['min'] for i in items), 'max': max(i['max'] for i in items)} if items else None
    evidence['all_guarded_runs_health'] = aggregate_health
    chronological_health = sorted((h for h in observed_health if h.get('first_utc') is not None and h.get('guest_swap_first')), key=lambda h: h['first_utc'])
    if chronological_health:
        evidence['guest_swap_across_run_span'] = {
            'first_observed_counters': chronological_health[0]['guest_swap_first'],
            'last_observed_counters': chronological_health[-1]['guest_swap_last'],
            'interpretation': 'Cumulative counter differences span gaps between runs as well as runs. Per-run zero deltas do not erase an earlier active-swap guard stop.',
        }
    trained_runs = [r for r in candidates if r['recorded_optimizer_updates']]
    evidence['independent_training_runs'] = []
    for r in trained_runs:
        before = load(Path(r['path']) / 'adapter-before.json')
        same_initial = [other['name'] for other in trained_runs if other['name'] != r['name'] and before and before == load(Path(other['path']) / 'adapter-before.json')]
        evidence['independent_training_runs'].append({'run': r['name'], 'optimizer_updates': r['recorded_optimizer_updates'], 'status': r['status'], 'same_initial_adapter_hashes_as_runs': same_initial, 'guard_stop': r['guard_stop'], 'two_update_proof': r['two_update_proof'], 'independent_training_bitwise_equivalence_failure': r['independent_training_bitwise_equivalence_failure'], 'explicit_checkpoint_reload_failure': r['explicit_checkpoint_reload_failure'], 'missing_or_nonfinite_gradient_failure': r['missing_or_nonfinite_gradient_failure'], 'bounded_numerical_failure': r['bounded_numerical_failure'], 'overflow_summary': r['overflow_summary'], 'last_safe_checkpoint_receipt': r['last_safe_checkpoint_receipt']})
    evidence['independent_checkpoint_validation'] = next((r for r in runs if r['name'] == 'checkpoint-validation'), None)
    pair_path = root / 'data-audit/pairs.json'
    if pair_path.is_file():
        pairs = load(pair_path, {})
        evidence['pairs_sha256_now'] = digest(pair_path)
        evidence['pairs_hash_matches_preparation_receipt'] = isinstance(prep, dict) and prep.get('pairs_sha256') == evidence['pairs_sha256_now']
        evidence['observed_pair_counts'] = {key: {'pairs': len(pairs.get(key, [])), 'positive': sum(r.get('y') == 1 for r in pairs.get(key, [])), 'negative': sum(r.get('y') == 0 for r in pairs.get(key, [])), 'queries': len({r.get('query_id') for r in pairs.get(key, [])})} for key in ('train', 'eval')}

    lines = ['SearchJev-0.8B / SkillRet local adaptation report', f"Generated: {evidence['generated_utc']}", '']
    if select and select['final_result_claim_supported']:
        lines.append(f"Outcome: completed a verified {select['recorded_optimizer_updates']}-optimizer-update pilot in runs/{select['name']}.")
        lines.append('This is continuation of the released rank16 SearchJev adapter, subject to saved loaded-adapter verification and configuration evidence; it is not full model training or a200-update run.')
    elif select and select['recorded_optimizer_updates']:
        lines.append(f"Outcome: {select['recorded_optimizer_updates']} optimizer update records exist in runs/{select['name']}, but a complete pilot result is not supported. Status: {select['status']}.")
    else:
        lines.append('Outcome: no completed optimizer updates are supported by an updates.json artifact. Do not describe inference, loading, or baseline evaluation as completed training.')
    if select:
        lines.append(f"Selected run status: {select['status']}; recorded updates: {select['recorded_optimizer_updates']}.")
        if select['guard_stop']:
            lines.append(f"Guard stop: {select['guard_stop']}.")
        if select['exit']:
            lines.append(f"Worker exit receipt: {select['exit']}.")
        if select['latest_event']:
            lines.append('Latest recorded event: ' + json.dumps(select['latest_event'], ensure_ascii=False))
        if select['failure_log_tail']:
            errors = [line for line in select['failure_log_tail'].splitlines() if any(mark in line for mark in ('Error:', 'AssertionError', 'Exception:', 'CUDA out of memory'))]
            if errors:
                lines.append('Worker error evidence: ' + ' | '.join(errors[-3:]))
    lines.append('Independent training runs (do not add these into one continuous run): ' + json.dumps(evidence['independent_training_runs'], ensure_ascii=False))
    if len(trained_runs) > 1:
        lines.append('Each attempt starts from the released adapter when confirmed by matching saved initial-adapter hashes. Separate attempts must not be added into one continuous training trajectory.')
    for r in candidates:
        if r['independent_training_bitwise_equivalence_failure']:
            lines.append(f"runs/{r['name']} stopped after{r['recorded_optimizer_updates']} saved finite updates because its newly trained checkpoint was not bitwise identical to the checkpoint produced by a separate smoke run. This was a cross-run training reproducibility assertion, not a failed save/reload of the same checkpoint. No resource failure is inferred from that assertion. Differences between these FP16 training runs were observed; their precise cause was not isolated.")
        if r['missing_or_nonfinite_gradient_failure']:
            lines.append(f"runs/{r['name']} stopped at the missing/nonfinite-gradient assertion after{r['recorded_optimizer_updates']} completed optimizer updates, before the next optimizer step. This combines missing and nonfinite gradient checks; that combined assertion alone does not identify which parameter/check failed. It is separate from any checkpoint-reload proof or resource guard outcome.")
        if r['bounded_numerical_failure']:
            lines.append(f"runs/{r['name']} exhausted its bounded numerical retry policy after{r['recorded_optimizer_updates']} saved completed updates. Unsafe rejected attempts are not optimizer updates. Last-safe receipt: " + json.dumps(r['last_safe_checkpoint_receipt']))
        if r['overflow_events']:
            lines.append(f"runs/{r['name']} overflow evidence: " + json.dumps(r['overflow_summary']) + '. Raw parameter names, NaN/Inf counts, update/attempt numbers, skip and unchanged-adapter assertions are preserved in overflow-events.json and evidence-summary.json.')
        proof = r.get('two_update_proof')
        if isinstance(proof, dict) and proof.get('optimizer_parameter_identity_preserved') is True:
            lines.append(f"runs/{r['name']} recorded a named-adapter reload check using the same verified frozen base, tensor/logit comparisons, and preserved optimizer parameter identities. Validation details and any separate full-base fresh-process proof are in its two-update-proof.json. Recorded gate passed={proof.get('passed')}.")
    if trained_runs:
        per_attempt = {r['name']: r['recorded_optimizer_updates'] for r in trained_runs}
        lines.append('Observed independent attempt counts: ' + json.dumps(per_attempt) + '. These are separate trajectories. Their arithmetic sum is not a continuous training count; the selected final attempt alone defines its checkpoint trajectory.')
    validation = evidence['independent_checkpoint_validation']
    if validation:
        lines.append('Independent fresh-process checkpoint-validation proof: ' + json.dumps(validation['two_update_proof'], ensure_ascii=False))
        lines.append('Independent checkpoint-validation exit: ' + json.dumps(validation['exit'], ensure_ascii=False))
    lines += ['', 'Quality and update evidence']
    if select:
        lines.append('Two-update proof: ' + json.dumps(select['two_update_proof'], ensure_ascii=False))
        lines.append(f"Finite/nonzero logged adapter gradients: {select['logged_gradients_all_finite_and_nonzero']}; consecutive update IDs: {select['updates_consecutive_from_one']}.")
        lines.append('Rejected overflow attempts (excluded from completed-update count): ' + json.dumps(select['overflow_summary']))
        lines.append('Last-safe checkpoint receipt: ' + json.dumps(select['last_safe_checkpoint_receipt']))
        lines.append(f"Saved frozen-base parameter hashes identical at update2: {select['saved_base_hashes_equal_at_update2']}.")
        if select['baseline_metrics']:
            lines.append('Baseline metrics: ' + json.dumps(select['baseline_metrics'], ensure_ascii=False))
        if select['adapted_metrics']:
            lines.append('Adapted metrics: ' + json.dumps(select['adapted_metrics'], ensure_ascii=False))
            lines.append('Recomputed adapted minus baseline: ' + json.dumps(select.get('recomputed_metric_delta'), ensure_ascii=False))
            lines.append(f"Before/after pair IDs, order, labels identical: {select.get('matched_before_after_pair_order_and_labels')}.")
        else:
            lines.append('No complete matched adapted quality artifact is available.')
        lines.append(f"Saved update compute time: {fmt(select['training_update_seconds'], 2)} seconds; mean update: {fmt(select['mean_update_seconds'], 2)} seconds. These durations include retries inside completed updates when recorded; an unfinished failed update has no completed-update duration.")
        lines.append(f"Linear200-update compute-only estimate: {fmt(select['linear_200_update_compute_seconds_estimate'], 2)} seconds, excluding load/evaluation/reload/guards and without asserting sustained feasibility.")
        result = select.get('result')
        if isinstance(result, dict):
            lines.append(f"Worker-declared final frozen base unchanged: {result.get('frozen_base_unchanged')}; peak CUDA allocated/reserved MiB: {result.get('peak_cuda_allocated_mib')}/{result.get('peak_cuda_reserved_mib')}.")
        config = select.get('configuration') or {}
        fields = ('max_length', 'pad_multiple', 'batch_size', 'gradient_accumulation', 'learning_rate', 'clip_norm', 'optimizer', 'precision', 'gradient_checkpointing', 'use_cache', 'workers', 'cpu_threads', 'loss', 'body_used')
        lines.append('Actual saved configuration: ' + json.dumps({k: config.get(k) for k in fields}, ensure_ascii=False))

    lines += ['', 'Data, coverage, and exposure']
    if isinstance(prep, dict):
        lines.append('Preserved gold coverage: ' + json.dumps(prep.get('coverage'), ensure_ascii=False))
        lines.append(f"Actual train/eval candidate-family overlap: {prep.get('actual_train_eval_family_overlap_count')}; training overlap with excluded exposed positive families: {prep.get('actual_train_exposed_positive_family_overlap_count')}.")
        lines.append('Selection: ' + str(prep.get('selection')))
        lines.append('Excluded training family sources: ' + json.dumps(prep.get('excluded_train_family_sources')))
        lines.append('Exposure: ' + str(prep.get('exposure')))
        lines.append('Evaluation design: ' + str(prep.get('eval_design')))
    lines.append('All4392 public test queries were historically exposed; prior local development and later round2/scaling panels are not blind. Historical256 dev exact IDs at another revision remain unavailable in the examined ledger.')
    lines.append('This balanced binary pair diagnostic cannot establish full-corpus retrieval performance, independent holdout performance, or200-update generalization.')
    lines.append('Pairs file SHA256 matches preparation receipt: ' + str(evidence.get('pairs_hash_matches_preparation_receipt')))

    lines += ['', 'Pins and runtime']
    lines.append(json.dumps(evidence['pins'], ensure_ascii=False))
    for item in manifest if isinstance(manifest, list) else []:
        if '.safetensors' in str(item.get('file')) and not str(item.get('file')).endswith('.json'):
            lines.append(f"Recorded weight pin: {item.get('repo')}@{item.get('revision')} | {item.get('file')} | SHA256 {item.get('sha256')} | bytes {item.get('bytes')}.")
    runtime_runs = [r for r in runs if r['runtime']]
    if runtime_runs:
        preferred = select if select and select.get('runtime') else runtime_runs[-1]
        lines.append('Verified runtime artifact: ' + json.dumps(preferred['runtime'], ensure_ascii=False))
    lines.append('Earlier runtime probe metadata AttributeErrors are listed below separately; they are not evidence of a model forward/backward failure.')

    lines += ['', 'Health evidence across every recorded run']
    lines.append('All guarded runs aggregate: ' + json.dumps(evidence['all_guarded_runs_health'], ensure_ascii=False))
    lines.append('Guest swap across run span: ' + json.dumps(evidence.get('guest_swap_across_run_span'), ensure_ascii=False))
    for r in runs:
        h = r['health']
        def metric(key, bound, scale=1):
            value = h.get(key)
            return fmt(value[bound] / scale, 3) if value else 'unavailable'
        lines.append(
            f"{r['name']}: status={r['status']}; samples={h['samples']}; min host/guest GiB={metric('host_available_bytes','min',1024**3)}/{metric('guest_available_bytes','min',1024**3)}; max VRAM MiB={metric('gpu_used_mib','max')}; max GPU C={metric('gpu_temp_c','max')}; max host CPU%={metric('host_cpu_percent','max')}; max CPU>80 streak={h['max_consecutive_unique_host_samples_above_80_percent']}; max host age seconds={metric('host_age_seconds','max')}; guest swap deltas={h['guest_swap_counter_deltas']}; guard_stop={r['guard_stop']}.")
    mh = evidence['host_monitor_health']
    lines.append('Entire host monitor: ' + json.dumps(mh, ensure_ascii=False))
    lines.extend(evidence['guard_observation_limitations'])
    lines += ['', 'PC release and files']
    lines.append('Explicit PC release confirmed by receipt: ' + str(release_confirmed))
    lines.append('Release evidence: ' + json.dumps(evidence['pc_release'], ensure_ascii=False))
    if select:
        lines.append('Saved adapter checkpoints: ' + json.dumps(select['checkpoint_files'], ensure_ascii=False))
    lines.append('Raw update/evaluation/telemetry logs remain under runs/. Data evidence is under data-audit/. Model and official-review manifests preserve exact revisions and hashes.')
    report_path = root / 'SearchJev-SkillRet-Report.txt'
    summary_path = root / 'evidence-summary.json'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    evidence['english_report'] = {'path': str(report_path), 'sha256': digest(report_path)}
    summary_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(report_path), 'summary': str(summary_path), 'selected_run': evidence['selected_training_run'], 'outcome': evidence['selected_outcome'], 'recorded_updates': select['recorded_optimizer_updates'] if select else 0, 'pc_release_confirmed': release_confirmed}, indent=2))


if __name__ == '__main__':
    main()
