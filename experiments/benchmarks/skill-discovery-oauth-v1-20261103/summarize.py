"""Summarize private details.csv; preserves real zeros and explicit context blocks."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path


def yes(value):
    return str(value).lower() == 'true'


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], int(row['library_size']))].append(row)
    result = []
    for (model, n), group in sorted(groups.items()):
        total = len(group)
        valid = [r for r in group if not r['error'] and not yes(r['context_limit'])]
        attempted = [r for r in group if not yes(r['context_limit'])]
        tp = sum(yes(r['exact_match']) and not r['error'] for r in group)
        result.append({'model': model, 'library_size': n, 'queries': total,
            'exact_set_accuracy_all': tp/total,
            'selection_precision_valid': tp/len(valid) if valid else None,
            'selection_recall_all': tp/total,
            'context_overflow_rate': sum(yes(r['context_limit']) for r in group)/total,
            'noncontext_accuracy': sum(yes(r['exact_match']) and not r['error'] for r in attempted)/len(attempted) if attempted else None,
            'error_rate': sum(bool(r['error']) for r in group)/total,
            'noncontext_error_rate': sum(bool(r['error']) and not yes(r['context_limit']) for r in group)/total,
            'mean_recall_at_10_all': sum(float(r['recall@10']) for r in group)/total,
            'mean_precision_at_10_all': sum(float(r['precision@10']) for r in group)/total,
            'real_zero_noncontext': bool(valid) and not any(yes(r['exact_match']) for r in valid)})
    return result


def plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = sorted({r['model'] for r in summary})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for model in models:
        rs = [r for r in summary if r['model'] == model]
        # NaN gaps for ALL blocked sizes, keeping genuine non-context zero at zero.
        xs = [r['library_size'] for r in rs]
        ys = [float('nan') if r['selection_precision_valid'] is None else r['selection_precision_valid'] for r in rs]
        line, = axes[0].plot(xs, ys, marker='o', label=model)
        for r in rs:
            if r['context_overflow_rate']:
                axes[0].scatter([r['library_size']], [1.04], marker='x', color=line.get_color())
                axes[0].axvline(r['library_size'], linestyle=':', alpha=.15, color=line.get_color())
        axes[1].plot(xs, [r['context_overflow_rate'] for r in rs], marker='x', label=model)
        axes[2].plot(xs, [r['noncontext_error_rate'] for r in rs], marker='s', label=model)
    axes[0].set_title('Top-1 accuracy, valid non-error queries')
    axes[0].set_ylabel('Accuracy; X above plot = context block present')
    axes[1].set_title('Context overflow rate, all queries')
    axes[2].set_title('Other error rate, all queries')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_ylim(-.03, 1.09)
        ax.set_xlabel('Library size')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--png')
    args = p.parse_args()
    rows = []
    seen = set()
    for path in args.csv:
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                key = (row['model'], row['library_size'], row['query'])
                if key in seen:
                    raise SystemExit('Duplicate trials across CSV files; do not aggregate repeated/resumed exports.')
                seen.add(key)
                rows.append(row)
    result = summarize(rows)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.png:
        plot(result, args.png)
