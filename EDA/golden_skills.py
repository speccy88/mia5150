#!/usr/bin/env python3
"""Validate golden labels and refresh their distribution in the saved EDA report."""

import argparse
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DATA = REPO / "skillret-local-metadata-20261005/data"
START = "<!-- GOLDEN SKILLS START -->"
END = "<!-- GOLDEN SKILLS END -->"
PLOTLY_URL = "https://cdn.plot.ly/plotly-4.1.1.min.js"
LAYOUT_CSS = """/* Golden skill report layout */
  [id] { scroll-margin-top: 9rem; }
  @media (max-width: 768px) {
    .nav { position: static; }
    [id] { scroll-margin-top: 1rem; }
  }
"""


def analyze_rows(queries, qrels):
    """Count distinct positive skill IDs for every query, including zero labels."""
    by_id = {}
    for query in queries:
        query_id = query["id"]
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("Query IDs must be nonempty strings")
        if query_id in by_id:
            raise ValueError("Duplicate query ID")
        by_id[query_id] = query
    if not by_id:
        raise ValueError("Query split is empty")

    golden = defaultdict(set)
    positive_rows = 0
    for qrel in qrels:
        query_id, skill_id = qrel["query_id"], qrel["skill_id"]
        if query_id not in by_id:
            raise ValueError("Qrel references an unknown query")
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError("Skill IDs must be nonempty strings")
        relevance = qrel["relevance"]
        if (isinstance(relevance, bool) or not isinstance(relevance, (int, float))
                or not math.isfinite(relevance)):
            raise ValueError("Relevance must be a finite number")
        if relevance > 0:
            golden[query_id].add(skill_id)
            positive_rows += 1

    for query_id, query in by_id.items():
        labels = query["skill_ids"]
        if not isinstance(labels, list) or any(not isinstance(s, str) for s in labels):
            raise ValueError("skill_ids must be a list of strings")
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate skill ID in query labels")
        if set(labels) != golden[query_id]:
            raise ValueError("skill_ids disagrees with positive qrels")
        if query["k"] != len(golden[query_id]):
            raise ValueError("k disagrees with positive qrels")

    distribution = Counter(len(golden[query_id]) for query_id in by_id)
    pairs = sum(k * count for k, count in distribution.items())
    return {
        "queries": len(by_id),
        "qrel_rows": len(qrels),
        "positive_qrel_rows": positive_rows,
        "distinct_golden_pairs": pairs,
        "duplicate_positive_qrel_rows": positive_rows - pairs,
        "distribution": dict(sorted(distribution.items())),
        "mean_golden_skills": pairs / len(by_id),
        "max_golden_skills": max(distribution),
        "validation": {"unique_query_ids": True, "no_orphan_qrels": True,
                       "skill_ids_match_qrels": True, "k_matches_qrels": True},
    }


def analyze_dataset(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir.parent / "evidence/data-manifest.json").read_text())
    result = {
        "dataset": "ThakiCloud/SKILLRET", "revision": manifest["revision"],
        "definition": "Distinct skill_id values with relevance > 0 per query_id; all queries included",
        "source_files": {}, "splits": {},
    }
    for split in ("train", "test"):
        records = {}
        for kind in ("queries", "qrels"):
            path = data_dir / kind / f"{split}.jsonl"
            raw = path.read_bytes()
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            records[kind] = rows
            result["source_files"][f"data/{kind}/{split}.jsonl"] = {
                "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows),
            }
            if len(rows) != manifest["files"][f"data/{kind}/{split}.jsonl"]["rows"]:
                raise ValueError("Source row count disagrees with the pinned dataset manifest")
        result["splits"][split] = analyze_rows(records["queries"], records["qrels"])
    totals = Counter()
    for split in result["splits"].values():
        totals.update(split["distribution"])
    n = sum(totals.values())
    result["combined"] = {
        "queries": n, "distribution": dict(sorted(totals.items())),
        "distinct_golden_pairs": sum(k * count for k, count in totals.items()),
    }
    return result


def summary_text(stats):
    test, train = (stats["splits"][split] for split in ("test", "train"))
    one = test["distribution"].get(1, 0)
    multi = sum(count for k, count in test["distribution"].items() if k > 1)
    train_multi = sum(count for k, count in train["distribution"].items() if k > 1)
    return (
        f'Test: {one:,} of {test["queries"]:,} queries ({one / test["queries"] * 100:.2f}%) '
        f'have exactly one golden skill; {multi:,} ({multi / test["queries"] * 100:.2f}%) '
        f'have multiple. Train: {train_multi:,} of {train["queries"]:,} '
        f'({train_multi / train["queries"] * 100:.2f}%) have multiple golden skills.'
    )


def distribution_svg(split, stats, bins):
    """Use the same 0–100% axis for both splits, with exact counts on bars."""
    color = "#e67e22" if split == "train" else "#3498db"
    parts = [f'<svg viewBox="0 0 480 320" role="img" aria-label="{split.title()} golden skill distribution" '
             f'style="width:100%;height:auto;font-family:system-ui,sans-serif">',
             f'<title>{split.title()} queries: share by number of golden skills</title>',
             '<text x="240" y="22" text-anchor="middle" font-size="18" font-weight="600">'
             f'{split.title()} ({stats["queries"]:,} queries)</text>']
    for percent in range(0, 101, 20):
        y = 250 - 1.9 * percent
        parts += [f'<line x1="52" y1="{y}" x2="464" y2="{y}" stroke="#dce2e6"/>',
                  f'<text x="44" y="{y + 5}" text-anchor="end" font-size="14">{percent}%</text>']
    step = 412 / len(bins)
    for index, k in enumerate(bins):
        count = stats["distribution"].get(k, 0)
        percent = count / stats["queries"] * 100
        height = percent * 1.9
        center = 52 + (index + 0.5) * step
        parts += [f'<rect x="{center - step * .3:.2f}" y="{250 - height:.2f}" '
                  f'width="{step * .6:.2f}" height="{height:.2f}" fill="{color}">'
                  f'<title>{k} golden skills: {count:,} queries ({percent:.2f}%)</title></rect>',
                  f'<text x="{center:.2f}" y="{240 - height:.2f}" text-anchor="middle" '
                  f'font-size="14">{count:,}</text>',
                  f'<text x="{center:.2f}" y="274" text-anchor="middle" font-size="15">{k}</text>']
    parts += ['<text x="258" y="307" text-anchor="middle" font-size="15">Golden skills per query</text>', '</svg>']
    return "\n".join(parts)


def render_section(stats):
    splits = stats["splits"]
    combined = stats["combined"]
    bins = list(range(max(combined["distribution"]) + 1))
    rows = []
    for k in bins:
        cells = []
        for split in (splits["train"], splits["test"], combined):
            count = split["distribution"].get(k, 0)
            cells.append(f'<td>{count:,} ({count / split["queries"] * 100:.2f}%)</td>')
        rows.append(f'<tr><th scope="row">{k}</th>{"".join(cells)}</tr>')
    totals = "".join(f'<td>{split["queries"]:,} (100.00%)</td>'
                     for split in (splits["train"], splits["test"], combined))
    source = "https://github.com/speccy88/mia5150/blob/main/skillret-local-metadata-20261005"
    duplicates = sum(s["duplicate_positive_qrel_rows"] for s in splits.values())
    return f'''{START}
<section class="section" id="golden-skills">
  <h2>Golden Skills per Query</h2>
  <p>A <strong>golden skill</strong> is a distinct <code>skill_id</code> with a positive relevance label
     (<code>relevance &gt; 0</code>) for that query. We count all queries in each split, including queries
     with zero golden skills. Duplicate positive qrel rows count only once.</p>
  <div class="highlight">{summary_text(stats)}</div>
  <div style="overflow-x:auto;">
    <table class="stats" id="golden-skills-table" style="min-width:620px;">
      <caption>Query counts and percentages within each split (all {combined["queries"]:,} queries)</caption>
      <thead><tr><th scope="col">Golden skills</th><th scope="col">Train queries</th>
        <th scope="col">Test queries</th><th scope="col">All queries</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
      <tfoot><tr><th scope="row">Total</th>{totals}</tr></tfoot>
    </table>
  </div>
  <div class="two-col">
    <div>{distribution_svg("train", splits["train"], bins)}</div>
    <div>{distribution_svg("test", splits["test"], bins)}</div>
  </div>
  <p>Mean golden skills per query: <strong>{splits["train"]["mean_golden_skills"]:.3f}</strong> in train;
     <strong>{splits["test"]["mean_golden_skills"]:.3f}</strong> in test.
     The maximum is {splits["train"]["max_golden_skills"]} in train and {splits["test"]["max_golden_skills"]} in test.</p>
  <p><strong>Validation:</strong> query IDs are unique; no qrels refer to unknown queries;
     every query's <code>skill_ids</code> and <code>k</code> agree with the positive-qrel set.
     Duplicate positive qrel rows: {duplicates:,}. The histogram totals match every query,
     and its weighted totals match {splits["train"]["distinct_golden_pairs"]:,} train and
     {splits["test"]["distinct_golden_pairs"]:,} test distinct golden pairs.</p>
  <p><strong>Sources:</strong> <a href="{source}/data/queries">query splits</a>,
     <a href="{source}/data/qrels">qrel splits</a>, and the
     <a href="{source}/evidence/data-manifest.json">dataset manifest</a> for
     <a href="https://huggingface.co/datasets/ThakiCloud/SKILLRET/tree/{html.escape(stats["revision"])}">revision
     <code>{html.escape(stats["revision"])}</code></a>.
     <a href="https://github.com/speccy88/mia5150/blob/main/EDA/golden_skills_summary.json">Download verified counts and input hashes</a>.</p>
  <p><strong>Scope:</strong> these are the dataset's labeled targets, not an exhaustive inventory of
     every skill that could satisfy a real request. Public query text was privacy-redacted;
     the query IDs, golden skill IDs, and relevance labels were preserved.</p>
</section>
{END}'''


def update_report(report, stats):
    section = render_section(stats)
    if START in report:
        report, count = re.subn(re.escape(START) + r".*?" + re.escape(END),
                               lambda _: section, report, flags=re.S)
        if count != 1:
            raise ValueError("Expected one golden-skills section")
    else:
        anchor = "<!-- Section 6: Qrels -->"
        if report.count(anchor) != 1:
            raise ValueError("Cannot locate the qrels section")
        report = report.replace(anchor, section + "\n\n" + anchor)
    if '<a href="#golden-skills">' not in report:
        report = report.replace('<div class="nav">', '<div class="nav"><a href="#golden-skills">Golden Skills per Query</a>', 1)
    report = report.replace("https://cdn.plot.ly/plotly-latest.min.js", PLOTLY_URL)
    if "/* Golden skill report layout */" not in report:
        report = report.replace("</style>", LAYOUT_CSS + "</style>", 1)
    report = re.sub(r'<strong>6\. (?:Mostly single-skill queries|Golden skill counts differ by split):</strong>.*?(?=\s*</div>)',
                    lambda _: '<strong>6. Golden skill counts differ by split:</strong> ' + summary_text(stats),
                    report, flags=re.S)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=HERE / "eda_report.html")
    parser.add_argument("--summary", type=Path, default=HERE / "golden_skills_summary.json")
    parser.add_argument("--check", action="store_true", help="Verify saved outputs without changing them")
    args = parser.parse_args()
    stats = analyze_dataset(args.data_dir)
    summary = json.dumps(stats, indent=2, sort_keys=True) + "\n"
    original = args.report.read_text(encoding="utf-8")
    updated = update_report(original, stats)
    if args.check:
        if original != updated or args.summary.read_text(encoding="utf-8") != summary:
            raise SystemExit("Golden skill report or summary is stale")
    else:
        args.report.write_text(updated, encoding="utf-8")
        args.summary.write_text(summary, encoding="utf-8")
    print(summary_text(stats))
    print("Golden skill labels, histogram totals, and saved outputs verified.")


if __name__ == "__main__":
    main()
