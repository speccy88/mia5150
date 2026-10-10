"""Build the three-page PDF from saved evidence only; no model/network calls.

Local report dependencies: reportlab==5.0.1, pypdf==6.19.0.
Run: .venv/bin/python scripts/make_short_report.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/MIA5150_SkillRet_Short_Report.pdf"
SOURCES = [
    "reports/OPENROUTER_SCALING.json",
    "results/openrouter/scaling/trial.jsonl",
    "results/baseline_summary.jsonl",
    "reports/OPENROUTER_MODELS.json",
]
BLUE = colors.HexColor("#2866A5")
NAVY = colors.HexColor("#18334F")
INK = colors.HexColor("#243649")
MUTED = colors.HexColor("#5C6B7A")
GRAY = colors.HexColor("#D7E0E8")
LIGHT = colors.HexColor("#F1F5F8")
GOLD = colors.HexColor("#E4B75E")
WHITE = colors.white
W, H = A4
M = 42
CW = W - 2 * M


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_evidence():
    data = json.loads((ROOT / SOURCES[0]).read_text())
    raw = read_jsonl(ROOT / SOURCES[1])
    baseline = read_jsonl(ROOT / SOURCES[2])[-1]
    integration = json.loads((ROOT / SOURCES[3]).read_text())
    assert len(raw) == len({r["case_key"] for r in raw}) == data["planned"] == 180
    assert sum(r["status"] == "valid" for r in raw) == data["valid"] == 121
    assert (
        sum(r.get("selection", {}).get("exact_set_match", 0) for r in raw)
        == data["exact"]
        == 24
    )
    assert sum(r.get("source_reused", False) for r in raw) == data["reused_cases"] == 53
    assert all(r["mode"] == "live" for r in raw)
    assert baseline["query_count"] == 200 and baseline["corpus_size"] == 6006
    assert len(data["query_ids"]) == 3
    assert all(m["canary_passed"] for m in integration["models"])
    valid_large = [r for r in raw if r["N"] == 1000 and r["status"] == "valid"]
    overlap = sum(
        bool(set(r["returned_ids"]) & set(r["evaluator_only_gold"]))
        for r in valid_large
    )
    extras = sum(
        bool(set(r["returned_ids"]) - set(r["evaluator_only_gold"]))
        for r in valid_large
    )
    assert (len(valid_large), overlap, extras) == (35, 22, 34)
    return data, baseline, integration


class Report:
    def __init__(self):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        self.c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
        self.c.setTitle("MIA5150 - Skill discovery trial: short results report")
        self.c.setAuthor("MIA5150 project")
        self.c.setSubject("Saved OpenRouter benchmark results, costs and workflow")

    def text(self, text, x, top, size=10, color=INK, bold=False):
        self.c.setFillColor(color)
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.c.drawString(x, H - top - size, text)

    def para(
        self, text, x, top, width=CW, size=10.3, color=INK, leading=None, center=False
    ):
        style = ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=size,
            leading=leading or size * 1.38,
            textColor=color,
            alignment=TA_CENTER if center else 0,
        )
        p = Paragraph(text, style)
        _, height = p.wrap(width, H)
        assert top + height < H - 38, (text[:60], top, height)
        p.drawOn(self.c, x, H - top - height)
        return top + height

    def rect(self, x, top, width, height, fill=LIGHT, stroke=None, radius=6):
        self.c.setFillColor(fill)
        self.c.setStrokeColor(stroke or fill)
        self.c.roundRect(
            x, H - top - height, width, height, radius, fill=1, stroke=bool(stroke)
        )

    def line(self, x1, t1, x2, t2, color=GRAY, width=0.8):
        self.c.setStrokeColor(color)
        self.c.setLineWidth(width)
        self.c.line(x1, H - t1, x2, H - t2)

    def arrow(self, x1, t1, x2, t2):
        self.line(x1, t1, x2, t2, MUTED, 1)
        # The report uses downward vertical arrows only.
        path = self.c.beginPath()
        path.moveTo(x2, H - t2)
        path.lineTo(x2 - 3, H - t2 + 5)
        path.lineTo(x2 + 3, H - t2 + 5)
        path.close()
        self.c.setFillColor(MUTED)
        self.c.drawPath(path, stroke=0, fill=1)

    def header(self, page, title, subtitle):
        self.text("MIA5150  /  SKILL DISCOVERY PILOT", M, 25, 8.2, BLUE, True)
        self.text("26 SEPTEMBER 2026", W - M - 106, 25, 8.2, MUTED)
        self.text(title, M, 53, 25, NAVY, True)
        self.para(subtitle, M, 91, size=10, color=MUTED)
        self.line(M, H - 33, W - M, H - 33)
        self.text(
            "Saved experimental evidence | No new model calls for this report",
            M,
            H - 25,
            7.5,
            MUTED,
        )
        self.text(f"{page} / 3", W - M - 21, H - 25, 7.5, MUTED)

    def box(self, x, top, width, height, title, body, dark=False):
        self.rect(x, top, width, height, NAVY if dark else LIGHT)
        self.para(
            f"<b>{title}</b>",
            x + 12,
            top + 9,
            width - 24,
            10.1,
            WHITE if dark else NAVY,
            center=True,
        )
        self.para(
            body, x + 12, top + 28, width - 24, 9.1, WHITE if dark else INK, center=True
        )


def page_one(r, d, baseline):
    r.header(
        1,
        "Skill discovery trial",
        "Five OpenRouter models | Four methods | A small, real integration test",
    )
    r.text("Executive Summary", M, 126, 13, NAVY, True)
    r.para(
        "We tested how five models choose useful skills from the pinned SkillRet dataset, using "
        "LangChain Deep Agents' real skill middleware. The trial compared listing every skill, "
        "keyword search, meaning-based search and category-first search on three requests, with "
        "libraries of 50, 250 and 1,000 skills. The integration worked, but only 24 selections "
        "matched the reference answers exactly. This sample is too small to choose a winning model.",
        M,
        149,
        size=10.5,
    )
    top, gap = 253, 10
    tile_w = (CW - 2 * gap) / 3
    for i, (number, title, sub) in enumerate(
        [
            (str(d["planned"]), "planned cases", "138 attempted; 42 skipped"),
            (str(d["valid"]), "valid selections", "Correct format and allowed IDs"),
            (
                str(int(d["exact"])),
                "exact matches",
                "All and only the reference skills",
            ),
        ]
    ):
        x = M + i * (tile_w + gap)
        r.rect(x, top, tile_w, 84)
        r.text(number, x + 12, top + 9, 26, BLUE, True)
        r.text(title, x + 12, top + 41, 10, NAVY, True)
        r.para(sub, x + 12, top + 59, tile_w - 24, 8.0, MUTED)
    r.text("Results by model", M, 358, 14, NAVY, True)
    r.para(
        "Each model had 36 planned cases. A valid selection can still be wrong. "
        "Exact counts below use only valid selections; coverage differs.",
        M,
        381,
        size=9.6,
    )
    models = [
        ("qwen38-27b", "Qwen 3.8 27B", "Mostly paused: rate limits"),
        ("qwen38-flash", "Qwen 3.8 Flash", "Nearly complete"),
        ("deepseek41-flash", "DeepSeek V4.1 Flash", "Nearly complete"),
        ("mimo-v2-6-pro", "MiMo-V2.6-Pro", "Invalid IDs; provider overload"),
        ("glm53-flash", "GLM-5.3-Flash", "Nearly complete"),
    ]
    table_rows = [["Model", "Valid / attempted", "Exact / valid", "Coverage note"]]
    for key, name, note in models:
        rows = [s for s in d["summary"] if s["model"] == key]
        attempted, valid, exact = [
            int(sum(s[k] for s in rows)) for k in ("attempted", "valid", "exact")
        ]
        table_rows.append([name, f"{valid} / {attempted}", f"{exact} / {valid}", note])
    tab = Table(
        table_rows, colWidths=[135, 100, 79, CW - 314], rowHeights=[27] + [29] * 5
    )
    tab.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.6),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    _, th = tab.wrap(CW, H)
    tab.drawOn(r.c, M, H - 419 - th)
    r.para(
        "Of 180 cases, 53 reuse earlier matching runs. The 59 cases without valid selections "
        "comprise 13 invalid-ID outputs, 4 provider errors, 32 provider-paused cases and "
        "10 prerequisite-gated cases.",
        M,
        604,
        size=9.2,
        color=MUTED,
    )
    r.rect(M, 658, CW, 72)
    recall = baseline["metrics"]["recall_at_5"] * 100
    r.text(f"Offline baseline: {recall:.2f}% Recall@5", M + 12, 667, 12, NAVY, True)
    r.para(
        "Before paid calls, full-document BM25 found an average 51.67% of the annotated "
        "skills in its top five results: 200 queries against 6,006 skills. This is a different "
        "test from the metadata-only selections above.",
        M + 12,
        688,
        CW - 24,
        9.2,
    )
    r.para(
        f"<b>Cost:</b> expansion US${d['incremental_actual_usd']:.4f}; whole project "
        f"US${d['budget']['actual_usd']:.4f} (estimated CAD${d['estimated_cad']:.4f} at 1.50). "
        "Costs are provider-reported. The report itself used no paid model calls.",
        M,
        742,
        size=9.3,
    )
    r.c.showPage()


def page_two(r, d):
    r.header(
        2,
        "Results in two graphs",
        "Selection quality and text volume answer different questions.",
    )
    r.text("Selection outcomes by library size", M, 130, 15, NAVY, True)
    r.para(
        "60 planned cases per row: 5 models x 3 requests x 4 methods. Counts, not percentages.",
        M,
        155,
        size=9.6,
        color=MUTED,
    )
    # Directly drawn vector bars stay crisp at any PDF zoom.
    x0, plot_w = M + 70, CW - 77
    for tick in range(0, 61, 15):
        x = x0 + tick / 60 * plot_w
        r.line(x, 212, x, 342, LIGHT)
        r.text(str(tick), x - 4, 347, 8, MUTED)
    for i, (label, color, x) in enumerate(
        [
            ("Exact match", BLUE, M),
            ("Valid, but not exact", GRAY, M + 123),
            ("Invalid / failed / skipped", GOLD, M + 301),
        ]
    ):
        r.rect(x, 185, 9, 9, color, radius=0)
        r.text(label, x + 14, 183, 8.4, MUTED)
    for i, n in enumerate((50, 250, 1000)):
        rows = [s for s in d["summary"] if s["N"] == n]
        valid, exact = [int(sum(s[k] for s in rows)) for k in ("valid", "exact")]
        segments = [(exact, BLUE), (valid - exact, GRAY), (60 - valid, GOLD)]
        top, offset = 217 + i * 44, 0
        r.text(f"{n:,} skills", M, top + 7, 9.3, INK, True)
        for count, color in segments:
            width = count / 60 * plot_w
            if count:
                r.rect(x0 + offset, top, width, 28, color, radius=0)
                value = str(count)
                r.text(
                    value,
                    x0 + offset + width / 2 - len(value) * 2.7,
                    top + 7,
                    10,
                    WHITE if color == BLUE else INK,
                    True,
                )
            offset += width
    r.para(
        "At 1,000 skills, <b>0 of 35 valid selections were exact</b>. Still, 22 included a "
        "reference skill, while 34 included extra IDs. These three requests do not establish "
        "a reliable scaling trend, and failures change which models contribute to each row.",
        M,
        379,
        size=10.1,
    )
    r.text("Text processed at 1,000 skills", M, 459, 15, NAVY, True)
    r.para(
        "DeepSeek V4.1 Flash example | Mean input + output tokens per workflow",
        M,
        484,
        size=9.6,
        color=MUTED,
    )
    samples = {
        s["method"]: s
        for s in d["summary"]
        if s["N"] == 1000 and s["model"] == "deepseek41-flash"
    }
    x0, plot_w, max_x = M + 103, CW - 132, 125000
    for tick in (0, 40000, 80000, 120000):
        x = x0 + tick / max_x * plot_w
        r.line(x, 514, x, 665, LIGHT)
        r.text("0" if not tick else f"{tick // 1000}k", x - 7, 672, 8, MUTED)
    for i, (method, label) in enumerate(
        [
            ("listing", "Listing all skills"),
            ("bm25", "BM25 keywords"),
            ("vector", "Vector meaning"),
            ("domain_vector", "Domain + vector"),
        ]
    ):
        row = samples[method]
        value = row["mean_logical_tokens"]
        width = value / max_x * plot_w
        top = 521 + 37 * i
        r.text(label, M, top + 4, 9.1, INK)
        r.rect(x0, top, width, 23, BLUE, radius=0)
        if method == "listing":
            r.text(f"{value:,.0f}", x0 + width - 51, top + 5, 9.6, WHITE, True)
        else:
            r.text(f"{value:,.0f}", x0 + width + 6, top + 5, 9.6, BLUE, True)
    r.para(
        "Tokens are pieces of text processed by a model. In this example, keyword search "
        "used about <b>1/70 of the tokens</b> needed to list every skill. Lower text volume "
        "alone does not prove better selections or an equal reduction in billed cost.",
        M,
        704,
        size=10.0,
    )
    r.para(
        "Sample sizes: 3 completed responses each for listing, BM25 and vector; 2 for domain-vector. "
        "Includes invalid responses with usage data and category-routing tokens, even when reused. "
        "This is workflow text volume, not newly billed traffic.",
        M,
        756,
        size=8.3,
        color=MUTED,
    )
    r.c.showPage()


def page_three(r, d, baseline, integration):
    r.header(
        3,
        "How the test works",
        "The model chooses skill IDs; downloaded skill instructions are never executed.",
    )
    r.box(
        M,
        130,
        CW,
        57,
        "1. Prepare the same test library and request",
        "Pinned SkillRet data | 50, 250 or 1,000 skills | Three fixed request texts",
    )
    r.para(
        "Reference labels keep relevant skills in each test library and are used for scoring. "
        "They never enter model prompts or model state.",
        M,
        198,
        size=9.0,
        color=MUTED,
    )
    center = W / 2
    r.line(center, 232, center, 244, MUTED, 1)
    gap, bw, bt, bh = 8, (CW - 24) / 4, 258, 78
    centers = [M + i * (bw + gap) + bw / 2 for i in range(4)]
    r.line(centers[0], 244, centers[-1], 244, MUTED, 1)
    methods = [
        ("Listing", "Show all skill<br/>descriptions"),
        ("BM25", "Keyword search<br/>Keep top 5"),
        ("Vector", "Meaning similarity<br/>Keep top 5"),
        ("Domain-vector", "Predict a category;<br/>search it for top 5"),
    ]
    for i, (name, body) in enumerate(methods):
        r.arrow(centers[i], 244, centers[i], bt)
        r.box(M + i * (bw + gap), bt, bw, bh, name, body)
        r.line(centers[i], bt + bh, centers[i], 350, MUTED, 1)
    r.line(centers[0], 350, centers[-1], 350, MUTED, 1)
    r.arrow(center, 350, center, 368)
    r.box(
        M,
        368,
        CW,
        65,
        "2. Load candidates with real Deep Agents SkillsMiddleware",
        "Only candidate metadata enters the main selection prompt.<br/>"
        "An isolated StateBackend holds candidates; the host filesystem is not mounted.",
        dark=True,
    )
    r.arrow(center, 433, center, 451)
    r.box(
        M,
        451,
        CW,
        57,
        "3. Ask one pinned OpenRouter model to select 0-5 skill IDs",
        "Each method and model is tested separately. The five models do not vote together.",
    )
    r.arrow(center, 508, center, 526)
    r.box(
        M,
        526,
        CW,
        57,
        "4. Validate, score and save evidence",
        "Compare IDs with reference labels; retain sanitized outputs, errors, tokens and costs.",
    )
    r.para(
        "<b>Controls:</b> all model calls require local paid-run permission and a persistent "
        "budget guard. Caps: US$3 / CAD$5 and 300 requests. Providers are pinned; no silent "
        "fallback or context truncation. Category-first cases run only after prerequisite checks pass.",
        M,
        599,
        size=9.0,
    )
    reads = sum(m["actual_native_reads"] for m in integration["models"])
    r.text("What was verified separately", M, 657, 12, NAVY, True)
    r.para(
        f"All five models passed a live integration canary. The earlier activation check "
        f"recorded <b>{reads} actual native SKILL.md reads across 25 cases</b>; 15 cases completed "
        "the workflow. A successful read is not proof that a skill can execute a task.",
        M,
        679,
        size=9.5,
    )
    r.para(
        "<b>Next step:</b> resolve provider availability, then test more fixed requests with "
        "comparable coverage. The present results support integration and cost observations, "
        "not a general model ranking.",
        M,
        731,
        size=9.1,
    )
    r.para(
        "<b>Evidence:</b> reports/OPENROUTER_SCALING.json; results/openrouter/scaling/trial.jsonl; "
        "results/baseline_summary.jsonl; reports/OPENROUTER_MODELS.json.<br/>"
        f"SkillRet revision: {baseline['spec']['dataset_revision']}",
        M,
        774,
        size=7.2,
        color=MUTED,
        leading=9,
    )
    r.c.showPage()


def main():
    data, baseline, integration = load_evidence()
    report = Report()
    page_one(report, data, baseline)
    page_two(report, data)
    page_three(report, data, baseline, integration)
    report.c.save()
    pdf = PdfReader(OUT)
    text = " ".join(" ".join(page.extract_text() for page in pdf.pages).split())
    assert len(pdf.pages) == 3
    for expected in (
        "51.67%",
        "200 queries",
        "6,006 skills",
        "24",
        "121",
        "114,877",
        "16 actual native",
    ):
        assert expected in text, expected
    assert "\ufffd" not in text
    evidence = {
        "pdf": str(OUT.relative_to(ROOT)),
        "pages": len(pdf.pages),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES
        },
        "source_checks_passed": True,
        "new_model_calls": 0,
        "initial_inspection_note": "A read-only inspection used status=ok and raised StopIteration; corrected to the saved status=valid. No experiment data changed.",
        "authoring_check_note": "Initial PDF text check failed on a line break in '200 queries'; validation now normalizes whitespace. Source numbers were unchanged.",
        "dependencies": {"reportlab": "5.0.1", "pypdf": "6.19.0"},
    }
    (ROOT / "evidence/short_pdf_report.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"pdf": str(OUT), "pages": len(pdf.pages), "bytes": OUT.stat().st_size}
        )
    )


if __name__ == "__main__":
    main()
