#!/usr/bin/env python3
"""
SKILLRET Dataset - Exploratory Data Analysis
Generates a self-contained HTML report with embedded charts.
"""

import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA = Path("dataset/data")
OUT = Path("eda_report.html")

# ── 1. Load all data ──────────────────────────────────────────────────────────

print("Loading skills.jsonl (full corpus) ...")
skills_all = [json.loads(l) for l in open(DATA / "skills.jsonl")]
print(f"  -> {len(skills_all):,} records")

print("Loading skills splits ...")
skills_train = [json.loads(l) for l in open(DATA / "skills/train.jsonl")]
skills_test  = [json.loads(l) for l in open(DATA / "skills/test.jsonl")]
print(f"  -> train: {len(skills_train):,}  test: {len(skills_test):,}")

print("Loading queries ...")
queries_train = [json.loads(l) for l in open(DATA / "queries/train.jsonl")]
queries_test  = [json.loads(l) for l in open(DATA / "queries/test.jsonl")]
print(f"  -> train: {len(queries_train):,}  test: {len(queries_test):,}")

print("Loading qrels ...")
qrels_train = [json.loads(l) for l in open(DATA / "qrels/train.jsonl")]
qrels_test  = [json.loads(l) for l in open(DATA / "qrels/test.jsonl")]
print(f"  -> train: {len(qrels_train):,}  test: {len(qrels_test):,}")

print("Loading taxonomy ...")
taxonomy = json.load(open(DATA / "taxonomy.json"))

# Convert to DataFrames
df_skills = pd.DataFrame(skills_all)
df_skills_train = pd.DataFrame(skills_train)
df_skills_test  = pd.DataFrame(skills_test)
df_queries_train = pd.DataFrame(queries_train)
df_queries_test  = pd.DataFrame(queries_test)
df_qrels_train = pd.DataFrame(qrels_train)
df_qrels_test  = pd.DataFrame(qrels_test)

# ── 2. Skills analysis ────────────────────────────────────────────────────────

total_skills = len(skills_all)

# ── 2a. Skill structure analysis ──────────────────────────────────────────────
# Per the Agent Skills spec (agentskills.io/specification), a skill is a
# directory containing SKILL.md (required) + optional subfolders:
#   scripts/     – executable code
#   references/  – additional documentation
#   assets/      – templates, resources
#
# The dataset only captured SKILL.md content (body == skill_md for all records).
# We infer the original folder structure from references *inside* SKILL.md text.
# ─────────────────────────────────────────────────────────────────────────────

df_skills["skill_md_len"] = df_skills["skill_md"].str.len()

pat_scripts    = re.compile(r'\bscripts/', re.IGNORECASE)
pat_references = re.compile(r'\breferences/', re.IGNORECASE)
pat_assets     = re.compile(r'\bassets/', re.IGNORECASE)

# Markdown links to local files: [text](path/file.ext) excluding URLs and anchors
pat_md_link_local = re.compile(
    r'\]\('
    r'(?!https?://|#)'
    r'([^)]+\.'
    r'(?:py|sh|bash|js|ts|rb|go|rs|json|yaml|yml|toml|csv|sql|md))'
    r'\)'
)

def classify_structure(body):
    """Classify a skill's folder structure from its SKILL.md text."""
    has_scripts    = bool(pat_scripts.search(body))
    has_references = bool(pat_references.search(body))
    has_assets     = bool(pat_assets.search(body))
    has_any_sub    = has_scripts or has_references or has_assets
    # Also detect markdown links to local companion files with path separators
    md_links = pat_md_link_local.findall(body)
    local_links = [l for l in md_links
                   if not l.upper().startswith(('HTTP', 'SKILL.MD', 'LICENSE'))
                   and '/' in l]
    has_local_links = len(local_links) > 0
    return {
        "has_scripts": has_scripts,
        "has_references": has_references,
        "has_assets": has_assets,
        "has_any_subfolder": has_any_sub,
        "has_companion_files": has_any_sub or has_local_links,
    }

struct = df_skills["skill_md"].apply(classify_structure)
df_skills["has_scripts"]    = struct.apply(lambda x: x["has_scripts"])
df_skills["has_references"] = struct.apply(lambda x: x["has_references"])
df_skills["has_assets"]     = struct.apply(lambda x: x["has_assets"])
df_skills["has_any_subfolder"]  = struct.apply(lambda x: x["has_any_subfolder"])
df_skills["has_companion_files"] = struct.apply(lambda x: x["has_companion_files"])
df_skills["is_bare_skill_md"]   = ~df_skills["has_companion_files"]

bare_count   = df_skills["is_bare_skill_md"].sum()
companion_count = (~df_skills["is_bare_skill_md"]).sum()
scripts_count    = df_skills["has_scripts"].sum()
references_count = df_skills["has_references"].sum()
assets_count     = df_skills["has_assets"].sum()
any_sub_count    = df_skills["has_any_subfolder"].sum()

print(f"\n=== Skill Structure (inferred from SKILL.md text) ===")
print(f"  Bare SKILL.md only:          {bare_count:>6,} ({bare_count/total_skills*100:.1f}%)")
print(f"  With companion files/folders: {companion_count:>6,} ({companion_count/total_skills*100:.1f}%)")
print(f"    - with scripts/:            {scripts_count:>6,} ({scripts_count/total_skills*100:.1f}%)")
print(f"    - with references/:          {references_count:>6,} ({references_count/total_skills*100:.1f}%)")
print(f"    - with assets/:              {assets_count:>6,} ({assets_count/total_skills*100:.1f}%)")

# Subfolder combination breakdown
def combo_label(row):
    parts = []
    if row["has_scripts"]:    parts.append("scripts/")
    if row["has_references"]: parts.append("references/")
    if row["has_assets"]:     parts.append("assets/")
    if not parts:
        return "other local file refs"
    return " + ".join(parts)

df_with_sub = df_skills[df_skills["has_companion_files"]].copy()
df_with_sub["combo"] = df_with_sub.apply(combo_label, axis=1)
combo_counts = df_with_sub["combo"].value_counts()

# By-category breakdown
cat_bare = df_skills[df_skills["is_bare_skill_md"]].groupby("major").size()
cat_comp = df_skills[~df_skills["is_bare_skill_md"]].groupby("major").size()
cat_total = df_skills.groupby("major").size()

# 2d. Taxonomy breakdown
major_counts = df_skills["major"].value_counts()
sub_counts   = df_skills["sub"].value_counts()
major_sub    = df_skills.groupby(["major", "sub"]).size().reset_index(name="count")

# 2e. Primary actions & objects
action_counts = df_skills["primary_action"].value_counts()
object_counts = df_skills["primary_object"].value_counts()
domain_counts = df_skills["domain"].value_counts()

# 2f. Author & repo stats
author_counts = df_skills["author"].value_counts()
repo_counts   = df_skills["repo"].value_counts()
license_counts = df_skills["license"].value_counts()

# 2g. Stars & installs distributions
stars_series   = df_skills["stars"]
installs_series = df_skills["installs"]

# 2h. Skill description / body length stats
df_skills["desc_len"] = df_skills["description"].str.len()

# ── 3. Queries analysis ──────────────────────────────────────────────────────

df_queries_test["query_len"]    = df_queries_test["query"].str.len()
df_queries_test["query_words"]  = df_queries_test["query"].str.split().str.len()
df_queries_test["num_skills"]   = df_queries_test["skill_ids"].apply(len)
df_queries_test["k_value"]      = df_queries_test["k"]

df_queries_train["query_len"]   = df_queries_train["query"].str.len()
df_queries_train["query_words"] = df_queries_train["query"].str.split().str.len()
df_queries_train["num_skills"]  = df_queries_train["skill_ids"].apply(len)
df_queries_train["k_value"]     = df_queries_train["k"]

# Generator model breakdown
gen_model_test  = df_queries_test["generator_model"].value_counts()
gen_model_train = df_queries_train["generator_model"].value_counts()

# ── 4. Qrels analysis ────────────────────────────────────────────────────────

# Skills per query in qrels
qrels_test_per_query  = df_qrels_test.groupby("query_id").size()
qrels_train_per_query = df_qrels_train.groupby("query_id").size()

# ── 5. Build charts ──────────────────────────────────────────────────────────

import math

def nice_xaxis(fig, max_val, num_ticks=6):
    """Set explicit tick values with comma-formatted labels on the x-axis."""
    step = math.ceil(max_val / num_ticks / 100) * 100  # round step to nearest 100
    if step == 0:
        step = 1
    vals = list(range(0, int(max_val + step), step))
    fig.update_layout(xaxis=dict(
        tickmode="array",
        tickvals=vals,
        ticktext=[f"{v:,}" for v in vals],
    ))

charts = []

# -- Chart 1: Dataset overview (table-like)
fig_overview = go.Figure(data=[go.Table(
    header=dict(values=["Component", "Split", "Records", "Description"],
                fill_color="#2c3e50", font=dict(color="white", size=13),
                align="left"),
    cells=dict(values=[
        ["Skills (full)", "Skills", "Skills", "Queries", "Queries", "Qrels", "Qrels"],
        ["all", "train", "test", "train", "test", "train", "test"],
        [f"{len(skills_all):,}", f"{len(skills_train):,}", f"{len(skills_test):,}",
         f"{len(queries_train):,}", f"{len(queries_test):,}",
         f"{len(qrels_train):,}", f"{len(qrels_test):,}"],
        ["Combined skill corpus", "Training skill split", "Evaluation skill split",
         "Synthetic training queries (Qwen3.5-122B-A10B)", "Evaluation queries (Claude Opus 4.6)",
         "Training relevance labels", "Evaluation relevance labels"],
    ],
    fill_color=[["#ecf0f1", "#f9f9f9"]*4],
    align="left", font=dict(size=12))
)])
fig_overview.update_layout(title="Dataset Overview", height=300, margin=dict(t=40, b=10))
charts.append(("Dataset Overview", fig_overview))

# -- Chart 2: Major category breakdown (horizontal bar)
major_df = pd.DataFrame({"Category": major_counts.index, "Count": major_counts.values})
major_df["Label"] = major_df["Count"].apply(lambda v: f"{v:,}")
fig_major = px.bar(
    major_df, x="Count", y="Category",
    orientation="h",
    labels={"Count": "Number of Skills", "Category": "Major Category"},
    title="Skills by Major Category",
    color="Count",
    color_continuous_scale="Viridis",
    text="Label",
)
fig_major.update_layout(showlegend=False, coloraxis_showscale=False, height=400,
                        yaxis=dict(autorange="reversed"))
nice_xaxis(fig_major, major_df["Count"].max())
fig_major.update_traces(textposition="outside")
charts.append(("Major Categories", fig_major))

# -- Chart 3: Sub-category breakdown (sunburst)
# Build explicit ids/labels/parents/values for the sunburst so color maps correctly
sun_ids, sun_labels, sun_parents, sun_values = [], [], [], []
# Add major-level entries
for major_name, grp in major_sub.groupby("major"):
    total = grp["count"].sum()
    sun_ids.append(major_name)
    sun_labels.append(major_name)
    sun_parents.append("")
    sun_values.append(total)
# Add sub-level entries
for _, row in major_sub.iterrows():
    sun_ids.append(f"{row['major']} - {row['sub']}")
    sun_labels.append(row["sub"])
    sun_parents.append(row["major"])
    sun_values.append(row["count"])

fig_sun = go.Figure(go.Sunburst(
    ids=sun_ids, labels=sun_labels, parents=sun_parents, values=sun_values,
    branchvalues="total",
    marker=dict(colors=sun_values, colorscale="Viridis", showscale=True,
                colorbar=dict(title="Skills")),
    textinfo="label+value",
    hovertemplate="<b>%{label}</b><br>Skills: %{value:,}<extra></extra>",
))
fig_sun.update_layout(title="Taxonomy Sunburst: Major -> Sub-Category", height=600)
charts.append(("Taxonomy Sunburst", fig_sun))

# -- Chart 4: Sub-category bar chart
sub_df = pd.DataFrame({"SubCategory": sub_counts.index, "Count": sub_counts.values})
sub_df["Label"] = sub_df["Count"].apply(lambda v: f"{v:,}")
fig_sub = px.bar(
    sub_df, x="Count", y="SubCategory",
    orientation="h",
    labels={"Count": "Number of Skills", "SubCategory": "Sub-Category"},
    title="Skills by Sub-Category",
    color="Count",
    color_continuous_scale="Tealgrn",
    text="Label",
)
fig_sub.update_layout(showlegend=False, coloraxis_showscale=False, height=500,
                      yaxis=dict(autorange="reversed"))
nice_xaxis(fig_sub, sub_df["Count"].max())
fig_sub.update_traces(textposition="outside")
charts.append(("Sub-Categories", fig_sub))

# -- Chart 5: Bare SKILL.md vs skills with companion files (donut)
fig_structure = go.Figure(data=[go.Pie(
    labels=["Bare SKILL.md only", "With companion files/folders"],
    values=[bare_count, companion_count],
    hole=0.45,
    marker_colors=["#3498db", "#e74c3c"],
    textinfo="label+value+percent",
    textfont=dict(size=14),
    pull=[0, 0.04],
)])
fig_structure.update_layout(
    title="Skill Directory Structure: Bare SKILL.md vs With Sub-Folders/Files<br>"
          "<span style='font-size:0.8em;color:#666'>(inferred from references inside SKILL.md text)</span>",
    height=420,
)
charts.append(("Skill Structure", fig_structure))

# -- Chart 6: Breakdown of companion folder types (bar)
folder_type_data = pd.DataFrame({
    "Folder Type": ["scripts/", "references/", "assets/"],
    "Count": [scripts_count, references_count, assets_count],
})
folder_type_data = folder_type_data.sort_values("Count", ascending=True)
folder_type_data["Label"] = folder_type_data["Count"].apply(lambda v: f"{v:,}")
fig_folders = px.bar(
    folder_type_data, x="Count", y="Folder Type", orientation="h",
    title="Skills with Standard Sub-Folders (per Agent Skills spec)",
    text="Label", color="Count", color_continuous_scale="Oranges",
)
fig_folders.update_layout(showlegend=False, coloraxis_showscale=False, height=300)
nice_xaxis(fig_folders, folder_type_data["Count"].max())
fig_folders.update_traces(textposition="outside")
charts.append(("Sub-Folder Types", fig_folders))

# -- Chart 7: Sub-folder combination breakdown (bar)
combo_df = combo_counts.reset_index()
combo_df.columns = ["Combination", "Count"]
combo_df = combo_df.sort_values("Count", ascending=True)
combo_df["Label"] = combo_df["Count"].apply(lambda v: f"{v:,}")
fig_combos = px.bar(
    combo_df, x="Count", y="Combination", orientation="h",
    title="Sub-Folder Combinations in Skills with Companion Files",
    text="Label", color="Count", color_continuous_scale="Tealgrn",
)
fig_combos.update_layout(showlegend=False, coloraxis_showscale=False,
                         height=max(300, len(combo_df)*45+100))
nice_xaxis(fig_combos, combo_df["Count"].max())
fig_combos.update_traces(textposition="outside")
charts.append(("Folder Combos", fig_combos))

# -- Chart 8: Bare vs companion by major category (stacked bar)
struct_by_cat = pd.DataFrame({
    "Bare SKILL.md": cat_bare,
    "With companions": cat_comp,
}).fillna(0).astype(int)
struct_by_cat = struct_by_cat.sort_values("With companions", ascending=True)

bare_vals = struct_by_cat["Bare SKILL.md"]
comp_vals = struct_by_cat["With companions"]
total_vals = bare_vals + comp_vals

fig_cat_struct = go.Figure()
fig_cat_struct.add_trace(go.Bar(
    y=struct_by_cat.index, x=bare_vals,
    name="Bare SKILL.md", orientation="h", marker_color="#3498db",
    text=[f"{v:,}" for v in bare_vals],
    textposition="inside", textfont=dict(color="white"),
))
fig_cat_struct.add_trace(go.Bar(
    y=struct_by_cat.index, x=comp_vals,
    name="With companions", orientation="h", marker_color="#e74c3c",
    text=[f"{v:,}" for v in comp_vals],
    textposition="inside", textfont=dict(color="white"),
))
# Add total labels at the end of each stacked bar
fig_cat_struct.add_trace(go.Scatter(
    y=struct_by_cat.index, x=total_vals + total_vals.max() * 0.02,
    mode="text",
    text=[f"{v:,}" for v in total_vals],
    textposition="middle right",
    textfont=dict(size=12, color="#333"),
    showlegend=False,
))
fig_cat_struct.update_layout(
    title="Bare SKILL.md vs With Companions by Major Category",
    barmode="stack", height=400,
    xaxis_title="Number of Skills",
    legend=dict(orientation="h", y=1.1),
)
nice_xaxis(fig_cat_struct, total_vals.max() * 1.1)
charts.append(("Structure by Category", fig_cat_struct))

# -- Chart 8: Primary action breakdown
top_actions = action_counts.head(15)
actions_df = pd.DataFrame({"Action": top_actions.index, "Count": top_actions.values})
actions_df["Label"] = actions_df["Count"].apply(lambda v: f"{v:,}")
fig_actions = px.bar(
    actions_df, x="Count", y="Action", orientation="h",
    title="Top 15 Primary Actions",
    text="Label", color="Count",
    color_continuous_scale="Purp",
)
fig_actions.update_layout(showlegend=False, coloraxis_showscale=False, height=450,
                          yaxis=dict(autorange="reversed"))
nice_xaxis(fig_actions, actions_df["Count"].max())
fig_actions.update_traces(textposition="outside")
charts.append(("Primary Actions", fig_actions))

# -- Chart 9: Primary object breakdown
top_objects = object_counts.head(15)
objects_df = pd.DataFrame({"Object": top_objects.index, "Count": top_objects.values})
objects_df["Label"] = objects_df["Count"].apply(lambda v: f"{v:,}")
fig_objects = px.bar(
    objects_df, x="Count", y="Object", orientation="h",
    title="Top 15 Primary Objects",
    text="Label", color="Count",
    color_continuous_scale="Teal",
)
fig_objects.update_layout(showlegend=False, coloraxis_showscale=False, height=450,
                          yaxis=dict(autorange="reversed"))
nice_xaxis(fig_objects, objects_df["Count"].max())
fig_objects.update_traces(textposition="outside")
charts.append(("Primary Objects", fig_objects))

# -- Chart 10: Domain breakdown
top_domains = domain_counts.head(20)
domains_df = pd.DataFrame({"Domain": top_domains.index, "Count": top_domains.values})
domains_df["Label"] = domains_df["Count"].apply(lambda v: f"{v:,}")
fig_domains = px.bar(
    domains_df, x="Count", y="Domain", orientation="h",
    title="Top 20 Domains",
    text="Label", color="Count",
    color_continuous_scale="Mint",
)
fig_domains.update_layout(showlegend=False, coloraxis_showscale=False, height=550,
                          yaxis=dict(autorange="reversed"))
nice_xaxis(fig_domains, domains_df["Count"].max())
fig_domains.update_traces(textposition="outside")
charts.append(("Domains", fig_domains))

# -- Chart 11: License distribution
fig_license = px.pie(
    names=license_counts.index, values=license_counts.values,
    title="License Distribution",
    hole=0.35,
)
fig_license.update_layout(height=400)
charts.append(("Licenses", fig_license))

# -- Chart 12: Stars distribution (log scale)
fig_stars = go.Figure(go.Histogram(x=df_skills["stars"].values.tolist(), nbinsx=80,
                                    marker_color="#3498db"))
fig_stars.update_layout(title="GitHub Stars Distribution (all skills)", height=350,
                        xaxis_title="GitHub Stars", yaxis_title="count", yaxis_type="log")
nice_xaxis(fig_stars, df_skills["stars"].max())
charts.append(("Stars Distribution", fig_stars))

# -- Chart 13: Installs distribution
fig_installs = go.Figure(go.Histogram(x=df_skills["installs"].values.tolist(), nbinsx=80,
                                       marker_color="#e67e22"))
fig_installs.update_layout(title="Install Count Distribution (all skills)", height=350,
                           xaxis_title="Installs", yaxis_title="count", yaxis_type="log")
nice_xaxis(fig_installs, df_skills["installs"].max())
charts.append(("Installs Distribution", fig_installs))

# -- Chart 14: Top authors
top_authors = author_counts.head(20)
authors_df = pd.DataFrame({"Author": top_authors.index, "Count": top_authors.values})
authors_df["Label"] = authors_df["Count"].apply(lambda v: f"{v:,}")
fig_authors = px.bar(
    authors_df, x="Count", y="Author", orientation="h",
    title="Top 20 Skill Authors (by number of skills)",
    text="Label", color="Count",
    color_continuous_scale="Blues",
)
fig_authors.update_layout(showlegend=False, coloraxis_showscale=False, height=550,
                          yaxis=dict(autorange="reversed"))
nice_xaxis(fig_authors, authors_df["Count"].max())
fig_authors.update_traces(textposition="outside")
charts.append(("Top Authors", fig_authors))

# -- Chart 15: Skill body length (SKILL.md size) distribution
fig_bodylen = go.Figure(go.Histogram(x=df_skills["skill_md_len"].values.tolist(), nbinsx=100,
                                      marker_color="#9b59b6"))
fig_bodylen.update_layout(title="SKILL.md Content Length Distribution (characters)", height=350,
                          xaxis_title="SKILL.md Length (chars)", yaxis_title="count", yaxis_type="log")
nice_xaxis(fig_bodylen, df_skills["skill_md_len"].max())
charts.append(("SKILL.md Length", fig_bodylen))

# -- Chart 16: Description length distribution
fig_desclen = go.Figure(go.Histogram(x=df_skills["desc_len"].values.tolist(), nbinsx=80,
                                      marker_color="#1abc9c"))
fig_desclen.update_layout(title="Skill Description Length Distribution (characters)", height=350,
                          xaxis_title="Description Length (chars)", yaxis_title="count")
nice_xaxis(fig_desclen, df_skills["desc_len"].max())
charts.append(("Desc Length", fig_desclen))

# -- Chart 17: Query length distributions (test vs train)
fig_qlen = make_subplots(rows=1, cols=2, subplot_titles=["Test Queries (Claude Opus 4.6)", "Train Queries (Qwen3.5-122B-A10B)"])
fig_qlen.add_trace(go.Histogram(x=df_queries_test["query_words"].values.tolist(), nbinsx=60, name="Test", marker_color="#3498db"), row=1, col=1)
fig_qlen.add_trace(go.Histogram(x=df_queries_train["query_words"].values.tolist(), nbinsx=60, name="Train", marker_color="#e67e22"), row=1, col=2)
# Explicit x-axis ticks for subplots
test_max = int(df_queries_test["query_words"].max())
train_max = int(df_queries_train["query_words"].max())
test_ticks = list(range(0, test_max + 100, 100))
train_ticks = list(range(0, train_max + 50, 50))
fig_qlen.update_xaxes(title_text="Query Length (words)", tickmode="array",
                       tickvals=test_ticks, ticktext=[str(v) for v in test_ticks], row=1, col=1)
fig_qlen.update_xaxes(title_text="Query Length (words)", tickmode="array",
                       tickvals=train_ticks, ticktext=[str(v) for v in train_ticks], row=1, col=2)
fig_qlen.update_yaxes(title_text="Count", row=1, col=1)
fig_qlen.update_layout(title="Query Length Distribution (word count)", height=400, showlegend=False)
charts.append(("Query Lengths", fig_qlen))

# -- Chart 18: Number of relevant skills per query (k=1,2,3 - use bar chart)
test_k_vc = df_queries_test["k_value"].value_counts().sort_index()
train_k_vc = df_queries_train["k_value"].value_counts().sort_index()
fig_k = make_subplots(rows=1, cols=2, subplot_titles=["Test Queries", "Train Queries"])
fig_k.add_trace(go.Bar(x=[str(k) for k in test_k_vc.index], y=test_k_vc.values.tolist(),
                        name="Test", marker_color="#2ecc71",
                        text=[f"{v:,}" for v in test_k_vc.values], textposition="outside"), row=1, col=1)
fig_k.add_trace(go.Bar(x=[str(k) for k in train_k_vc.index], y=train_k_vc.values.tolist(),
                        name="Train", marker_color="#9b59b6",
                        text=[f"{v:,}" for v in train_k_vc.values], textposition="outside"), row=1, col=2)
fig_k.update_xaxes(title_text="k (relevant skills per query)", row=1, col=1)
fig_k.update_xaxes(title_text="k (relevant skills per query)", row=1, col=2)
fig_k.update_yaxes(title_text="Count", tickmode="array",
                   tickvals=list(range(0, int(test_k_vc.max()) + 1000, 500)),
                   ticktext=[f"{v:,}" for v in range(0, int(test_k_vc.max()) + 1000, 500)], row=1, col=1)
fig_k.update_yaxes(tickmode="array",
                   tickvals=list(range(0, int(train_k_vc.max()) + 5000, 5000)),
                   ticktext=[f"{v:,}" for v in range(0, int(train_k_vc.max()) + 5000, 5000)], row=1, col=2)
fig_k.update_layout(title="Relevant Skills Per Query (k distribution)", height=400, showlegend=False)
charts.append(("Skills per Query", fig_k))

# -- Chart 19: Qrels per query distribution
test_qrel_vc = qrels_test_per_query.value_counts().sort_index()
train_qrel_vc = qrels_train_per_query.value_counts().sort_index()
fig_qrels_dist = make_subplots(rows=1, cols=2, subplot_titles=["Test Qrels", "Train Qrels"])
fig_qrels_dist.add_trace(go.Bar(x=[str(k) for k in test_qrel_vc.index], y=test_qrel_vc.values.tolist(),
                                 name="Test", marker_color="#1abc9c",
                                 text=[f"{v:,}" for v in test_qrel_vc.values], textposition="outside"), row=1, col=1)
fig_qrels_dist.add_trace(go.Bar(x=[str(k) for k in train_qrel_vc.index], y=train_qrel_vc.values.tolist(),
                                 name="Train", marker_color="#e74c3c",
                                 text=[f"{v:,}" for v in train_qrel_vc.values], textposition="outside"), row=1, col=2)
fig_qrels_dist.update_xaxes(title_text="Qrels per Query", row=1, col=1)
fig_qrels_dist.update_xaxes(title_text="Qrels per Query", row=1, col=2)
fig_qrels_dist.update_yaxes(title_text="Count", tickmode="array",
                            tickvals=list(range(0, int(test_qrel_vc.max()) + 1000, 500)),
                            ticktext=[f"{v:,}" for v in range(0, int(test_qrel_vc.max()) + 1000, 500)], row=1, col=1)
fig_qrels_dist.update_yaxes(tickmode="array",
                            tickvals=list(range(0, int(train_qrel_vc.max()) + 5000, 5000)),
                            ticktext=[f"{v:,}" for v in range(0, int(train_qrel_vc.max()) + 5000, 5000)], row=1, col=2)
fig_qrels_dist.update_layout(title="Relevance Judgments Per Query", height=400, showlegend=False)
charts.append(("Qrels per Query", fig_qrels_dist))

# -- Chart 20: Train vs test split comparison by major category
train_major = df_skills_train["major"].value_counts().rename("train")
test_major  = df_skills_test["major"].value_counts().rename("test")
split_df = pd.concat([train_major, test_major], axis=1).fillna(0).astype(int)
split_df = split_df.sort_values("train", ascending=True)

fig_split = go.Figure()
fig_split.add_trace(go.Bar(y=split_df.index, x=split_df["train"].values.tolist(), name="Train", orientation="h", marker_color="#3498db",
                            text=[f"{v:,}" for v in split_df["train"]], textposition="outside"))
fig_split.add_trace(go.Bar(y=split_df.index, x=split_df["test"].values.tolist(), name="Test", orientation="h", marker_color="#e74c3c",
                            text=[f"{v:,}" for v in split_df["test"]], textposition="outside"))
fig_split.update_layout(
    title="Train vs Test Split by Major Category",
    barmode="group", height=400,
    xaxis_title="Number of Skills",
)
nice_xaxis(fig_split, split_df.max().max())
charts.append(("Train/Test Split", fig_split))


# ── 6. Assemble HTML report ──────────────────────────────────────────────────

# Compute summary stats for narrative
total_skills    = len(skills_all)
total_authors   = df_skills["author"].nunique()
total_repos     = df_skills["repo"].nunique()
median_stars    = int(stars_series.median())
median_installs = int(installs_series.median())
mean_body_len   = int(df_skills["skill_md_len"].mean())
median_body_len = int(df_skills["skill_md_len"].median())
max_body_len    = int(df_skills["skill_md_len"].max())
pct_bare        = bare_count / total_skills * 100
pct_companion   = companion_count / total_skills * 100

# Query stats
test_q_med_words  = int(df_queries_test["query_words"].median())
test_q_mean_words = int(df_queries_test["query_words"].mean())
train_q_med_words = int(df_queries_train["query_words"].median())
train_q_mean_words= int(df_queries_train["query_words"].mean())

# JSON samples (truncated for display)
import html as html_mod

def truncate_val(v, maxlen=120):
    if isinstance(v, str) and len(v) > maxlen:
        return v[:maxlen] + "..."
    return v

def sample_json(records, maxlen=120):
    rec = records[0].copy() if isinstance(records[0], dict) else json.loads(records[0])
    return json.dumps({k: truncate_val(v, maxlen) for k, v in rec.items()}, indent=2)

sample_skill_json = html_mod.escape(sample_json(skills_all))
sample_query_json = html_mod.escape(sample_json(queries_test))
sample_qrel_json  = html_mod.escape(sample_json(qrels_test))

# Build HTML -- generate each chart block individually
chart_blocks = []
for title, fig in charts:
    block = f'<div class="chart-container" id="{title.lower().replace(" ", "-")}">\n'
    block += fig.to_html(full_html=False, include_plotlyjs=False)
    block += "\n</div>\n"
    chart_blocks.append(block)

def charts_html(start, end=None):
    """Return concatenated chart HTML for charts[start:end]."""
    if end is None:
        return "".join(chart_blocks[start:])
    return "".join(chart_blocks[start:end])

# Navigation
nav_items = "".join(
    f'<a href="#{t.lower().replace(" ", "-")}">{t}</a>' for t, _ in charts
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SKILLRET Dataset - Exploratory Data Analysis</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  :root {{ --bg: #f8f9fa; --card: #ffffff; --accent: #2c3e50; --accent2: #3498db; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
          background: var(--bg); color: #333; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #2c3e50, #3498db);
             color: white; padding: 2.5rem 2rem; text-align: center; }}
  .header h1 {{ font-size: 2.2rem; margin-bottom: 0.3rem; }}
  .header p  {{ opacity: 0.85; font-size: 1.05rem; }}
  .nav {{ background: #2c3e50; padding: 0.6rem 1rem; display: flex;
          flex-wrap: wrap; gap: 0.4rem; justify-content: center;
          position: sticky; top: 0; z-index: 100; }}
  .nav a {{ color: #ecf0f1; text-decoration: none; padding: 0.3rem 0.7rem;
            border-radius: 4px; font-size: 0.82rem; transition: background 0.2s; }}
  .nav a:hover {{ background: rgba(255,255,255,0.15); }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 1.5rem; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 1rem; margin-bottom: 2rem; }}
  .kpi {{ background: var(--card); border-radius: 10px; padding: 1.2rem;
          text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
          border-top: 4px solid var(--accent2); }}
  .kpi .value {{ font-size: 1.9rem; font-weight: 700; color: var(--accent); }}
  .kpi .label {{ font-size: 0.85rem; color: #666; margin-top: 0.2rem; }}
  .section {{ margin-bottom: 2.5rem; }}
  .section h2 {{ font-size: 1.5rem; color: var(--accent); margin-bottom: 0.8rem;
                 border-left: 4px solid var(--accent2); padding-left: 0.8rem; }}
  .section p, .section ul {{ margin-bottom: 0.8rem; }}
  .section ul {{ padding-left: 1.5rem; }}
  .section li {{ margin-bottom: 0.3rem; }}
  .chart-container {{ background: var(--card); border-radius: 10px; padding: 1rem;
                      margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .highlight {{ background: #eaf2f8; border-left: 4px solid #3498db; padding: 1rem 1.2rem;
                border-radius: 6px; margin: 1rem 0; }}
  .highlight strong {{ color: var(--accent); }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  table.stats {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0; }}
  table.stats th, table.stats td {{ padding: 0.5rem 0.8rem; border: 1px solid #ddd; text-align: left; }}
  table.stats th {{ background: #2c3e50; color: white; font-size: 0.9rem; }}
  table.stats tr:nth-child(even) {{ background: #f5f5f5; }}
  .footer {{ text-align: center; padding: 2rem; color: #999; font-size: 0.85rem; }}
  code {{ background: #eee; padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>

<div class="header">
  <h1>SKILLRET Dataset &mdash; Exploratory Data Analysis</h1>
  <p>A retrieval benchmark for matching natural-language user requests to AI agent skills</p>
  <p style="opacity:0.7; font-size:0.9rem; margin-top:0.3rem;">
    Source: <a href="https://huggingface.co/datasets/ThakiCloud/SKILLRET" style="color:#ecf0f1;">huggingface.co/datasets/ThakiCloud/SKILLRET</a>
  </p>
</div>

<div class="nav">{nav_items}</div>

<div class="container">

<!-- KPIs -->
<div class="kpi-row">
  <div class="kpi"><div class="value">{total_skills:,}</div><div class="label">Total Skills</div></div>
  <div class="kpi"><div class="value">{len(queries_train)+len(queries_test):,}</div><div class="label">Total Queries</div></div>
  <div class="kpi"><div class="value">{len(qrels_train)+len(qrels_test):,}</div><div class="label">Relevance Judgments</div></div>
  <div class="kpi"><div class="value">{total_authors:,}</div><div class="label">Unique Authors</div></div>
  <div class="kpi"><div class="value">{total_repos:,}</div><div class="label">GitHub Repos</div></div>
  <div class="kpi"><div class="value">{pct_bare:.1f}%</div><div class="label">Bare SKILL.md Only</div></div>
</div>

<!-- Section 1: What is this dataset? -->
<div class="section">
  <h2>1. What Is SKILLRET?</h2>
  <p><strong>SkillRet</strong> is a retrieval benchmark for matching natural-language user requests to AI agent skills.
     It contains a curated library of <strong>{total_skills:,}</strong> public agent skills harvested from GitHub,
     along with synthetic training queries and human-quality evaluation queries.</p>
  <p>The dataset consists of three core components:</p>
  <ul>
    <li><strong>Skills</strong> &mdash; structured metadata + full SKILL.md content for each agent skill</li>
    <li><strong>Queries</strong> &mdash; natural-language user requests that should match one or more skills</li>
    <li><strong>Qrels</strong> &mdash; binary relevance judgments linking queries to their correct skills</li>
  </ul>
  <p>Additionally, the dataset includes a <strong>taxonomy</strong> with 6 major categories and 18 sub-categories
     that classify every skill.</p>
</div>

{charts_html(0, 1)}

<!-- Section 1b: Data Structure Examples -->
<div class="section">
  <h2>Data Structure</h2>
  <p>Each JSONL file stores one record per line. Below are example records (long text values truncated with &hellip;).</p>

  <h3>Skill Record <code>(skills.jsonl)</code></h3>
  <p>18 fields per skill: identifiers, metadata, taxonomy labels, and the SKILL.md content.</p>
  <pre style="background:#f4f6f8; padding:1rem; border-radius:6px; overflow-x:auto; font-size:0.85rem; line-height:1.5; border:1px solid #ddd;">{sample_skill_json}</pre>

  <h3 style="margin-top:1.5rem;">Query Record <code>(queries/*.jsonl)</code></h3>
  <p>7 fields: the natural-language request, its matching skill IDs, the number of relevant skills (k), and the generator model.</p>
  <pre style="background:#f4f6f8; padding:1rem; border-radius:6px; overflow-x:auto; font-size:0.85rem; line-height:1.5; border:1px solid #ddd;">{sample_query_json}</pre>

  <h3 style="margin-top:1.5rem;">Qrel Record <code>(qrels/*.jsonl)</code></h3>
  <p>3 fields: a binary relevance judgment linking a query to a skill.</p>
  <pre style="background:#f4f6f8; padding:1rem; border-radius:6px; overflow-x:auto; font-size:0.85rem; line-height:1.5; border:1px solid #ddd;">{sample_qrel_json}</pre>
</div>

<!-- Section 2: Skills deep dive -->
<div class="section">
  <h2>2. Skill Directory Structure</h2>
  <p>Per the <a href="https://agentskills.io/specification">Agent Skills specification</a>, a skill is a
     <strong>directory</strong> containing a required <code>SKILL.md</code> file plus optional companion folders:</p>
  <ul>
    <li><code>scripts/</code> &mdash; executable code the agent can run</li>
    <li><code>references/</code> &mdash; additional documentation loaded on demand</li>
    <li><code>assets/</code> &mdash; templates, data files, static resources</li>
  </ul>
  <p>The dataset only captured the <code>SKILL.md</code> content for each skill (<code>body == skill_md</code>
     for all {total_skills:,} records). To determine how many skills are <strong>bare SKILL.md</strong> (nothing
     else in the directory) versus skills that have companion sub-folders, we analyzed the SKILL.md text
     for references to <code>scripts/</code>, <code>references/</code>, <code>assets/</code>, and other local file paths.</p>

  <div class="highlight">
    <strong>Key Finding: {pct_bare:.1f}% ({bare_count:,}) of skills are bare SKILL.md with no sub-folders.</strong><br>
    The remaining <strong>{companion_count:,}</strong> skills ({pct_companion:.1f}%) reference companion files
    in their SKILL.md text, indicating they were designed with a richer directory structure:
    <strong>{scripts_count:,}</strong> reference <code>scripts/</code>,
    <strong>{references_count:,}</strong> reference <code>references/</code>, and
    <strong>{assets_count:,}</strong> reference <code>assets/</code>.
  </div>

  <p><strong>Important caveat:</strong> While these skills reference sub-folders and companion files in their
     SKILL.md instructions (e.g., &ldquo;Run <code>scripts/extract.py</code>&rdquo;), the actual content of
     those files is <strong>not included in the dataset</strong>. The dataset only provides the SKILL.md text.
     This means the retrieval benchmark operates exclusively on SKILL.md content, missing the richer
     context that ~{pct_companion:.0f}% of skills were designed to include.</p>
</div>

{charts_html(1, 8)}

<div class="section">
  <h2>3. Taxonomy & Categories</h2>
  <p>The SKILLRET taxonomy organizes skills into <strong>6 major categories</strong> and <strong>18 sub-categories</strong>:</p>
  <table class="stats">
    <tr><th>Major Category</th><th>Sub-Categories</th><th>Skill Count</th></tr>
    {"".join(f'<tr><td><strong>{cat["major"]}</strong></td><td>{", ".join(s["sub"] for s in cat["subs"])}</td><td>{major_counts.get(cat["major"], 0):,}</td></tr>' for cat in taxonomy["taxonomy"])}
  </table>
  <p style="margin-top:0.8rem;"><strong>Software Engineering</strong> dominates with {major_counts.get("Software Engineering", 0):,} skills
     ({major_counts.get("Software Engineering", 0)/total_skills*100:.1f}%), followed by
     <strong>AI Agents</strong> ({major_counts.get("AI Agents", 0):,}) and
     <strong>Data &amp; ML</strong> ({major_counts.get("Data & ML", 0):,}).</p>
</div>

{charts_html(8, 12)}

<div class="section">
  <h2>4. Skill Metadata Statistics</h2>
  <div class="two-col">
    <div>
      <h3>GitHub Stars</h3>
      <table class="stats">
        <tr><th>Stat</th><th>Value</th></tr>
        <tr><td>Mean</td><td>{stars_series.mean():,.0f}</td></tr>
        <tr><td>Median</td><td>{median_stars:,}</td></tr>
        <tr><td>Max</td><td>{stars_series.max():,}</td></tr>
        <tr><td>Min</td><td>{stars_series.min():,}</td></tr>
        <tr><td>Std Dev</td><td>{stars_series.std():,.0f}</td></tr>
      </table>
    </div>
    <div>
      <h3>Install Count</h3>
      <table class="stats">
        <tr><th>Stat</th><th>Value</th></tr>
        <tr><td>Mean</td><td>{installs_series.mean():,.0f}</td></tr>
        <tr><td>Median</td><td>{median_installs:,}</td></tr>
        <tr><td>Max</td><td>{installs_series.max():,}</td></tr>
        <tr><td>Min</td><td>{installs_series.min():,}</td></tr>
        <tr><td>Std Dev</td><td>{installs_series.std():,.0f}</td></tr>
      </table>
    </div>
  </div>
  <div class="two-col" style="margin-top:1rem;">
    <div>
      <h3>SKILL.md Content Length</h3>
      <table class="stats">
        <tr><th>Stat</th><th>Value</th></tr>
        <tr><td>Mean</td><td>{mean_body_len:,} chars</td></tr>
        <tr><td>Median</td><td>{median_body_len:,} chars</td></tr>
        <tr><td>Max</td><td>{max_body_len:,} chars</td></tr>
        <tr><td>P25</td><td>{int(df_skills["skill_md_len"].quantile(0.25)):,} chars</td></tr>
        <tr><td>P75</td><td>{int(df_skills["skill_md_len"].quantile(0.75)):,} chars</td></tr>
      </table>
    </div>
    <div>
      <h3>Description Length</h3>
      <table class="stats">
        <tr><th>Stat</th><th>Value</th></tr>
        <tr><td>Mean</td><td>{df_skills["desc_len"].mean():.0f} chars</td></tr>
        <tr><td>Median</td><td>{int(df_skills["desc_len"].median()):,} chars</td></tr>
        <tr><td>Max</td><td>{int(df_skills["desc_len"].max()):,} chars</td></tr>
        <tr><td>P25</td><td>{int(df_skills["desc_len"].quantile(0.25)):,} chars</td></tr>
        <tr><td>P75</td><td>{int(df_skills["desc_len"].quantile(0.75)):,} chars</td></tr>
      </table>
    </div>
  </div>
</div>

{charts_html(12, 17)}

<!-- Section 5: Queries -->
<div class="section">
  <h2>5. Queries Component</h2>
  <p>Queries are synthetic natural-language user requests designed to match one or more skills.
     They were generated by two different LLMs:</p>
  <ul>
    <li><strong>Test split ({len(queries_test):,} queries)</strong> &mdash; generated by <strong>Claude Opus 4.6</strong>.
        These are the evaluation queries, typically longer and more complex.</li>
    <li><strong>Train split ({len(queries_train):,} queries)</strong> &mdash; generated by <strong>Qwen3.5-122B-A10B</strong>.
        These are the synthetic training queries.</li>
  </ul>

  <div class="two-col">
    <div>
      <h3>Test Query Stats</h3>
      <table class="stats">
        <tr><th>Stat</th><th>Value</th></tr>
        <tr><td>Count</td><td>{len(queries_test):,}</td></tr>
        <tr><td>Mean length</td><td>{test_q_mean_words} words</td></tr>
        <tr><td>Median length</td><td>{test_q_med_words} words</td></tr>
        <tr><td>Mean chars</td><td>{df_queries_test["query_len"].mean():,.0f}</td></tr>
        <tr><td>Generator</td><td>Claude Opus 4.6</td></tr>
      </table>
    </div>
    <div>
      <h3>Train Query Stats</h3>
      <table class="stats">
        <tr><th>Stat</th><th>Value</th></tr>
        <tr><td>Count</td><td>{len(queries_train):,}</td></tr>
        <tr><td>Mean length</td><td>{train_q_mean_words} words</td></tr>
        <tr><td>Median length</td><td>{train_q_med_words} words</td></tr>
        <tr><td>Mean chars</td><td>{df_queries_train["query_len"].mean():,.0f}</td></tr>
        <tr><td>Generator</td><td>Qwen3.5-122B-A10B</td></tr>
      </table>
    </div>
  </div>

  <p style="margin-top:1rem;">Each query record includes:</p>
  <ul>
    <li><code>id</code> / <code>original_id</code> &mdash; unique query identifier</li>
    <li><code>query</code> &mdash; the full natural-language request text</li>
    <li><code>skill_ids</code> &mdash; list of relevant skill UUIDs</li>
    <li><code>skill_names</code> &mdash; human-readable names of relevant skills</li>
    <li><code>k</code> &mdash; number of relevant skills for this query</li>
    <li><code>generator_model</code> &mdash; which LLM generated the query</li>
  </ul>
</div>

{charts_html(17, 20)}

<!-- Section 6: Qrels -->
<div class="section">
  <h2>6. Qrels (Relevance Judgments)</h2>
  <p>Qrels (query relevance judgments) are the ground-truth labels that link queries to their correct skills.
     Each qrel is a triple: <code>(query_id, skill_id, relevance)</code> where relevance is binary (1 = relevant).</p>

  <div class="two-col">
    <div>
      <h3>Test Qrels</h3>
      <table class="stats">
        <tr><th>Stat</th><th>Value</th></tr>
        <tr><td>Total judgments</td><td>{len(qrels_test):,}</td></tr>
        <tr><td>Unique queries</td><td>{df_qrels_test["query_id"].nunique():,}</td></tr>
        <tr><td>Unique skills</td><td>{df_qrels_test["skill_id"].nunique():,}</td></tr>
        <tr><td>Mean qrels/query</td><td>{qrels_test_per_query.mean():.1f}</td></tr>
        <tr><td>Max qrels/query</td><td>{qrels_test_per_query.max()}</td></tr>
      </table>
    </div>
    <div>
      <h3>Train Qrels</h3>
      <table class="stats">
        <tr><th>Stat</th><th>Value</th></tr>
        <tr><td>Total judgments</td><td>{len(qrels_train):,}</td></tr>
        <tr><td>Unique queries</td><td>{df_qrels_train["query_id"].nunique():,}</td></tr>
        <tr><td>Unique skills</td><td>{df_qrels_train["skill_id"].nunique():,}</td></tr>
        <tr><td>Mean qrels/query</td><td>{qrels_train_per_query.mean():.1f}</td></tr>
        <tr><td>Max qrels/query</td><td>{qrels_train_per_query.max()}</td></tr>
      </table>
    </div>
  </div>
</div>

{charts_html(20)}

<!-- Section 7: Train/Test Split -->
<div class="section">
  <h2>7. Train / Test Split Design</h2>
  <p>The dataset uses a <strong>disjoint skill split</strong>: train and test skills do NOT overlap.
     This ensures that evaluation measures true generalization to unseen skills, not memorization.</p>
  <ul>
    <li><strong>Train skills:</strong> {len(skills_train):,} skills with {len(queries_train):,} queries and {len(qrels_train):,} qrels</li>
    <li><strong>Test skills:</strong> {len(skills_test):,} skills with {len(queries_test):,} queries and {len(qrels_test):,} qrels</li>
    <li><strong>Out-of-split skills:</strong> {total_skills - len(skills_train) - len(skills_test):,} skills
        in the full corpus that appear in neither train nor test splits (serve as distractors)</li>
  </ul>
</div>

<!-- Section 8: Key findings summary -->
<div class="section">
  <h2>8. Key Findings Summary</h2>
  <div class="highlight">
    <strong>1. Majority of skills are bare SKILL.md:</strong> {pct_bare:.1f}% ({bare_count:,}) of skills consist of only
    a SKILL.md file with no companion sub-folders. The remaining {pct_companion:.1f}% ({companion_count:,}) were designed
    with <code>scripts/</code>, <code>references/</code>, or <code>assets/</code> sub-folders, but the dataset only
    captured the SKILL.md content &mdash; the actual sub-file contents are not included.
  </div>
  <div class="highlight">
    <strong>2. <code>references/</code> is the most common sub-folder:</strong> {references_count:,} skills reference a
    <code>references/</code> directory ({references_count/total_skills*100:.1f}%), followed by <code>scripts/</code>
    ({scripts_count:,}, {scripts_count/total_skills*100:.1f}%) and <code>assets/</code> ({assets_count:,},
    {assets_count/total_skills*100:.1f}%).
  </div>
  <div class="highlight">
    <strong>3. Software Engineering dominance:</strong> {major_counts.get("Software Engineering", 0)/total_skills*100:.1f}%
    of all skills fall under Software Engineering, with Development being the largest sub-category.
    AI Agents skills are the second largest group.
  </div>
  <div class="highlight">
    <strong>4. Long-tail author distribution:</strong> {total_authors:,} unique authors contributed skills,
    but the distribution is heavily skewed &mdash; a few prolific authors contribute hundreds of skills
    while most authors contribute just one or two.
  </div>
  <div class="highlight">
    <strong>5. Query complexity differs by split:</strong> Test queries (Claude Opus 4.6) average {test_q_mean_words} words,
    while train queries (Qwen3.5-122B-A10B) average {train_q_mean_words} words.
    Both tend to be detailed, multi-sentence requests rather than simple keyword searches.
  </div>
  <div class="highlight">
    <strong>6. Mostly single-skill queries:</strong> The vast majority of queries map to just 1-2 relevant skills,
    making this a precision-focused retrieval benchmark.
  </div>
  <div class="highlight">
    <strong>7. Dataset limitation &mdash; no sub-file content:</strong> The dataset provides <code>body</code> and
    <code>skill_md</code> fields that are byte-identical for all 17,810 records. Skills that were designed
    with scripts, references, or assets folders lose that context in this dataset. A retrieval model trained
    on SKILLRET can only match based on SKILL.md text, not the full skill package.
  </div>
</div>

</div>

<div class="footer">
  SKILLRET EDA Report &mdash; Generated from ThakiCloud/SKILLRET dataset
</div>

</body>
</html>"""

OUT.write_text(html)
print(f"\nReport written to {OUT} ({len(html):,} chars)")
