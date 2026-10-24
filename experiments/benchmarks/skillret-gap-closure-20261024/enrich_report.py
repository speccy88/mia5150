"""Add sensitivity tables, readable typography and reviewable static slide source."""
import json,pathlib
HERE=pathlib.Path(__file__).resolve().parent;OUT=HERE/'report';ROOT=HERE.parents[2]
def main():
    data=json.loads((OUT/'summary.json').read_text())
    p=OUT/'RESEARCH_DRAFT.md';text=p.read_text()
    replacements={'contains200':'contains 200','Additional24':'Additional 24','sensitivity,12':'sensitivity, 12','full4392':'full 4,392','At6006':'At 6,006','topfive':'top five','matched9/200':'matched 9/200','with7/200':'with 7/200','with11/200':'with 11/200','returned4/200':'returned 4/200','constrained0.6B':'constrained 0.6B','mean411':'mean 411','at6006':'at 6,006','all200':'all 200','configured8192':'configured 8,192','At100':'At 100','skills,196':'skills, 196','difference at6006':'difference at 6,006','about+1.02':'about +1.02','stratified95%':'stratified 95%','official1/2/3':'official 1/2/3','weights2143':'weights 2143','denominator200':'denominator 200','coverage1':'coverage 1','RTX2060SUPER':'RTX 2060 SUPER','with8GB':'with 8 GB','and16GB':'and 16 GB','at6GB':'at 6 GB','with2GB':'with 2 GB','difference1.23':'difference 1.23','original80':'original 80','All4392':'All 4,392','this200-query':'this 200-query','balanced24-query':'balanced 24-query','DeepSeekV4Flash0731/DeepInfraFP8':'DeepSeek V4 Flash 0731 / DeepInfra FP8','tighterUS$':'tighter US$','complete24-query':'complete 24-query','reserves14032062':'reserves 14,032,062','and74112':'and 74,112','over579':'over 579','estimatesUS$':'estimates US$','before25%':'before 25%','aliases:deepujoy88,speccy88,vsimard':'aliases: deepujoy88, speccy88, vsimard'}
    for a,b in replacements.items():text=text.replace(a,b)
    sections=[('Matched balanced model sensitivity at 6,006 skills',[r for r in data['selection'] if r['scope']=='balanced24_seed42' and r['size']==6006]),('Generated silver-label probes',[r for r in data['selection'] if r['scope']=='generated12_silver'])]
    for title,rows in sections:
        text+='\n## '+title+'\n\n|Model|N|Method|Planned|Valid|Unattempted|Exact sets|Macro F1|\n|---|---:|---|---:|---:|---:|---:|---:|\n'
        for r in rows:
            score='pending' if r['not_run'] else f"{r['macro_f1']:.4f}"
            text+=f"|{r['model']}|{r['size']}|{r['method']}|{r['planned']}|{r['valid']}|{r['not_run']}|{r['exact_successes']}|{score}|\n"
    text+='\nBalanced sensitivity is distinct from the primary prevalence-weighted estimate. Seeds do not add independent requests. Generated labels are silver unions; neither table establishes workflow order or human agreement. Inspect per-cell attempt status before interpreting any pending row.\n'
    p.write_text(text)
    paper=ROOT/'paper/final_report.tex';tex=paper.read_text()
    for a,b in replacements.items():tex=tex.replace(a,b)
    tex=tex.replace(r'\section{Method}',r'\section{Related work}SkillRet supplies the retrieval benchmark \cite{kang2026skillret}. Skill retrieval augmentation \cite{su2026skill}, skill shadowing \cite{song2026shadowing} and dependency-aware structural retrieval \cite{liu2026graph} motivate distinct retrieval, selection and workflow questions. Their published results are not new measurements here.'+'\n'+r'\section{Method}')
    tex=tex.replace(r'\end{document}',r'\bibliographystyle{plain}\bibliography{references}'+'\n'+r'\appendix\section{Audit and reproduction}The study contains frozen protocols, exact catalogs, source snapshots, full raw outputs and compact independently scoreable cells. Run aggregate.py, plot.py, compact\_audit.py and make\_report.py. The original 80-cell crosswalk and 30-row closure matrix retain pending human and hosted work. The balanced model/seed and generated-probe tables are in the accompanying report.'+'\n'+r'\end{document}')
    paper.write_text(tex)
    prefix='../../experiments/benchmarks/skillret-gap-closure-20261024/report/'
    slides=[r'\documentclass[aspectratio=169]{beamer}',r'\usepackage{graphicx}',r'\usetheme{Boadilla}',r'\setbeamertemplate{navigation symbols}{}',r'\title{Skill Discovery: Evidence and Remaining Gaps}',r'\author{deepujoy88 \and speccy88 \and vsimard}',r'\date{Research draft --- 24 October 2026}',r'\begin{document}\frame{\titlepage}']
    frames=[('Question and proposal','Five methods; original catalog sizes 10/100/1,000/6,006.\\Separate retrieval, final selection and execution.\\Original 80 hosted cells remain explicitly tracked.'),('Frozen design','200 exposed requests; 98/78/24 positive-count strata.\\Pinned metadata, models and per-query catalogs.\\Strict opaque IDs; no tools or filesystem.'),('Final selection','selection-heatmap.png'),('Retrieval versus selection','candidate-recall.png'),('Context and tokens','input-growth.png'),('Paired uncertainty','paired-differences.png'),('Sensitivity and scenarios','Balanced 24-query model/seed sensitivity is separate.\\Generated 12-probe labels are silver unions.\\No dependency order, human agreement or task success is claimed.'),('Reproducibility and resources','Raw outputs, source hashes, strict scoring and small per-cell audit files.\\One GPU job; 8 GB VRAM; CPU encoder; preserved RAM interruptions.\\Hosted stage blocked on credentials; no paid calls.'),('Next evidence and team review','Fresh independent requests and human semantic agreement.\\Workflow ordering and activation/execution instrumentation.\\Team contributions and speaking allocation require confirmation.')]
    for title,body in frames:
        if body.endswith('.png'):body=r'\centering\includegraphics[height=.75\textheight]{'+prefix+body+'}'
        slides += [r'\begin{frame}{'+title+'}',body,r'\end{frame}']
    slides += [r'\end{document}'];(ROOT/'paper/presentation/research_draft.tex').write_text('\n'.join(slides)+'\n')
    print('Sensitivity tables, bibliography, audit appendix and static Beamer draft added')
if __name__=='__main__':main()
