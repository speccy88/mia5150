"""Sequential evidence-only finalization. Never starts training or downloads."""
import pathlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parent;py=str(R/'.venv/bin/python')
for script in ['compare.py','statistics_report.py','retrieval_audit.py','performance_report.py','audit_finish.py','finish_exposure.py','make_demo.py','scope_receipts.py','resource_summary.py','write_report.py']:
 print('Finalizing',script,flush=True)
 subprocess.run([py,script],cwd=R,check=True)
subprocess.run([py,'-m','pytest','-q'],cwd=R,check=True)
print('Analysis finalized; visual QA and packaging still required.',flush=True)
