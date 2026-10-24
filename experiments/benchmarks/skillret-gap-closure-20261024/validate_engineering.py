"""Record actual local test/compile outcomes without model or API execution."""
import datetime,json,pathlib,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
def main():
    commands=[('pytest',[sys.executable,'-m','pytest','-q',str(HERE)]),('python_compileall',[sys.executable,'-m','compileall','-q',str(HERE)])]
    results=[]
    for name,args in commands:
        run=subprocess.run(args,cwd=HERE,capture_output=True,text=True)
        results.append({'check':name,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
    receipt={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python_version':sys.version.split()[0],'status':'passed' if all(r['exit_code']==0 for r in results) else 'failed','checks':results,'scope':'Synthetic/local scoring, equivalence, persistence and hosted-budget contract tests plus Python compilation. No inference, paid request, human validation or TeX-build success claimed.'}
    (HERE/'report/engineering-validation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));raise SystemExit(0 if receipt['status']=='passed' else 1)
if __name__=='__main__':main()
