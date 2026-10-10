import subprocess,sys
import pytest
@pytest.mark.parametrize('script',['fetch','build','train','infer','score','pilot_report','demo','export_results'])
def test_cli_help_does_not_start_work(script):
    p=subprocess.run([sys.executable,'-m',f'scripts.{script}','--help'],capture_output=True,text=True)
    assert p.returncode==0 and 'usage:' in p.stdout.lower()

def test_training_requires_approval(tmp_path):
    p=subprocess.run([sys.executable,'-m','scripts.train','--data','not-real','--out',str(tmp_path/'run'),
                      '--config','configs/pilot.json'],capture_output=True,text=True)
    assert p.returncode!=0 and 'approve-gpu-run' in p.stderr and not (tmp_path/'run').exists()
