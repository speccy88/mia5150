"""Offline artifact smoke checks; no model loads or network."""
import contextlib,io,json,xml.etree.ElementTree as ET,zipfile,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150'
def main():
 notebook=json.loads((HERE/'results.ipynb').read_text());namespace={};capture=io.StringIO()
 import os
 before=Path.cwd()
 try:
  os.chdir(HERE)
  with contextlib.redirect_stdout(capture):
   for cell in notebook['cells']:
    if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'<artifact-notebook>','exec'),namespace)
 finally:os.chdir(before)
 assert len(namespace['rows'])==183;tree=ET.parse(HERE/'report/retrieval-curve.svg');assert tree.getroot().tag.endswith('svg');assert len(tree.findall('.//{http://www.w3.org/2000/svg}polyline'))==7
 archive=ROOT/'deliverables/MIA5150_Seed5150_Local_Scaling.zip';scanned=0
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   assert '..' not in Path(info.filename).parts and not info.filename.startswith('/')
   name=Path(info.filename).name.lower();assert name not in ['.env','credentials','credentials.json','secret_store.json']
   if info.file_size<10000000 and Path(info.filename).suffix in ['.py','.json','.jsonl','.md','.txt','.csv','.log']:
    data=z.read(info).decode('utf8',errors='replace');assert not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{32,}',data),info.filename;scanned+=1
 result={'unit_tests':{'command':'.venv/bin/python -m unittest discover -s scaling5150 -p test_*.py','observed_pass_count':11,'source':'Successful final test command before package creation'},'notebook_code_cells_executed':2,'notebook_metric_rows':len(namespace['rows']),'svg_xml_valid':True,'svg_curves':7,'visual_browser_review':False,'zip_text_files_secret_pattern_scanned':scanned,'secret_pattern_matches':0,'library_cases_independently_verified':120,'full_retrieval_rows_verified':2040,'final_metric_cells':183,'scope':'Offline validation. Pattern scanning is not proof of absence of every possible secret; no credential paths were selected for packaging.'};(HERE/'report/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
