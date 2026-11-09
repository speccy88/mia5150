"""Source selection must preserve the entire query and reserved scoring suffix."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from render import *
from transformers import AutoTokenizer

def fixture_renderer(body=None):
 tok=AutoTokenizer.from_pretrained(CACHE/'models/reranker',local_files_only=True,padding_side='left')
 r=object.__new__(Renderer);r.tok=tok;r.idf={'rareword':5.};r.prefix=tok.encode(PREFIX_R,add_special_tokens=False);r.suffix=tok.encode(SUFFIX,add_special_tokens=False)
 if body is None:body=('An ordinary paragraph. '*300)+' rareword '+('Different evidence. '*300)
 enc=tok(body,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];offsets=enc['offset_mapping'];windows=[]
 for start in range(0,len(ids),224):
  end=min(len(ids),start+256);a,b=offsets[start][0],offsets[end-1][1];windows.append({'start':a,'end':b,'token_start':start,'tokens':ids[start:end],'term_counts':dict(Counter(words(body[a:b]))),'sha256':digest(body[a:b])})
  if end==len(ids):break
 record={'body_sha256':digest(body),'metadata_ids':tok.encode('skill | description | ',add_special_tokens=False),'body_characters':len(body),'body_ids':ids,'offsets':offsets,'windows':windows};r.record=lambda sid:record
 return r,body

def test_selected_windows_are_label_blind_and_query_complete():
 r,body=fixture_renderer();query='rareword relevant evidence';a=r.render(query,'positive',1024);b=r.render(query,'negative',1024);assert a==b
 ids,meta=a[0];assert ids[-len(r.suffix):]==r.suffix and len(ids)<=1024
 full_query=r.tok.encode(f'<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: ',add_special_tokens=False);assert ids[len(r.prefix):len(r.prefix)+len(full_query)]==full_query
 assert any('rareword' in body[x:y] for x,y in meta['spans']);assert all(0<=x<y<=len(body) for x,y in meta['spans'])

def test_full_coverage_windows_cover_all_source_tokens_without_window_labels():
 r,body=fixture_renderer();pieces=r.render('rareword','any',1024,True);coverage=[]
 for ids,meta in pieces:
  assert len(ids)<=1024 and ids[-len(r.suffix):]==r.suffix;assert 'positive' not in meta
  coverage+=meta['spans']
 assert coverage[0][0]==0 and coverage[-1][1]==len(body)
 assert all(a[1]>=b[0] for a,b in zip(coverage,coverage[1:]))

def test_complete_short_skill_body_is_retained():
 r,body=fixture_renderer('Read CSV rows.')
 record=r.record('any')
 assert len(record['body_ids'])<32
 ids,meta=r.render('Read a table','any',1024)[0]
 assert meta['spans']==[[0,len(body)]]
 assert ids[-len(r.suffix)-len(record['body_ids']):-len(r.suffix)]==record['body_ids']
