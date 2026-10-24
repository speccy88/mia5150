import json,math
import numpy as np
import pytest
from core import *

def test_per_query_nested_library_and_display():
    ids=[str(x) for x in range(100)]
    a=library(ids,['99'],'q',10,42);b=library(ids,['99'],'q',50,42)
    assert len(a)==10 and set(a)<=set(b) and '99' in a
    assert len(library(ids,[str(x) for x in range(20)],'other',20,42))==20
    assert display('q',a,42)==display('q',a,42)
    assert library(ids,['99'],'q',10,42)!=library(ids,['99'],'q',10,43)

def test_duplicate_names_do_not_collide_or_truncate_300():
    docs={'one':'same | '+'z'*900,'two':'same | '+'y'*900}
    text,m=select_prompt('task',list(docs),docs,'q',42)
    assert len(m)==2 and 'z'*900 in text and alias('q','one')!=alias('q','two')

@pytest.mark.parametrize('text',['["a","a"]','["unknown"]','{"ids":["a"]}','```json\n["a"]\n```','[1]','["a","b","c","d","e","f"]'])
def test_reject_bad_output(text):
    with pytest.raises((ValueError,TypeError)):parse_selection(text,{'a':'one'})

def test_set_and_ranking_known_answers():
    assert retrieval_metrics(['a','a','b'],['b','c'])['recall5']==.5
    assert retrieval_metrics(['a','a','b'],['b'])['mrr5']==.5
    assert selection_metrics(['a','x'],['a','b'])['f1']==.5
    assert selection_metrics(['a'],['a'],'invalid_selection')['exact']==0
    assert selection_metrics([],['a'])['fn']==1

def test_bm25_idf_is_catalog_local_and_ties_by_id():
    m=BM25({'a':'foo foo','b':'bar','c':'foo bar','d':'bar bar'})
    assert m.search('foo',['a','b'])==['a','b']
    assert m.search('absent',['d','a','c'])==['a','c','d']
    # Independent scalar formula oracle, with two query terms.
    docs={'a':'foo foo','b':'bar','c':'foo bar','d':'bar bar'};scores={}
    for d,text in docs.items():
        tokens=text.split();scores[d]=0
        for term in ['foo','bar']:
            df=sum(term in x.split() for x in docs.values());tf=tokens.count(term)
            scores[d]+=math.log(1+(4-df+.5)/(df+.5))*tf*2.5/(tf+1.5*(.25+.75*len(tokens)/1.75))
    assert m.search('foo bar',list(docs))==rank(scores,list(docs))

def test_rrf_and_empty_domain():
    assert rrf(['a','b'],['b','c'])[0]=='b'
    assert rrf([],[])==[]
    assert retrieval_metrics([],['a'])['recall5']==0

def test_leakage_boundary():
    s={'id':'a','name':'name','description':'short','skill_md':'BODY_SECRET','skill_ids':['GOLD_SECRET'],'k':4}
    p,m=select_prompt('canonical task',['a'],{'a':render(s)},'qid',42)
    assert 'BODY_SECRET' not in p and 'GOLD_SECRET' not in p and '"k"' not in p
    assert 'short' in p and s['id'] not in m

def test_description_cap_preserves_name():
    assert render({'name':'longname','description':'a'*1100})=='longname | '+'a'*1024

def test_failed_cases_remain_zero_denominator():
    values=[selection_metrics(['a'],['a']),selection_metrics([],['a'],'timeout')]
    assert sum(x['exact'] for x in values)/2==.5

def test_cutoff_five_vs_six_and_three_positives():
    assert retrieval_metrics(['a','b','c','d','gold','other'],['gold'])['mrr5']==.2
    assert retrieval_metrics(['a','b','c','d','other','gold'],['gold'])['mrr5']==0
    assert retrieval_metrics(['a','b'],['a','b','c'])['recall5']==2/3
    assert selection_metrics(['a','b','x'],['a','b','c'])['exact']==0

def test_unknown_not_corrected_and_name_normalization_collision():
    d={'uuid1':'Foo-Bar | skill one','uuid2':'foo bar | skill two'}
    p,m=select_prompt('request',list(d),d,'q',42)
    assert len(m)==2
    with pytest.raises(ValueError):parse_selection('["'+next(iter(m))+'typo"]',m)
