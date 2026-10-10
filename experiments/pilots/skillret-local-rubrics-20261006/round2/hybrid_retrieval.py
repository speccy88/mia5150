"""Reusable SkillRet + Jev Score RRF60; label-free, dependency-free core.

Production fallback and strict benchmark output are separate. No transport,
credential source or budget is implicit. Cached replay never opens the network.
"""
from dataclasses import dataclass
from decimal import Decimal
import json,math,os,time
from pathlib import Path
from probe_contract import build_request,MODEL

def validate_pool(pool):
    if len(pool)!=20 or len(set(pool))!=20 or not all(isinstance(x,str) and x for x in pool):
        raise ValueError('Exactly20 unique skill IDs required')

def score_order(pool,scores):
    validate_pool(pool)
    if len(scores)!=len(pool) or any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) or not 0<=s<=3 for s in scores):
        raise ValueError('Complete finite Score values required')
    return sorted(pool,key=lambda d:(-scores[pool.index(d)],pool.index(d)))

def rrf60(baseline,reranked):
    validate_pool(baseline)
    if len(reranked)!=20 or set(reranked)!=set(baseline):raise ValueError('Candidate pool changed')
    scores={d:1/(61+i) for i,d in enumerate(baseline)}
    for i,d in enumerate(reranked):scores[d]+=1/(61+i)
    return sorted(baseline,key=lambda d:(-scores[d],baseline.index(d)))

def score_request(query,headers):
    packet=build_request(query,headers)
    packet['questions']={k:v for k,v in packet['questions'].items() if k.startswith('s')}
    return packet

def parse_scores(response,pool):
    validate_pool(pool)
    if response.get('model')!=MODEL:raise ValueError('Unpinned model')
    tokens=response['usage']['input_tokens']
    if isinstance(tokens,bool) or not isinstance(tokens,int) or not 0<=tokens<=64000:raise ValueError('Invalid token usage')
    answers=response['answers']
    if set(answers)!={f's{i}' for i in range(20)}:raise ValueError('Missing or extra answers')
    values=[]
    for i in range(20):
        answer=answers[f's{i}']
        if answer.get('type')!='score':raise ValueError('Wrong answer type')
        values.append(answer['score'])
    return score_order(pool,values),values,tokens

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

class SharedBudget:
    """WSL-only batch reservation. Hold the shared lock for the entire batch.

    Requires an existing ledger; reserves before any transmission and records
    each attempt before dispatch. Reopening a used run ID is refused. Reservations
    never release automatically, including failed or uncertain calls.
    """
    def __init__(self,ledger_path,run_id,reserve_usd,max_calls):
        self.path=Path(ledger_path);self.run_id=run_id;self.amount=Decimal(str(reserve_usd));self.max_calls=max_calls;self.stopped=False;self.active=False
        if not self.amount.is_finite() or self.amount<=0 or isinstance(max_calls,bool) or not isinstance(max_calls,int) or max_calls<1:raise ValueError('Invalid reservation')
        # Pinned worst-case usage plus conservative headroom; no rate override.
        if self.amount<Decimal(max_calls)*64000*Decimal('0.042')/1000000:raise ValueError('Reservation below token bound')
    def __enter__(self):
        import fcntl
        self.lock=self.path.with_name('budget.lock').open('a')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.ledger=json.loads(self.path.read_text())
            if self.ledger.get('paid_tests_paused',False):raise ValueError('Paid experiments paused pending review')
            cap=Decimal(self.ledger['cap_usd']);committed=Decimal(self.ledger['committed_usd']);external=Decimal(self.ledger['external_spend_usd'])
            if any(not x.is_finite() or x<0 for x in (cap,committed,external)) or cap!=4 or committed+external+self.amount>cap:raise ValueError('Shared budget denied')
            if self.run_id in self.ledger['runs']:raise ValueError('Run ID already reserved')
            self.entry={'reserved_usd':str(self.amount),'status':'reserved','attempt_ids':[],'max_calls':self.max_calls}
            self.ledger['runs'][self.run_id]=self.entry;self.ledger['committed_usd']=str(committed+self.amount)
            atomic_json(self.path,self.ledger);self.active=True;return self
        except BaseException:self.lock.close();raise
    def before_call(self,query_id):
        if not self.active or self.stopped:raise ValueError('Inactive or stopped budget')
        if query_id in self.entry['attempt_ids'] or len(self.entry['attempt_ids'])>=self.max_calls:raise ValueError('Duplicate or excess call')
        self.entry['attempt_ids'].append(query_id);atomic_json(self.path,self.ledger)
    def stop(self):self.stopped=True
    def __exit__(self,*exc):
        try:
            self.entry['status']='finished_or_stopped';atomic_json(self.path,self.ledger)
        finally:self.active=False;self.lock.close()

class CachedRetriever:
    """Injected ranking cache; works with any existing dense retrieval backend."""
    def __init__(self,rankings):self.rankings=rankings
    def retrieve(self,query):
        pool=list(self.rankings[query['id']][:20]);validate_pool(pool);return pool

class CachedJev:
    def __init__(self,rows):self.rows={r['id']:r for r in rows}
    def rank(self,query,headers):
        row=self.rows[query['id']];pool=[x['id'] for x in headers]
        if row['status']!='ok' or row['candidate_ids']!=pool:raise ValueError('Missing/mismatched cache')
        scores=row['scores']['score'] if isinstance(row['scores'],dict) else row['scores']
        return score_order(pool,scores),{'source':'cached','input_tokens':row['input_tokens'],'historical_api_seconds':row.get('api_seconds',row.get('seconds'))}

class JevScoreAdapter:
    """Explicit transport(packet)->dict and active SharedBudget required.

    Adapter does not read credentials, choose providers, retry or load labels.
    Bind the already approved fixed TypeSafe transport only in a reviewed runner.
    """
    def __init__(self,transport,budget):self.transport=transport;self.budget=budget
    def rank(self,query,headers):
        packet=score_request(query,headers);self.budget.before_call(query['id'])
        start=time.monotonic()
        try:
            ranking,values,tokens=parse_scores(self.transport(packet),[h['id'] for h in headers])
            return ranking,{'source':'live','input_tokens':tokens,'api_seconds':time.monotonic()-start}
        except BaseException as error:
            self.budget.stop()
            # Close HTTP errors without reflecting response/error contents.
            if hasattr(error,'close'):error.close()
            raise RuntimeError('Jev adapter stopped') from None

@dataclass
class HybridResult:
    baseline:list
    reranked:list
    fused:list
    served:list
    strict_benchmark:list
    status:str
    metadata:dict

class HybridRetriever:
    def __init__(self,retriever,header_by_id,jev):self.retriever=retriever;self.headers=header_by_id;self.jev=jev
    def search(self,query):
        if set(query)!={'id','query'}:raise ValueError('Canonical query only; labels forbidden')
        baseline=self.retriever.retrieve(query)
        try:
            headers=[{k:self.headers[d][k] for k in ('id','name','description')} for d in baseline]
            reranked,metadata=self.jev.rank(query,headers);fused=rrf60(baseline,reranked)
            return HybridResult(baseline,reranked,fused,fused,fused,'ok',metadata)
        except Exception:
            # User-serving fallback is never disguised as successful reranking.
            return HybridResult(baseline,[],[],baseline,[],'baseline_fallback',{'error':'Rerank unavailable or invalid; sanitized','fallback_used':True})
