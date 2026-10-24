"""Locked reservations; unknown charges stop unless explicitly bounded by evidence."""
import contextlib,decimal,fcntl,json,os,pathlib
D=decimal.Decimal
class BudgetStop(RuntimeError):pass
class Budget:
    def __init__(self,path,usd_cap,cad_cap,cad_per_usd,call_cap,model_caps=None,legacy_model=None):
        self.path=pathlib.Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.usd_cap=D(str(usd_cap));self.cad_cap=D(str(cad_cap));self.fx=D(str(cad_per_usd));self.call_cap=int(call_cap)
        self.model_caps={k:D(str(v)) for k,v in (model_caps or {}).items()};self.legacy_model=legacy_model
    @contextlib.contextmanager
    def lock(self):
        with self.path.with_suffix('.lock').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)
    def events(self):
        if not self.path.exists():return []
        with self.path.open() as f:return [json.loads(x) for x in f if x.strip()]
    def append(self,event):
        with self.path.open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
    def state(self):
        current={}
        for event in self.events():
            if event['kind']=='reserve':current[event['case_key']]=dict(event)
            else:current[event['case_key']].update(event)
        held=sum((D(x.get('actual_usd') or x['reserve_usd']) for x in current.values()),D(0))
        return current,held
    def reserve(self,key,amount,model=None,run_id=None):
        amount=D(str(amount))
        if not amount.is_finite() or amount<0:raise BudgetStop('Invalid reserve')
        with self.lock():
            current,held=self.state()
            if key in current:raise BudgetStop('Case already reserved/attempted; no duplicate generation')
            if any(x.get('kind') in ['unknown','reserve'] for x in current.values()):raise BudgetStop('Unknown/in-flight charge retained; reconcile before expansion')
            if len(current)>=self.call_cap:raise BudgetStop('Request cap')
            if held+amount>self.usd_cap or (held+amount)*self.fx>self.cad_cap:raise BudgetStop('USD/CAD cap')
            if self.model_caps:
                if model not in self.model_caps:raise BudgetStop('Model not allocated')
                model_held=sum((D(x.get('actual_usd') or x['reserve_usd']) for x in current.values() if (x.get('model') or self.legacy_model)==model),D(0))
                if model_held+amount>self.model_caps[model]:raise BudgetStop('Per-model cap')
            self.append({'kind':'reserve','case_key':key,'reserve_usd':str(amount),'actual_usd':None,'model':model,'run_id':run_id})
    def settle(self,key,actual):
        with self.lock():
            current,held=self.state()
            if key not in current:raise BudgetStop('Missing reservation')
            if actual is None:self.append({'kind':'unknown','case_key':key,'actual_usd':None});raise BudgetStop('Unknown charge retained; expansion halted')
            value=D(str(actual))
            if isinstance(actual,bool) or not value.is_finite() or value<0:raise BudgetStop('Invalid returned cost')
            self.append({'kind':'settled','case_key':key,'actual_usd':str(value)})
            _,total=self.state()
            if total>self.usd_cap or total*self.fx>self.cad_cap:raise BudgetStop('Returned charge exceeded cap; stop')
            if self.model_caps:
                current,_=self.state();model=current[key].get('model') or self.legacy_model
                model_total=sum((D(x.get('actual_usd') or x['reserve_usd']) for x in current.values() if (x.get('model') or self.legacy_model)==model),D(0))
                if model not in self.model_caps or model_total>self.model_caps[model]:raise BudgetStop('Returned charge exceeded model allocation; stop')
    def bound_unknown(self,key,amount,evidence_sha256):
        """Append an evidence-backed liability hold; never turn it into actual spend."""
        import re
        value=D(str(amount))
        if not value.is_finite() or value<0 or not re.fullmatch('[a-f0-9]{64}',evidence_sha256):raise BudgetStop('Invalid bound evidence')
        with self.lock():
            current,held=self.state();old=current.get(key)
            if old is None or old.get('kind')!='unknown' or old.get('actual_usd') is not None:raise BudgetStop('Only a recorded unknown charge can be bounded')
            if value<D(old['reserve_usd']):raise BudgetStop('A conservative amendment cannot reduce the hold')
            total=held-D(old['reserve_usd'])+value
            if total>self.usd_cap or total*self.fx>self.cad_cap:raise BudgetStop('Bound exceeds cumulative cap')
            if self.model_caps:
                model=old.get('model') or self.legacy_model
                model_total=sum((D(x.get('actual_usd') if x.get('actual_usd') is not None else x['reserve_usd']) for x in current.values() if (x.get('model') or self.legacy_model)==model),D(0))-D(old['reserve_usd'])+value
                if model not in self.model_caps or model_total>self.model_caps[model]:raise BudgetStop('Bound exceeds model allocation')
            self.append({'kind':'bounded_unknown','case_key':key,'reserve_usd':str(value),'actual_usd':None,'evidence_sha256':evidence_sha256,'meaning':'Permanent unknown liability hold; not settled or measured cost'})
    def raise_unknown_hold(self,key,amount,evidence_sha256):
        """Increase a failed request's hold while retaining the generation STOP."""
        import re
        value=D(str(amount))
        if not value.is_finite() or value<0 or not re.fullmatch('[a-f0-9]{64}',evidence_sha256):raise BudgetStop('Invalid hold evidence')
        with self.lock():
            current,held=self.state();old=current.get(key)
            if old is None or old.get('kind')!='unknown' or old.get('actual_usd') is not None:raise BudgetStop('Expected an unknown stopped request')
            if value<D(old['reserve_usd']):raise BudgetStop('Hold cannot shrink')
            total=held-D(old['reserve_usd'])+value
            if total>self.usd_cap or total*self.fx>self.cad_cap:raise BudgetStop('Hold exceeds global cap')
            model=old.get('model') or self.legacy_model
            if self.model_caps and (model not in self.model_caps or value>self.model_caps[model]):raise BudgetStop('Hold exceeds model cap')
            self.append({'kind':'unknown','case_key':key,'reserve_usd':str(value),'actual_usd':None,'evidence_sha256':evidence_sha256,'meaning':'Increased conservative liability; generation remains stopped. Not settled, not permission to continue.'})
