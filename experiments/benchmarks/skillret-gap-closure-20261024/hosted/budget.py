"""Persistent locked reservations. Unknown charges halt further generation."""
import contextlib,decimal,fcntl,json,os,pathlib
D=decimal.Decimal
class BudgetStop(RuntimeError):pass
class Budget:
    def __init__(self,path,usd_cap,cad_cap,cad_per_usd,call_cap):
        self.path=pathlib.Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.usd_cap=D(str(usd_cap));self.cad_cap=D(str(cad_cap));self.fx=D(str(cad_per_usd));self.call_cap=int(call_cap)
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
    def reserve(self,key,amount):
        amount=D(str(amount))
        if not amount.is_finite() or amount<0:raise BudgetStop('Invalid reserve')
        with self.lock():
            current,held=self.state()
            if key in current:raise BudgetStop('Case already reserved/attempted; no duplicate generation')
            if any(x.get('kind')=='unknown' for x in current.values()):raise BudgetStop('Unknown charge retained; reconcile before expansion')
            if len(current)>=self.call_cap:raise BudgetStop('Request cap')
            if held+amount>self.usd_cap or (held+amount)*self.fx>self.cad_cap:raise BudgetStop('USD/CAD cap')
            self.append({'kind':'reserve','case_key':key,'reserve_usd':str(amount),'actual_usd':None})
    def settle(self,key,actual):
        with self.lock():
            current,held=self.state()
            if key not in current:raise BudgetStop('Missing reservation')
            if actual is None:self.append({'kind':'unknown','case_key':key,'actual_usd':None});raise BudgetStop('Unknown charge retained; expansion halted')
            value=D(str(actual))
            if not value.is_finite() or value<0:raise BudgetStop('Invalid returned cost')
            self.append({'kind':'settled','case_key':key,'actual_usd':str(value)})
            _,total=self.state()
            if total>self.usd_cap or total*self.fx>self.cad_cap:raise BudgetStop('Returned charge exceeded cap; stop')
