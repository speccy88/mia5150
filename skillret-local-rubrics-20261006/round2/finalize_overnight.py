"""Finalize nonsecret accounting and explicitly pause further paid work."""
import json,fcntl
from decimal import Decimal
from pathlib import Path
from jev_probe import save
HERE=Path(__file__).resolve().parent
with (HERE/'budget.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 path=HERE/'budget-ledger.json';ledger=json.loads(path.read_text())
 assert Decimal(ledger['committed_usd'])+Decimal(ledger['external_spend_usd'])==Decimal('1.77')
 ledger['paid_tests_paused']=True;ledger['pause_reason']='Final authorized two-case order check completed. No new paid tests until next-day review.';save(path,ledger)
known=Decimal(0);attempts=0
for folder,key in [('jev-typed-12-v1','known_usage_usd_at_pinned_rate'),('openrouter-listwise-12-v1','known_reported_cost_usd'),('selective-jev-8-v1','known_usage_usd'),('fusion-confirmation-8-v1','known_usage_usd'),('order-sensitivity-2-v1','known_usage_usd')]:
 r=json.loads((HERE/folder/'results.json').read_text());known+=Decimal(r[key]);attempts+=r['attempts']
status={'status':'paid_tests_paused_pending_review','committed_usd':'1.77','cap_usd':'4.00','known_successful_usage_usd':str(known),'paid_attempts':attempts,'unknown_charge':'One failed OpenRouter attempt; reservation retained, no retry.','reserve_refunds_applied':False,'key_files_included':False}
save(HERE/'PAID_EXPERIMENTS_PAUSED.json',status);save(HERE/'FINAL_BUDGET_STATUS.json',status);print(json.dumps(status))
