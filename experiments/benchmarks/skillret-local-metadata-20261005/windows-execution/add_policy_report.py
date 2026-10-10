from pathlib import Path
p=Path('write_report.py');s=p.read_text();needle="lines+=['## Scaling and full-catalog prompting'"
insert="""native_policy=read('policy-native')
if native_policy:
 lines+=['Actual frozen-policy validation: '+str(native_policy['latency_seconds'])+' seconds warm query-to-ranking; two-model startup '+str(native_policy['startup_seconds'])+' seconds; escalation rate '+str(native_policy['escalation_rate'])+'. '+native_policy['note'],'']
"""
assert needle in s;s=s.replace(needle,insert+needle);p.write_text(s)
