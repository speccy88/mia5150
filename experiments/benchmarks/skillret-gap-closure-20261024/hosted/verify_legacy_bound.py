"""Offline request reconstruction. No credentials, HTTP, or ledger mutation."""
import argparse,decimal,json,pathlib
from runner import MODEL,PROVIDER,request_body
from core import digest

HERE=pathlib.Path(__file__).resolve().parent
D=decimal.Decimal

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',required=True);args=p.parse_args()
    private=pathlib.Path(args.runtime_root)/'hosted-execution'
    calls=[json.loads(x) for x in (private/'calls.jsonl').read_text().splitlines() if x.strip()]
    assert len(calls)==1,'Unexpected number of recorded generation attempts'
    old=calls[0]
    body=request_body([{'role':'system','content':'Return only the JSON array requested. No tools.'},{'role':'user','content':'Return ["s000000000000"] for this non-evaluation format check.'}],'selection',[])
    assert digest(body)==old['request_sha256'],'Reconstruction differs from actual recorded request'
    assert body['model']==MODEL and body['provider']==PROVIDER and body['max_tokens']==128
    assert old['reserved_input_tokens']==8192 and old['output_token_limit']==128
    endpoint=json.loads((HERE/'deepinfra-endpoint-snapshot.json').read_text())['endpoint']
    assert endpoint['tag']=='deepinfra/fp8' and endpoint['context_length']==1048576
    assert endpoint['max_completion_tokens']==384000
    rates=endpoint['pricing'];assert rates=={'prompt':'0.00000006','completion':'0.00000018','input_cache_read':'0.000000015','discount':0}
    input_rate=D(rates['prompt']);output_rate=D(rates['completion'])
    context_input=D(endpoint['context_length'])*input_rate
    sent_cap=context_input+D(body['max_tokens'])*output_rate
    endpoint_cap=context_input+D(endpoint['max_completion_tokens'])*output_rate
    receipt={'status':'request_reconstruction_passed_bound_candidate_pending_review','generation_post_attempts_recorded':1,'additional_generation_retries_in_source':0,'request_sha256':old['request_sha256'],'request_hash_matches_reconstruction':True,'model':MODEL,'provider_pin':PROVIDER,'sent_max_tokens':128,'sent_reasoning':body['reasoning'],'recorded_estimated_input_tokens':old['estimated_input_tokens'],'old_input_allowance':8192,'endpoint_context_tokens':endpoint['context_length'],'endpoint_max_completion_tokens':endpoint['max_completion_tokens'],'endpoint_pricing':rates,'full_context_plus_sent_output_usd':str(sent_cap),'full_context_plus_entire_endpoint_completion_usd':str(endpoint_cap),'candidate_permanent_unknown_hold_usd':'0.14','actual_billed_cost_usd':None,'candidate_reason':'Full input context plus the entire endpoint completion ceiling, beyond the actual sent 128 output cap. This avoids relying solely on a provider-specific reasoning/max_tokens interpretation. No request fee or tier was listed in the saved endpoint pricing. One recorded POST, no retries or redirects in transport. Requires independent review before any ledger amendment or new POST.','limitations':['Observed request count and source do not independently prove provider-side billing correctness.','A protocol hold is a conservative liability reserve, not known spend.','Saved endpoint metadata is historical; future requests need fresh rates and separate caps.'],'sources':['https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion','https://openrouter.ai/docs/guides/best-practices/reasoning-tokens','https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints']}
    (HERE/'reports/legacy-unknown-bound.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
