"""Resource summaries from actual samples; distinguish invocation wall and pauses."""
import csv,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
def main():
    result=[]
    for model in ['qwen3_06_q8','qwen35_4b_q4']:
        with (HERE/f'results/{model}-resources.jsonl').open() as f:rows=[json.loads(x) for x in f]
        receipt=json.loads((HERE/f'results/{model}-execution.json').read_text())
        result.append({'model':model,'sample_count':len(rows),'sampling_target_seconds':2,'peak_gpu_used_mib':max(x['gpu_used_mib'] for x in rows),'minimum_gpu_free_mib':min(x['gpu_free_mib'] for x in rows),'peak_gpu_temperature_c':max(x['gpu_temperature_c'] for x in rows),'minimum_host_available_gib':min(x['host_available_bytes'] for x in rows)/1024**3,'minimum_guest_available_gib':min(x['guest_available_bytes'] for x in rows)/1024**3,'peak_server_rss_gib':max(x.get('server_rss_bytes',0) for x in rows)/1024**3,'cumulative_invocation_wall_seconds':receipt['cumulative_elapsed_seconds'],'latest_startup_seconds':receipt['startup_seconds'],'scope':'Successful watcher samples only; triggering sub-reserve events remain in resource-stops/execution receipts. Cumulative invocation wall includes conservative crash bound and excludes between-invocation operator pauses. Not GPU-active time or energy.'})
    with (HERE/'report/resource-summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    print(json.dumps(result))
if __name__=='__main__':main()
