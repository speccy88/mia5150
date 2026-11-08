"""Check selected-adapter save/load on its original heldout training proxy."""
import json
import time
import numpy as np
from common import *
from engine import baseline,Monitor,attach_lora
from train import Dataset,infer_batch

class ReloadMonitor(Monitor):
    def loop(self):
        with (STUDY/'results/adapter_reload_telemetry.jsonl').open('a') as stream:
            while not self.stop.is_set():
                try:stream.write(json.dumps(self.sample())+'\n');stream.flush()
                except Exception as error:self.problem='Reload monitor unavailable: '+type(error).__name__
                self.stop.wait(1)

def check():
    data=Dataset();selection=read(STUDY/'results/selected_adapter.json');started=time.perf_counter()
    expected=read(STUDY/'results'/f"validation-step{selection['optimizer_step']:05d}.json")
    adapter=CACHE/selection['adapter_relative_path']
    assert file_hash(adapter/'adapter_model.safetensors')==selection['adapter_sha256']
    monitor=ReloadMonitor('postrun:adapter_reload_parity')
    try:
        encoder=baseline.Encoder(CACHE,monitor,frozen=False);model,_=attach_lora(encoder,adapter)
        model.eval();torch=encoder.torch;vectors=[]
        for start in range(0,len(data.val_ids),8):
            ids=data.val_ids[start:start+8];monitor.check()
            vectors.append(infer_batch(encoder,[data.passage(sid) for sid in ids],[384]*len(ids)))
        matrix=torch.cat(vectors);ties=np.asarray([data.documents[sid]['source_sha256'] for sid in data.val_ids]);rankings={};gold={}
        for start in range(0,len(data.valid),8):
            batch=data.valid[start:start+8]
            query=infer_batch(encoder,[data.query(e) for e in batch],[256]*len(batch))
            scores=(query@matrix.T).cpu().numpy()
            for example,score in zip(batch,scores):
                qid=example['query_id'];order=np.lexsort((ties,-score))[:50]
                rankings[qid]=[data.val_ids[i] for i in order];gold[qid]={sid:1 for sid in example['all_positive_ids']}
        same=sum(rankings[qid]==expected['rankings'][qid] for qid in gold)
        result=evaluate(gold,rankings)
        compare_evaluation(result,{qid:independent(g,rankings[qid]) for qid,g in gold.items()},gold,rankings)
        differences={m:result['aggregate'][m]-expected['evaluation']['aggregate'][m] for m in result['aggregate']}
        receipt={'status':'passed' if same==128 and max(map(abs,differences.values()))<1e-12 else 'failed',
            'checkpoint_step':selection['optimizer_step'],'adapter_sha256':selection['adapter_sha256'],
            'validation_queries':128,'validation_skills':len(data.val_ids),'identical_top50_rankings':same,
            'metric_differences':differences,'reloaded_validation_ndcg10':result['aggregate']['NDCG@10'],
            'original_validation_ndcg10':expected['evaluation']['aggregate']['NDCG@10'],
            'source':'Original fixed training-derived heldout proxy; no test scores select or alter the adapter',
            'wall_s':time.perf_counter()-started,'weights_or_checkpoint_selection_changed':False}
        atomic_json(STUDY/'results/adapter_reload_parity.json',receipt)
        assert receipt['status']=='passed',receipt
        print('Saved adapter exactly reproduces all 128 selected-checkpoint validation rankings and metrics',flush=True)
    finally:monitor.close()

if __name__=='__main__':check()
