"""Exact immutable baseline runner with the separately identified LoRA encoder."""
import time
from common import BASE,CACHE,STUDY,module,verify_baseline,read,atomic_json
import engine

class MeasuredEncoder(engine.Encoder):
    def __init__(self,*args,**kwargs):
        started=time.perf_counter();super().__init__(*args,**kwargs)
        path=STUDY/'results/model.json';model=read(path)
        model['construction_wall_s_including_hash_checks_and_adapter']=time.perf_counter()-started
        model['model_loading_s_scope']='Backbone load only; complete construction time additionally includes pin verification, PEFT adapter loading and metadata persistence'
        atomic_json(path,model)

def main():
    verify_baseline()
    assert read(STUDY/'results/training_summary.json')['status']=='completed'
    assert read(STUDY/'results/selected_adapter.json')['official_test_inspected'] is False
    assert read(CACHE/'selection.json')==read(STUDY/'results/selected_adapter.json'),'Cache does not contain the validation-selected adapter'
    runner=module('immutable_bge_inference_runner',BASE/'src/run.py')
    runner.Encoder=MeasuredEncoder
    runner.main(CACHE)

if __name__=='__main__':main()
