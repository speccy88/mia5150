"""Exact immutable baseline runner with the separately identified LoRA encoder."""
from common import BASE,CACHE,STUDY,module,verify_baseline,read
import engine

def main():
    verify_baseline()
    assert read(STUDY/'results/training_summary.json')['status']=='completed'
    assert read(STUDY/'results/selected_adapter.json')['official_test_inspected'] is False
    runner=module('immutable_bge_inference_runner',BASE/'src/run.py')
    runner.main(CACHE)

if __name__=='__main__':main()
