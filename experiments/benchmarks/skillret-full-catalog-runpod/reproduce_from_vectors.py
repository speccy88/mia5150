"""Recompute all five score matrices on a RunPod GPU from archived vectors."""
import argparse
from pathlib import Path
from train_eval_stage import evaluate

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    a=p.parse_args()
    evaluate(a.root)
