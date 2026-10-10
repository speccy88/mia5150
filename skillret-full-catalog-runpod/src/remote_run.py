"""Run the fixed experiment on one RunPod GPU pod.

The local guard owns pod shutdown. This process checks its wall deadline before
each stage and keeps every completed artifact on the persistent /workspace disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from data_stage import write_json

STAGES = ["data", "bge", "qwen", "train", "evaluate", "report"]


def stage(root: Path, name: str, deadline: float):
    if name == "data":
        from data_stage import prepare
        prepare(root)
    elif name == "bge":
        from embedding_stage import bge
        bge(root)
    elif name == "qwen":
        from embedding_stage import qwen
        qwen(root, deadline)
    elif name == "train":
        from train_eval_stage import train
        train(root, deadline)
    elif name == "evaluate":
        from train_eval_stage import evaluate
        evaluate(root)
    elif name == "report":
        from report_stage import main
        main(root)
    else:
        raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--deadline-epoch", type=float, required=True)
    ap.add_argument("--stage", choices=STAGES+["all"], default="all")
    a = ap.parse_args()
    root = a.root.resolve()
    assert platform.system() == "Linux", "experiment computation must run on RunPod Linux"
    assert root.is_dir() and str(root).startswith("/workspace/")
    os.environ.setdefault("HF_HOME",str(root/"model-cache"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
    os.environ.setdefault("VLLM_NO_USAGE_STATS","1")
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    env={"python":sys.version,"platform":platform.platform(),"pytorch":torch.__version__,
         "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
         "gpu_vram_bytes":torch.cuda.get_device_properties(0).total_memory}
    write_json(root/"evidence/remote_environment.json",env)
    (root/"evidence/requirements.freeze.txt").write_text(
        subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    names=STAGES if a.stage=="all" else [a.stage]
    for name in names:
        if time.time() > a.deadline_epoch-900:
            raise TimeoutError("wall-clock export reserve reached before stage "+name)
        marker=root/"evidence"/f"stage_{name}.json"
        if marker.exists() and json.loads(marker.read_text()).get("status")=="complete":
            continue
        started=time.time()
        try:
            stage(root,name,a.deadline_epoch)
            write_json(marker,{"status":"complete","started_epoch":started,
                               "finished_epoch":time.time()})
        except Exception as exc:
            write_json(marker,{"status":"failed","started_epoch":started,
                               "finished_epoch":time.time(),"exception":type(exc).__name__,
                               "message":str(exc)[:500]})
            (root/"logs"/f"{name}_failure.txt").write_text(traceback.format_exc())
            write_json(root/"evidence/completion.json",{"status":"PARTIAL","failed_stage":name})
            raise
    write_json(root/"evidence/completion.json",{"status":"COMPLETE","stages":STAGES})


if __name__=="__main__":
    main()
