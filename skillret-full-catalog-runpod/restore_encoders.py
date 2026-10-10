"""Opt-in restoration of the pinned public encoder weights; never starts compute."""
from __future__ import annotations
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

REVISIONS = {
    "qwen": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "bge": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
}

def main():
    p=argparse.ArgumentParser(description="Download a pinned public encoder snapshot")
    p.add_argument("model",choices=sorted(REVISIONS))
    p.add_argument("--destination",type=Path,required=True)
    a=p.parse_args()
    repo,revision=REVISIONS[a.model]
    a.destination.mkdir(parents=True,exist_ok=True)
    print(snapshot_download(repo_id=repo,revision=revision,local_dir=a.destination))

if __name__=="__main__": main()
