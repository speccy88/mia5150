"""Load the retained CLM head; input must be pinned Qwen3-8B pre-head vectors."""
from pathlib import Path
import numpy as np

from train_eval_stage import load_heads, project


def load_and_project(checkpoint: str | Path, query_prehead: np.ndarray,
                     skill_prehead: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Return normalized state and action projections for cached 4096-d vectors."""
    _,state,action,_ = load_heads(Path(checkpoint))
    if query_prehead.shape[1]!=4096 or skill_prehead.shape[1]!=4096:
        raise ValueError("Expected pinned Qwen3-8B 4096-d pre-head inputs")
    return project(query_prehead,state),project(skill_prehead,action)
