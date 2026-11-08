# Portable LoRA adapters

`best/` contains the training-derived validation-selected adapter; `final/`
contains the last trained adapter. Only the 786,432 intended FP32 query/value
LoRA parameters are saved, through Git LFS. Base weights are excluded. Configs
pin `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181` without
local paths. The adapter manifest and tensor validation verify shapes, finite
values, checksums, rank 8, alpha 16 and dropout 0.05.

Use the study's pinned PEFT/Transformers environment and inference wrapper to
preserve official CLS pooling, normalization, complete source spans and the
same mixed adapter/base dtypes. See `results/selected_adapter.json` for the exact
checkpoint and validation selection evidence. Optimizer/scheduler/scaler/RNG
recovery states remain outside Git. The base model's own terms are available
from its [official model card](https://huggingface.co/BAAI/bge-m3).
