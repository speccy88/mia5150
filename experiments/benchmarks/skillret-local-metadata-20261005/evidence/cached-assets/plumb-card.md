---
license: apache-2.0
base_model: crh225/plumb-4b
language: [en]
tags: [gguf, llama.cpp, ollama, decision-model, calibration]
---

# Plumb-4B GGUF

[Plumb-4B](https://huggingface.co/crh225/plumb-4b) for llama.cpp and Ollama, on almost any GPU or a CPU.

| File | Quantisation | Size |
|---|---|---:|
| `plumb-4b-v5-Q8_0.gguf` | 8-bit, closest to the full model | ~4.5 GB |
| `plumb-4b-v5-Q4_K_M.gguf` | 4-bit | 2.7 GB |

Plumb-4B is a decision model, not a chat model: it answers with one option letter, and its value is in
the **probabilities of those letters**, softened by its calibration temperature **T = 2.07**.

## With Ollama

`Modelfile` in this repo holds the decision prompt as the template, so an ordinary chat request carrying
the question as JSON (evidence, criterion, lettered options) renders the right prompt:

```bash
ollama create plumb-4b -f Modelfile      # next to the downloaded .gguf
```

Ask for one token with `logprobs` and `top_logprobs`, send `reasoning_effort: "none"`, read the option
letters' log-probabilities, divide by 2.07 and renormalise.

## With llama.cpp

```bash
llama-server -m plumb-4b-v5-Q8_0.gguf -c 8192 -ngl 99
pip install --no-deps "jevk5 @ git+https://github.com/allebee/jevk5@v0.2.1"   # standard library only
```

```python
from jevk5 import JevK5GGUF

model = JevK5GGUF(temperature=2.07)   # llama-server on :8080
model.decide("Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
             {"type": "choice", "instructions": "What happened to the parcel?",
              "criteria": ["delivered", "misdelivered", "unknown"]})
```

Converted with llama.cpp (`convert_hf_to_gguf.py --no-mtp`, then `llama-quantize`). Apache-2.0; see the
main model's card for credits.
