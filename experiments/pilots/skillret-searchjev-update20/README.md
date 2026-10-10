# SearchJev: verified 20-update local adaptation

Start with [the report](SearchJev-SkillRet-Report.txt), [raw evidence](evidence-summary.json), and [paired analysis](paired-quality-analysis.json). The completed run is `runs/pilot20-dynamic`; other attempts are independent and must not be combined. The 60-example development evaluation was exposed. No full training or 200-update result is claimed.

This expanded canonical bundle matches SHA-256 `4c3418dad6f7f34c9ac0a6daf98b402a4d73a5b0663245771cba614f28f70ece`. It includes `runs/pilot20-dynamic/checkpoint-update20/adapter_model.safetensors` (43,340,856 bytes), its configuration, optimizer-update evidence, and [licenses](licenses/manifest.json). All 196 canonical files are included before this README/audit addition.

The multi-gigabyte Qwen3.5 base and released initial adapter are upstream downloads. Their repositories, immutable revisions, byte hashes, and expected paths are in [models/manifest.json](models/manifest.json); download_models.py records reconstruction. Review the pinned dependencies and original absolute paths before using the scripts. Training/runtime/reload scripts load models and are **not** consolidation validation commands; none was rerun here.
