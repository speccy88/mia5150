# Retained SkillRet CLM head

The best checkpoint is selected by train-catalog validation Recall@5, with MRR
as the tiebreaker, from genuinely updated copies of the released CLM state and
action heads. Training labels come from SkillRet train only. The Qwen3-8B
encoder stays frozen. The checkpoint is a `torch.save` dictionary containing
`state_head`, `action_head`, `logit_scale`, `cfg`, and `skillret_epoch`.

To apply it, produce Qwen/Qwen3-8B pre-head vectors with revision
`b968826d9c46dd6066d109eabc6255188de91218`, BF16 inference,
last-token pooling, 2048-token cap, and float32 L2 normalization. Use canonical
query text or skill `name: description` directly, without extra instructions.
`src/load_clm_head.py` then projects query vectors through the state head and
skill vectors through the action head. Cosine scores rank the whole catalog.

The pretrained source head is `Contrastive-LM/CLM-v0.1-8B` revision
`e939398d4556fcd9400c76fa8c5a513202f42b0a`, SHA256
`b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`.
The pinned CLM implementation is Git commit
`bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`.

Results and limitations appear in `REPORT.pdf` only after a completed run.
