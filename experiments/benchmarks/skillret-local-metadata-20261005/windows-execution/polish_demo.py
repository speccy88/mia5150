from pathlib import Path
p=Path('make_demo.py');s=p.read_text()
s=s.replace('<h1>Skill discovery lab</h1><p>', '<h1>Skill discovery lab</h1><p><strong>Full-query finding:</strong> fusion was essentially flat across all4,392 public queries (NDCG@10 +0.000136; paired interval includes zero). This viewer explains the smaller development sample, not a confirmed overall gain.</p><p>')
s=s.replace("t=template.replace('SkillRet embeddings'", "t=template.replace('Full-query finding:</strong> fusion was essentially flat across all4,392 public queries (NDCG@10 +0.000136; paired interval includes zero). This viewer explains the smaller development sample, not a confirmed overall gain.', 'Transfer finding:</strong> the development gain did not transfer: public32-query NDCG@10 changed from0.762598 to0.759710. Both development and transfer paired intervals include zero. This viewer shows the development cases.').replace('SkillRet embeddings'")
p.write_text(s)
