"""Train released CLM heads and score five complete SkillRet test catalogs."""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from clm.heads import make_head
from data_stage import read_jsonl, write_json, write_jsonl
from reference_checks import multi_positive_loss, positive_mask, retrieval_metrics


def vec(root: Path, model: str, split: str, kind: str):
    base = root / "vectors" / model / f"{split}_{kind}"
    x = np.load(str(base) + ".npy", mmap_mode="r")
    meta = json.loads(Path(str(base) + ".json").read_text())
    assert list(x.shape) == meta["shape"] and len(meta["ids"]) == x.shape[0]
    return x, meta["ids"]


def gold_map(root: Path, split: str):
    gold = defaultdict(set)
    for r in read_jsonl(root / "data" / f"{split}_qrels.jsonl"):
        gold[r["query_id"]].add(r["skill_id"])
    return gold


def load_heads(checkpoint: Path, device="cuda"):
    # Only the SHA-pinned official checkpoint or our own generated descendant is loaded.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    kw = dict(width=cfg["width"], depth=cfg["depth"],
              proj=ck.get("projection_dim", cfg.get("projection_dim", 512)),
              activation=cfg.get("activation", "gelu"),
              layernorm=cfg.get("layernorm", False), residual=cfg.get("residual", False),
              hidden=cfg.get("hidden_size", 4096))
    sh, ah = make_head(**kw).to(device), make_head(**kw).to(device)
    sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
    scale = torch.nn.Parameter(torch.as_tensor(ck["logit_scale"], device=device).float().clone())
    return ck, sh, ah, scale


def project(x, head, batch=1024):
    head.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(x), batch):
            z = F.normalize(head(torch.from_numpy(np.asarray(x[i:i+batch])).to("cuda")), dim=1)
            out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def score_dense(q, s, block=128):
    q = np.asarray(q, dtype=np.float32); s = np.asarray(s, dtype=np.float32)
    assert np.isfinite(q).all() and np.isfinite(s).all()
    assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=2e-3)
    assert np.allclose(np.linalg.norm(s, axis=1), 1, atol=2e-3)
    bank = torch.from_numpy(s).to("cuda")
    out = np.empty((len(q), len(s)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(q), block):
            y = torch.from_numpy(q[i:i+block]).to("cuda") @ bank.T
            out[i:i+block] = y.cpu().numpy()
    return out


def validate(root, sh, ah, val_ids, qx, qids, sx, sids, gold):
    lookup = {qid: i for i, qid in enumerate(qids)}
    q = qx[[lookup[i] for i in val_ids]]
    zq, zs = project(q, sh), project(sx, ah)
    scores = score_dense(zq, zs)
    order_key = np.asarray(sids, dtype=str)
    recalls, rrs = [], []
    for i, qid in enumerate(val_ids):
        ranking = np.lexsort((order_key, -scores[i]))
        relevant = np.isin(ranking, [sids.index(x) for x in gold[qid]])
        recalls.append(float(relevant[:5].sum() / len(gold[qid])))
        first = np.flatnonzero(relevant)
        rrs.append(float(1 / (first[0] + 1)) if len(first) else 0)
    return {"recall_at_5": float(np.mean(recalls)), "mrr": float(np.mean(rrs)),
            "validation_queries": len(val_ids), "catalog_size": len(sids)}


def save_head(path: Path, original: dict, sh, ah, scale, epoch: int):
    ck = copy.deepcopy(original)
    ck["state_head"] = {k: v.detach().cpu() for k,v in sh.state_dict().items()}
    ck["action_head"] = {k: v.detach().cpu() for k,v in ah.state_dict().items()}
    ck["logit_scale"] = scale.detach().cpu()
    ck["skillret_epoch"] = epoch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)


def train(root: Path, deadline: float):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    original_path = root / "checkpoints/clm_pretrained/CLM_v0.1-8B.pt"
    original_sha = hashlib.sha256(original_path.read_bytes()).hexdigest()
    original, sh, ah, scale = load_heads(original_path)
    qx, qids = vec(root, "qwen_prehead", "train", "queries")
    sx, sids = vec(root, "qwen_prehead", "train", "skills")
    assert qx.shape[1] == sx.shape[1] == 4096
    cohort = json.loads((root / "data/training_cohort.json").read_text())
    selected = set(cohort["selected_training_query_ids"])
    val_ids = cohort["validation_query_ids"]
    assert selected.isdisjoint(val_ids)
    gold = gold_map(root, "train")
    train_ids = sorted(selected)
    qindex = {v:i for i,v in enumerate(qids)}
    sindex = {v:i for i,v in enumerate(sids)}
    assert all(x in qindex for x in train_ids+val_ids)
    assert all(gold[q].issubset(sindex) for q in train_ids+val_ids)
    history = []
    baseline = validate(root, sh, ah, val_ids, qx, qids, sx, sids, gold)
    history.append({"epoch": 0, **baseline})
    write_json(root / "results/validation_epoch0.json", baseline)
    nparams = sum(p.numel() for p in list(sh.parameters())+list(ah.parameters())) + 1
    recipe = {"seed":42,"optimizer":"AdamW","learning_rate":1e-4,"weight_decay":.01,
              "batch_unique_queries":256,"random_negatives":512,"max_epochs":10,
              "patience":2,"gradient_clip_norm":1.0,"loss":"uniform-positive bidirectional cross-entropy",
              "trainable_parameters":nparams,"cohort_groups":cohort["selected_training_groups"]}
    write_json(root / "checkpoints/clm_skillret/training_config.json", recipe)
    params = list(sh.parameters())+list(ah.parameters())+[scale]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    # TRAIN-only gradient/change smoke, followed by reset to the released state.
    smoke_ids = train_ids[:8]
    smoke_sids = sorted({s for q in smoke_ids for s in gold[q]})
    a = torch.from_numpy(np.asarray(qx[[qindex[q] for q in smoke_ids]])).cuda()
    b = torch.from_numpy(np.asarray(sx[[sindex[s] for s in smoke_sids]])).cuda()
    m = torch.from_numpy(positive_mask(smoke_ids, smoke_sids, gold)).cuda()
    before = next(sh.parameters()).detach().clone()
    loss = multi_positive_loss((F.normalize(sh(a),dim=1) @ F.normalize(ah(b),dim=1).T) * scale.exp(), m)
    loss.backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    opt.step(); assert not torch.equal(before, next(sh.parameters()).detach())
    write_json(root / "evidence/train_smoke.json", {"loss":float(loss),"gradient_finite":True,"weights_changed":True})
    original, sh, ah, scale = load_heads(original_path)
    params = list(sh.parameters())+list(ah.parameters())+[scale]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    best_key = (-1.,-1.); stale = 0
    rng = random.Random(42)
    all_sid_set = set(sids)
    for epoch in range(1,11):
        if time.time() > deadline - 1800: raise TimeoutError("training stopped for report and transfer reserve")
        sh.train(); ah.train()
        order = train_ids.copy(); rng.shuffle(order)
        losses=[]
        t0=time.monotonic()
        for off in range(0,len(order),256):
            qbatch=order[off:off+256]
            positives={s for q in qbatch for s in gold[q]}
            negative_pool=sorted(all_sid_set-positives)
            extra=rng.sample(negative_pool,min(512,len(negative_pool)))
            candidates=sorted(positives | set(extra))
            pm=torch.from_numpy(positive_mask(qbatch,candidates,gold)).to("cuda")
            q=torch.from_numpy(np.asarray(qx[[qindex[x] for x in qbatch]])).cuda()
            s=torch.from_numpy(np.asarray(sx[[sindex[x] for x in candidates]])).cuda()
            logits=(F.normalize(sh(q),dim=1) @ F.normalize(ah(s),dim=1).T)*scale.exp().clamp(max=100)
            loss=multi_positive_loss(logits,pm)
            if not torch.isfinite(loss): raise ValueError("nonfinite training loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.0)
            if not torch.isfinite(norm): raise ValueError("nonfinite training gradient")
            opt.step()
            with torch.no_grad(): scale.clamp_(max=math.log(100))
            losses.append(float(loss.detach()))
            if time.time() > deadline - 1800: raise TimeoutError("training reserve reached")
        v=validate(root,sh,ah,val_ids,qx,qids,sx,sids,gold)
        row={"epoch":epoch,"loss_mean":float(np.mean(losses)),"train_seconds":time.monotonic()-t0,**v}
        history.append(row);write_json(root/"results/validation_history.json",history)
        save_head(root/"checkpoints/clm_skillret/last_head.pt",original,sh,ah,scale,epoch)
        key=(v["recall_at_5"],v["mrr"])
        if key>best_key:
            best_key=key;stale=0
            save_head(root/"checkpoints/clm_skillret/best_head.pt",original,sh,ah,scale,epoch)
            write_json(root/"checkpoints/clm_skillret/best_selection.json",row)
        else: stale+=1
        torch.save({"epoch":epoch,"optimizer":opt.state_dict(),"torch_rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state(),"python_rng":rng.getstate()},
                   root/"checkpoints/clm_skillret/resume_state.pt")
        if stale>=2: break
    assert hashlib.sha256(original_path.read_bytes()).hexdigest()==original_sha
    best=root/"checkpoints/clm_skillret/best_head.pt"
    assert best.exists() and hashlib.sha256(best.read_bytes()).hexdigest()!=original_sha
    _, rh, ra, rs=load_heads(best)
    sample_q=np.asarray(qx[:10]);sample_s=np.asarray(sx[:50])
    q1,s1=project(sample_q,rh),project(sample_s,ra)
    scores1=score_dense(q1,s1)
    # When best was earlier than last, compare two fresh loads of the saved best.
    _, rh2, ra2, _=load_heads(best)
    q2,s2=project(sample_q,rh2),project(sample_s,ra2)
    scores2=score_dense(q2,s2)
    assert np.allclose(scores1,scores2,atol=1e-6)
    assert np.array_equal(np.argsort(-scores1,axis=1,kind="stable"),
                          np.argsort(-scores2,axis=1,kind="stable"))
    write_json(root/"evidence/checkpoint_reload.json",{"best_sha256":hashlib.sha256(best.read_bytes()).hexdigest(),
               "original_unchanged":True,"reload_scores_equal":True,"reload_rankings_equal":True,
               "sample_queries":len(sample_q),"sample_skills":len(sample_s),
               "best_epoch":int(json.loads((root/"checkpoints/clm_skillret/best_selection.json").read_text())["epoch"])})


def evaluate(root: Path):
    from rank_bm25 import BM25Okapi
    qrows=read_jsonl(root/"data/test_queries.jsonl")
    srows=read_jsonl(root/"data/test_skills.jsonl")
    qids=[x["id"] for x in qrows];sids=[x["id"] for x in srows]
    assert len(qids)==len(set(qids)) and len(sids)==len(set(sids))
    token=lambda s: re.findall(r"\w+",s.lower())
    t=time.monotonic()
    bm=BM25Okapi([token(x["text"]) for x in srows],k1=1.5,b=.75)
    bm_scores=np.asarray([bm.get_scores(token(x["text"])) for x in qrows],dtype=np.float64)
    write_json(root/"evidence/bm25_config.json",{"package":"rank-bm25","class":"BM25Okapi",
               "tokenizer":"lowercase Unicode word regex","k1":1.5,"b":.75,
               "scoring_seconds":time.monotonic()-t,"catalog_size":len(sids)})
    bm_latency=[]
    for row in qrows[:20]:
        started=time.perf_counter();_ = np.argmax(bm.get_scores(token(row["text"])))
        bm_latency.append({"query_id":row["id"],"text_to_top1_seconds":time.perf_counter()-started})
    write_json(root/"results/bm25_online_latency.json",bm_latency)
    scores={"bm25_full":bm_scores}
    timing={"bm25_build_and_full_score_seconds":time.monotonic()-t}
    bq,bqi=vec(root,"bge_m3","test","queries");bs,bsi=vec(root,"bge_m3","test","skills")
    qq,qqi=vec(root,"qwen_prehead","test","queries");qs,qsi=vec(root,"qwen_prehead","test","skills")
    assert qids==bqi==qqi and sids==bsi==qsi
    t=time.monotonic();scores["bge_m3_dense_full"]=score_dense(bq,bs)
    timing["bge_full_score_seconds"]=time.monotonic()-t
    t=time.monotonic();scores["qwen3_raw_full"]=score_dense(qq,qs)
    timing["qwen_raw_full_score_seconds"]=time.monotonic()-t
    timing["index_bytes"]={"bge_test_skills":int(bs.nbytes),"qwen_test_skills":int(qs.nbytes)}
    for label,path in [("clm_pretrained_full",root/"checkpoints/clm_pretrained/CLM_v0.1-8B.pt"),
                       ("clm_skillret_full",root/"checkpoints/clm_skillret/best_head.pt")]:
        t=time.monotonic()
        _,sh,ah,_=load_heads(path)
        torch.cuda.synchronize()
        timing[f"{label}_load_seconds"]=time.monotonic()-t
        t=time.monotonic()
        qz=project(qq,sh);sz=project(qs,ah)
        torch.cuda.synchronize()
        timing[f"{label}_project_test_seconds"]=time.monotonic()-t
        namespace="clm_pretrained" if "pretrained" in label else "clm_skillret"
        from embedding_stage import put_vectors
        put_vectors(root,namespace,"test","queries",qrows,qz,{"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"head":"state"})
        put_vectors(root,namespace,"test","skills",srows,sz,{"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"head":"action"})
        t=time.monotonic();scores[label]=score_dense(qz,sz)
        timing[f"{label}_full_score_seconds"]=time.monotonic()-t
    gold=gold_map(root,"test")
    agg={};idkey=np.asarray(sids,dtype=str)
    for name,mat in scores.items():
        assert mat.shape==(len(qids),len(sids)) and np.isfinite(mat).all()
        np.save(root/"results"/f"{name}_scores.npy",mat)
        rankings=np.empty(mat.shape,dtype=np.int32)
        per=[]
        for i,qid in enumerate(qids):
            ix=np.lexsort((idkey,-mat[i]))
            rankings[i]=ix
            ranked=[sids[j] for j in ix]
            m=retrieval_metrics(ranked,gold[qid])
            m["recall_at_10"]=len(set(ranked[:10])&gold[qid])/len(gold[qid])
            per.append({"query_id":qid,**m,"top_10":ranked[:10]})
        np.save(root/"results"/f"{name}_rankings.npy",rankings)
        write_jsonl(root/"results"/f"{name}_per_query.jsonl",per)
        keys=["hit_at_1","recall_at_1","recall_at_5","recall_at_10","reciprocal_rank","ndcg_at_5","completeness_at_5"]
        agg[name]={k:float(np.mean([r[k] for r in per])) for k in keys}
        agg[name].update({"queries":len(qids),"catalog_size":len(sids)})
    write_json(root/"results/aggregate.json",agg)
    write_json(root/"results/score_index.json",{"query_ids":qids,"skill_ids":sids,
               "rankings":"int32 row indices into skill_ids; scores float32 except BM25 float64",
               "ties":"descending exact saved score, ascending stable skill ID"})
    import resource
    timing["process_peak_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(root/"evidence/evaluation_timing.json",timing)
