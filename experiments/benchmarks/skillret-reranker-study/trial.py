"""Bounded experiment orchestration. Labels are used only by sampler/evaluator."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from pilot import (ROOT, config, read_json, save_json, append_row, rows, digest, load_data,
                   gold_map, bm25_rank, baseline, make_nested_catalogs, retrieval_metrics, selection_metrics)
from agent_runner import (GuardedModel, Blocked, preflight, require_permission, run_agent,
                          fixtures, offline_check, Budget, DOMAINS, ROUTER_POLICY, route_domain, sanitize, context_estimate)


def vectors():
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4)
    cfg=config()
    model_id=cfg["embedding"]["model"]
    manifest=ROOT/"manifests/embedding.json"
    sha=read_json("manifests/embedding.json")["revision"] if manifest.exists() else HfApi().model_info(model_id).sha
    skills,queries,qrels=load_data()
    pids=read_json("manifests/pilot_query_ids.json")
    spec={"model":model_id,"revision":sha,"device":"cpu","normalize":True,"dimension":384,
          "metadata_token_cap":128,"metadata_character_cap":1024,"query_prefix":cfg["embedding"]["query_prefix"],
          "dataset_revision":read_json("manifests/dataset.json")["revision"],"query_ids":pids,
          "representation":"original name + ': ' + description; 1024 character then 128 BGE token common cap; full bodies excluded"}
    key=digest(spec)
    if (ROOT/"data/vectors.npz").exists() and manifest.exists():
        saved=read_json("manifests/embedding.json")
        if saved["spec_hash"]!=key: raise Blocked("embedding_cache_config_mismatch")
        if digest(rows("data/metadata.jsonl"))!=saved["metadata_hash"]: raise Blocked("metadata_cache_checksum")
        import hashlib
        if hashlib.sha256((ROOT/"data/vectors.npz").read_bytes()).hexdigest()!=saved["vectors_sha256"]: raise Blocked("vector_cache_checksum")
        return saved
    start=time.monotonic()
    model=SentenceTransformer(model_id,revision=sha,device="cpu",trust_remote_code=False,
                              cache_folder=str(ROOT/".cache/embeddings"),model_kwargs={"use_safetensors":True})
    tokenizer=model.tokenizer
    capped=[]; transformations=[]
    for s in skills:
        raw=s["name"]+": "+s["description"]
        tokens=tokenizer.encode(raw[:1024],add_special_tokens=False)
        text=tokenizer.decode(tokens[:126],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        while len(tokenizer.encode(text))>128:
            text=tokenizer.decode(tokenizer.encode(text,add_special_tokens=False)[:-1],skip_special_tokens=True)
        capped.append({"id":s["id"],"metadata":text,"major":s["major"]})
        transformations.append({"id":s["id"],"original_chars":len(raw),"capped_chars":len(text),
            "tokens_before_token_cap":len(tokens),"final_tokens":len(tokenizer.encode(text)),
            "char_capped":len(raw)>1024,"token_capped":len(tokens)>126,"metadata_hash":digest(text)})
    # 512 is the pinned encoder context; already capped documents do not truncate here.
    model.max_seq_length=512
    docs=model.encode([s["metadata"] for s in capped],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
    qby={q["id"]:q for q in queries}
    query_texts=[cfg["embedding"]["query_prefix"]+qby[q]["query"] for q in pids]
    query_lengths=[len(tokenizer.encode(q,truncation=False)) for q in query_texts]
    qvec=model.encode(query_texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
    (ROOT/"data/metadata.jsonl").write_text("".join(json.dumps(s,ensure_ascii=False)+"\n" for s in capped))
    np.savez(ROOT/"data/vectors.npz",docs=docs,queries=qvec,skill_ids=np.array([s["id"] for s in skills]),query_ids=np.array(pids))
    import hashlib
    saved={**spec,"spec_hash":key,"metadata_hash":digest(capped),
        "vectors_sha256":hashlib.sha256((ROOT/"data/vectors.npz").read_bytes()).hexdigest(),
        "wall_seconds":time.monotonic()-start,
        "skills_character_capped":sum(r["char_capped"] for r in transformations),
        "skills_token_capped_after_char_cap":sum(r["token_capped"] for r in transformations),
        "queries_truncated_at_512_tokens":[q for q,n in zip(pids,query_lengths) if n>512]}
    save_json("manifests/embedding.json",saved)
    save_json("manifests/metadata_transformations.json",transformations)
    gold=gold_map(qrels)
    metrics=[]
    for i,q in enumerate(pids):
        ordering=np.argsort(-(docs@qvec[i]),kind="stable")[:10]
        ranking=[skills[j]["id"] for j in ordering]
        m={**retrieval_metrics(ranking,gold[q],5),**retrieval_metrics(ranking,gold[q],10)}
        append_row("results/vector_baseline_rows.jsonl",{"run_id":key,"query_id":q,"ranking":ranking,"evaluator_only_gold":sorted(gold[q]),"metrics":m})
        metrics.append(m)
    append_row("results/vector_baseline_summary.jsonl",{"run_id":key,"query_count":len(pids),"corpus_size":len(skills),
         "protocol":"capped metadata; not comparable to full-document BM25 as a method-only comparison",
         "metrics":{k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]}})
    # A matched metadata BM25 reference separates representation from retrieval method.
    ranked=bm25_rank([s["metadata"] for s in capped],[s["id"] for s in capped],[qby[q]["query"] for q in pids])
    ms=[]
    for q,rank in zip(pids,ranked):
        m={**retrieval_metrics([x[0] for x in rank],gold[q],5),**retrieval_metrics([x[0] for x in rank],gold[q],10)}
        ms.append(m)
        append_row("results/metadata_bm25_rows.jsonl",{"run_id":key,"query_id":q,"ranking":rank,"evaluator_only_gold":sorted(gold[q]),"metrics":m})
    append_row("results/metadata_bm25_summary.jsonl",{"run_id":key,"query_count":len(pids),"corpus_size":len(skills),
         "protocol":"same capped metadata as vectors","metrics":{k:sum(m[k] for m in ms)/len(ms) for k in ms[0]}})
    return saved


def experiment_data():
    skills,queries,qrels=load_data()
    metadata=rows("data/metadata.jsonl")
    by={s["id"]:{"id":s["id"],"metadata":m["metadata"],"skill_md":s["skill_md"]} for s,m in zip(skills,metadata)}
    if [s["id"] for s in skills]!=[s["id"] for s in metadata]: raise Blocked("metadata_order_mismatch")
    v=np.load(ROOT/"data/vectors.npz",allow_pickle=False)
    indices={str(s):i for i,s in enumerate(v["skill_ids"])}
    qindices={str(q):i for i,q in enumerate(v["query_ids"])}
    return by,{q["id"]:q for q in queries},gold_map(qrels),metadata,v,indices,qindices


def run_config(cfg,pin):
    # No timestamp/pricing snapshot path in case identity; all behavioral inputs are pinned.
    import importlib.metadata as md
    from agent_runner import POLICY,ACTIVATION_POLICY,SKILLS_TEMPLATE
    import inspect
    from local_runner import local_contract,UUID_CONTRACT
    return {"config":cfg,"dataset":read_json("manifests/dataset.json"),
        "embedding_spec":read_json("manifests/embedding.json")["spec_hash"],
        "llm":{k:v for k,v in pin.items() if k not in {"snapshot","pricing"}},
        "prompts":digest([POLICY,ACTIVATION_POLICY,SKILLS_TEMPLATE,ROUTER_POLICY]),
        "uuid_contract":digest([UUID_CONTRACT,inspect.getsource(local_contract)]) if cfg.get('uuid_contract_revision')==2 else None,
        "code":{p:digest((ROOT/p).read_text()) for p in ["pilot.py","agent_runner.py","trial.py","reference_metrics.py"]},
        "versions":{p:md.version(p) for p in ["deepagents","langchain","langchain-openrouter","bm25s","sentence-transformers"]}}


def result_file(cfg,kind):
    label=cfg.get('run_label')
    if label:
        import re
        if not re.fullmatch(r'[a-z0-9-]+',label):raise Blocked('invalid_run_label')
        return f'results/openrouter/{label}/{kind}.jsonl'
    return f'results/{kind}.jsonl'


def configured_preflight(cfg):
    return preflight(cfg,model=cfg['llm'].get('model_override'),
                     manifest_path=cfg['llm'].get('pin_manifest','manifests/llm.json'))


def smoke(live=False,cfg=None):
    cfg=cfg or config()
    offline=offline_check()
    if not live: return offline
    require_permission()
    pin=configured_preflight(cfg)
    key=digest({"phase":"canary","config":cfg,"pin":{k:v for k,v in pin.items() if k not in {"snapshot","pricing"}},"policy":(ROOT/"agent_runner.py").read_text()})
    prior=[r for r in rows(result_file(cfg,'smoke')) if r["case_key"]==key]
    if prior: return prior[-1]
    model=GuardedModel(pin=pin,cfg=cfg,case_id=key)
    try:
        result=run_agent("Add two numbers.",fixtures()[:2],model)
    except Blocked as e:
        result={"status":"blocked","selected":[],"error_category":str(e),"calls":sanitize(model.calls)}
    result.update(case_key=key,phase="hand_authored_canary",evidence_class="live OpenRouter integration, not SkillRet benchmark")
    result["canary_correct"]=result["status"]=="valid" and result["selected"]==[fixtures()[0]["id"]]
    append_row(result_file(cfg,'smoke'),result)
    return result


def route(query,model):
    message=model.bind_tools([route_domain]).invoke([{"role":"system","content":ROUTER_POLICY},{"role":"user","content":query}])
    calls=message.tool_calls
    if message.invalid_tool_calls or len(calls)!=1 or calls[0]["name"]!="route_domain": raise Blocked("router_malformed")
    major=calls[0]["args"].get("major")
    if major not in DOMAINS: raise Blocked("router_unknown_category")
    return major


def usage_totals(calls):
    totals={"input_tokens":0,"output_tokens":0,"cached_tokens":0,"reasoning_tokens":0,"actual_usd":0.0}
    for call in calls:
        usage=call.get("usage") or {}
        for key in ["input_tokens","output_tokens"]: totals[key]+=usage.get(key,0)
        totals["cached_tokens"]+=(usage.get("input_token_details") or {}).get("cache_read",0)
        totals["reasoning_tokens"]+=(usage.get("output_token_details") or {}).get("reasoning",0)
        if call.get("actual_usd") is None: totals["actual_usd"]=None
        elif totals["actual_usd"] is not None: totals["actual_usd"]+=call["actual_usd"]
    return totals


def save_memberships(run_id,membership):
    path=f"manifests/catalogs-{run_id}.json"
    previous=read_json(path) if (ROOT/path).exists() else []
    merged={(r["query_id"],r["N"]):r for r in previous}
    for row in membership:
        key=(row["query_id"],row["N"])
        if key in merged and row!=merged[key]: raise Blocked("catalog_membership_changed")
        merged[key]=row
    save_json(path,list(merged.values()))


def run_trial(cfg=None,quick=False,live=False):
    cfg=cfg or config()
    vectors()
    offline_check()
    permitted=False; blocker="live_flag_not_set"
    pin=configured_preflight(cfg)
    if live:
        try: require_permission(); permitted=True
        except Blocked as e: blocker=str(e)
    if permitted:
        canary=smoke(True,cfg)
        if not canary.get("canary_correct"): raise Blocked("canary_failed; no expansion")
    by,qby,gold,metadata,v,indices,qindices=experiment_data()
    mby={m["id"]:m for m in metadata}
    spec=run_config(cfg,pin)
    run_id=digest(spec)
    save_json(f"manifests/runs/{run_id}.json",spec)
    qids=read_json("manifests/live_query_ids.json")[:5 if quick else 20]
    sizes=[50] if quick else cfg["experiment"]["library_sizes"]
    # Five-query N=50 pilot must finish before larger paid expansion.
    if permitted and not quick:
        first=run_trial(cfg,quick=True,live=True)
        if not first.get("prerequisites_ok"): raise Blocked("quick_trial_incomplete; no expansion")
    prior={r["case_key"]:r for r in rows(result_file(cfg,'trial')) if r.get("mode")=="live"}
    route_cache={r["case_key"]:r for r in rows(result_file(cfg,'routes'))}
    membership=[]; completed=[]
    for qid in qids:
        catalogs=make_nested_catalogs(list(by),gold[qid],cfg["experiment"]["library_sizes"],qid,cfg["experiment"]["seed"])
        for n in sizes:
            library=list(catalogs[n]); membership.append({"query_id":qid,"N":n,"ordered_ids":library})
            for strategy in cfg["experiment"]["strategies"]:
                key=digest([run_id,qid,n,strategy])
                if permitted and key in prior:
                    completed.append(prior[key]); continue
                start=time.monotonic(); route_row=None; route_cached=False; selector=None
                base={"case_key":key,"run_id":run_id,"query_id":qid,"dataset_revision":spec["dataset"]["revision"],
                    "N":n,"k":5,"strategy":strategy,"seed":42,"mode":"live" if permitted else "offline",
                    "evaluator_only_gold":sorted(gold[qid]),"status":"blocked","error_category":None,"attempt_count":0,
                    "cad_per_usd":float(os.getenv("CAD_PER_USD",str(cfg["budget"]["cad_per_usd"]))),"candidate_ids":[],"returned_ids":[]}
                try:
                    if strategy=="listing": candidates=library
                    elif strategy=="bm25": candidates=[x[0] for x in bm25_rank([mby[s]["metadata"] for s in library],library,[qby[qid]["query"]],5)[0]]
                    elif strategy=="vector":
                        subset=np.array([indices[s] for s in library]); scores=v["docs"][subset]@v["queries"][qindices[qid]]
                        candidates=[library[i] for i in np.argsort(-scores,kind="stable")[:5]]
                    else:
                        if not permitted: raise Blocked(blocker)
                        # Only proceed after listing/BM25/vector have valid responses for this query/N.
                        first_three=[r for r in completed if r["query_id"]==qid and r["N"]==n and r["strategy"] in {"listing","bm25","vector"}]
                        if len(first_three)!=3 or any(r["status"]!="valid" for r in first_three): raise Blocked("selector_prerequisite_failed")
                        rkey=digest([run_id,qid,"router"])
                        if rkey in route_cache: route_row=route_cache[rkey]; route_cached=True
                        else:
                            router=GuardedModel(pin=pin,cfg=cfg,case_id=rkey,max_output=96,allowed={"route_domain"})
                            rstart=time.monotonic()
                            try:
                                major=route(qby[qid]["query"],router)
                                route_row={"case_key":rkey,"query_id":qid,"major":major,"status":"valid"}
                            except Blocked as e:
                                route_row={"case_key":rkey,"query_id":qid,"status":"blocked","error_category":str(e)}
                            route_row.update(calls=sanitize(router.calls),wall_seconds=time.monotonic()-rstart)
                            append_row(result_file(cfg,'routes'),route_row); route_cache[rkey]=route_row
                        if route_row.get('status','valid')!='valid':raise Blocked(route_row['error_category'])
                        pool=[s for s in library if mby[s]["major"]==route_row["major"]]
                        base.update(predicted_major=route_row["major"],routing_retention=len(gold[qid]&set(pool))/len(gold[qid]))
                        if not pool:
                            base.update(status="empty_domain",error_category="routing_failure_no_fallback",first_stage=retrieval_metrics([],gold[qid],5),selection=selection_metrics([],gold[qid]))
                            raise Blocked("empty_domain")
                        scores=v["docs"][[indices[s] for s in pool]]@v["queries"][qindices[qid]]
                        candidates=[pool[i] for i in np.argsort(-scores,kind="stable")[:5]]
                    # Preserve the same query/ID-based display order independently of retrieval rank.
                    chosen=set(candidates)
                    display=[s for s in library if s in chosen]
                    base.update(candidate_ids=display,retrieval_ranking=candidates,
                        first_stage=retrieval_metrics(candidates,gold[qid],n if strategy=="listing" else 5),
                        candidate_recall=len(chosen&gold[qid])/len(gold[qid]),candidate_completeness=float(gold[qid]<=chosen),
                        retrieval_seconds=time.monotonic()-start-(route_row["wall_seconds"] if route_row and not route_cached else 0))
                    if not permitted:
                        # Real native loading on real candidates, but a scripted empty
                        # response: effective-prompt evidence only, never selection scores.
                        from langchain_core.messages import AIMessage
                        scripted=AIMessage(content="",tool_calls=[{"name":"select_skills","args":{"skill_ids":[]},"id":"offline-preflight"}])
                        preview=run_agent(qby[qid]["query"],[by[s] for s in display],GuardedModel(responses=[scripted],case_id=key))
                        call=preview["calls"][0]
                        base["effective_prompt_preflight"]={k:call[k] for k in ["input_token_upper_bound","input_tokens_estimated","estimate_method"] if k in call}
                        base["effective_prompt_preflight"].update(native_metadata_count=len(display),tool_names=[x["function"]["name"] for x in call["tools"]],
                            context_preflight_tokens=context_estimate(call),
                            conservative_context_fits=context_estimate(call)<=pin["max_prompt_tokens"] and context_estimate(call)+512<=pin["context_length"])
                        if qid==qids[0]: save_json(f"evidence/effective_prompts/{key}.json",preview)
                        raise Blocked(blocker)
                    selector=GuardedModel(pin=pin,cfg=cfg,case_id=key,max_output=cfg["llm"]["max_output_tokens"])
                    decision=run_agent(qby[qid]["query"],[by[s] for s in display],selector)
                    base.update(status=decision["status"],returned_ids=decision["selected"],decision=decision,
                        selection=selection_metrics(decision["selected"],gold[qid]),attempt_count=len(selector.calls))
                    if decision["status"]!="valid": base["error_category"]=decision["status"]
                    selector_usage=usage_totals(selector.calls)
                    logical=usage_totals(selector.calls+(route_row["calls"] if route_row else []))
                    base.update(selector_usage=selector_usage,logical_usage=logical,
                        billed_usage=usage_totals(selector.calls+(route_row["calls"] if route_row and not route_cached else [])),
                        routing_cached=route_cached,route_case_key=route_row["case_key"] if route_row else None,
                        routing_seconds=route_row["wall_seconds"] if route_row else 0,
                        selector_seconds=sum(c.get("latency_seconds",0) for c in selector.calls))
                except Blocked as e:
                    base["error_category"]=str(e)
                    if base["status"]!="empty_domain": base["status"]="context_limit" if str(e)=="context_limit" else "blocked"
                    if selector is not None:
                        base.update(attempt_count=len(selector.calls),failed_calls=sanitize(selector.calls))
                base["workflow_wall_seconds"]=time.monotonic()-start
                base["logical_workflow_seconds"]=base["workflow_wall_seconds"]+(route_row["wall_seconds"] if route_row and route_cached else 0)
                append_row(result_file(cfg,'trial'),sanitize(base)); completed.append(base)
                print(json.dumps({'query':qid,'N':n,'method':strategy,'status':base['status'],'exact':base.get('selection',{}).get('exact_set_match')}),flush=True)
                # Ambiguous paid failures stop all expansion, not repeated across cases.
                if permitted and base["status"]=="blocked" and base['error_category'] not in {'selector_prerequisite_failed','router_malformed','router_unknown_category'}:
                    save_memberships(run_id,membership)
                    raise Blocked(base["error_category"])
    save_memberships(run_id,membership)
    return {"run_id":run_id,"cases":len(completed),"valid":sum(r["status"]=="valid" for r in completed),
        "prerequisites_ok":all(r["status"]=="valid" for r in completed if r["strategy"] in {"listing","bm25","vector"}),
        "mode":"live" if permitted else "offline retrieval only","blocker":None if permitted else blocker}


def activation_smoke(live=False,cfg=None):
    cfg=cfg or config()
    if live:
        require_permission()
        canary=smoke(True,cfg)
        if not canary.get("canary_correct"): raise Blocked("canary_failed")
    pin=configured_preflight(cfg) if live else {"model":"scripted-disclosure-only"}
    by,qby,gold,metadata,v,indices,qindices=experiment_data()
    tasks=[("fixture-"+str(i),text,fixtures()) for i,text in enumerate(["Add two numbers.","Translate a French paragraph.","Draw a square."])]
    for qid in read_json("manifests/live_query_ids.json")[:2]:
        library=make_nested_catalogs(list(by),gold[qid],[50],qid)[50]
        scores=v["docs"][[indices[s] for s in library]]@v["queries"][qindices[qid]]
        chosen={library[i] for i in np.argsort(-scores,kind="stable")[:5]}
        tasks.append((qid,qby[qid]["query"],[by[s] for s in library if s in chosen]))
    output=[]
    for qid,query,candidates in tasks:
        key=digest([run_config(cfg,pin),"activation",qid])
        destination=result_file(cfg,'activation') if live else "evidence/activation_offline.jsonl"
        existing=[r for r in rows(destination) if r["case_key"]==key]
        if existing: output.append(existing[-1]);continue
        model=GuardedModel(pin=pin,cfg=cfg,case_id=key)
        if not live:
            from langchain_core.messages import AIMessage
            sid=candidates[0]["id"]
            model.responses=[AIMessage(content="",tool_calls=[{"name":"read_file","args":{"file_path":f"/skills/s-{sid}/SKILL.md","offset":0,"limit":100},"id":"offline-read"}]),
                             AIMessage(content="",tool_calls=[{"name":"select_skills","args":{"skill_ids":[sid]},"id":"offline-select"}])]
        try:
            result=run_agent(query,candidates,model,True)
        except Blocked as e:
            result={"status":"blocked","selected":[],"reads":[],"calls":sanitize(model.calls),"error_category":str(e)}
        row={"case_key":key,"query_id":qid,**result,"evidence_class":"live native disclosure, NOT task execution success" if live else "Actual native read with scripted model; NOT live agent or semantic accuracy"}
        append_row(destination,row);output.append(row)
        if live:print(json.dumps({'activation':qid,'status':row['status'],'reads':len(row['reads'])}),flush=True)
        # Policy denials are recorded outcomes. Never retry that case; other
        # independent cases may proceed. Transport/accounting failures stop.
        policy_failures={'parallel_reads_forbidden','read_window_limit','read_outside_candidates','per_workflow_call_cap','read_count_limit'}
        if live and result["status"]=="blocked" and result.get('error_category') not in policy_failures: break
    return {"tasks":len(output),"native_reads":sum(len(r["reads"]) for r in output),"statuses":[r["status"] for r in output]}


def dispatch(a):
    try:
        if a.command=="offline-check": return offline_check()
        if a.command=="vectors": return vectors()
        if a.command=="smoke": return smoke(a.live,config(a.config))
        if a.command=="run": return run_trial(config(a.config),a.quick,a.live)
        if a.command=="activation-smoke": return activation_smoke(a.live,config(a.config))
        from handoff import report,package
        return {"report":report,"package":package}[a.command]()
    except Blocked as e:
        row={"command":a.command,"status":"BLOCKED","reason":str(e),"time":time.time(),"live_requested":a.live}
        append_row("results/blockers.jsonl",row)
        return row
