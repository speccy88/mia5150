pilot=json.loads((r/'fixtures/skillret-pilot.json').read_text());index=next(i for i,q in enumerate(pilot['queries']) if q['id']=='q-02348');query=pilot['queries'][index];shapes=[]
def capture(module,args,kwargs):
 if kwargs.get('input_ids') is not None:shapes.append(list(kwargs['input_ids'].shape))
 elif kwargs.get('inputs_embeds') is not None:shapes.append(list(kwargs['inputs_embeds'].shape[:2]))
handle=native.language_model.register_forward_pre_hook(capture,with_kwargs=True)
vectors=np.stack([embed(query['query'],'SearchQuery',f'sensitivity/{repeat}') for repeat in range(3)]);handle.remove();np.save(o/'query-1024.npy',vectors)
assert len(shapes)==3 and all(x==[1,836] for x in shapes),shapes
old=r/'evaluation-compatible';docs=np.load(old/'skillret-docs.npy');original=np.load(old/'skillret-queries-0.npy');updated=original.copy();updated[index]=vectors[0];np.save(o/'skillret-queries-sensitive.npy',updated)
ids=[d['id'] for d in pilot['documents']];before=[ids[j] for j in np.argsort(-(docs@original[index]),kind='stable')];after=[ids[j] for j in np.argsort(-(docs@vectors[0]),kind='stable')];gold=set(pilot['qrels'][query['id']]);save('sensitivity',{'query_id':query['id'],'index':index,'actual_input_shapes':shapes,'before_top20':before[:20],'after_top20':after[:20],'before_recall':{k:len(set(before[:k])&gold)/len(gold) for k in [1,5,10,20]},'after_recall':{k:len(set(after[:k])&gold)/len(gold) for k in [1,5,10,20]},'query_cosine_before_after':float(original[index]@vectors[0]),'repeat_max_absolute_difference':float(np.max(np.abs(vectors-vectors[0]))),'changed_rows':1,'original_scores_preserved':True})
save('torch-memory',{'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved()});stage('done');print('SENSITIVITY_COMPLETE',flush=True)
