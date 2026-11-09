"""Independent old loader versus zero-adapter new loader on non-test text."""
from engine import *

def main():
 historical=module('archived_loader_reference',OLD/'run_skillret_baseline.py');assert historical.PREFIX==PREFIX;mp=CACHE/'models/embedding';tok=AutoTokenizer.from_pretrained(mp,local_files_only=True,padding_side='left');texts=[PREFIX+'Find a Python CSV parser for transaction records',PREFIX+'Plot café sales by month and export a figure'];m,start=historical.load_model(mp);reference=[historical.encode_one(m,tok,text,32768)[0] for text in texts];del m;torch.cuda.empty_cache();m,tok=load(train=True);m.eval();errors=[]
 for text,ref in zip(texts,reference):
  with torch.inference_mode():actual=vector(m,tok,text).cpu().numpy()
  errors.append(float(np.max(np.abs(actual-ref))));assert np.array_equal(actual,ref)
 save(STUDY/'results/independent_loader_parity.json',{'status':'passed','fixture':'Two synthetic non-test queries, one Unicode; original archived load_model+encode_one versus new zero-LoRA loader','archive_loader_sha256':sha(OLD/'run_skillret_baseline.py'),'new_engine_sha256':sha(SRC/'engine.py'),'model_revision':MODEL_REV,'max_absolute_error':max(errors),'exact_numeric_vectors_equal':True,'training_updates':0})
 print('Independent original-loader/zero-LoRA parity passed',flush=True)
if __name__=='__main__':main()
