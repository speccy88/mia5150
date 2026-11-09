from engine import *
import sys

def main():
 part=sys.argv[1] if len(sys.argv)>1 else 'train'
 if part=='train':
  m,tok=load();s,q,g=data('train');encode_cache(m,tok,[fulltext(x) for x in s],[x['id'] for x in s],'train-documents',32768)
  if not (CACHE/'vectors/train-documents/complete.json').exists():return
  membership=read(STUDY/'manifests/split_membership.json');eligible=set(membership['train_query_ids']+membership['selection_query_ids']+membership['confirmation_query_ids']);queries=[x for x in q if x['id'] in eligible]
  budget=read(STUDY/'results/data_audit.json')['query_budget'];encode_cache(m,tok,[PREFIX+x['query'] for x in queries],[x['id'] for x in queries],'train-queries',budget)
  docs,dids=load_vectors('train-documents');base,bids=load_vectors('train-queries');bmap={x:i for i,x in enumerate(bids)}
  for panel in ('selection','confirmation'):
   if panel=='confirmation':continue # Confirmation remains sealed until challenger selection lock.
   ids=membership[panel+'_query_ids'];r={i:retrieve(base[bmap[i]],docs,dids)[0] for i in ids};save(STUDY/'results'/f'frozen-{panel}.json',{'evaluation':evaluate({i:g[i] for i in ids},r),'rankings':r})
 else:raise ValueError(part)
if __name__=='__main__':main()
