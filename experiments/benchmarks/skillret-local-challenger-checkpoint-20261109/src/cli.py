"""Low-memory local entry points; all model execution is local CUDA."""
import argparse,subprocess,sys
from common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['index','train-query','train-reranker','evaluate','verify']);a,rest=p.parse_known_args()
 scripts={'index':'index.py','train-query':'train_query.py','train-reranker':'train_reranker.py','evaluate':'evaluate_final.py','verify':'verify.py'}
 sys.exit(subprocess.call([sys.executable,str(SRC/scripts[a.command])]+rest))
if __name__=='__main__':main()
