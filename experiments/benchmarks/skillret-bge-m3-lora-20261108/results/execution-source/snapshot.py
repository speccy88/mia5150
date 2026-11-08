"""Use the baseline's exact lossless format and model-free ranking replay."""
import argparse
from common import BASE,CACHE,module

snapshot=module('immutable_bge_snapshot',BASE/'src/snapshot.py')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('export','replay'))
    args=parser.parse_args()
    snapshot.export(CACHE) if args.action=='export' else snapshot.replay()
