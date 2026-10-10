from provider_auth_checks import main
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Read-only RunPod access check; no pods created or changed.')
 parser.add_argument('--legacy',action='store_true',help='Explicitly check the documented v1 endpoint instead of v2')
 args=parser.parse_args()
 raise SystemExit(main('runpod',legacy=args.legacy))
