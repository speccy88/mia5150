"""Update only changed publication hashes from index bytes, preserving LFS hashes."""
import argparse,hashlib,pathlib,subprocess
root=pathlib.Path(__file__).resolve().parents[1]
manifest_path='docs/publication/PUBLICATION_MANIFEST.sha256'
def git(*args):return subprocess.check_output(['git','-C',str(root),*args])
def main():
    a=argparse.ArgumentParser();a.add_argument('--base',default='HEAD');args=a.parse_args()
    text=git('show',args.base+':'+manifest_path).decode();entries={}
    for line in text.splitlines():
        if line and not line.startswith('#'):
            h,p=line.split('  ',1);entries[p]=h
    changed=git('diff','--cached','--name-only','--diff-filter=ACMR',args.base).decode().splitlines()
    deleted=git('diff','--cached','--name-only','--diff-filter=D',args.base).decode().splitlines()
    for p in deleted:entries.pop(p,None)
    for p in changed:
        if p==manifest_path:continue
        data=git('show',':'+p)
        if data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            import re
            entries[p]=re.search(rb'oid sha256:([0-9a-f]{64})',data).group(1).decode()
        else:entries[p]=hashlib.sha256(data).hexdigest()
    entries.pop(manifest_path,None)
    out=['# SHA-256 of the sanitized publication; paths relative to repository root. Hydrate Git LFS before checking.']
    out += [entries[p]+'  '+p for p in sorted(entries)]
    (root/manifest_path).write_text('\n'.join(out)+'\n')
    print('Manifest entries:',len(entries),'changed files:',len(changed),'source: normalized index bytes')
if __name__=='__main__':main()
