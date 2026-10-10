"""Create one portable results ZIP before Pod termination. Whitelist paths; never read environment secrets."""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,type=Path)
    p.add_argument('--data',action='append',required=True,type=Path)
    p.add_argument('--extra',action='append',default=[],type=Path,help='Explicit additional public evidence directories, e.g. scoring outputs')
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    root=Path(__file__).resolve().parents[1]
    items={}
    for folder in ('configs','docs','scripts','skillret_decider','tests','notebooks','evidence','examples'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                items['project/'+path.relative_to(root).as_posix()]=path
    for name in ('START_HERE.md','HERMES_HANDOFF.md','requirements.txt','PROTOCOL.md','STATUS.md'):
        items['project/'+name]=root/name
    for category,folders in [('runs',args.run),('prepared_data',args.data),('extra_evidence',args.extra)]:
        for i,folder in enumerate(folders):
            if not folder.is_dir(): raise ValueError(f'Not a directory: {folder}')
            for path in folder.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    if path.is_symlink(): raise ValueError('Do not follow symlinks outside the requested evidence')
                    items[f'{category}/{i:02d}_{folder.name}/'+path.relative_to(folder).as_posix()]=path
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in items.items()}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(items.items()): z.write(path,name)
        z.writestr('SHA256SUMS',''.join(f'{checksum}  {name}\n' for name,checksum in sorted(hashes.items())))
        z.writestr('START_HERE.md','# Results export\n\nRead project/START_HERE.md and runs/. Review STATUS.md: CPU package tests are not GPU validation.\n\nModel backbone and embedding caches are omitted; immutable upstream revisions and prepared inputs are retained. Verify hashes after copying off the Pod. Terminate only after an off-Pod copy is verified.\n')
    with zipfile.ZipFile(args.out) as z:
        for name,checksum in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=checksum: raise RuntimeError('ZIP verification failed')
    print(json.dumps({'zip':str(args.out),'files_verified':len(hashes),'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
        'billing_warning':'Local archive creation does NOT stop Pod billing; verify a downloaded copy, then terminate.'},indent=2))

if __name__=='__main__':main()
