"""File-specific clean-cache hints for this task's completed files only."""
import pathlib,os
roots=[pathlib.Path('/home/user/.cache/uv'),pathlib.Path('/home/user/code/mia5150-local')]
count=0
for root in roots:
 for p in root.rglob('*'):
  if p.is_file() and p.stat().st_size>1048576 and not p.name.endswith('.partial'):
   with p.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
   count+=1
print(count,'completed files hinted')
