"""User-owned plaintext WSL credential store. Never echo or log values."""
import getpass,os,stat,sys,warnings
from pathlib import Path
STORE=Path.home()/'.local/share/mia5150-secrets'
NAMES={'jev':'jev.key','openrouter':'openrouter.key','runpod':'runpod.key'}
def validate(value):
 if not value or '\r' in value or '\n' in value or len(value)>8192:raise ValueError('Invalid key input')
 return value
def directory(root,create=False):
 if os.name!='posix':raise ValueError('WSL/Linux required')
 root=Path(root)
 if not root.is_absolute():raise ValueError('Absolute store required')
 for p in list(root.parents)[::-1]+[root]:
  if p.is_symlink():raise ValueError('Symlink path refused')
  if not p.exists():
   if not create:raise FileNotFoundError('Store unavailable')
   p.mkdir(mode=0o700)
  s=p.stat()
  if not stat.S_ISDIR(s.st_mode) or s.st_mode&0o022:raise ValueError('Writable-by-others path refused')
 s=root.stat()
 if s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o700:raise ValueError('Store must be owner-only mode700')
 return root
def save_key(provider,value,root=STORE):
 value=validate(value);root=directory(root,True);path=root/NAMES[provider]
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:f.write(value);f.flush();os.fsync(f.fileno())
def load_key(provider,root=STORE):
 root=directory(root);fd=os.open(root/NAMES[provider],os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'r') as f:
  s=os.fstat(f.fileno())
  if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o600 or s.st_nlink!=1:raise ValueError('Unsafe credential file refused')
  return validate(f.read(8193))
def obtain_key(provider,prompt):
 # Existing but unsafe stores/files fail closed, never silently overwritten.
 if STORE.exists() or STORE.is_symlink():
  directory(STORE)
  path=STORE/NAMES[provider]
  if path.exists() or path.is_symlink():return load_key(provider)
 if not sys.stdin.isatty():raise ValueError('Interactive input required')
 with warnings.catch_warnings():
  warnings.simplefilter('error',getpass.GetPassWarning);return validate(getpass.getpass(prompt).strip())
def main():
 if not sys.stdin.isatty():print('Run in an interactive Ubuntu terminal.');return 2
 print('Save keys locally to '+str(STORE)+' (directory700, separate plaintext .key files600).')
 print('Outside project, Windows mount and packages. No network calls. Your WSL account/root can read these keys; this is not encryption or protection from a compromised account or whole-WSL backups. Existing files are never overwritten.')
 directory(STORE,True)
 for provider in NAMES:
  path=STORE/NAMES[provider]
  if path.exists() or path.is_symlink():
   # Check metadata only, not existing credential contents during setup.
   s=path.lstat()
   if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o600 or s.st_nlink!=1:raise ValueError('Unsafe existing file')
   print(provider+': existing file preserved.');continue
  with warnings.catch_warnings():
   warnings.simplefilter('error',getpass.GetPassWarning);value=getpass.getpass(provider+' key (hidden; Enter skips): ').strip()
  if value:save_key(provider,value);print(provider+': saved.')
  else:print(provider+': skipped.')
  del value
 print('Finished. Spending confirmations still apply. No credential was transmitted.');return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (Exception,KeyboardInterrupt):print('Stopped; existing files preserved. No sensitive details displayed.');raise SystemExit(1)
