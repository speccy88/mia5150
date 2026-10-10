import os,tempfile,unittest
from pathlib import Path
from secret_store import save_key,load_key
@unittest.skipUnless(os.name=='posix','WSL permission tests')
class StoreTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);self.root=Path(self.temp.name)/'store'
 def tearDown(self):self.temp.cleanup()
 def test_roundtrip_permissions_and_no_overwrite(self):
  save_key('jev','DUMMY-NOT-A-CREDENTIAL',self.root)
  self.assertEqual(load_key('jev',self.root),'DUMMY-NOT-A-CREDENTIAL')
  self.assertEqual(self.root.stat().st_mode&0o777,0o700)
  self.assertEqual((self.root/'jev.key').stat().st_mode&0o777,0o600)
  with self.assertRaises(FileExistsError):save_key('jev','different-dummy',self.root)
  with self.assertRaises(FileNotFoundError):load_key('openrouter',self.root)
 def test_unsafe_permissions_refused(self):
  save_key('jev','dummy',self.root);(self.root/'jev.key').chmod(0o644)
  with self.assertRaises(ValueError):load_key('jev',self.root)
 def test_symlink_and_hardlink_refused(self):
  save_key('jev','dummy',self.root);(self.root/'runpod.key').symlink_to(self.root/'jev.key')
  with self.assertRaises(OSError):load_key('runpod',self.root)
  os.link(self.root/'jev.key',self.root/'openrouter.key')
  with self.assertRaises(ValueError):load_key('openrouter',self.root)
 def test_symlink_directory_refused(self):
  self.root.symlink_to(self.temp.name,target_is_directory=True)
  with self.assertRaises(ValueError):save_key('jev','dummy',self.root)
