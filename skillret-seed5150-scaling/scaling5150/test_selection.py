import unittest
from selection import alias,parse_set
class Selection(unittest.TestCase):
 def test_validation(self):
  self.assertEqual(parse_set('["x"]',{'x':'skill'}),['skill']);self.assertEqual(parse_set('[]',{'x':'skill'}),[])
  for s in ['["x","x"]','["z"]','{}','[1]','["a","b","c","d","e","f"]']:
   with self.assertRaises((ValueError,AssertionError)):parse_set(s,{x:x for x in 'abcdefx'})
 def test_alias_is_query_specific_and_deterministic(self):
  self.assertEqual(alias('q','d'),alias('q','d'));self.assertNotEqual(alias('q','d'),alias('other','d'))
if __name__=='__main__':unittest.main()
