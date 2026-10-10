import unittest
from request_clause_view import request_view
class ViewTests(unittest.TestCase):
 def test_discards_background_retains_requirement(self):self.assertEqual(request_view('We sell shoes. Please build an API. It must log errors.'),'Please build an API. It must log errors.')
 def test_no_cue_preserves_original(self):self.assertEqual(request_view('A short context-only example.'),'A short context-only example.')
