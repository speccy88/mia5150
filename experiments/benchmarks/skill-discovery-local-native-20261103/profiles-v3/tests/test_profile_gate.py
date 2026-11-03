import unittest
from unittest.mock import patch
import bench_profiles_v3 as b


class ProfileGateTests(unittest.TestCase):
    def trial(self):
        return {'key':'500:0','n':500,'qi':0,'query':{'query':'task','gold_skill':'gold','gold_id':'g'},
            'candidates':[{'id':'g','name':'gold','description':'d'}], 'messages':[]}

    def test_old_estimate_does_not_discard_an_exact_token_profile(self):
        with patch.object(b,'infer',side_effect=b.BenchError('SYNTHETIC_REACHED_INFER')) as infer:
            row,_=b.evaluate({'backend':'llama','model':'test','context_window':1024},self.trial())
        infer.assert_called_once()
        self.assertEqual(row['error'],'SYNTHETIC_REACHED_INFER')

    def test_real_tokens_and_output_reserve_prevent_generation(self):
        config={'backend':'llama','model':'test','context_window':1024,'base_url':'http://127.0.0.1:28085'}
        def request(url,*args,**kwargs):
            if url.endswith('/apply-template'): return {'prompt':'test'}
            if url.endswith('/tokenize'): return {'tokens':list(range(525))}
            self.fail('Generation must not occur when 525+500 exceeds context1024')
        with patch.object(b,'check_gpu_proof'),patch.object(b,'request',side_effect=request):
            with self.assertRaises(b.BenchError) as error: b.infer(config,self.trial())
        self.assertEqual(error.exception.code,'CONTEXT_LIMIT_TOKENIZED')


if __name__=='__main__': unittest.main()
