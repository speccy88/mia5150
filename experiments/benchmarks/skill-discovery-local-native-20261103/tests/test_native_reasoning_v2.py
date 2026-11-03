import unittest
from unittest.mock import patch

import bench_native_v2 as b


class NativeReasoningV2Tests(unittest.TestCase):
    def test_prefilled_think_is_parse_state_not_rank_one(self):
        text, error = b.final_answer('Internal analysis.\n1. gold\n</think>\n1. wrong\n2. gold', '<think>\n')
        self.assertIsNone(error)
        candidates=[{'name':'gold'},{'name':'wrong'}]
        ranked=b.parse_ranked(text,candidates)
        self.assertEqual([x['name'] for x in ranked],['wrong','gold'])

    def test_unfinished_thinking_cannot_score_an_internal_gold_mention(self):
        text, error=b.final_answer('Choose:\n1. gold', '<think>\n')
        self.assertEqual(text,'')
        self.assertEqual(error,'REASONING_NO_FINAL_ANSWER')

    def test_gemma_thought_channel_is_not_final_answer(self):
        text,error=b.final_answer('<|channel>thought\n1. gold\n<channel|>\n1. wrong\n2. gold')
        self.assertIsNone(error)
        self.assertEqual(text.strip(),'1. wrong\n2. gold')

    def test_unfinished_gemma_channel_has_no_final_answer(self):
        self.assertEqual(b.final_answer('1. gold','<|channel>thought\n'),('', 'REASONING_NO_FINAL_ANSWER'))

    def test_invalid_final_slot_is_still_preserved(self):
        text,error=b.final_answer('<think>Analyze.\n</think>\n1. unknown\n2. gold')
        self.assertIsNone(error)
        ranked=b.parse_ranked(text,[{'name':'gold'}])
        self.assertIsNone(ranked[0]['name'])
        self.assertEqual(ranked[1]['name'],'gold')

    def test_prefix_detection_is_limited_to_template_suffix(self):
        self.assertEqual(b.reasoning_prefix('system <think> example\nassistant\n'),'')
        self.assertEqual(b.reasoning_prefix('assistant\n<think>\n'),'<think>\n')

    def test_failed_reasoning_keeps_real_token_usage(self):
        config={'backend':'llama','model':'synthetic','context_window':32768}
        trial={'n':2,'query':{'query':'task','gold_skill':'gold'},'candidates':[{'name':'gold'}]}
        response={'text':'1. gold','usage':{'input_tokens':100,'output_tokens':500},
                  'llm_s':1.0,'e2e_s':1.1,'preflight_tokens':100,
                  'response_audit':{'reasoning_prefix':'<think>\n'}}
        with patch.object(b,'infer',return_value=response):
            row,audit=b.evaluate(config,trial)
        self.assertEqual(row['error'],'REASONING_NO_FINAL_ANSWER')
        self.assertEqual(row['input_tokens'],100)
        self.assertEqual(row['output_tokens'],500)
        self.assertEqual(row['rank_of_gold'],0)
        self.assertFalse(row['exact_match'])


if __name__=='__main__':
    unittest.main()
