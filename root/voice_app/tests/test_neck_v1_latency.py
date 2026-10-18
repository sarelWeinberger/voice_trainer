"""Latency/error contract tests, with no provider calls or patient data."""
import io
import json
import urllib.error
import unittest
from unittest.mock import patch
from neck_v1_openai import interpret, NeckProviderError
from neck_v1_scoring import NeckEffortConfig
from test_neck_effort_engine import context, prediction

class LatencyTests(unittest.TestCase):
    def test_reduced_evidence_and_output_budget(self):
        self.assertEqual(NeckEffortConfig().max_evidence_frames,8)
        def transport(payload):
            self.assertEqual(payload['max_output_tokens'],1600)
            return dict(status='completed',output=[{'content':[{'type':'output_text','text':prediction().model_dump_json()}]}])
        interpret(context(),[],transport)

    def test_http_errors_are_actionable_without_leaking_body(self):
        for code in (401,403,429,500):
            error=urllib.error.HTTPError('https://api.openai.com',code,'sensitive provider body',{},io.BytesIO(b'patient image secret'))
            with patch.dict('os.environ',{'OPENAI_API_KEY':'private-key'}), patch('urllib.request.urlopen',side_effect=error):
                with self.assertRaises(NeckProviderError) as raised:interpret(context(),[])
                self.assertIn(str(code),str(raised.exception))
                self.assertNotIn('private-key',str(raised.exception))
                self.assertNotIn('patient',str(raised.exception))

    def test_429_preserves_safe_code_without_provider_message(self):
        for code in ('credit_balance_exhausted','project_spend_limit_exceeded','rate_limit_exceeded'):
            body=json.dumps({'error':{'code':code,'type':'insufficient_quota','message':'patient private-key'}}).encode()
            error=urllib.error.HTTPError('https://api.openai.com',429,'sensitive',{},io.BytesIO(body))
            with patch.dict('os.environ',{'OPENAI_API_KEY':'private-key'}), patch('urllib.request.urlopen',side_effect=error):
                with self.assertRaises(NeckProviderError) as raised:interpret(context(),[])
                self.assertIn('['+code+']',str(raised.exception))
                self.assertNotIn('private-key',str(raised.exception))
                self.assertNotIn('patient',str(raised.exception))

    def test_429_unknown_code_not_exposed_and_quota_fallback(self):
        for kind, expected in [('insufficient_quota',True),('unknown',False)]:
            body=json.dumps({'error':{'code':'private-key','type':kind}}).encode()
            error=urllib.error.HTTPError('https://api.openai.com',429,'sensitive',{},io.BytesIO(body))
            with patch.dict('os.environ',{'OPENAI_API_KEY':'private-key'}), patch('urllib.request.urlopen',side_effect=error):
                with self.assertRaises(NeckProviderError) as raised:interpret(context(),[])
                self.assertNotIn('private-key',str(raised.exception))
                self.assertEqual('[insufficient_quota]' in str(raised.exception),expected)

    def test_timeout_is_explicit_and_network_timeout_is_bounded(self):
        with patch.dict('os.environ',{'OPENAI_API_KEY':'private-key'}), patch('urllib.request.urlopen',side_effect=TimeoutError) as network:
            with self.assertRaisesRegex(NeckProviderError,'timed out'):interpret(context(),[])
            self.assertEqual(network.call_args.kwargs['timeout'],45)

if __name__=='__main__':unittest.main()
