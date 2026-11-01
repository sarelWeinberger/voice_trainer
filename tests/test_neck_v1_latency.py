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
        self.assertEqual(NeckEffortConfig().max_evidence_frames,16)
        def transport(payload):
            self.assertEqual(payload['max_output_tokens'],3200 if payload['model'].startswith('gpt-6') else 1600)
            return dict(status='completed',output=[{'content':[{'type':'output_text','text':prediction().model_dump_json()}]}])
        interpret(context(),[],transport)

    def test_sampling_parameter_matches_model_family(self):
        for model, supports_temperature in [('gpt-6.1-sol',False),('gpt-6-luna',False),('gpt-4.1-2025-04-14',True)]:
            with self.subTest(model=model), patch.dict('os.environ',{'NECK_OPENAI_MODEL':model}):
                def transport(payload):
                    self.assertEqual(payload['model'],model)
                    self.assertEqual(payload['max_output_tokens'],3200 if model.startswith('gpt-6') else 1600)
                    self.assertEqual('temperature' in payload,supports_temperature)
                    if supports_temperature:self.assertEqual(payload['temperature'],0)
                    return dict(status='completed',output=[{'content':[{'type':'output_text','text':prediction().model_dump_json()}]}])
                interpret(context(),[],transport)

    def test_output_limit_is_explicit_and_never_accepts_partial_scores(self):
        for text in ('{"dimensions":', prediction().model_dump_json()):
            response=dict(status='incomplete',incomplete_details={'reason':'max_output_tokens'},
                          output=[{'content':[{'type':'output_text','text':text}]}])
            with patch('neck_v1_openai.ModelOutput.model_validate_json') as validate:
                with self.assertRaisesRegex(NeckProviderError,r'\[max_output_tokens\]'):
                    interpret(context(),[],lambda payload:response)
                validate.assert_not_called()

    def test_other_incomplete_responses_are_not_misreported_as_token_limits(self):
        response=dict(status='incomplete',incomplete_details={'reason':'content_filter'},output=[])
        with self.assertRaisesRegex(NeckProviderError,'did not complete') as raised:
            interpret(context(),[],lambda payload:response)
        self.assertNotIn('max_output_tokens',str(raised.exception))

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
