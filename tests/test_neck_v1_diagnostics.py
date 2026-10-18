"""Explain abstentions without loosening scientific acceptance rules."""
import unittest
from test_neck_effort_engine import prediction, context
from neck_v1_scoring import aggregate, DIMENSIONS, NeckAnalysis
class DiagnosticsTests(unittest.TestCase):
    def test_zero_jaw_is_not_a_focus_and_confounded_neck_remains_unavailable(self):
        output=prediction((1,1,1,1,0),independent=False)
        output.dimensions.posture_compensation.continuous_score=1.1
        output.dimensions.shoulder_recruitment.speech_related=False
        result=aggregate(output,context())
        self.assertEqual(result['overall']['continuous_score'],.4714)
        self.assertEqual(result['overall']['category'],0)
        self.assertEqual(result['dimensions']['jaw_compensation']['score'],0)
        focus=result['summary'].split('.')[0]
        self.assertIn('תנוחת ראש וצוואר',focus)
        self.assertNotIn('לסת',focus)
        for name in ('scm_lateral_recruitment','anterior_submental_activity'):
            self.assertIsNone(result['dimensions'][name]['score'])
            self.assertEqual(result['diagnostics']['dimensions'][name]['rejection_reason'],
                             'Local activity not distinguishable from jaw/head mechanics')

    def test_shoulders_only_is_partial_and_does_not_turn_missing_scores_into_zero(self):
        output=prediction()
        for name in DIMENSIONS:
            item=getattr(output.dimensions,name)
            if name!='shoulder_recruitment':
                item.observable=False;item.score=None;item.continuous_score=None
                item.evidence=['לא נראה שינוי חריג באזור']
        shoulder=output.dimensions.shoulder_recruitment
        shoulder.continuous_score=.9;shoulder.confidence=.57
        result=aggregate(output,context())
        NeckAnalysis.model_validate(result)
        self.assertEqual(result['overall']['continuous_score'],.9)
        self.assertEqual(result['overall']['confidence'],.057)
        coverage=result['diagnostics']['aggregation']
        self.assertTrue(coverage['partial'])
        self.assertEqual(coverage['assessed_dimensions'],['shoulder_recruitment'])
        self.assertEqual(coverage['original_weight_coverage'],.1)
        self.assertIn('סיכום חלקי',result['summary'])
        for name in DIMENSIONS:
            if name!='shoulder_recruitment':self.assertIsNone(result['dimensions'][name]['score'])

    def test_supported_zero_observations_do_not_require_positive_local_activity(self):
        result=aggregate(prediction((0,0,0,0,0),independent=False),context())
        self.assertTrue(all(item['score']==0 for item in result['dimensions'].values()))
        self.assertFalse(result['diagnostics']['aggregation']['partial'])
        self.assertEqual(result['diagnostics']['aggregation']['original_weight_coverage'],1.0)
        output=prediction((0,0,0,0,0),independent=False)
        output.dimensions.scm_lateral_recruitment.speech_related=False
        result=aggregate(output,context())
        self.assertIsNone(result['dimensions']['scm_lateral_recruitment']['score'])

    def test_sparse_sampling_alone_can_preclude_every_score(self):
        ctx=context();ctx['temporal_quality']=.0952
        result=aggregate(prediction(),ctx)
        self.assertIsNone(result['overall']['continuous_score'])
        for item in result['diagnostics']['dimensions'].values():
            self.assertLessEqual(item['adjusted_confidence'],.0952)
            self.assertEqual(item['rejection_reason'],'Insufficient evidence confidence')

    def test_model_and_cv_rejections_are_separate_and_model_evidence_survives(self):
        output=prediction();item=output.dimensions.posture_compensation
        item.observable=False;item.score=None;item.continuous_score=None
        item.evidence=['ההקלטה אינה מספקת הקשר דיבור ברור']
        result=aggregate(output,context())
        self.assertEqual(result['diagnostics']['dimensions']['posture_compensation']['rejection_reason'],'Model found insufficient observable evidence')
        self.assertEqual(result['diagnostics']['dimensions']['posture_compensation']['model_evidence'],item.evidence)
        ctx=context();ctx['dimension_quality']['posture_compensation']=.1
        result=aggregate(output,ctx)
        self.assertEqual(result['diagnostics']['dimensions']['posture_compensation']['rejection_reason'],'Tracking quality below threshold')
    def test_sparse_recording_skips_provider_without_fabricating_scores(self):
        from unittest.mock import Mock
        import numpy as np
        from neck_effort_engine import NeckEffortEngine
        from test_neck_effort_engine import FakeProcessor
        provider=Mock(side_effect=AssertionError('Must not call provider'))
        engine=NeckEffortEngine(processor=FakeProcessor(),interpreter=provider)
        try:
            engine.start_recording_session('sparse-test')
            for i in range(12):engine.process_frame(np.zeros((100,100,3),np.uint8),timestamp=i*1.8)
            result=engine.finalize()
            provider.assert_not_called()
            self.assertIsNone(result['score'])
            self.assertTrue(result['neckAnalysis']['diagnostics']['provider_skipped'])
            self.assertEqual(result['neckAnalysis']['diagnostics']['captured_frames'],12)
        finally:engine.close()

    def test_provider_timeout_retains_frame_diagnostics(self):
        import numpy as np
        from neck_effort_engine import NeckEffortEngine
        from neck_v1_openai import NeckProviderError
        from test_neck_effort_engine import FakeProcessor
        def fail(*args):raise NeckProviderError('OpenAI analysis timed out')
        engine=NeckEffortEngine(processor=FakeProcessor(),interpreter=fail)
        try:
            engine.start_recording_session('timeout-test')
            for i in range(30):engine.process_frame(np.zeros((100,100,3),np.uint8),timestamp=i/5)
            result=engine.finalize()
            self.assertEqual(result['mode'],'error')
            self.assertEqual(result['neckAnalysis']['diagnostics']['captured_frames'],30)
            self.assertIsNotNone(result['neckAnalysis']['diagnostics']['mean_cv_processing_seconds'])
            self.assertEqual(len(result['timeline']),30)
            self.assertEqual(engine.images,[])
        finally:engine.close()

if __name__=='__main__':unittest.main()
