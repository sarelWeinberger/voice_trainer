"""V1 neck-only regression: mocked interpretation and synthetic image controls."""
import base64
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
from pydantic import ValidationError
sys.path.insert(0,str((Path(__file__).resolve().parents[1] / "root" / "voice_app")))
from neck_effort_engine import NeckEffortEngine, NeckEffortConfig, evidence_context
from neck_v1_scoring import DIMENSIONS, ModelOutput, NeckAnalysis, aggregate, category
from neck_v1_cv import fit_rigid, residual_flow, jaw_residualization, select_frames
from neck_v1_openai import interpret, NeckProviderError


def prediction(scores=(1,1,1,1,1), independent=True):
    return ModelOutput.model_validate(dict(dimensions={key:dict(observable=True,score=score,
        continuous_score=float(score),confidence=.95,evidence=['נצפה שינוי בפריימים'],
        frame_indices=[0,1],speech_related=True,independent_local_activity=independent,confounds=[])
        for key,score in zip(DIMENSIONS,scores)},summary='תיאור התנהגות נצפית',limitations=[]))


def context(quality=1.0):
    return dict(selected_indices=[0,1],dimension_quality={k:quality for k in DIMENSIONS},
                temporal_quality=1.0,speech_status='active',jaw_explained_fraction=0,
                quality={'overall':quality,'warnings':[]},versions={'analyzer':'neck-analysis-v1'})


def row(t):
    return dict(t=t,features={'jaw_velocity':0.0,'anterior_motion':0.0},visibility={
        k:1.0 for k in ('face','shoulders','left_lateral','right_lateral','anterior','submental','jaw')},
        rois={},lighting=1.0,resolution_quality=1.0,camera_stability=1.0,landmarks={})

class FakeProcessor:
    def reset(self): pass
    def close(self): pass
    def process(self,image,t): return row(t)

class NeckV1Tests(unittest.TestCase):
    def test_weighted_formula(self):
        result=aggregate(prediction((0,1,2,3,1)),context())
        self.assertAlmostEqual(result['overall']['continuous_score'],1.35)
        self.assertEqual(result['overall']['category'],1)
        NeckAnalysis.model_validate(result)

    def test_missing_shoulders_renormalizes(self):
        ctx=context();ctx['dimension_quality']['shoulder_recruitment']=0
        result=aggregate(prediction((0,1,2,3,1)),ctx)
        self.assertIsNone(result['dimensions']['shoulder_recruitment']['score'])
        self.assertAlmostEqual(result['overall']['continuous_score'],round(1.05/.9,4))
        self.assertLess(result['overall']['confidence'],.95)

    def test_all_unavailable_is_not_zero(self):
        result=aggregate(prediction(),context(0))
        self.assertIsNone(result['overall']['category'])
        self.assertEqual(result['overall']['confidence'],0)

    def test_category_boundaries(self):
        for x,expected in [(0,0),(.49999,0),(.5,1),(1.49999,1),(1.5,2),(2.5,3),(3,3)]:
            self.assertEqual(category(x),expected)

    def test_score_ranges_and_strict_parser(self):
        for field,bad in [('score',4),('score',True),('continuous_score',3.1),('confidence',1.1),('confidence',float('nan'))]:
            raw=prediction().model_dump();raw['dimensions'][DIMENSIONS[0]][field]=bad
            with self.assertRaises(ValidationError): ModelOutput.model_validate(raw)
        raw=prediction().model_dump();raw['diagnosis']='MTD'
        with self.assertRaises(ValidationError): ModelOutput.model_validate(raw)

    def test_null_required_when_unobservable(self):
        raw=prediction().model_dump();raw['dimensions'][DIMENSIONS[0]]['observable']=False
        with self.assertRaises(ValidationError): ModelOutput.model_validate(raw)

    def test_unknown_frame_reference_rejected(self):
        out=prediction();out.dimensions.posture_compensation.frame_indices=[99]
        with self.assertRaises(ValueError): aggregate(out,context())

    def test_confidence_limited_by_quality(self):
        good=aggregate(prediction(),context())['overall']['confidence']
        bad=aggregate(prediction(),context(.4))['overall']['confidence']
        self.assertLess(bad,good)

    def test_head_and_jaw_do_not_automatically_raise_neck(self):
        for scores in [(2,2,2,0,0),(0,0,2,0,2)]:
            result=aggregate(prediction(scores,False),context())
            self.assertIsNone(result['dimensions']['anterior_submental_activity']['score'])
            if scores[1]: self.assertIsNone(result['dimensions']['scm_lateral_recruitment']['score'])

    def test_silent_movement_has_no_effort_score(self):
        ctx=context();ctx['speech_status']='silent'
        result=aggregate(prediction((3,3,3,3,3)),ctx)
        self.assertIsNone(result['overall']['category'])

    def test_lateral_visible_independent_change_can_raise(self):
        result=aggregate(prediction((0,2,0,0,0)),context())
        self.assertEqual(result['dimensions']['scm_lateral_recruitment']['score'],2)

    def test_anterior_jaw_fit_does_not_penalize_lateral_evidence(self):
        out=prediction((0,1,1,0,0));ctx=context()
        out.dimensions.scm_lateral_recruitment.confidence=.17
        out.dimensions.anterior_submental_activity.confidence=.17
        ctx['jaw_explained_fraction']=.95
        result=aggregate(out,ctx)
        self.assertEqual(result['dimensions']['scm_lateral_recruitment']['score'],1)
        self.assertAlmostEqual(result['dimensions']['scm_lateral_recruitment']['confidence'],.17)
        self.assertIsNone(result['dimensions']['anterior_submental_activity']['score'])
        self.assertEqual(result['diagnostics']['dimensions']['anterior_submental_activity']['agreement_factor'],.7)

    def test_lower_confidence_threshold_cannot_remove_local_ambiguity(self):
        for threshold in (.15,.10,.05):
            out=prediction((0,1,1,0,0),False)
            result=aggregate(out,context(),NeckEffortConfig(min_dimension_confidence=threshold))
            for name in DIMENSIONS[1:3]:
                self.assertIsNone(result['dimensions'][name]['score'])
                decision=result['diagnostics']['dimensions'][name]
                self.assertEqual(decision['observation']['status'],'ambiguous_local_change')
                self.assertEqual(decision['observation']['frame_indices'],[0,1])
                self.assertFalse(decision['observation']['effort_assessable'])
                self.assertFalse(decision['model_independent_local_activity'])

    def test_subcategory_positive_local_change_still_requires_separability(self):
        out=prediction((0,0,0,0,0),False)
        out.dimensions.scm_lateral_recruitment.continuous_score=.4
        result=aggregate(out,context())
        self.assertIsNone(result['dimensions']['scm_lateral_recruitment']['score'])

    def test_clear_supported_neck_and_shoulder_changes_are_not_forced_to_zero(self):
        result=aggregate(prediction((0,2,2,2,0)),context())
        for name in DIMENSIONS[1:4]:
            self.assertEqual(result['dimensions'][name]['score'],2)
            self.assertNotIn('observation',result['diagnostics']['dimensions'][name])

    def test_regression_fixture_cases(self):
        fixtures=json.loads(Path(__file__).with_name('fixtures').joinpath('neck_v1_cases.json').read_text())
        for case in fixtures:
            with self.subTest(case=case['name']):
                ctx=context(case.get('quality',1.0));ctx['speech_status']=case.get('speech','active')
                if case.get('neck_hidden'):
                    for name in DIMENSIONS[1:3]:ctx['dimension_quality'][name]=0
                if case.get('shoulders_hidden'):ctx['dimension_quality']['shoulder_recruitment']=0
                result=aggregate(prediction(case['scores'],case['local_independent']),ctx)
                NeckAnalysis.model_validate(result)
                if case['name']=='relaxed_speech':self.assertEqual(result['overall']['category'],0)
                if case.get('shoulders_hidden'):self.assertIsNone(result['dimensions']['shoulder_recruitment']['score'])

    def test_similarity_compensation(self):
        source=np.array([[10,10],[30,10],[30,30],[10,30]],float)
        target=source+np.array([8,4])
        transform=fit_rigid(source,target)
        np.testing.assert_allclose(transform,[[1,0,8],[0,1,4]],atol=1e-6)
        rng=np.random.default_rng(0)
        image=(rng.random((96,96))*255).astype(np.uint8)
        moved=cv2.warpAffine(image,transform,(96,96))
        flow,support=residual_flow(image,moved,transform)
        self.assertLess(np.median(np.linalg.norm(flow,axis=2)[support]),.05)

    def test_local_deformation_survives_rigid_control(self):
        rng=np.random.default_rng(1)
        image=(rng.random((96,96))*255).astype(np.uint8)
        changed=image.copy();changed[30:70,30:70]=np.roll(image[30:70,30:70],3,axis=1)
        flow,_=residual_flow(image,changed,np.array([[1,0,0],[0,1,0]],np.float32))
        self.assertGreater(np.mean(np.linalg.norm(flow[35:65,35:65],axis=2)),.1)

    def test_jaw_regression_removes_explained_activity(self):
        jaw=np.linspace(-1,1,50)
        motion=2+3*abs(jaw)+jaw
        residual,fraction=jaw_residualization(motion,jaw)
        self.assertGreater(fraction,.99)
        self.assertLess(np.mean(residual),.01)

    def test_frame_selection_deterministic_and_bounded(self):
        rows=[row(i/5) for i in range(150)]
        rows[80]['features']['lateral_motion']=100
        a=select_frames(rows);b=select_frames(rows)
        self.assertEqual(a,b);self.assertLessEqual(len(a),24)
        self.assertIn(0,a);self.assertIn(149,a);self.assertIn(80,a)

    def test_camera_stability_and_short_duration_lower_quality(self):
        rows=[row(i/5) for i in range(30)]
        good=evidence_context(rows,[0,1],NeckEffortConfig())
        for item in rows:item['camera_stability']=.1
        bad=evidence_context(rows,[0,1],NeckEffortConfig())
        self.assertLess(bad['dimension_quality'][DIMENSIONS[1]],good['dimension_quality'][DIMENSIONS[1]])
        self.assertIn('camera_motion',bad['quality']['warnings'])

    def test_audio_silence_and_no_overlap(self):
        rows=[row(i/5) for i in range(30)]
        audio=[dict(t=0,duration=6,speech_active=False)]
        ctx=evidence_context(rows,[0,1],NeckEffortConfig(),audio,'caller_declared_offset')
        self.assertEqual(ctx['speech_status'],'silent')
        audio[0]['t']=30
        ctx=evidence_context(rows,[0,1],NeckEffortConfig(),audio,'caller_declared_offset')
        self.assertEqual(ctx['speech_status'],'unknown')

    def test_container_timing_retains_sync_uncertainty(self):
        rows=[row(i/5) for i in range(30)]
        audio=[dict(t=0,duration=6,speech_active=True)]
        estimated=evidence_context(rows,[0,1],NeckEffortConfig(),audio,'container_estimated_offset')
        declared=evidence_context(rows,[0,1],NeckEffortConfig(),audio,'caller_declared_offset')
        self.assertAlmostEqual(estimated['temporal_quality'],declared['temporal_quality']*.8)
        self.assertIn('speech_timing_unverified',estimated['quality']['warnings'])
        self.assertEqual(estimated['speech_status'],'active')

    def test_provider_payload_and_parsing(self):
        captured=[]
        def transport(payload):
            captured.append(payload)
            return dict(status='completed',output=[{'content':[{'type':'output_text','text':prediction().model_dump_json()}]}])
        result=interpret(context(),[dict(index=0,t=0,kind='full_frame',url='data:image/jpeg;base64,AA==')],transport)
        self.assertIsInstance(result,ModelOutput)
        self.assertFalse(captured[0]['store'])
        self.assertTrue(captured[0]['text']['format']['strict'])
        self.assertNotIn('overall',captured[0]['text']['format']['schema']['properties'])

    def test_missing_key_never_calls_provider(self):
        with patch.dict('os.environ',{},clear=True), patch('urllib.request.urlopen') as network:
            with self.assertRaises(NeckProviderError): interpret(context(),[])
            network.assert_not_called()

    def test_provider_refusal_malformed_incomplete(self):
        for response in [{'status':'incomplete'},dict(status='completed',output=[{'content':[{'type':'refusal'}]}]),
                         dict(status='completed',output=[{'content':[{'type':'output_text','text':'not json'}]}])]:
            with self.assertRaises(NeckProviderError): interpret(context(),[],lambda payload:response)

    def engine(self,interpreter):
        return NeckEffortEngine(processor=FakeProcessor(),interpreter=interpreter)

    def test_session_finalizes_once_and_clears_images(self):
        calls=[]
        def interpreter(ctx,images):
            calls.append((ctx,len(images)))
            output=prediction()
            for name in DIMENSIONS:
                getattr(output.dimensions,name).frame_indices=ctx['selected_indices'][:2]
            return output
        engine=self.engine(interpreter)
        try:
            engine.start_recording_session('abc')
            for i in range(30):engine.process_frame(np.ones((100,100,3),np.uint8)*100,timestamp=i/5)
            result=engine.finalize();again=engine.finalize()
            self.assertIs(result,again);self.assertEqual(len(calls),1)
            self.assertEqual(result['score'],1)
            self.assertEqual(len(result['neckAnalysis']['dimensions']),5)
            self.assertFalse(engine.images)
            self.assertNotIn('landmarks',engine.export_recording()[0])
            self.assertLessEqual(calls[0][1],30)
        finally:engine.close()

    def test_provider_failure_no_fabricated_score(self):
        def fail(*args):raise NeckProviderError('provider unavailable')
        engine=self.engine(fail)
        try:
            engine.start_recording_session()
            for i in range(30):engine.process_frame(np.zeros((100,100,3),np.uint8),timestamp=i/5)
            result=engine.finalize()
            self.assertFalse(result['ok']);self.assertIsNone(result['score'])
            self.assertFalse(engine.images)
            self.assertEqual(result['neckAnalysis']['observable_change']['version'],'neck-observable-change-v1')
            self.assertGreater(len(result['neckAnalysis']['observable_change']['windows']),0)
        finally:engine.close()

    def test_not_recording_and_invalid_timestamps(self):
        engine=self.engine(lambda *args:prediction())
        try:
            image=np.zeros((100,100,3),np.uint8)
            self.assertEqual(engine.process_frame(image)['mode'],'no_score')
            engine.start_recording_session();engine.process_frame(image,timestamp=0)
            self.assertFalse(engine.process_frame(image,timestamp=0)['ok'])
        finally:engine.close()

if __name__=='__main__':unittest.main()
