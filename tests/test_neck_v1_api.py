"""Existing neck routes and storage contract; no network/API key required."""
import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0,str((Path(__file__).resolve().parents[1] / "root" / "voice_app")))
os.environ['DATABASE_URL']='sqlite:///'+str(Path(tempfile.mkdtemp(prefix='neck-v1-api-'))/'test.db')
from test_neck_effort_engine import FakeProcessor, prediction
from neck_effort_engine import NeckEffortEngine
import app as appmod

def memory_upload(data):
    file=tempfile.SpooledTemporaryFile(max_size=1024*1024)
    file.write(data);file.seek(0)
    return appmod.UploadFile(file=file)

class DummyDB:
    def __init__(self):self.saved=[]
    def add(self,item):self.saved.append(item)
    def commit(self):pass
    def rollback(self):pass

class APITests(unittest.TestCase):
    def setUp(self):
        self.db=DummyDB()
        def interpreter(ctx,images):
            result=prediction()
            for item in result.dimensions.__dict__.values():item.frame_indices=ctx['selected_indices'][:2]
            return result
        self.engine=NeckEffortEngine(processor=FakeProcessor(),interpreter=interpreter)
        self.user=SimpleNamespace(id=999)
        appmod.app.dependency_overrides[appmod.get_current_user]=lambda:self.user
        appmod.app.dependency_overrides[appmod.get_db]=lambda:self.db
        appmod.neck_effort_engines[999]=self.engine

    def tearDown(self):
        self.engine.close()
        appmod.neck_effort_engines.pop(999,None)
        appmod.app.dependency_overrides.clear()

    def test_existing_recording_routes_and_persistence(self):
        started=appmod.start_neck_effort_recording(self.user)
        self.assertTrue(started['session_id'])
        for i in range(30):self.engine.process_frame(np.ones((100,100,3),np.uint8)*100,timestamp=i/5)
        ok,jpeg=cv2.imencode('.jpg',np.ones((100,100,3),np.uint8)*100)
        self.engine.state.recording_started_at -= 6
        upload=memory_upload(jpeg.tobytes())
        try:
            frame=asyncio.run(appmod.analyze_neck_effort_frame(upload,self.user))
        finally:
            upload.file.close()
        parsed=appmod.NeckEffortOut(**frame)
        self.assertEqual(parsed.model_version,'neck-analysis-v1')
        packet=appmod.stop_neck_effort_recording(self.user,self.db)['neck_effort']
        self.assertEqual(packet['score'],1)
        self.assertEqual(len(packet['neckAnalysis']['dimensions']),5)
        self.assertEqual(len(self.db.saved),1)
        summary=json.loads(self.db.saved[0].summary)
        self.assertIn('neckAnalysis',summary)
        self.assertNotIn('data:image',self.db.saved[0].summary)
        self.assertNotIn('landmarks',self.db.saved[0].feature_series)
        appmod.NeckEffortOut(**packet)
        self.assertIn('/api/neck-effort',appmod.app.openapi()['paths'])

    def test_history_recovers_final_neck_result_when_audio_saved_first(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from database import Base, Result, NeckRecording
        database=create_engine('sqlite:///:memory:')
        Base.metadata.create_all(database)
        with Session(database) as db:
            db.add(Result(user_id=self.user.id, title='Joined analysis',session_id='joined',
                          raw_scores=json.dumps({'scores':{'G':1},'neck_effort':None})))
            db.add(NeckRecording(user_id=self.user.id,session_id='joined',feature_series='[]',
                                 summary=json.dumps({'neckAnalysis':{'status':'complete'}})))
            db.add(NeckRecording(user_id=self.user.id+1,session_id='joined',feature_series='[]',
                                 summary=json.dumps({'private':'other-user'})))
            db.commit()
            history=appmod.get_history(db,self.user)['history']
            self.assertEqual(len(history),1)
            self.assertEqual(history[0]['session_id'],'joined')
            self.assertEqual(history[0]['neck_effort'],{'neckAnalysis':{'status':'complete'}})
            self.assertEqual(history[0]['scores'],{'G':1})
        database.dispose()

    def test_history_returns_ten_latest_results_without_deleting_older_results(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from database import Base, Result
        from datetime import datetime, timedelta
        database=create_engine('sqlite:///:memory:');Base.metadata.create_all(database)
        with Session(database) as db:
            for i in range(12):
                db.add(Result(user_id=self.user.id,title='Saved analysis',
                    created_at=datetime(2026,1,1)+timedelta(seconds=i),
                    raw_scores=json.dumps({'scores':{'G':i}})))
            db.add(Result(user_id=self.user.id+1,title='Another account',raw_scores='{}'))
            db.commit()
            history=appmod.get_history(db,self.user)['history']
            self.assertEqual(len(history),10)
            self.assertEqual([entry['scores']['G'] for entry in history],list(range(11,1,-1)))
            self.assertEqual(db.query(Result).filter(Result.user_id==self.user.id).count(),12)
        database.dispose()

    def test_neck_only_history_survives_engine_restart_and_stays_user_scoped(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from database import Base, NeckRecording, Result
        database=create_engine('sqlite:///:memory:');Base.metadata.create_all(database)
        self.engine.start_recording_session('neck-without-audio')
        for i in range(30):self.engine.process_frame(np.ones((100,100,3),np.uint8)*100,timestamp=i/5)
        with Session(database) as db:
            stopped=appmod.stop_neck_effort_recording(self.user,db,session_id='neck-without-audio')
            self.assertTrue(stopped['history_saved'])
            self.assertEqual(db.query(Result).count(),0)
            db.add(NeckRecording(user_id=self.user.id+1,session_id='private-other-account',
                                feature_series='[]',summary='{"private":true}'))
            db.commit()
        # New DB session, and no in-memory recording engine available.
        with Session(database) as db, patch.dict(appmod.neck_effort_engines,{},clear=True):
            history=appmod.get_history(db,self.user)['history']
            self.assertEqual(len(history),1)
            self.assertEqual(history[0]['session_id'],'neck-without-audio')
            self.assertEqual(history[0]['scores'],{})
            self.assertEqual(history[0]['neck_effort'],stopped['neck_effort'])
            self.assertEqual(history[0]['model_type'],'neck_only')
        database.dispose()

    def test_neck_storage_failure_is_reported_without_discarding_analysis(self):
        self.engine.start_recording_session('storage-error')
        for i in range(30):self.engine.process_frame(np.ones((100,100,3),np.uint8)*100,timestamp=i/5)
        with patch.object(self.db,'commit',side_effect=RuntimeError('database unavailable')):
            result=appmod.stop_neck_effort_recording(self.user,self.db,session_id='storage-error')
        self.assertFalse(result['history_saved'])
        self.assertEqual(result['neck_effort']['neckAnalysis']['status'],'complete')

    def test_provider_failure_remains_structured_and_null(self):
        from neck_v1_openai import NeckProviderError
        def fail(*args):raise NeckProviderError('provider failed')
        self.engine.interpreter=fail
        self.engine.start_recording_session('test')
        for i in range(30):self.engine.process_frame(np.zeros((100,100,3),np.uint8),timestamp=i/5)
        result=appmod.stop_neck_effort_recording(self.user,self.db)['neck_effort']
        self.assertFalse(result['ok']);self.assertIsNone(result['score'])
        appmod.NeckEffortOut(**result)

    def test_motion_trace_persists_even_when_provider_fails(self):
        from head_motion import VERSION
        from test_head_motion import trace
        from neck_v1_openai import NeckProviderError
        self.engine.start_recording_session('motion-session')
        for i in range(30):self.engine.process_frame(np.zeros((100,100,3),np.uint8),timestamp=i/5)
        self.engine.interpreter=lambda *args: (_ for _ in ()).throw(NeckProviderError('provider failed'))
        raw=json.dumps(dict(version='head-motion-trace-v1',source='browser_patch_tracking',samples=trace()))
        result=appmod.stop_neck_effort_recording(self.user,self.db,session_id='motion-session',motion_trace=raw)['neck_effort']
        motion=result['neckAnalysis']['movement_stability']
        self.assertEqual(motion['version'],VERSION)
        self.assertEqual(motion['status'],'measured')
        self.assertEqual(result['neckAnalysis']['status'],'error')
        self.assertIsNone(motion['effort_score'])
        self.assertEqual(result['neckAnalysis']['validation_status'],'not_clinically_validated')
        self.assertEqual(result['neckAnalysis']['confidence_kind'],'uncalibrated_evidence_index')
        saved=json.loads(self.db.saved[0].summary)
        self.assertEqual(saved['neckAnalysis']['movement_stability']['status'],'measured')
        appmod.NeckEffortOut(**result)

    def test_motion_requires_current_session_and_valid_payload(self):
        from fastapi import HTTPException
        self.engine.start_recording_session('current')
        raw=json.dumps(dict(version='head-motion-trace-v1',source='browser_patch_tracking',samples=[]))
        for session,payload,expected in [(None,raw,400),('old',raw,409),('current','{}',400)]:
            with self.assertRaises(HTTPException) as caught:
                appmod.stop_neck_effort_recording(self.user,self.db,session_id=session,motion_trace=payload)
            self.assertEqual(caught.exception.status_code,expected)
        self.assertEqual(len(self.db.saved),0)

    def test_recovery_trace_keeps_segments_and_reinitialization_diagnostics(self):
        from test_head_motion import trace
        self.engine.start_recording_session('recovered-session')
        for i in range(30):self.engine.process_frame(np.zeros((100,100,3),np.uint8),timestamp=i/5)
        samples=[row+[0 if i<45 else 1] for i,row in enumerate(trace())]
        raw=json.dumps(dict(version='head-motion-trace-v2',source='browser_patch_tracking',samples=samples,
            tracking_diagnostics=dict(reinitializations=1,remaining_head_points=12)))
        result=appmod.stop_neck_effort_recording(self.user,self.db,session_id='recovered-session',motion_trace=raw)['neck_effort']
        report=result['neckAnalysis']['movement_stability']
        self.assertEqual(report['tracking_diagnostics']['reinitializations'],1)
        self.assertEqual(report['windows'][0]['reason'],'tracking_reinitialized')
        self.assertTrue(any(w['status']=='measured' and w['start']>=2 for w in report['windows']))
        saved=json.loads(self.db.saved[0].summary)['neckAnalysis']['movement_stability']
        self.assertEqual(saved,report)

    def test_upload_route_uses_v1_and_existing_contract(self):
        with patch.object(appmod,'analyze_video_audio',return_value={'ok':True,'neckAnalysis':{'version':'neck-analysis-v1'}}) as analyze:
            result=appmod.analyze_synchronized_effort(
                appmod.UploadFile(file=io.BytesIO(b'video')),
                appmod.UploadFile(file=io.BytesIO(b'audio')),.1,self.user)
            self.assertEqual(result['neckAnalysis']['version'],'neck-analysis-v1')
            self.assertEqual(analyze.call_args.kwargs['audio_offset_seconds'],.1)

    def test_live_audio_reaches_provider_context(self):
        captured=[]
        def provider(ctx,images):
            captured.append(ctx)
            result=prediction()
            for item in result.dimensions.__dict__.values():item.frame_indices=ctx['selected_indices'][:2]
            return result
        self.engine.interpreter=provider
        self.engine.start_recording_session('live-audio')
        for i in range(25):self.engine.process_frame(np.zeros((100,100,3),np.uint8),timestamp=i/5)
        audio=[dict(t=.04,duration=5,speech_active=True,vad_available=True,clipped_fraction=0)]
        upload=memory_upload(b'wav')
        try:
            with patch.object(appmod,'decode_live_neck_audio',return_value=audio) as decode:
                response=appmod.stop_neck_effort_recording(self.user,self.db,upload,.04,'live-audio')
        finally:upload.file.close()
        self.assertTrue(response['neck_effort']['ok'])
        decode.assert_called_once_with(b'wav',.04)
        self.assertEqual(captured[0]['speech_status'],'active')
        self.assertEqual(captured[0]['audio_timing'],'browser_estimated_offset')
        self.assertAlmostEqual(captured[0]['usable_speech_duration'],4.76)
        self.assertIn('speech_timing_unverified',captured[0]['quality']['warnings'])
        self.assertAlmostEqual(captured[0]['temporal_quality'],.96*.8)
        self.assertNotIn('wav',self.db.saved[0].summary)

    def test_stale_audio_and_bad_audio_cannot_finalize_current_recording(self):
        self.engine.start_recording_session('current')
        upload=memory_upload(b'not a wav')
        try:
            with self.assertRaises(appmod.HTTPException) as error:
                appmod.stop_neck_effort_recording(self.user,self.db,upload,0,'current')
            self.assertEqual(error.exception.status_code,400)
            self.assertTrue(self.engine.state.recording_active)
            with patch.object(appmod,'decode_live_neck_audio',return_value=[]):
                with self.assertRaises(appmod.HTTPException) as error:
                    appmod.stop_neck_effort_recording(self.user,self.db,upload,0,'old')
            self.assertEqual(error.exception.status_code,409)
            self.assertTrue(self.engine.state.recording_active)
        finally:upload.file.close()

    def test_silent_live_audio_skips_provider_and_keeps_null_scores(self):
        from unittest.mock import Mock
        import soundfile as sf
        data=io.BytesIO();sf.write(data,np.zeros(16000*5),16000,format='WAV',subtype='PCM_16')
        upload=memory_upload(data.getvalue())
        self.engine.interpreter=Mock(side_effect=AssertionError('Silent recording must not call provider'))
        self.engine.start_recording_session('silent')
        for i in range(25):self.engine.process_frame(np.zeros((100,100,3),np.uint8),timestamp=i/5)
        try:result=appmod.stop_neck_effort_recording(self.user,self.db,upload,0,'silent')['neck_effort']
        finally:upload.file.close()
        self.engine.interpreter.assert_not_called()
        self.assertIsNone(result['score'])
        self.assertEqual(result['neckAnalysis']['diagnostics']['speech_status'],'silent')
        self.assertTrue(result['neckAnalysis']['diagnostics']['provider_skipped'])

if __name__=='__main__':unittest.main()

class BatchTests(unittest.TestCase):
    setUp = APITests.setUp
    tearDown = APITests.tearDown
    def test_retry_of_accepted_batch_returns_receipt_without_duplicate_frames(self):
        session=appmod.start_neck_effort_recording(self.user)['session_id']
        ok,jpeg=cv2.imencode('.jpg',np.ones((100,100,3),np.uint8)*100)
        files=[memory_upload(jpeg.tobytes()) for _ in range(5)]
        try:
            receipt=appmod.analyze_neck_effort_frames(files,'[0,0.2,0.4,0.6,0.8]',session,self.user,'batch1')
            for file in files:file.file.seek(0)
            retried=appmod.analyze_neck_effort_frames(files,'[0,0.2,0.4,0.6,0.8]',session,self.user,'batch1')
            self.assertEqual(receipt,retried);self.assertEqual(receipt['batch_id'],'batch1')
            self.assertEqual(len(self.engine.rows),5)
            for file in files:file.file.seek(0)
            with self.assertRaises(appmod.HTTPException) as caught:
                appmod.analyze_neck_effort_frames(files,'[1,1.2,1.4,1.6,1.8]',session,self.user,'batch1')
            self.assertEqual(caught.exception.status_code,409)
            appmod.start_neck_effort_recording(self.user)
            self.assertEqual(self.engine.upload_batches,{})
            for file in files:file.file.seek(0)
            with self.assertRaises(appmod.HTTPException) as caught:
                appmod.analyze_neck_effort_frames(files,'[0,0.2,0.4,0.6,0.8]',session,self.user,'batch1')
            self.assertEqual(caught.exception.status_code,409)
        finally:
            for file in files:file.file.close()

    def test_retry_resumes_partially_processed_batch(self):
        session=appmod.start_neck_effort_recording(self.user)['session_id']
        ok,jpeg=cv2.imencode('.jpg',np.ones((100,100,3),np.uint8)*100)
        files=[memory_upload(jpeg.tobytes()) for _ in range(5)]
        original=self.engine.process_frame
        def interrupted(image,timestamp):
            if timestamp==.4:raise RuntimeError('transient processing failure')
            return original(image,timestamp=timestamp)
        try:
            with patch.object(self.engine,'process_frame',side_effect=interrupted):
                with self.assertRaises(appmod.HTTPException) as caught:
                    appmod.analyze_neck_effort_frames(files,'[0,0.2,0.4,0.6,0.8]',session,self.user,'part')
                self.assertEqual(caught.exception.status_code,503)
            self.assertEqual(len(self.engine.rows),2)
            for file in files:file.file.seek(0)
            with self.assertRaises(appmod.HTTPException) as caught:
                appmod.analyze_neck_effort_frames(files,'[1,1.2,1.4,1.6,1.8]',session,self.user,'later')
            self.assertEqual(caught.exception.status_code,409)
            for file in files:file.file.seek(0)
            result=appmod.analyze_neck_effort_frames(files,'[0,0.2,0.4,0.6,0.8]',session,self.user,'part')
            self.assertEqual(result['accepted_frames'],5)
            self.assertEqual([r['t'] for r in self.engine.rows],[0,.2,.4,.6,.8])
        finally:
            for file in files:file.file.close()
    def test_batches_use_capture_timestamps_and_reject_old_session(self):
        started=appmod.start_neck_effort_recording(self.user)
        ok,jpeg=cv2.imencode('.jpg',np.ones((100,100,3),np.uint8)*100)
        files=[memory_upload(jpeg.tobytes()) for _ in range(5)]
        try:
            response=appmod.analyze_neck_effort_frames(files,json.dumps([0,.2,.4,.6,.8]),started['session_id'],self.user)
            self.assertEqual(response['accepted_frames'],5)
            self.assertEqual([row['t'] for row in self.engine.rows],[0,.2,.4,.6,.8])
            for file in files:file.file.seek(0)
            with self.assertRaises(appmod.HTTPException) as error:
                appmod.analyze_neck_effort_frames(files,json.dumps([1,1.2,1.4,1.6,1.8]),'old-session',self.user)
            self.assertEqual(error.exception.status_code,409)
            self.assertEqual(len(self.engine.rows),5)
        finally:
            for file in files:file.file.close()

    def test_bad_timestamps_do_not_mutate_recording(self):
        started=appmod.start_neck_effort_recording(self.user)
        file=memory_upload(b'invalid')
        try:
            with self.assertRaises(appmod.HTTPException):
                appmod.analyze_neck_effort_frames([file],'[true]',started['session_id'],self.user)
            self.assertEqual(len(self.engine.rows),0)
        finally:file.file.close()
