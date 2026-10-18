"""Neck Analysis V1. Legacy severity heuristics have been removed."""
import base64
from collections import deque
from dataclasses import dataclass, field
import math
import os
import threading
import time
import cv2
import numpy as np
from neck_v1_cv import CVProcessor, jaw_residualization, select_frames
from neck_v1_scoring import NeckEffortConfig, VERSION, DIMENSIONS, NeckAnalysis, aggregate, unavailable
from neck_v1_openai import interpret, NeckProviderError, PROMPT_VERSION, DEFAULT_MODEL
from neck_observable_change import compare_observable_windows
from neck_capture_context import movement_evidence, transport_evidence

@dataclass
class NeckEffortState:
    recording_active: bool = False
    recording_session_id: str | None = None
    recording_started_at: float | None = None
    recording_stopped_at: float | None = None
    recording_results: deque = field(default_factory=lambda: deque(maxlen=300))
    calibrated: bool = False


def packet(mode='collecting', result=None, confidence=0.0, reasons=None):
    overall=(result or {}).get('overall',{})
    return dict(ok=mode!='error', mode=mode, calibrated=mode in ('running','complete'),
                calibration_progress=1.0 if mode in ('running','complete') else 0.0,
                score=overall.get('category'), score_label=overall.get('label'),
                continuous_score=overall.get('continuous_score'), confidence=overall.get('confidence',confidence),
                features=None, reasons=reasons or [], model_version=VERSION, neckAnalysis=result)


def evidence_context(rows, selected, cfg, audio=None, timing='unverified',movement=None,capture_transport=None):
    observable_change=compare_observable_windows(rows,audio,timing)
    features=[r['features'] for r in rows]
    motion=[f.get('anterior_motion',0.0) for f in features]
    jaw=[f.get('jaw_velocity',0.0) for f in features]
    residual,fraction=jaw_residualization(motion,jaw)
    visibility=lambda key: float(np.mean([r['visibility'].get(key,0) for r in rows]))
    stability=float(np.mean([r['camera_stability'] for r in rows]))
    lighting=float(np.mean([r['lighting'] for r in rows]))
    resolution=float(np.mean([r['resolution_quality'] for r in rows]))
    duration=rows[-1]['t']-rows[0]['t'] if len(rows)>1 else 0
    deltas=np.diff([r['t'] for r in rows])
    rate=min(1/float(np.median(deltas)), (len(rows)-1)/duration) if len(deltas) and np.median(deltas)>0 and duration>0 else 0
    # Live flow already samples at 5fps; not equivalent to dense 30fps analysis.
    temporal=min(1.0,duration/5)*min(1.0,rate/5)
    warnings=[]
    warnings.extend(sorted({warning for row in rows for warning in row.get('warnings',[])}))
    if duration<5: warnings.append('short_recording')
    if rate<5: warnings.append('sparse_frame_rate')
    if lighting<.5: warnings.append('poor_lighting')
    if resolution<.75: warnings.append('low_resolution')
    if stability<.5: warnings.append('camera_motion')
    if visibility('shoulders')<.25: warnings.append('shoulders_not_reliably_visible')
    if min(visibility('anterior'),visibility('submental'))<.25: warnings.append('neck_not_reliably_visible')
    if not any(r.get('camera_measurement_available',False) for r in rows): warnings.append('camera_stability_not_independently_verified')
    if timing in ('unverified','browser_estimated_offset','container_estimated_offset'): warnings.append('speech_timing_unverified')
    speech_status='unknown'
    usable_duration=None
    audio_context=None
    if audio is not None and timing!='unverified':
        overlap=[r for r in audio if r['t']<rows[-1]['t'] and r['t']+r['duration']>rows[0]['t']]
        audio_context=overlap
        if overlap:
            usable_duration=sum(max(0,min(rows[-1]['t'],r['t']+r['duration'])-max(rows[0]['t'],r['t']))
                                for r in overlap if r.get('speech_active',False))
            speech_status='active' if usable_duration>0 else ('silent' if all(r.get('clipped_fraction',0)<.01 and r.get('vad_available',True) for r in overlap) else 'unknown')
        else: warnings.append('audio_video_do_not_overlap')
    if speech_status=='active':
        temporal*=min(1.0,usable_duration/3)
    elif speech_status=='unknown':
        temporal*=.8
    if timing in ('browser_estimated_offset','container_estimated_offset') and speech_status!='unknown':
        temporal*=.8  # Device latency and callback timing have not been independently verified.
    quality=dict(overall=round((visibility('face')*lighting*stability*resolution)**.25,4),
                 face_visibility=visibility('face'), neck_visibility=min(visibility('anterior'),visibility('submental')),
                 shoulder_visibility=visibility('shoulders'),camera_stability=stability,
                 lighting_quality=lighting,warnings=warnings)
    dimension_quality={
        'posture_compensation':visibility('face'),
        'scm_lateral_recruitment':max(visibility('left_lateral'),visibility('right_lateral')),
        'anterior_submental_activity':min(visibility('anterior'),visibility('submental')),
        'shoulder_recruitment':visibility('shoulders'),
        'jaw_compensation':visibility('jaw')}
    dimension_quality={k:float(v*math.sqrt(stability*lighting*resolution)) for k,v in dimension_quality.items()}
    timeline=[]
    for index,row in enumerate(rows):
        window=math.floor(row['t'])
        if not timeline or timeline[-1]['start']!=window:
            timeline.append(dict(start=window,end=window+1,values=[]))
        timeline[-1]['values'].append(float(residual[index]))
    for window in timeline:
        window['neck_activity']=float(np.mean(window.pop('values')))
    diagnostics = dict(captured_frames=len(rows), selected_frames=len(selected),
                       selected_frame_indices=list(selected),selected_frame_times=[round(rows[i]['t'],3) for i in selected],
                       frame_selection='temporal_coverage_then_extrema_v2',
                       feature_available_frames={key:sum(key in row['features'] for row in rows)
                           for key in ('head_pitch_deg','shoulder_motion','shoulders_head_relative_motion')},
                       observed_span_seconds=round(duration, 3), effective_fps=round(rate, 3),
                       target_fps=5, temporal_quality=round(temporal, 4),
                       speech_status=speech_status,audio_timing=timing,
                       usable_speech_duration=round(usable_duration,3) if usable_duration is not None else None,
                       vad_available=any(r.get('vad_available',False) for r in (audio_context or [])),
                       audio_offset_seconds=audio[0]['t'] if audio else None,
                       dimension_quality=dimension_quality,
                       mean_cv_processing_seconds=round(float(np.mean([r['processing_seconds'] for r in rows if 'processing_seconds' in r])), 4)
                           if any('processing_seconds' in r for r in rows) else None)
    head_movement=movement_evidence(movement,rows[0]['t'],rows[-1]['t'])
    if head_movement is not None:
        diagnostics['head_motion_context_supplied']=True
        diagnostics['head_motion_context']=head_movement
    transport=transport_evidence(capture_transport,len(rows))
    if transport is not None:
        diagnostics['capture_transport']=transport
        if not transport['complete']:quality['warnings'].append('capture_frames_missing')
    context=dict(diagnostics=diagnostics,quality=quality,dimension_quality=dimension_quality,temporal_quality=temporal,
                observable_change=observable_change,
                jaw_explained_fraction=fraction, jaw_residual_motion=residual,
                timeline=timeline, selected_indices=selected, speech_status=speech_status,
                usable_speech_duration=usable_duration,audio_timing=timing,audio=audio_context,
                frames=[dict(index=i,**rows[i]) for i in selected],
                limitations=['ROIs approximate image-space regions, not isolated muscles',
                             'Flow and jaw regression are motion evidence, not force or causality',
                             'WebRTC VAD speech timing is provisional, not ground truth or diagnosis'],
                versions=dict(analyzer=VERSION,preprocessing='neck-cv-v1.1',prompt=PROMPT_VERSION,
                              weights='neck-weights-v1.0',scoring='neck-scoring-v1.1',
                              observable_change='neck-observable-change-v1',selection='temporal-coverage-v2',
                              model=os.environ.get('NECK_OPENAI_MODEL',DEFAULT_MODEL)))
    if head_movement is not None:context['head_movement']=head_movement
    return context


class NeckEffortEngine:
    def __init__(self,config=None,interpreter=None,processor=None):
        self.cfg=config or NeckEffortConfig()
        self.processor=processor or CVProcessor()
        self.interpreter=interpreter or interpret
        self.lock=threading.RLock()
        self.expiry_timer=None
        self.reset_session()

    def reset_session(self):
        with self.lock:
            if self.expiry_timer is not None:
                self.expiry_timer.cancel()
                self.expiry_timer=None
            self.state=NeckEffortState(recording_results=deque(maxlen=self.cfg.max_frames))
            self.rows=[]
            self.images=[]
            self.result=None
            self.latest=None
            self.debug_evidence=None
            self.upload_batches={}
            self.processor.reset()

    def start_recording_session(self,session_id=None):
        self.reset_session()
        self.state.recording_active=True
        self.state.recording_session_id=session_id
        self.state.recording_started_at=time.monotonic()
        session=self.state.recording_started_at
        def expire():
            with self.lock:
                if self.state.recording_started_at==session:
                    self.images.clear()
                    self.debug_evidence=None
                    self.processor.reset()
                    self.state.recording_active=False
        self.expiry_timer=threading.Timer(120,expire)
        self.expiry_timer.daemon=True
        self.expiry_timer.start()

    def stop_recording_session(self):
        with self.lock:
            self.state.recording_active=False
            self.state.recording_stopped_at=time.monotonic()

    def is_calibrated(self):
        return bool(self.rows)

    def calibration_progress(self):
        return min(1.0,len(self.rows)/10)

    def latest_result(self,max_age_seconds=None):
        if self.latest is None: return None
        if max_age_seconds is not None and time.monotonic()-self.latest_at>max_age_seconds: return None
        return self.latest

    def process_frame(self,frame_bgr,timestamp=None):
        with self.lock:
            if not self.state.recording_active:
                return packet('no_score',reasons=['recording_required'])
            if frame_bgr is None or frame_bgr.size==0:
                return packet('error',reasons=['empty_frame'])
            if len(self.rows)>=self.cfg.max_frames:
                return packet('running',reasons=['frame_limit_reached'])
            t=timestamp if timestamp is not None else time.monotonic()-self.state.recording_started_at
            if not math.isfinite(t) or t<0 or (self.rows and t<=self.rows[-1]['t']):
                return packet('error',reasons=['invalid_frame_timestamp'])
            # Bound processing/memory resolution, preserve aspect ratio.
            height,width=frame_bgr.shape[:2]
            factor=min(1,960/max(height,width))
            image=cv2.resize(frame_bgr,(max(1,round(width*factor)),max(1,round(height*factor))))
            processing_started=time.monotonic()
            row=self.processor.process(image,float(t))
            row['processing_seconds']=time.monotonic()-processing_started
            ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok: return packet('error',reasons=['frame_encoding_failed'])
            self.rows.append(row);self.images.append(jpeg.tobytes())
            current=packet('running',confidence=row['visibility'].get('face',0),reasons=['collecting_neck_evidence'])
            self.latest=current;self.latest_at=time.monotonic()
            self.state.recording_results.append({'at':self.latest_at,'result':dict(t=t,**current)})
            return current

    def export_recording(self):
        # Metadata only. No JPEG bytes, raw landmark arrays or patient images.
        return [dict(t=row['t'],features=row['features'],visibility=row['visibility'],processing_seconds=row.get('processing_seconds')) for row in self.rows]

    def recording_summary(self,max_age_seconds=None):
        ended=self.state.recording_stopped_at
        if max_age_seconds is not None and (ended is None or time.monotonic()-ended>max_age_seconds): return None
        return self.result

    def finalize(self,audio=None,timing='unverified',movement=None,capture_transport=None):
        with self.lock:
            if self.result is not None: return self.result
            self.stop_recording_session()
            context=None
            try:
                if len(self.rows)<2 or len(self.images)!=len(self.rows):
                    raise ValueError('Insufficient or expired video frames')
                selected=select_frames(self.rows,self.cfg.max_evidence_frames)
                context=evidence_context(self.rows,selected,self.cfg,audio,timing,movement,capture_transport)
                if os.environ.get('NECK_DEBUG')=='1':
                    self.debug_evidence={'context':context,'sampled_frames':[self.images[i] for i in selected]}
                ceilings={k:q*context['temporal_quality'] for k,q in context['dimension_quality'].items()}
                insufficient_sampling=max(ceilings.values())<self.cfg.min_dimension_confidence
                silent=context['speech_status']=='silent'
                if insufficient_sampling or silent:
                    reason='No usable speech/phonation evidence in aligned audio' if silent else 'Insufficient temporal sampling confidence'
                    diagnostics=dict(context['diagnostics'], provider_skipped=True,
                        dimensions={k:dict(tracking_quality=context['dimension_quality'][k],
                            maximum_possible_confidence=round(ceilings[k],4),
                            required_confidence=self.cfg.min_dimension_confidence,
                            rejection_reason=reason,model_evidence=[]) for k in DIMENSIONS})
                    analysis=dict(version=VERSION,status='insufficient_quality',quality=context['quality'],
                        dimensions={k:unavailable(reason) for k in DIMENSIONS},diagnostics=diagnostics,
                        overall=dict(continuous_score=None,category=None,confidence=0.0,label='Not reliably assessable'),
                        summary=('לא זוהתה פעילות דיבור שימושית באודיו החופף לצילום; לא נשלחה בקשה ל־OpenAI.' if silent else
                                 'איכות וקצב הפריימים אינם מאפשרים להגיע לסף הביטחון הנדרש; לא נשלחה בקשה ל־OpenAI.'),
                        limitations=context['limitations'],versions=context['versions'])
                elif max(context['dimension_quality'].values())<self.cfg.min_roi_quality:
                    analysis=dict(version=VERSION,status='insufficient_quality',quality=context['quality'],
                                  dimensions={k:unavailable('Insufficient visibility') for k in DIMENSIONS},
                                  overall=dict(continuous_score=None,category=None,confidence=0.0,label='Not reliably assessable'),
                                  summary='איכות הצילום אינה מספיקה להערכת התנהגות צוואר נצפית.',
                                  limitations=context['limitations'],versions=context['versions'])
                else:
                    images=[]
                    crop_indices=set(selected[:1]+sorted(selected,key=lambda i: -self.rows[i]['features'].get('anterior_motion',0))[:1])
                    for i in selected:
                        encoded=self.images[i]
                        images.append(dict(index=i,t=self.rows[i]['t'],kind='full_frame',url='data:image/jpeg;base64,'+base64.b64encode(encoded).decode()))
                        box=self.rows[i]['rois'].get('anterior')
                        if box and i in crop_indices:
                            frame=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
                            h,w=frame.shape[:2];x1,y1,x2,y2=box
                            # Context crop includes lateral neck, chin and clavicles.
                            margin=max(1,x2-x1)
                            crop=frame[max(0,y1-margin//2):min(h,y2+margin//2),max(0,x1-margin):min(w,x2+margin)]
                            if crop.size:
                                ok,jpeg=cv2.imencode('.jpg',crop)
                                if ok: images.append(dict(index=i,t=self.rows[i]['t'],kind='neck_context_crop',url='data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()))
                    output=self.interpreter(context,images)
                    analysis=aggregate(output,context,self.cfg)
                analysis.setdefault('diagnostics',context['diagnostics'])
                analysis['timeline']=context['timeline']
                analysis['observable_change']=context['observable_change']
                if movement is not None:analysis['movement_stability']=movement
                analysis['effective_weights']={k:self.cfg.weights[k]/sum(self.cfg.weights[d] for d in DIMENSIONS if analysis['dimensions'][d]['score'] is not None)
                                               for k in DIMENSIONS if analysis['dimensions'][k]['score'] is not None}
                analysis=NeckAnalysis.model_validate(analysis).model_dump()
                self.result=packet('complete',analysis)
                self.result.update(recording_summary=True,captured_frame_count=len(self.rows),timeline=self.export_recording())
                return self.result
            except (NeckProviderError,ValueError,RuntimeError) as exc:
                # Sanitized provider error; no patient/provider bodies in ordinary logs.
                self.result=packet('error',reasons=[str(exc)])
                self.result['neckAnalysis']=dict(version=VERSION,status='error',
                    validation_status='not_clinically_validated',confidence_kind='uncalibrated_evidence_index',
                    dimensions={k:unavailable('Analysis failed') for k in DIMENSIONS},
                    overall=dict(continuous_score=None,category=None,confidence=0.0,label='Analysis unavailable'),
                    quality=context['quality'] if context else {'overall':0.0,'warnings':['analysis_failed']},
                    observable_change=context['observable_change'] if context else compare_observable_windows(self.rows,audio,timing),
                    diagnostics=context['diagnostics'] if context else {'captured_frames':len(self.rows)},summary='ניתוח הצוואר אינו זמין.',limitations=[str(exc)])
                if movement is not None:self.result['neckAnalysis']['movement_stability']=movement
                self.result.update(recording_summary=True,captured_frame_count=len(self.rows),timeline=self.export_recording())
                return self.result
            finally:
                # Media never survives finalization, even when the provider fails.
                self.images.clear()
                self.processor.reset()
                if self.expiry_timer is not None and self.debug_evidence is None: self.expiry_timer.cancel()

    def close(self):
        if self.expiry_timer is not None: self.expiry_timer.cancel()
        self.images.clear()
        self.debug_evidence=None
        self.processor.close()
