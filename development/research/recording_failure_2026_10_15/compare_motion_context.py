"""Controlled real-provider comparison: identical images/prompt, motion context on/off.

No effort-stage label or desired answer is supplied. Raw frames remain in memory.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import cv2
import imageio_ffmpeg
import soundfile as sf
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[2]
FOLDER=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from head_motion import analyze_payload
from neck_effort_engine import NeckEffortEngine
from neck_v1_openai import interpret,NeckProviderError,PROMPT_VERSION,DEFAULT_MODEL,PROVIDER_TIMEOUT_SECONDS
from neck_v1_scoring import aggregate
from observable_effort import neck_audio_timing


def main():
    load_dotenv(ROOT.parents[1]/'.env',override=True)
    assert os.environ.get('OPENAI_API_KEY'),'Server key is missing'
    video=ROOT/'data/WIN_20261015_21_04_57_Pro.mp4'
    movement=analyze_payload(json.dumps(json.loads((FOLDER/'tracking_after.json').read_text())['trace']))
    with tempfile.TemporaryDirectory(prefix='neck-context-audio-') as folder:
        wav=Path(folder)/'audio.wav'
        conversion=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-vn','-ac','1','-ar','16000','-y',str(wav)],capture_output=True)
        assert conversion.returncode==0
        audio,sr=sf.read(wav)
    trials=[]
    def interpreter(context,images):
        assert context['head_movement']['status']=='measured','Dense context must be usable before spending API tokens'
        fingerprint=hashlib.sha256(''.join(i['url'] for i in images).encode()).hexdigest()
        outputs=[]
        for include_motion in (False,True):
            candidate=copy.deepcopy(context)
            if not include_motion:
                candidate.pop('head_movement',None)
                for key in ('head_motion_context','head_motion_context_supplied'):candidate['diagnostics'].pop(key,None)
            info={}
            def transport(payload):
                request=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(payload).encode(),method='POST',
                    headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'],'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=PROVIDER_TIMEOUT_SECONDS) as response:raw=json.load(response)
                info['usage']=raw.get('usage');return raw
            before=time.monotonic()
            try:
                prediction=interpret(candidate,images,transport=transport)
                result=aggregate(prediction,candidate)
                entry=dict(with_dense_motion=include_motion,status='completed',seconds=time.monotonic()-before,
                    image_fingerprint=fingerprint,image_count=len(images),selected_indices=context['selected_indices'],
                    raw_model=prediction.model_dump(),accepted_scores={k:v['score'] for k,v in result['dimensions'].items()},
                    rejection_reasons={k:v['rejection_reason'] for k,v in result['diagnostics']['dimensions'].items()},**info)
                outputs.append(prediction)
            except NeckProviderError as error:
                entry=dict(with_dense_motion=include_motion,status='error',seconds=time.monotonic()-before,error=str(error),**info)
            trials.append(entry)
            (FOLDER/'motion_context_comparison.json').write_text(json.dumps(dict(source_file=video.name,
                prompt=PROMPT_VERSION,model=os.environ.get('NECK_OPENAI_MODEL',DEFAULT_MODEL),trials=trials,
                motion_context=context.get('head_movement'),limitations=[
                    'One clip, no independent muscle-effort labels; not a clinical accuracy evaluation',
                    'Same images and updated prompt in both arms; only dense derived motion context differs',
                    'No participant-reported effort stage or desired grade supplied',
                    'Single call per arm; ordering and provider nondeterminism can confound differences']),ensure_ascii=False,indent=2))
            print(json.dumps({k:entry.get(k) for k in ('with_dense_motion','status','seconds','accepted_scores','error')},ensure_ascii=False),flush=True)
        if not outputs:raise NeckProviderError('Both paired requests failed')
        return outputs[-1]
    engine=NeckEffortEngine(interpreter=interpreter)
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    try:
        engine.start_recording_session('paired-motion-evidence')
        for n in range(int(duration*5)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round(n/5*fps));ok,frame=cap.read();assert ok
            frame=cv2.resize(frame,(480,270));ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
            engine.process_frame(cv2.imdecode(jpeg,cv2.IMREAD_COLOR),n/5)
        engine.finalize(audio=neck_audio_timing(audio,sr),timing='container_estimated_offset',movement=movement)
        assert len(trials)==2,'Paired comparison did not run'
    finally:cap.release();engine.close()


if __name__=='__main__':main()
