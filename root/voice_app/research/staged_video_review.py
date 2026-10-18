"""Paired segment review with real provider, local audio and no retained images.

Participant-reported effort timing is not sent to the model and is not ground truth.
Use --provider only when sending selected identifiable frames is authorized.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
import cv2
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neck_effort_engine import NeckEffortEngine
import neck_effort_engine as engine_module
from neck_v1_openai import interpret
from neck_v1_scoring import NeckEffortConfig, aggregate
from observable_effort import neck_audio_timing, acoustic_timeline


def review(video, output, provider=False, whole=False, edge=480, indices=None, repeats=1, provider_timeout=15):
    import imageio_ffmpeg
    load_dotenv(Path(__file__).resolve().parents[3]/'.env',override=True)
    if provider and not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('No provider key configured')
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    specs=[(0.,duration)] if whole else [(0.,5.),(5.,10.),(10.,duration)]
    if indices is not None:
        def fixed_selection(rows,limit):
            if len(indices)>limit or len(set(indices))!=len(indices) or any(i<0 or i>=len(rows) for i in indices):
                raise ValueError('Invalid fixed evidence indices')
            return sorted(indices)
        engine_module.select_frames=fixed_selection
    with tempfile.TemporaryDirectory(prefix='neck-stage-audio-') as directory:
        wav=Path(directory)/'audio.wav'
        conversion=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),
            '-vn','-ac','1','-ar','16000','-y',str(wav)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if conversion.returncode:
            raise ValueError('Local audio extraction failed')
        audio,sr=sf.read(wav)
    stages=[]
    try:
        for start,end in specs:
            details={};samples=audio[round(start*sr):round(end*sr)]
            timing=neck_audio_timing(samples,sr)
            def transport(payload):
                request=urllib.request.Request('https://api.openai.com/v1/responses',
                    data=json.dumps(payload).encode(),method='POST',
                    headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'],
                             'Content-Type':'application/json'})
                before=time.monotonic()
                with urllib.request.urlopen(request,timeout=provider_timeout) as response:
                    result=json.load(response)
                details['provider_seconds']=time.monotonic()-before
                details['usage']=result.get('usage')
                details['provider_status']=result.get('status')
                details.setdefault('provider_calls',[]).append(dict(seconds=details['provider_seconds'],
                    usage=result.get('usage'),status=result.get('status')))
                return result
            def interpreter(context,images):
                details['selected_indices']=context['selected_indices']
                details['images_sent']=len(images)
                if not provider:
                    raise ValueError('Provider not enabled; local measurements only')
                result=interpret(context,images,transport=transport)
                details['model_output']=result.model_dump()
                details['jaw_explained_fraction']=context['jaw_explained_fraction']
                repeated=[aggregate(result,context)]
                for repeat in range(1,repeats):
                    print(f'Repeating identical evidence/prompt {repeat+1}/{repeats}',flush=True)
                    candidate=interpret(context,images,transport=transport)
                    repeated.append(aggregate(candidate,context))
                if repeats>1:
                    details['repeatability']=dict(repeats=repeats,identical_input=True,
                        accepted_dimensions=[{name:dict(score=value['score'],continuous_score=value['continuous_score'])
                            for name,value in run['dimensions'].items()} for run in repeated],
                        status='consistent_scores' if all(
                            all(run['dimensions'][name]['score']==repeated[0]['dimensions'][name]['score']
                                for name in run['dimensions']) for run in repeated[1:]) else 'inconsistent_scores')
                details['threshold_comparison']={str(threshold):{
                    name:dict(score=value['score'],reason=scored['diagnostics']['dimensions'][name]['rejection_reason'])
                    for name,value in scored['dimensions'].items()}
                    for threshold in (.15,.10,.05)
                    for scored in [aggregate(result,context,NeckEffortConfig(min_dimension_confidence=threshold))]}
                return result
            engine=NeckEffortEngine(interpreter=interpreter)
            before=time.monotonic()
            try:
                engine.start_recording_session()
                n=0
                while start+n/5 < end-.05:
                    index=round((start+n/5)*fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                    ok,frame=cap.read()
                    if not ok:break
                    h,w=frame.shape[:2];scale=min(1,edge/max(w,h))
                    frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
                    ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95])
                    if not ok:raise ValueError('JPEG encoding failed')
                    engine.process_frame(cv2.imdecode(jpeg,cv2.IMREAD_COLOR),index/fps-start)
                    n+=1
                print(f'Segment {start:.1f}–{end:.1f}: {n} measured frames; finalizing',flush=True)
                result=engine.finalize(audio=timing,timing='container_estimated_offset')
                stage=dict(start=start,end=end,elapsed_seconds=time.monotonic()-before,
                    result=result,acoustic_timeline=acoustic_timeline(samples,sr),
                    vad_windows=timing,**details)
                stages.append(stage)
                print(json.dumps(dict(start=start,end=end,status=result['neckAnalysis']['status'],
                    scores={k:v['score'] for k,v in result['neckAnalysis']['dimensions'].items()},
                    provider_seconds=details.get('provider_seconds'),reasons=result.get('reasons')),ensure_ascii=False),flush=True)
                output.parent.mkdir(parents=True,exist_ok=True)
                output.write_text(json.dumps(dict(source_file=video.name,source_fps=fps,duration=duration,max_edge=edge,
                    provider_timeout_seconds=provider_timeout,
                    stages=stages,limitations=['One participant; no physiological effort ground truth',
                    'Audio/video share container timing; synchronization not independently verified',
                    'OpenCV resize/JPEG approximates browser encoding; not browser throughput test',
                    'VAD speech windows do not validate vocal effort',
                    'No reported effort-stage labels were sent to the provider']),ensure_ascii=False,indent=2))
            finally:
                engine.close()
    finally:
        cap.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--provider',action='store_true')
    parser.add_argument('--whole',action='store_true')
    parser.add_argument('--edge',type=int,choices=(480,960),default=480)
    parser.add_argument('--indices',type=lambda value:[int(i) for i in value.split(',')])
    parser.add_argument('--repeats',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--provider-timeout',type=int,choices=(15,30,45),default=15,
                        help='Research transport timeout only; production timeout is unchanged')
    args=parser.parse_args()
    review(args.video,args.output,args.provider,args.whole,args.edge,args.indices,args.repeats,args.provider_timeout)
