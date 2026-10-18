"""Real Chromium capture and multipart app routes with a lost acknowledgement.

Actual image preprocessing/FaceMesh; interpretation is a controlled zero stub.
All users and saved results live in a temporary isolated database.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlsplit

from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[2]
FOLDER=Path(__file__).resolve().parent


async def check(app,client,token,user_id):
    from neck_v1_scoring import ModelOutput,DIMENSIONS
    video=(ROOT/'data/WIN_20261015_21_04_57_Pro.mp4').read_bytes()
    def controlled_interpretation(context,images):
        return ModelOutput.model_validate(dict(dimensions={name:dict(observable=True,score=0,continuous_score=0.,
            confidence=.9,evidence=['תשובה מדומה לבדיקת העברה בלבד'],frame_indices=context['selected_indices'][:2],
            speech_related=True,independent_local_activity=False,confounds=[]) for name in DIMENSIONS},summary='בדיקת העברה',limitations=[]))
    results=[]
    async with async_playwright() as playwright:
        browser=await playwright.chromium.launch(headless=True,args=['--no-sandbox'])
        for mobile in (False,True):
            page=await browser.new_page(viewport={'width':390 if mobile else 1280,'height':900})
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            calls=[];lost=False
            await page.route('**/__transport_clip',lambda route:route.fulfill(status=200,content_type='video/mp4',body=video))
            async def route_api(route):
                nonlocal lost
                request=route.request;path=urlsplit(request.url).path
                response=await asyncio.to_thread(client.request,request.method,path,headers=request.headers,content=request.post_data_buffer)
                if path.endswith('/recording/start') and response.status_code==200:
                    app.neck_effort_engines[user_id].interpreter=controlled_interpretation
                calls.append(dict(path=path,status=response.status_code))
                if path.endswith('/frames') and response.status_code==200 and not lost:
                    lost=True
                    # Backend accepted the first batch. Hold and discard only
                    # its response while actual camera sampling continues.
                    await asyncio.sleep(6.2)
                    await route.abort('connectionreset')
                else:
                    await route.fulfill(status=response.status_code,content_type='application/json',body=response.content)
            await page.route('**/api/neck-effort/**',route_api)
            await page.goto('http://127.0.0.1:5000'+('/mobile.html' if mobile else '/'),wait_until='domcontentloaded')
            await page.wait_for_function('typeof NeckCapture==="function" && typeof FaceMesh==="function"')
            outcome=await asyncio.wait_for(page.evaluate('''async ({token,mobile})=>{
                if(mobile)document.querySelectorAll('.m-section').forEach(s=>s.classList.toggle('active',s.id==='tab-record'));
                else activateTab('record');
                const v=mobile?mVideo:video;v.srcObject=null;v.muted=true;v.src='/__transport_clip';
                await new Promise((resolve,reject)=>{v.onloadeddata=resolve;v.onerror=reject;});
                getAuthHeaders=()=>({Authorization:'Bearer '+token});_currentUser={role:3};
                faceMesh=new FaceMesh({locateFile:file=>'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/'+file});
                faceMesh.setOptions({maxNumFaces:1,refineLandmarks:false,minDetectionConfidence:.5,minTrackingConfidence:.5});
                faceMesh.onResults(onFaceResults);isRecording=false;
                await faceMesh.send({image:v});if(!neckMotionSeed)throw Error('Initial face missing');
                await startNeckEffortRecordingSession();if(!neckEffortSessionId)throw Error('Session did not start');
                recordingStart=Date.now();isRecording=true;
                neckSampler=new NeckCapture({video:v,canvas:document.createElement('canvas'),sessionId:neckEffortSessionId,
                    startedAt:recordingStart,motionSeed:neckMotionSeed,headers:getAuthHeaders,onResult:()=>{}});
                neckSampler.start();let busy=false;
                const refresh=setInterval(async()=>{
                    if(!busy&&neckSampler.requestMotionReference()){
                        busy=true;try{await faceMesh.send({image:v});}finally{busy=false;}
                    }
                },50);
                const ended=new Promise(resolve=>v.onended=resolve);await v.play();await ended;
                clearInterval(refresh);isRecording=false;
                const packet=await stopNeckEffortRecordingSession(null);
                for(let i=0;i<200&&busy;i++)await new Promise(resolve=>setTimeout(resolve,10));
                await faceMesh.close();
                const c=document.getElementById(mobile?'m-neck-v1-results':'neck-analysis-v1-results');
                return {transport:neckSampler.transportDiagnostics(),packet,rendered:c.textContent};
            }''',dict(token=token,mobile=mobile)),timeout=70)
            engine=app.neck_effort_engines[user_id]
            times=[r['t'] for r in engine.rows]
            stats=outcome['transport'];analysis=outcome['packet']['neckAnalysis']
            assert stats['captured_frames']>=30,stats
            assert stats['acknowledged_frames']==stats['captured_frames']==len(times),(stats,len(times))
            assert len(set(times))==len(times)
            assert stats['retry_count']==1 and stats['unconfirmed_frames']==0,stats
            assert analysis['diagnostics']['head_motion_context_supplied']
            assert analysis['diagnostics']['head_motion_context']['status']=='measured'
            assert analysis['diagnostics']['capture_transport']['complete']
            assert 'הניתוח חלקי: חלק מהתמונות' not in outcome['rendered']
            assert not errors,errors
            entry=dict(case='mobile' if mobile else 'desktop',transport=stats,server_received=len(times),
                timestamps_unique=True,http_calls=calls,dense_context=analysis['diagnostics']['head_motion_context'],
                movement_status=analysis['movement_stability']['status'],javascript_errors=errors,
                interpretation='controlled stub, not a real provider or effort-accuracy test')
            results.append(entry)
            (FOLDER/'browser_transport_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
            print(json.dumps(dict(case=entry['case'],captured=stats['captured_frames'],server_received=len(times),
                retries=stats['retry_count'],dense_context=entry['dense_context']['status']),ensure_ascii=False),flush=True)
            await page.close()
        await browser.close()


def main():
    with tempfile.TemporaryDirectory(prefix='neck-browser-transport-') as directory:
        os.environ['DATABASE_URL']='sqlite:///'+str(Path(directory)/'test.sqlite')
        sys.path.insert(0,str(ROOT))
        import app
        from database import SessionLocal,User
        from fastapi.testclient import TestClient
        with SessionLocal() as db:
            user=User(username='browser_transport_check',hashed_password='unused-test-password',role=3)
            db.add(user);db.commit();db.refresh(user);user_id=user.id
        token=app.create_access_token({'sub':'browser_transport_check'})
        with TestClient(app.app) as client:asyncio.run(check(app,client,token,user_id))
        for engine in app.neck_effort_engines.values():engine.close()


if __name__=='__main__':main()
