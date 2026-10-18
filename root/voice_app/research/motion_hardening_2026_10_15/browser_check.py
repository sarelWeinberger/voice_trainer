"""Actual browser FaceMesh, motion worker and offline stop-response fallback."""
import argparse,asyncio,json,sys
from pathlib import Path
from playwright.async_api import async_playwright
folder=Path(__file__).resolve().parent;root=folder.parents[1];sys.path.insert(0,str(root))
from head_motion import analyze_payload
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--video',type=Path,default=root/'data/WIN_20261011_10_33_38_Pro.mp4')
parser.add_argument('--output-dir',type=Path,default=folder)
parser.add_argument('--mobile-only',action='store_true')
parser.add_argument('--normal-only',action='store_true')
args=parser.parse_args()
video=args.video.read_bytes();folder=args.output_dir;folder.mkdir(parents=True,exist_ok=True)

async def main():
 results=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
  context=await browser.new_context()
  for mobile in ((True,) if '--mobile-only' in sys.argv else (False,True)):
   page=await context.new_page();await page.set_viewport_size({'width':390 if mobile else 1280,'height':900});errors=[]
   page.on('pageerror',lambda error:errors.append(str(error)))
   await page.route('**/__motion_test_clip',lambda route:route.fulfill(status=200,content_type='video/mp4',body=video))
   await page.goto('http://127.0.0.1:5000'+('/mobile.html' if mobile else '/'),wait_until='domcontentloaded')
   await page.wait_for_function('typeof NeckMotionCapture === "function" && typeof FaceMesh === "function"')
   for occlusion in ((False,) if '--normal-only' in sys.argv else (False,True)):
    outcome=await asyncio.wait_for(page.evaluate('''async ({mobile,occlusion})=>{
     if(mobile){document.querySelectorAll('.m-section').forEach(section=>section.classList.toggle('active',section.id==='tab-record'));}
     else activateTab('record');
     const v=mobile?mVideo:video;v.srcObject=null;v.muted=true;v.playsInline=true;v.src='/__motion_test_clip';
     await new Promise((resolve,reject)=>{v.onloadeddata=resolve;v.onerror=reject;});
     let detections=0,requests=0,busy=false;const inferenceMs=[],trackingMs=[];
     faceMesh=new FaceMesh({locateFile:file=>'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/'+file});
     faceMesh.setOptions({maxNumFaces:1,refineLandmarks:false,minDetectionConfidence:.5,minTrackingConfidence:.5});
     faceMesh.onResults(results=>{detections++;onFaceResults(results);});
     isRecording=false;
     await Promise.race([faceMesh.send({image:v}),new Promise((_,reject)=>setTimeout(()=>reject(Error('FaceMesh initialization timeout')),25000))]);
     if(!neckMotionSeed)throw Error('Real FaceMesh did not detect the initial face');
     recordingStart=Date.now();neckEffortSessionId='browser-verification';isRecording=true;
     neckSampler=new NeckCapture({video:v,canvas:document.createElement('canvas'),sessionId:neckEffortSessionId,startedAt:recordingStart,motionSeed:neckMotionSeed,headers:()=>({}),onResult:()=>{}});
     const capture=new NeckMotionCapture(v,recordingStart,neckMotionSeed);neckSampler.motion=capture;
     const captureFrame=capture.capture.bind(capture);capture.capture=meta=>{const before=performance.now();captureFrame(meta);trackingMs.push(performance.now()-before);};capture.start();
     let intercepted=false;
     const refresh=setInterval(async()=>{
      if(!busy&&neckSampler.requestMotionReference()){
       busy=true;requests++;
       const before=performance.now();try{await faceMesh.send({image:v});}finally{inferenceMs.push(performance.now()-before);busy=false;}
      }
     },50);
     const inject=setInterval(()=>{if(occlusion&&capture.tracker&&!intercepted){intercepted=true;const push=capture.tracker.push.bind(capture.tracker);
      capture.tracker.push=(gray,t,seed)=>push(t>.7&&t<.85?new Uint8Array(gray.length).fill(120):gray,t,seed);
     }},10);
     const ended=new Promise(resolve=>v.onended=resolve);await v.play();await ended;capture.stop();clearInterval(refresh);clearInterval(inject);isRecording=false;
     for(let i=0;i<100&&busy;i++)await new Promise(resolve=>setTimeout(resolve,10));
     const trace=capture.export(),local=await neckSampler.localMotionReport();
     await faceMesh.close();
     getAuthHeaders=()=>({Authorization:'Bearer local-browser-test'});_currentUser={role:3};
     const savedFetch=window.fetch;
     window.fetch=async(url,options)=>{if(String(url).includes('/api/neck-effort/recording/stop'))throw new TypeError('simulated disconnect');return savedFetch(url,options);};
     let fallback;
     try{fallback=await stopNeckEffortRecordingSession(new Blob(['test'],{type:'audio/wav'}),0);}finally{window.fetch=savedFetch;}
     const c=document.getElementById(mobile?'m-neck-v1-results':'neck-analysis-v1-results');
     return {trace,local,detections,requests,inferenceMs,trackingMs,fallback,rendered:c.textContent,hasMotionTable:!!c.querySelector('.neck-motion-table')};
    }''',dict(mobile=mobile,occlusion=occlusion)),timeout=75)
    report=analyze_payload(json.dumps(outcome['trace']))
    (folder/'last_browser_diagnostic.json').write_text(json.dumps(dict(mobile=mobile,occlusion=occlusion,report=report,inference_ms=outcome['inferenceMs'],tracking_ms=outcome['trackingMs']),indent=2))
    assert outcome['detections']>=3,outcome['detections']
    assert outcome['requests']>=2,outcome['requests']
    assert report['status']=='measured',report
    assert outcome['local']['version']==report['version']
    assert outcome['fallback']['neckAnalysis']['movement_stability']['source']=='browser_local'
    assert outcome['hasMotionTable'],outcome['rendered']
    if occlusion:assert report['tracking_diagnostics']['reinitializations']>=1,report
    assert not errors,errors
    label=('mobile' if mobile else 'desktop')+('_occlusion' if occlusion else '_normal')
    (folder/(label+'_trace.json')).write_text(json.dumps(outcome['trace']))
    entry=dict(case=label,detections=outcome['detections'],requests=outcome['requests'],inference_ms=outcome['inferenceMs'],tracking_ms=outcome['trackingMs'],server_report=report,local_report=outcome['local'],offline_fallback_visible=outcome['hasMotionTable'],javascript_errors=errors)
    results.append(entry);(folder/'browser_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(label,'real detections',outcome['detections'],'measured windows',sum(w['status']=='measured' for w in report['windows']),'supported periodic windows',sum(w.get('periodicity_status')=='supported' for w in report['windows']),flush=True)
   await page.close()
  await browser.close()

if __name__=='__main__':asyncio.run(main())
