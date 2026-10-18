import asyncio,base64,json,sys
from pathlib import Path
import cv2
from playwright.async_api import async_playwright
ROOT=Path('/home/elira/voice_trainer/root/voice_app')
cap=cv2.VideoCapture(str(ROOT/'data/WIN_20261008_18_21_53_Pro.mp4'))
fps=cap.get(cv2.CAP_PROP_FPS); frames=[]
for n in range(25):
 cap.set(cv2.CAP_PROP_POS_FRAMES,round(n/5*fps));ok,frame=cap.read()
 assert ok
 frame=cv2.resize(frame,(640,360))
 ok,png=cv2.imencode('.png',frame);assert ok
 frames.append('data:image/png;base64,'+base64.b64encode(png).decode())
cap.release()
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'])
  page=await browser.new_page();errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto('http://127.0.0.1:5099/',wait_until='domcontentloaded')
  await page.locator('#auth-username').fill('sarelw1@gmail.com');await page.locator('#auth-password').fill('admin1234');await page.locator('#login-btn').click()
  await page.locator('#record-btn').wait_for(state='visible')
  result=await page.evaluate('''async frames => {
    isRecording=true;
    const decoded=await Promise.all(frames.map(async src=>{const img=new Image();img.src=src;await img.decode();img.videoWidth=img.naturalWidth;img.videoHeight=img.naturalHeight;return img;}));
    const start=await fetch('/api/neck-effort/recording/start',{method:'POST',headers:getAuthHeaders()});
    if(!start.ok) throw Error('start failed'); const session=(await start.json()).session_id;
    const image=new Image();const canvas=document.createElement('canvas');
    const sampler=new NeckCapture({video:image,canvas,sessionId:session,startedAt:Date.now(),headers:getAuthHeaders,onResult(){}});
    sampler.active=true;const started=performance.now();
    for(let i=0;i<frames.length;i++){
      sampler.video=decoded[i];
      await new Promise(r=>setTimeout(r,Math.max(0,started+i*200-performance.now())));
      await sampler.capture();
    }
    await sampler.stop();const body=new FormData();body.append('session_id',session);
    const before=performance.now();const response=await fetch('/api/neck-effort/recording/stop',{method:'POST',headers:getAuthHeaders(),body});
    const data=await response.json();const packet=data.neck_effort;
    const analysis=packet.neckAnalysis;renderNeckAnalysis(document.querySelector('#neck-analysis-v1-results'),packet);
    return {status:response.status,analysis_status:analysis.status,diagnostics:analysis.diagnostics,reasons:packet.reasons,
      finalize_seconds:(performance.now()-before)/1000,rendered:document.querySelector('#neck-analysis-v1-results').innerText.length>0};
  }''',frames)
  assert result['status']==200,result
  assert result['diagnostics']['captured_frames']==25,result
  assert result['diagnostics']['feature_available_frames']['head_pitch_deg']>=20,result
  assert result['rendered'] and not errors,(result,errors)
  result['javascript_errors']=errors
  Path('/tmp/neck-user-browser-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
  print(json.dumps(result,ensure_ascii=False),flush=True)
  await browser.close()
asyncio.run(main())
