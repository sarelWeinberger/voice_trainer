"""Recheck the final estimator in a genuinely offline browser on captured live-FaceMesh traces."""
import asyncio,json,sys
from pathlib import Path
from playwright.async_api import async_playwright
folder=Path(__file__).resolve().parent;root=folder.parents[1];sys.path.insert(0,str(root))
from head_motion import analyze_payload

async def main():
 results=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
  context=await browser.new_context();page=await context.new_page()
  await page.goto('http://127.0.0.1:5000/',wait_until='domcontentloaded')
  await page.wait_for_function('typeof NeckMotionAnalysis === "object"')
  await context.set_offline(True)
  for case in ('desktop_normal','desktop_occlusion','mobile_normal','mobile_occlusion'):
   trace=json.loads((folder/(case+'_trace.json')).read_text());server=analyze_payload(json.dumps(trace))
   outcome=await page.evaluate('''async trace=>{
    const local=await NeckMotionAnalysis.analyzeAsync(trace);
    getAuthHeaders=()=>({Authorization:'Bearer offline-test'});_currentUser={role:3};neckEffortSessionId='offline-replay';
    neckSampler={stop:async()=>{},localMotionReport:async()=>local,motionTrace:()=>trace};
    const packet=await stopNeckEffortRecordingSession(new Blob(['test'],{type:'audio/wav'}),0);
    const container=document.getElementById('neck-analysis-v1-results');
    return {local,fallback:packet.neckAnalysis.status,motionVisible:!!container.querySelector('.neck-motion-table')};
   }''',trace)
   client=outcome['local'];assert client['status']==server['status'];assert len(client['windows'])==len(server['windows'])
   for a,b in zip(server['windows'],client['windows']):
    for key in ('status','reason','periodicity_status','dominant_frequency_hz'):assert a.get(key)==b.get(key),(case,key,a,b)
    for key in ('amplitude_percent_image_width','periodic_fraction','half_coherence','half_amplitude_ratio'):
     if a.get(key) is not None:assert abs(a[key]-b[key])<=.001,(case,key,a,b)
   assert outcome['fallback']=='error';assert outcome['motionVisible']
   results.append(dict(case=case,offline_browser=True,local_motion_visible=True,server_report=server,local_report=client))
  await browser.close()
 (folder/'final_replay_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
 print([{ 'case':r['case'],'measured':sum(w['status']=='measured' for w in r['server_report']['windows']),
  'supported':sum(w.get('periodicity_status')=='supported' for w in r['server_report']['windows']),'offline_visible':r['local_motion_visible']} for r in results])

if __name__=='__main__':asyncio.run(main())
