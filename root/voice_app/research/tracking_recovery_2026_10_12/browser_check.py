"""Real video playback with offline FaceMesh references and injected tracking loss.

This tests browser acquisition/recovery, not a physical camera or live FaceMesh.
"""
import asyncio,json,sys
from pathlib import Path
from playwright.async_api import async_playwright
folder=Path(__file__).resolve().parent
root=folder.parents[1];sys.path.insert(0,str(root))
from head_motion import analyze_payload
references=json.loads((folder/'face_reference_fixture.json').read_text())
video=(root/'data/WIN_20261011_10_33_38_Pro.mp4').read_bytes()
packet=json.loads((root/'research/recording_10_33_38_2026_10_11/video_results.json').read_text())['stages'][0]['result']

async def main():
    results=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
        for mobile in (False,True):
            for occlusion in (False,True):
                page=await browser.new_page(viewport={'width':390 if mobile else 1280,'height':900})
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                await page.route('**/__motion_test_clip',lambda route:route.fulfill(status=200,content_type='video/mp4',body=video))
                await page.goto('http://127.0.0.1:5000'+('/mobile.html' if mobile else '/'),wait_until='domcontentloaded')
                await page.wait_for_function('typeof NeckMotionCapture === "function"')
                outcome=await page.evaluate('''async ({references,occlusion})=>{
                    const v=document.createElement('video');v.muted=true;v.playsInline=true;v.style.width='256px';
                    v.src='/__motion_test_clip';document.body.appendChild(v);
                    await new Promise((resolve,reject)=>{v.onloadeddata=resolve;v.onerror=reject;});
                    const capture=new NeckMotionCapture(v,Date.now(),{at:Date.now(),points:references[0].points});
                    let requests=0,intercepted=false;
                    capture.start();
                    const refresh=setInterval(()=>{
                        if(capture.requestReference()){
                            requests++;
                            const ref=references.reduce((best,r)=>Math.abs(r.t-v.currentTime)<Math.abs(best.t-v.currentTime)?r:best,references[0]);
                            capture.updateReference({at:Date.now(),points:ref.points});
                        }
                    },100);
                    const injection=setInterval(()=>{
                        if(occlusion&&capture.tracker&&!intercepted){
                            intercepted=true;const push=capture.tracker.push.bind(capture.tracker);
                            capture.tracker.push=(gray,t,seed)=>push(t>.7&&t<.85?new Uint8Array(gray.length).fill(120):gray,t,seed);
                        }
                    },10);
                    const done=new Promise(resolve=>v.onended=resolve);await v.play();await done;
                    clearInterval(refresh);clearInterval(injection);capture.stop();
                    const trace=capture.export();v.remove();return {trace,requests};
                }''',dict(references=references,occlusion=occlusion))
                trace=outcome['trace'];report=analyze_payload(json.dumps(trace))
                assert report['status']=='measured',report
                measured=[w for w in report['windows'] if w['status']=='measured']
                assert len(measured)>=3,report
                if occlusion:
                    assert outcome['requests']>=1,outcome
                    assert report['tracking_diagnostics']['reinitializations']>=1,report
                    assert any(w.get('reason')=='tracking_reinitialized' for w in report['windows']),report
                    assert sum(r[5]>=.5 for r in trace['samples'] if r[0]>3)>=50
                current=json.loads(json.dumps(packet));current['neckAnalysis']['movement_stability']=report
                checks=await page.evaluate('''packet=>{
                    const c=document.createElement('div');document.body.appendChild(c);renderNeckAnalysis(c,packet,{viewer:{role:3}});
                    const checks={movement_visible:!!c.querySelector('.neck-motion-table'),no_admin_debug:!c.querySelector('.neck-admin-debug'),
                        group_grades_preserved:c.querySelectorAll('.neck-group-score').length===3};
                    packet.neckAnalysis.status='error';renderNeckAnalysis(c,packet,{viewer:{role:3}});
                    checks.local_motion_on_provider_failure=!!c.querySelector('.neck-motion-table');c.remove();return checks;
                }''',current)
                assert all(checks.values()),checks
                assert not errors,errors
                label=('mobile' if mobile else 'desktop')+('_occlusion' if occlusion else '_normal')
                (folder/(label+'_trace.json')).write_text(json.dumps(trace))
                results.append(dict(case=label,requests=outcome['requests'],report=report,checks=checks,javascript_errors=errors))
                print(label,'samples',len(trace['samples']),'measured',len(measured),'reinitializations',report['tracking_diagnostics'].get('reinitializations'),flush=True)
                await page.close()
        await browser.close()
    (folder/'browser_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':asyncio.run(main())
