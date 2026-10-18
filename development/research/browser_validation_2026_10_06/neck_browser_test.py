import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
OUT=Path('/tmp/neck-browser-results')
async def main():
 results=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'])
  for mobile in (False,True):
   context=await browser.new_context(viewport={'width':390 if mobile else 1280,'height':844 if mobile else 900},permissions=['camera','microphone'])
   page=await context.new_page(); errors=[]; responses=[]
   page.on('pageerror',lambda e:errors.append(str(e)))
   page.on('response',lambda r:responses.append({'url':r.url.split('5099')[-1],'status':r.status}) if '/api/' in r.url else None)
   await page.goto('http://127.0.0.1:5099'+('/mobile.html' if mobile else '/'),wait_until='domcontentloaded')
   await page.locator('#auth-username').fill('sarelw1@gmail.com'); await page.locator('#auth-password').fill('admin1234'); await page.locator('#login-btn').click()
   button=page.locator('#m-record-btn' if mobile else '#record-btn')
   await button.wait_for(state='visible'); await page.wait_for_timeout(7000)
   await button.click(); await page.wait_for_timeout(5500); await button.click()
   await page.wait_for_timeout(10000)
   container='#m-neck-v1-results' if mobile else '#neck-analysis-v1-results'
   real=await page.locator(container).inner_text()
   assert 'לא ניתן להעריך' in real,real
   assert len([r for r in responses if r['url']=='/api/neck-effort/recording/stop' and r['status']==200])>=1,responses
   keys=['posture_compensation','scm_lateral_recruitment','anterior_submental_activity','shoulder_recruitment','jaw_compensation']
   packet={'neckAnalysis':{'version':'neck-analysis-v1','status':'ok','dimensions':{k:{'score':i%4,'confidence':.8,'evidence':['עדות בדיקה מדומה <img src=x onerror=alert(1)>']} for i,k in enumerate(keys)},'overall':{'continuous_score':1.6,'category':2,'confidence':.8},'summary':'תוצאת בדיקה מדומה','quality':{'warnings':[]}}}
   await page.evaluate('([sel,packet])=>{const el=document.querySelector(sel);el.hidden=false;renderNeckAnalysis(el,packet)}',[container,packet])
   await page.locator('.m-nav-item[data-tab="results"]' if mobile else 'button.tab[data-tab="results"]').click()
   await page.locator(container).scroll_into_view_if_needed()
   assert await page.locator(container+' tr').count()==6
   assert await page.locator(container+' img').count()==0
   assert '1.6 / 3.0' in await page.locator(container).inner_text()
   overflow=await page.evaluate('({width:innerWidth,body:document.documentElement.scrollWidth,table:document.querySelector("'+container+' table").getBoundingClientRect().width})')
   await page.screenshot(path=str(OUT/('mobile-scores.png' if mobile else 'desktop-scores.png')),full_page=True)
   packet['neckAnalysis']['status']='error';packet['neckAnalysis']['dimensions']={};packet['neckAnalysis']['overall']={'continuous_score':None,'category':None,'confidence':0};packet['reasons']=['OPENAI_API_KEY missing']
   await page.evaluate('([sel,packet])=>renderNeckAnalysis(document.querySelector(sel),packet)',[container,packet])
   assert 'שירות הניתוח טרם הוגדר בשרת' in await page.locator(container).inner_text()
   assert '0 / 3' not in await page.locator(container).inner_text()
   results.append({'mode':'mobile' if mobile else 'desktop','real_recording_text':real,'api_responses':responses,'javascript_errors':errors,'layout':overflow,'mock_scores_and_error_passed':True})
   await context.close()
  await browser.close()
 OUT.joinpath('results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(results,ensure_ascii=False))
asyncio.run(main())
