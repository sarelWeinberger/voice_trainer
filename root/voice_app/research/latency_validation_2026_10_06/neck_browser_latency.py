import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
async def main():
 results=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'])
  for mobile in (False,True):
   context=await browser.new_context(viewport={'width':390 if mobile else 1280,'height':844 if mobile else 900},permissions=['camera','microphone'])
   page=await context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   await page.goto('http://127.0.0.1:5099'+('/mobile.html' if mobile else '/'),wait_until='domcontentloaded')
   await page.locator('#auth-username').fill('sarelw1@gmail.com');await page.locator('#auth-password').fill('admin1234');await page.locator('#login-btn').click()
   button=page.locator('#m-record-btn' if mobile else '#record-btn');await button.wait_for(state='visible');await page.wait_for_timeout(4000)
   seen={};keys=['posture_compensation','scm_lateral_recruitment','anterior_submental_activity','shoulder_recruitment','jaw_compensation']
   packet={'model_version':'neck-analysis-v1','neckAnalysis':{'version':'neck-analysis-v1','status':'ok','dimensions':{k:{'score':1,'confidence':.8,'evidence':['בדיקת תצוגה']} for k in keys},'overall':{'continuous_score':1.,'category':1,'confidence':.8},'summary':'בדיקת תצוגה'}}
   async def neck(route):
    seen['neck']=time.monotonic();await asyncio.sleep(2);await route.fulfill(json={'neck_effort':packet})
   async def voice(route):
    seen['voice']=time.monotonic();await asyncio.sleep(4);await route.fulfill(status=500,json={'detail':'כשל קול מדומה'})
   await page.route('**/api/neck-effort/recording/stop',neck);await page.route('**/api/analyze',voice)
   print("testing",mobile,flush=True)
   await button.click();await page.wait_for_timeout(10500)
   sel='#m-neck-v1-results' if mobile else '#neck-analysis-v1-results'
   initial_delta = abs(seen['neck']-seen['voice'])
   assert initial_delta<.7,seen
   assert await page.locator(sel).is_visible()
   assert '1.0 / 3.0' in await page.locator(sel).inner_text()
   await page.wait_for_function("!document.querySelector('" + ("#m-record-btn" if mobile else "#record-btn") + "').disabled", timeout=10000)
   # Slow provider: browser must stop waiting and show a visible error, with no score.
   await page.unroute('**/api/neck-effort/recording/stop',neck)
   async def slow(route):
    await asyncio.sleep(23)
    try:await route.fulfill(json={'neck_effort':packet})
    except Exception:pass
   await page.route('**/api/neck-effort/recording/stop',slow)
   await page.locator('.m-nav-item[data-tab="record"]' if mobile else 'button.tab[data-tab="record"]').click()
   await button.click();await page.wait_for_timeout(5000)
   await page.wait_for_function("document.querySelector('"+sel+"').innerText.includes('עבר את זמן ההמתנה')", timeout=30000)
   text=await page.locator(sel).inner_text()
   assert 'עבר את זמן ההמתנה' in text,text
   assert 'לא ניתן להעריך' in text
   await page.wait_for_function("!document.querySelector('" + ("#m-record-btn" if mobile else "#record-btn") + "').disabled", timeout=10000)
   results.append({'mode':'mobile' if mobile else 'desktop','parallel_start_seconds':initial_delta,'neck_visible_on_voice_failure':True,'automatic_5_second_stop':True,'20_second_timeout_visible':True,'errors':errors})
   assert not errors,errors
   await context.close()
  await browser.close()
 Path('/tmp/neck-latency-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(results,ensure_ascii=False),flush=True)
asyncio.run(main())
