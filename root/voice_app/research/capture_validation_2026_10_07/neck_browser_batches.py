import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
async def main():
 results=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'])
  for mobile in (False,True):
   page=await browser.new_page(viewport={'width':390 if mobile else 1280,'height':844 if mobile else 900});errors=[];responses=[]
   page.on('pageerror',lambda e:errors.append(str(e)))
   async def observe(response):
    if response.url.endswith('/api/neck-effort/recording/stop'):
     responses.append(await response.json())
   page.on('response',observe)
   await page.goto('http://127.0.0.1:5099'+('/mobile.html' if mobile else '/'),wait_until='domcontentloaded')
   await page.locator('#auth-username').fill('sarelw1@gmail.com');await page.locator('#auth-password').fill('admin1234');await page.locator('#login-btn').click()
   button=page.locator('#m-record-btn' if mobile else '#record-btn');await button.wait_for(state='visible');await page.wait_for_timeout(5000)
   async def delay(route):
    await asyncio.sleep(1.7);await route.continue_()
   await page.route('**/api/neck-effort/frames',delay)
   await button.click();await page.wait_for_timeout(12000)
   await page.wait_for_function('!document.querySelector("'+('#m-record-btn' if mobile else '#record-btn')+'").disabled',timeout=20000)
   assert responses,responses
   diag=responses[-1]['neck_effort']['neckAnalysis']['diagnostics']
   assert diag['captured_frames']>=20,diag
   assert diag['effective_fps']>=3.5,diag
   assert not errors,errors
   results.append({'mobile':mobile,'diagnostics':diag,'errors':errors});print(json.dumps(results[-1]),flush=True)
   await page.close()
  await browser.close()
 Path('/tmp/neck-batch-browser-results.json').write_text(json.dumps(results,indent=2))
asyncio.run(main())
