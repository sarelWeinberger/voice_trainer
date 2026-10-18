import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'])
  page=await browser.new_page();responses=[];errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('response',lambda r:responses.append([r.url.rsplit('/',1)[-1],r.status]) if '/api/' in r.url else None)
  await page.goto('http://127.0.0.1:5099/',wait_until='domcontentloaded');await page.locator('#auth-username').fill('sarelw1@gmail.com');await page.locator('#auth-password').fill('admin1234');await page.locator('#login-btn').click();await page.locator('#record-btn').wait_for(state='visible');await page.wait_for_timeout(4000)
  await page.locator('#record-btn').click();await page.wait_for_timeout(7000)
  await page.wait_for_function("!document.querySelector('#record-btn').disabled && document.querySelector('#neck-analysis-v1-results').innerText.includes('לא ניתן להעריך')",timeout=25000)
  assert ['analyze',200] in responses,responses
  assert ['stop',200] in responses,responses
  assert not errors,errors
  result={'actual_5_second_recording_api_responses':responses,'javascript_errors':errors}
  Path('/tmp/neck-latency-real.json').write_text(json.dumps(result));print(json.dumps(result),flush=True)
  await browser.close()
asyncio.run(main())
