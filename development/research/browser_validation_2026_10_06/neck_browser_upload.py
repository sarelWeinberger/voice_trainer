import asyncio,json
from pathlib import Path
import cv2,numpy as np,soundfile as sf
from playwright.async_api import async_playwright
OUT=Path('/tmp/neck-browser-results')
writer=cv2.VideoWriter(str(OUT/'blank.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
for _ in range(30):writer.write(np.zeros((240,320,3),dtype=np.uint8))
writer.release();sf.write(str(OUT/'silence.wav'),np.zeros(48000),16000)
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'])
  page=await browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto('http://127.0.0.1:5099/',wait_until='domcontentloaded')
  await page.locator('#auth-username').fill('sarelw1@gmail.com');await page.locator('#auth-password').fill('admin1234');await page.locator('#login-btn').click()
  await page.locator('#record-btn').wait_for(state='visible')
  await page.locator('#observable-upload-form input[name="video"]').set_input_files(str(OUT/'blank.mp4'))
  await page.locator('#observable-upload-form input[name="audio"]').set_input_files(str(OUT/'silence.wav'))
  await page.locator('#observable-upload-form button').click()
  await page.locator('#observable-download').wait_for(state='visible',timeout=60000)
  assert 'לא ניתן להעריך' in await page.locator('#observable-upload-report').inner_text()
  async with page.expect_download() as download_event:await page.locator('#observable-download').click()
  download=await download_event.value;await download.save_as(str(OUT/'uploaded-report.json'))
  data=json.loads(OUT.joinpath('uploaded-report.json').read_text())
  assert data['neckAnalysis']['overall']['continuous_score'] is None
  await page.route('**/api/observable-effort/recording',lambda route:route.fulfill(status=502,json={'detail':'בדיקת כשל שירות מדומה'}))
  await page.locator('#observable-upload-form button').click();await page.wait_for_timeout(500)
  assert 'בדיקת כשל שירות מדומה' in await page.locator('#observable-upload-status').inner_text()
  assert await page.locator('#observable-download').is_hidden()
  assert await page.locator('#observable-upload-form button').is_enabled()
  result={'real_upload':True,'json_download':True,'simulated_502':True,'javascript_errors':errors}
  OUT.joinpath('upload-results.json').write_text(json.dumps(result));print(json.dumps(result))
  await browser.close()
asyncio.run(main())
