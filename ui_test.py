from playwright.sync_api import sync_playwright
from pathlib import Path
import json
ROOT=Path(__file__).parent
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1280,'height':900},device_scale_factor=1)
 errors=[];page.on('pageerror',lambda e: errors.append(str(e)))
 page.goto((ROOT/'index.html').as_uri());page.wait_for_selector('#outputs .model')
 assert page.locator('#outputs .model').count()==2
 page.select_option('#sample','3-0');assert 'Electrode' in page.locator('#source').inner_text()
 page.fill('#note','Review electrode motion failure');page.click('#save');page.click('[data-tab=sessions]');assert 'Review electrode' in page.locator('#sessionRows').inner_text()
 page.reload();assert 'Review electrode' in page.locator('#sessionRows').inner_text()
 page.click('[data-replay="0"]');page.wait_for_selector('#outputs .model');assert page.locator('#outputs .model').count()==2
 page.locator('details summary').click();page.fill('#csvText','1\n2');page.click('#importCSV');assert '1,024' in page.locator('#csvError').inner_text()
 page.click('[data-tab=stress]');page.click('#stressRun');assert 'Probability shift' in page.locator('#stressOutputs').inner_text()
 page.select_option('#stressType','clip');page.click('#stressRun');assert 'clipping' not in errors
 page.click('[data-tab=evidence]');assert 'accepted' in page.locator('#coverageStats').inner_text();page.locator('#coverageThreshold').fill('0.9');assert '90%' in page.locator('#coverageValue').inner_text()
 page.click('[data-tab=errors]');assert 'matching windows' in page.locator('#errorCount').inner_text();page.select_option('#errorClass','2');assert page.locator('#errorRows .item').count()>0
 for width in [320,390,1280]:
  page.set_viewport_size({'width':width,'height':900})
  for route in ['lab','stress','evidence','errors','sessions','guide']:
   page.goto((ROOT/'index.html').as_uri()+'#'+route);page.wait_for_timeout(130)
   if route=='stress':page.click('#stressRun')
   assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),(width,route,'overflow')
   page.screenshot(path=str(ROOT/f'{route}-{width}.png'),full_page=True)
 assert not errors,errors
 # Download actual exported offline HTML and reopen it.
 page.goto((ROOT/'index.html').as_uri()+'#guide')
 with page.expect_download() as d:page.click('#saveOffline')
 d.value.save_as(str(ROOT/'SignalProof_Offline.html'))
 page.goto((ROOT/'SignalProof_Offline.html').as_uri()+'#lab');page.wait_for_selector('#outputs .model')
 assert page.locator('#outputs .model').count()==2
 b.close()
 print('UI PASS: all six routes at 320/390/1280; no overflow/errors; inference, validation, save/reload, replay, stress, evidence threshold, filters, offline export/reopen.')
