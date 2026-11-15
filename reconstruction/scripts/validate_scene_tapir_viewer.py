from playwright.sync_api import sync_playwright
from pathlib import Path
import json
O=Path('/home/ytung/Projects/do-as-i-do/reconstruction/known_scene/pipette_depth_03_hand_milestone')
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']);page=b.new_page(viewport={'width':1800,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto('http://127.0.0.1:8084/side_by_side.html?joint=1');page.wait_for_function('window.comparisonReady === true');assert page.locator('#fit').input_value()=='joint';assert page.locator('#fit option[value=contact]').count()==0
 assert 'TAPIR predict + joint correction' in page.locator('#fit').inner_text();assert page.locator('#hardNotice').count()==0;assert 'Soft collision penalties' in page.locator('#status').inner_text();assert page.evaluate('data.tracking !== undefined && !data.hard_status');assert 'TAPIR reference frame' in page.locator('#fittedStats').inner_text()
 keys=page.evaluate('renderers.map(r=>Object.keys(r.cache))');assert 'pipette' in keys[0] and 'pipette_joint' in keys[1];assert 'left_original' in keys[0] and 'left_joint' in keys[1]
 page.screenshot(path=str(O/'joint_tapir/side_by_side_preview.png'),full_page=True)
 old=page.locator('#fitted').evaluate('(c)=>c.toDataURL()');page.mouse.move(800,500);page.mouse.down();page.mouse.move(940,510,steps=6);page.mouse.up();assert old!=page.locator('#fitted').evaluate('(c)=>c.toDataURL()');page.locator('#reset').click()
 for i in [0,100,143,233]:
  page.locator('#slider').fill(str(i));page.locator('#slider').dispatch_event('input');fid=120+3*i;page.wait_for_function(f"document.querySelector('#frame').textContent.includes('Frame {fid} ·')");assert page.locator('#rgb').get_attribute('src').endswith(f'{fid:06d}.jpg')
 page.select_option('#side','right');page.locator('#objects').uncheck();page.locator('#wire').check();page.locator('#play').click();page.wait_for_timeout(1400);page.locator('#play').click();assert page.locator('#slider').input_value()!='233';assert not errors,errors
 (O/'joint_tapir/browser_validation.json').write_text(json.dumps({'page_errors':errors,'default_joint':True,'contact_option_removed':True,'distinct_original_joint_object_buffers':keys,'synchronized_rgb_frames':[120,420,549,819],'orbit_checked':True,'playback_checked':True},indent=2));b.close();print('Joint viewer checks passed')
