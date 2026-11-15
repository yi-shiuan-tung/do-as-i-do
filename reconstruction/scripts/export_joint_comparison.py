"""Publish the joint fit in the existing side-by-side viewer, preserving baselines."""
import json,shutil,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--run',default='joint');args=parser.parse_args()
R=Path(__file__).resolve().parents[1];O=R/'known_scene/pipette_depth_03_hand_milestone';J=O/args.run;V=O/'comparison';a=np.load(J/'joint_fit.npz');report=json.loads((J/'metrics.json').read_text());data=json.loads((V/'data.json').read_text());data['variants']=['original','joint'];data['joint_settings']=report['settings'];data['summary']={}
for side in ['left','right']:
 x=a[side+'_vertices'];assert np.isfinite(x).all();x.astype('<f4').tofile(V/f'{side}_joint.bin');h=data['hands'][side];h['metrics']={}
 for variant,source in [('original','baseline'),('joint','final')]:
  m=report[source][side];ix=h['fit_indices'];coll=report[source]['collision'];maxcoll=np.max(np.array([coll[key]['max_penetration_mm' if args.run=='joint_hard' else 'max_violation_mm'] for obj in ['pipette','tube'] for key in [side+'_'+obj,obj+'_'+side] if key in coll]),axis=0)
  h['metrics'][variant]={key:[values[i] for i in ix] for key,values in m.items()};h['metrics'][variant]['collision_violation_mm']=maxcoll[ix].tolist()
  crossings=J/('original_crossings.json' if variant=='original' else 'final_crossings.json')
  if crossings.exists():
   rows=json.loads(crossings.read_text())['rows'];h['metrics'][variant]['crossing_edges']=[sum(rows[i].get(side+'_'+obj) or 0 for obj in ['pipette','tube']) for i in ix]
  data['summary'][side+'_'+variant]={key:float(np.nanmedian([np.nan if z is None else z for z in vals])) for key,vals in h['metrics'][variant].items()}
for obj in ['pipette','tube']:
 a[obj+'_vertices'].astype('<f4').tofile(V/f'{obj}_joint.bin');data['objects'][obj]['joint_metrics']=report['final'][obj]
if args.run=='joint_hard':
 data['hard_status']=json.loads((J/'hard_status.json').read_text());shutil.copyfile(J/'hard_status.json',V/'hard_status.json')
if args.run=='joint_tapir':
 data['tracking']={str(row['frame']):row for row in report['history']}
(V/'joint_data.json').write_text(json.dumps(data,separators=(',',':')))
shutil.copyfile(J/'metrics.json',V/'joint_metrics.json')
s=(R/'scripts/templates/hand_side_by_side.html').read_text()
s=s.replace('<option value="contact">Contact-aware</option><option value="articulated">Articulated</option><option value="global">Global wrist fit</option>','<option value="joint">Joint RGB-D refinement</option>')
s=s.replace('Contact-aware fit <span','Joint RGB-D refinement <span').replace('Both 3D panels use identical object poses.','Left: initial object poses. Right: jointly refined object poses.')
s=s.replace('Contact-aware fitting uses hypothesized contacts and manual grasp intervals; intersections remain.','Joint refinement optimizes both wrists, finger poses, and both object poses against corrected hand masks, measured depth, and WiLoR RGB landmarks. No contact attraction, contact persistence, or manual grasp intervals are used. Collision constraints use sampled surfaces with a 0.25 mm tolerance; residual intersections can remain. Hand self-collisions and continuous-time collisions are not constrained. Original and fitted metrics both use the corrected masks. Tube collision geometry preserves its hollow bore.')
s=s.replace("fetch('data.json')","fetch('joint_data.json', {cache:'no-store'})")
s=s.replace("[...Object.keys(data.objects),...['left','right']", "[...Object.keys(data.objects).flatMap(o=>[o,o+'_joint']),...['left','right']")
s=s.replace("fetch(key+'.bin')", "fetch(key+'.bin', {cache:'no-store'})")
s=s.replace("mesh(name,obj.count,obj.faces,colors[name],false)","mesh(variant==='joint'?name+'_joint':name,obj.count,obj.faces,colors[name],false)")
s=s.replace("for(let [name,o] of Object.entries(data.objects))add(name,o.count);", "for(let [name,o] of Object.entries(data.objects)){add(name,o.count);add(name+'_joint',o.count);}")
s=s.replace("proxy max penetration ${m.proxy_max_penetration_mm[i].toFixed(1)} mm", "sampled collision violation ${m.collision_violation_mm[i].toFixed(1)} mm")
s=s.replace('<p><a href="joint_metrics.json">Detailed fitting metrics</a></p>','')
s=s.replace('<summary>Scope and limitations</summary>', '<summary>Scope and limitations</summary><p><a href="joint_metrics.json">Detailed fitting metrics</a></p>')
s=s.replace('Ready · Cameras and frames are synchronized · Contact-aware fit selected','Ready · Joint RGB-D reconstruction · No contact attraction or persistence')
if args.run=='joint_collision':
 s=s.replace('Joint RGB-D refinement','Joint RGB-D + collision penalties')
 s=s.replace('Ready · Joint RGB-D reconstruction · No contact attraction or persistence','Ready · Joint RGB-D + bidirectional collision penalties · All four entities optimized')
 s=s.replace('Collision constraints use sampled surfaces with a 0.25 mm tolerance; residual intersections can remain.', 'The same bidirectional collision penalty is applied to all six mesh pairs, with a 0.25 mm tolerance. Object surfaces inside hands are checked as well as hands inside objects. Surface-crossing counts come from independent edge/triangle tests. Residual intersections can remain.')
 s=s.replace('sampled collision violation ${m.collision_violation_mm[i].toFixed(1)} mm', 'bidirectional violation ${m.collision_violation_mm[i].toFixed(1)} mm · crossing edges ${m.crossing_edges?.[i] ?? "n/a"}')
 for filename in ['original_crossings.json','previous_crossings.json','final_crossings.json']:
  if (J/filename).exists():shutil.copyfile(J/filename,V/filename)
if args.run=='joint_hard':
 s=s.replace('Joint RGB-D + collision penalties','Joint RGB-D · hard 1 mm bound')
 s=s.replace('bidirectional violation ${m.collision_violation_mm[i].toFixed(1)} mm', 'checked penetration ${m.collision_violation_mm[i].toFixed(2)} mm')
 old='The same bidirectional collision penalty is applied to all six mesh pairs, with a 0.25 mm tolerance.'
 s=s.replace(old,'The same 1 mm hard acceptance bound applies to sampled bidirectional penetration for all six mesh pairs. Infeasible starting frames undergo feasibility restoration; larger initial translations can degrade alignment or create gaps. RGB-D steps are projected and backtracked, then rejected if the bound is exceeded.')
 s=s.replace('Residual intersections can remain.', 'The hard bound covers the fixed sampled collision queries, not all triangles or motion between frames. Shallow or unsampled intersections can remain.')
 s=s.replace("$('fittedStats').textContent=stats(fit);render();", "$('fittedStats').textContent=stats(fit);render();hardStatus();")
 s=s.replace("$('status').textContent='Ready · Joint RGB-D + bidirectional collision penalties · All four entities optimized';", 'hardStatus();')
 statusfn="""function hardStatus(){if(!data?.hard_status)return;const h=data.hard_status;const ok=h.feasible[sample];$('status').textContent=(ok?'PASS':'INFEASIBLE')+' · Frame '+data.ids[sample]+' · maximum checked penetration '+h.final_penetration_mm[sample].toFixed(3)+' mm / '+h.limit_mm.toFixed(1)+' mm limit · sampled constraints only';$('status').style.color=ok?'#8fe2bb':'#ff8e8e';}
"""
 if 'function hardStatus()' not in s:s=s.replace('async function start()',statusfn+'async function start()')
 for filename in ['original_crossings.json','previous_crossings.json','final_crossings.json']:
  if (J/filename).exists():shutil.copyfile(J/filename,V/filename)
if args.run=='joint_tapir':
 import re
 s=re.sub(r'<option value="joint">[^<]*</option>', '<option value="joint">TAPIR predict + joint correction</option>', s)
 s=re.sub(r'<h2 id="fitTitle">.*?<span', '<h2 id="fitTitle">TAPIR predict + joint correction <span', s)
 s=re.sub(r'<p id="hardNotice".*?</p>', '', s)
 s=re.sub(r'function hardStatus\(\).*?\n', '', s)
 s=s.replace('hardStatus();', '')
 s=s.replace('window.comparisonReady=true;', "$('status').textContent='Ready · TAPIR tracks both hands and objects · Soft collision penalties';window.comparisonReady=true;")
 s=s.replace('checked penetration', 'sampled violation above 0.25 mm')
 s=s.replace('Right: jointly refined object poses.', 'Middle: jointly refined object poses.')
 s=s.replace("trackingText=tr?' · TAPIR anchors '+tr.anchors[s]+(tr.track_residual_median_px[s]===undefined?'':' · residual '+tr.track_residual_median_px[s].toFixed(1)+' px'):'';", "trackingText=tr?(tr.source_frame===null?' · TAPIR reference frame':' · TAPIR anchors '+tr.anchors[s]+(tr.track_residual_median_px[s]===undefined?'':' · residual '+tr.track_residual_median_px[s].toFixed(1)+' px')):'';")
 s=s.replace('let i=h.fit_indices.indexOf(sample),m=h.metrics[variant],d=m.depth_median_mm[i];', "let i=h.fit_indices.indexOf(sample),m=h.metrics[variant],d=m.depth_median_mm[i],tr=variant==='joint'?data.tracking?.[data.ids[sample]]:null,trackingText=tr?(tr.source_frame===null?' · TAPIR reference frame':' · TAPIR anchors '+tr.anchors[s]+(tr.track_residual_median_px[s]===undefined?'':' · residual '+tr.track_residual_median_px[s].toFixed(1)+' px')):'';")
 s=s.replace('crossing_edges?.[i] ?? "n/a"}`', 'crossing_edges?.[i] ?? "n/a"}${trackingText}`')

 details='<details><summary>Scope and limitations</summary><p><a href="joint_metrics.json">Detailed fitting metrics</a> · <a href="tracking_provenance.json">Tracking provenance</a></p><p>234 sampled frames (120–819) from pipette_depth_03, played at 10 fps. Original HaWoR is on the left; TAPIR-guided reconstruction is in the middle; synchronized recorded RGB is on the right. Both 3D views share camera controls. Hands without usable observations are hidden.</p><p>Persistent TAPIR points track both hands and both objects. Points are associated with mesh triangles using fixed barycentric coordinates. Their 2D motion predicts the next articulated hand poses and rigid object poses; each frame is then corrected jointly against masks, measured RealSense depth, and WiLoR RGB landmarks. Confidence, segmentation consistency, and surface association gate tracks. Fits propagate forward and backward from frame 549; initialization and reappearing hands use the previous soft RGB-D fit. Tracking uses short context segments, but there is no overlapping-window pose refinement.</p><p>All six mesh pairs receive bidirectional soft collision penalties with a 0.25 mm tolerance. There is no hard penetration bound, inferred contact, contact attraction, or contact persistence. Residual intersections can remain. Hand self-collisions and motion between samples are not constrained. Hand shape and calibrated camera are fixed. Depth errors measure visible overlap only; lower errors do not alone establish better reconstruction.</p></details>'
 s=re.sub(r'<details>.*?</details>',details,s,flags=re.S)
 for filename in ['tracking_provenance.json','original_crossings.json','previous_crossings.json','final_crossings.json']:
  if (J/filename).exists():shutil.copyfile(J/filename,V/filename)
(V/'side_by_side.html').write_text(s)
# Save the published template as the default so rebuilding never restores contact-aware.
(R/'scripts/templates/hand_side_by_side.html').write_text(s)
(V/'joint_summary.json').write_text(json.dumps(data['summary'],indent=2));shutil.make_archive(str(O/'hand_comparison'),'zip',V)
print(json.dumps(data['summary'],indent=2));print('Published side_by_side.html')
