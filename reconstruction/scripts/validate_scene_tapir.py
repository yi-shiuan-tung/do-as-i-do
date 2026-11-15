"""Check correspondence integrity and report continuity without treating smoothness as accuracy."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1]
O=R/'known_scene/pipette_depth_03_hand_milestone'
J=O/'joint_tapir'
a=np.load(J/'joint_fit.npz');b=np.load(O/'joint_collision/joint_fit.npz');t=np.load(J/'tracks.npz')
r=json.loads((J/'metrics.json').read_text())
assert np.array_equal(a['frame_ids'],b['frame_ids']) and np.array_equal(t['frame_ids'],a['frame_ids'])
assert not r['settings']['hard_bound'] and not r['settings']['overlapping_window_refinement']
assert len(r['history'])==len(a['frame_ids'])
assert {x['frame'] for x in r['history']}==set(a['frame_ids'])
out={'frames':len(a['frame_ids']),'soft_collisions':True,'overlapping_window_refinement':False,'entities':{}}
for en,n in enumerate(['left','right','pipette','tube']):
 for row in t['track_ids'][:,en]:assert len(set(row[row>=0]))==sum(row>=0)
 for key in ['vertices','predicted_vertices']:assert np.isfinite(a[n+'_'+key]).all()
 valid=a[n+'_valid'].astype(bool) if n in ['left','right'] else np.ones(len(a['frame_ids']),bool)
 both=valid[1:]&valid[:-1];triples=valid[2:]&valid[1:-1]&valid[:-2]
 entity={}
 for stage,x in [('previous_soft',b),('tapir',a)]:
  pos=x[n+'_joints'][:,0] if n in ['left','right'] else x[n+'_T'][:,:3,3]
  step=np.linalg.norm(np.diff(pos,axis=0),axis=1)[both]*1000
  accel=np.linalg.norm(np.diff(pos,n=2,axis=0),axis=1)[triples]*1000
  entity[stage]={'median_step_mm':float(np.median(step)),'p95_step_mm':float(np.percentile(step,95)),'median_second_difference_mm':float(np.median(accel))}
 rows=[x for x in r['history'] if x['source_frame'] is not None and x['anchors'][n]>0]
 entity['transitions_with_surface_anchors']=len(rows)
 entity['median_surface_anchors']=float(np.median([x['anchors'][n] for x in rows]))
 entity['median_track_residual_px']=float(np.median([x['track_residual_median_px'][n] for x in rows]))
 out['entities'][n]=entity
out['interpretation']='Motion includes real motion; reduced jitter is not ground-truth accuracy. Only consecutive observed-hand samples are compared.'
(J/'temporal_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
