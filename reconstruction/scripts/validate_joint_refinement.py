"""Numerical checks for exported joint geometry and comparison semantics."""
from pathlib import Path
import json,numpy as np,argparse
parser=argparse.ArgumentParser();parser.add_argument('--run',default='joint');args=parser.parse_args()
R=Path(__file__).resolve().parents[1];O=R/'known_scene/pipette_depth_03_hand_milestone';J=O/args.run;a=np.load(J/'joint_fit.npz');e=np.load(O/'joint/evidence.npz');report=json.loads((J/'metrics.json').read_text());checks={}
assert np.array_equal(a['frame_ids'],e['frame_ids'])
for s in ['left','right']:
 assert np.array_equal(a[s+'_valid'],e[s+'_valid']);assert a[s+'_vertices'].shape==(len(a['frame_ids']),778,3)
 for key in ['vertices','joints','trans','rot','pose','betas']:assert np.isfinite(a[s+'_'+key]).all()
 valid=a[s+'_valid'];assert np.ptp(a[s+'_betas'][valid],axis=0).max()<1e-6
for obj in ['pipette','tube']:
 T=a[obj+'_T'];rot=T[:,:3,:3];assert np.allclose(rot@rot.transpose(0,2,1),np.eye(3),atol=1e-5);assert np.allclose(np.linalg.det(rot),1,atol=1e-5)
 delta=np.linalg.norm(T[:,:3,3]-e[obj+'_T'][:,:3,3],axis=1);assert delta.max()<=.01501;assert delta.max()>1e-6
 checks[obj+'_max_translation_correction_mm']=float(delta.max()*1000)
for key in ['contact_attraction','contact_persistence','manual_grasp_gates']:assert report['settings'][key] is False
checks['finite_geometry']=True;checks['rigid_objects']=True;checks['fixed_hand_shape']=True;checks['no_contact_terms']=True
checks['metrics']={}
for name in ['left','right','pipette','tube']:
 checks['metrics'][name]={}
 for stage in ['baseline','final']:
  checks['metrics'][name][stage]={k:float(np.nanmedian([np.nan if v is None else v for v in report[stage][name][k]])) for k in ['depth_median_mm','mask_iou','mask_coverage']}
checks['collision']={pair:{stage:dict(median_mm=float(np.median(report[stage]['collision'][pair]['max_violation_mm'])),max_mm=float(np.max(report[stage]['collision'][pair]['max_violation_mm']))) for stage in ['baseline','final']} for pair in report['final']['collision']}
checks['all_reported_violations_zero']=all(v['final']['max_mm']<1e-3 for v in checks['collision'].values())
if (J/'hard_status.json').exists():
 h=json.loads((J/'hard_status.json').read_text());assert all(h['feasible']);assert max(h['final_penetration_mm'])<=h['limit_mm'];assert all(x['max_accepted_penetration_mm'] is None or x['max_accepted_penetration_mm']<=h['limit_mm'] for x in h['accepted_steps']);checks['sampled_hard_bound_satisfied']=True;checks['hard_limit_mm']=h['limit_mm'];checks['max_checked_penetration_mm']=max(h['final_penetration_mm'])
(J/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
