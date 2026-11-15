"""Export a standalone four-way hand comparison, without model dependencies."""
from pathlib import Path
import json, shutil
import numpy as np
import trimesh
R=Path(__file__).resolve().parents[1]; O=R/'known_scene/pipette_depth_03_hand_milestone'; B=R/'known_scene/pipette_depth_03_tapir'; V=O/'comparison'; V.mkdir(exist_ok=True); (V/'rgb').mkdir(exist_ok=True)
names=['original','global','articulated','contact']; data={'variants':names,'hands':{},'objects':{},'summary':{}}
e=np.load(O/'evidence.npz'); ids=e['frame_ids']; data['ids']=ids.tolist(); data['K']=e['K'].tolist()
for side in ['left','right']:
 a=np.load(O/f'{side}_variants.npz'); metrics=json.loads((O/f'{side}_metrics.json').read_text()); data['hands'][side]={'valid':a['valid'].tolist(),'faces':a['faces'].tolist(),'reliability':a['reliability'].tolist(),'metrics':{},'fit_indices':a['fit_indices'].tolist()}; data['summary'][side]={}
 for name in names:
  x=a[name+'_vertices']; assert np.isfinite(x).all(); x.astype('<f4').tofile(V/f'{side}_{name}.bin'); data['hands'][side]['metrics'][name]=metrics[name]
  data['summary'][side][name]={key:float(np.nanmedian([np.nan if v is None else v for v in vals])) for key,vals in metrics[name].items()}
 data['hands'][side]['settings']=metrics['settings']
for obj in ['pipette','tube']:
 mesh=trimesh.load(B/obj/'mesh_local.ply',process=False); T=e[obj+'_T']; vertices=np.einsum('nij,vj->nvi',T[:,:3,:3],mesh.vertices)+T[:,None,:3,3]; vertices.astype('<f4').tofile(V/f'{obj}.bin'); data['objects'][obj]={'faces':mesh.faces.tolist(),'count':len(mesh.vertices)}
for i in ids: shutil.copyfile(B/'rgb'/f'{i:06d}.jpg',V/'rgb'/f'{i:06d}.jpg')
(V/'data.json').write_text(json.dumps(data,separators=(',',':')))
shutil.copyfile(R/'scripts/templates/hand_comparison.html',V/'index.html')
shutil.copyfile(R/'scripts/templates/hand_side_by_side.html',V/'side_by_side.html')
shutil.copyfile(R/'scripts/view_hand_comparison_3d.py',V/'view_3d.py')
(O/'comparison_summary.json').write_text(json.dumps(data['summary'],indent=2))
shutil.make_archive(str(O/'hand_comparison'),'zip',V)
print(json.dumps(data['summary'],indent=2));print('Viewer:',V)
