"""Independent edge/triangle crossing checks using Open3D ray casting.
Compares all six pairs with identical geometry, samples and tolerance. Counts
crossing mesh edges, not overlap volume; absence is not a continuous-time proof.
"""
import os
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import argparse,json,itertools
import numpy as np,trimesh,open3d as o3d
p=argparse.ArgumentParser();p.add_argument('--stages',nargs='+',default=['previous','final']);p.add_argument('--run',default='joint_collision');p.add_argument('--previous-run',default='joint');a=p.parse_args()
R=Path(__file__).resolve().parents[1];O=R/'known_scene/pipette_depth_03_hand_milestone';J=O/a.run;B=R/'known_scene/pipette_depth_03_tapir';before=np.load(O/a.previous_run/'joint_fit.npz');ids=before['frame_ids'];names=['left','right','pipette','tube'];faces={s:before[s+'_faces'] for s in ['left','right']};local={}
for obj in ['pipette','tube']:
 m=trimesh.load(O/'joint/tube_material_shell.ply' if obj=='tube' else B/obj/'mesh_local.ply',process=True);faces[obj]=m.faces;local[obj]=m.vertices
edges={n:np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0) for n,f in faces.items()};pairs=list(itertools.combinations(names,2))
def crossings(v,edge,scene,bounds):
 start=v[edge[:,0]];end=v[edge[:,1]];delta=end-start;length=np.linalg.norm(delta,axis=1);lo=np.minimum(start,end);hi=np.maximum(start,end);keep=(length>.0005)&(hi>=bounds[0]).all(1)&(lo<=bounds[1]).all(1)
 if not keep.any():return 0
 start=start[keep];delta=delta[keep];length=length[keep];rays=np.c_[start,delta/length[:,None]].astype('float32');hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)['t_hit'].numpy();return int(((hit>.00025)&(hit<length-.00025)).sum())
for stage in a.stages:
 source=before if stage=='previous' else np.load(J/'joint_fit.npz') if stage=='final' else None
 if stage=='original':
  raw=np.load(O/'hands_mano.npz');e=np.load(O/'joint/evidence.npz');source={s+'_vertices':raw[s+'_vertices'][ids] for s in ['left','right']};source.update({s+'_valid':before[s+'_valid'] for s in ['left','right']});source.update({obj+'_T':e[obj+'_T'] for obj in ['pipette','tube']})
 rows=[]
 for i,fid in enumerate(ids):
  verts={s:source[s+'_vertices'][i] for s in ['left','right']};valid={s:bool(source[s+'_valid'][i]) for s in ['left','right']}
  for obj in ['pipette','tube']:
   T=source[obj+'_T'][i];verts[obj]=local[obj]@T[:3,:3].T+T[:3,3];valid[obj]=True
  scenes={};bounds={}
  for n in names:
   if not valid[n]:continue
   scenes[n]=o3d.t.geometry.RaycastingScene(nthreads=4);scenes[n].add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(verts[n],np.float32)),o3d.core.Tensor(np.asarray(faces[n],np.uint32))));bounds[n]=np.array([verts[n].min(0),verts[n].max(0)])
  row={'frame':int(fid)}
  for x,y in pairs:
   key=x+'_'+y
   if not (valid[x] and valid[y]):row[key]=None;continue
   row[key]=crossings(verts[x],edges[x],scenes[y],bounds[y])+crossings(verts[y],edges[y],scenes[x],bounds[x])
  rows.append(row)
  if i%50==0:print(stage,i,'/',len(ids),flush=True)
 summary={}
 for x,y in pairs:
  key=x+'_'+y;v=np.array([r[key] for r in rows if r[key] is not None]);summary[key]=dict(median_crossing_edges=float(np.median(v)),mean_crossing_edges=float(np.mean(v)),max_crossing_edges=int(v.max()),frames_with_crossings=int((v>0).sum()),evaluated_frames=len(v))
 (J/f'{stage}_crossings.json').write_text(json.dumps(dict(method='Both mesh edge directions against the other surface; 0.25mm endpoint exclusion; hollow material tube mesh; no hand self or inter-frame checks',rows=rows,summary=summary),indent=2));print(stage,json.dumps(summary),flush=True)
