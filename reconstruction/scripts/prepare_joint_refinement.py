"""Correct hand-mask association and close tube material without sealing its bore."""
from pathlib import Path
import json,cv2,numpy as np,trimesh
from rgbd_known_geometry import raster
R=Path(__file__).resolve().parents[1];O=R/'known_scene/pipette_depth_03_hand_milestone';B=R/'known_scene/pipette_depth_03_tapir';J=O/'joint';J.mkdir(exist_ok=True)
e=np.load(O/'evidence.npz');raw=np.load(O/'hands_mano.npz');K=e['K'];ids=e['frame_ids'];C=json.loads((B/'calibration.json').read_text());data={k:e[k] for k in e.files};H,W=480,640;yy,xx=np.indices((H,W))
for side in ['left','right']:
 masks=[];deps=[];cont=[]
 for j,i in enumerate(ids):
  m=cv2.imread(str(O/f'masks/{side}/{i:06d}.png'),0)>0;v=raw[side+'_vertices'][i];joints=raw[side+'_joints'][i]
  if e[side+'_valid'][j]:
   z=raster(v,raw[side+'_faces'],K,(H,W));near=cv2.dilate(np.isfinite(z).astype('uint8'),np.ones((51,51),np.uint8))>0
   # Keep every nearby component, including palm separated from fingers by an object.
   m &= near
   uv=joints@K.T;uv=uv[:,:2]/uv[:,2:];wrist=uv[0];palm=uv[[5,9,13,17]].mean(0);axis=palm-wrist;axis/=max(np.linalg.norm(axis),1)
   m &= ((xx-wrist[0])*axis[0]+(yy-wrist[1])*axis[1]>-8)
  else:m[:]=False
  for obj in ['pipette','tube']:m &= ~(cv2.imread(str(B/f'masks/{obj}/{i:06d}.png'),0)>0)
  depth=cv2.imread(str(B/f'depth/{i:06d}.png'),-1)*C['depth_scale_m'];good=m&(depth>.15)&(depth<1.5)
  small=cv2.resize(m.astype('uint8'),(160,120),interpolation=cv2.INTER_NEAREST);d=cv2.resize(np.where(good,depth,0).astype('float32'),(160,120),interpolation=cv2.INTER_NEAREST)
  contours,_=cv2.findContours(small,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);pts=np.concatenate(contours)[:,0] if contours else np.zeros((1,2));cont.append(pts[np.linspace(0,len(pts)-1,160,dtype=int)]);masks.append(small);deps.append(d)
 data[side+'_mask']=np.array(masks);data[side+'_depth']=np.array(deps);data[side+'_contour']=np.array(cont,np.float32)
# Actual measured depth, including objects; never reuse a rendered depth as an observation.
data['scene_depth']=np.array([cv2.resize(cv2.imread(str(B/f'depth/{i:06d}.png'),-1).astype('float32')*C['depth_scale_m'],(160,120),interpolation=cv2.INTER_NEAREST) for i in ids])
np.savez_compressed(J/'evidence.npz',**data)
m=trimesh.load(B/'tube/mesh_local.ply',process=True);m.merge_vertices(digits_vertex=6);parts=m.split(only_watertight=False);outer=next(x for x in parts if len(x.faces)==1152);inner=next(x for x in parts if len(x.faces)==720);v=np.vstack([outer.vertices,inner.vertices]);f=np.vstack([outer.faces,inner.faces+len(outer.vertices)]).tolist()
def ring(mesh,top,offset):
 edges,counts=np.unique(mesh.edges_sorted,axis=0,return_counts=True);ix=np.unique(edges[counts==1]);z=mesh.vertices[ix,2];ix=ix[np.isclose(z,z.max() if top else z.min(),atol=1e-5)];angles=np.arctan2(mesh.vertices[ix,1],mesh.vertices[ix,0]);return ix[np.argsort(angles)]+offset
outtop=ring(outer,True,0);intop=ring(inner,True,len(outer.vertices));assert len(outtop)==len(intop)==72
for k in range(72):
 n=(k+1)%72;f.extend([[outtop[k],outtop[n],intop[k]],[outtop[n],intop[n],intop[k]]])
for mesh,offset in [(outer,0),(inner,len(outer.vertices))]:
 ringids=ring(mesh,False,offset);center=len(v);v=np.vstack([v,v[ringids].mean(0)])
 for k in range(len(ringids)):f.append([ringids[k],ringids[(k+1)%len(ringids)],center])
shell=trimesh.Trimesh(v,np.array(f),process=True);shell.fix_normals();assert shell.is_watertight and shell.is_volume; shell.export(J/'tube_material_shell.ply')
# Add watertight rim/ridge components as separate solids. Their union may overlap.
solids=[shell]
for part in parts:
 if part.is_watertight and len(part.faces)>100:
  part.fix_normals();
  if part.is_volume:solids.append(part)
import open3d as o3d
lo=m.bounds[0]-.045;hi=m.bounds[1]+.045;res=144;z,y,x=np.meshgrid(np.linspace(lo[2],hi[2],res),np.linspace(lo[1],hi[1],res),np.linspace(lo[0],hi[0],res),indexing='ij');pts=np.stack([x,y,z],-1).astype('float32');signed=np.full(pts.shape[:-1],np.inf,np.float32)
for solid in solids:
 scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(solid.vertices,np.float32)),o3d.core.Tensor(np.asarray(solid.faces,np.uint32))));signed=np.minimum(signed,scene.compute_signed_distance(o3d.core.Tensor(pts)).numpy())
np.savez_compressed(J/'tube_fields.npz',sdf=signed,lo=lo,hi=hi)
(J/'preparation.json').write_text(json.dumps(dict(hand_masks='All nearby disconnected SAM regions retained; HaWoR projection associates hand and clips forearm, object masks subtracted.',tube_geometry='Original inner and outer wall surfaces joined at rim and capped separately at bottom; bore remains open; closed decorative rings unioned as material.',shell_watertight=shell.is_watertight,shell_volume=float(shell.volume),solids=len(solids),grid_resolution=res),indent=2))
print('Joint evidence and hollow tube collision field ready',flush=True)
