"""Prepare measured hand evidence, explicit object reliability, and collision fields."""
import json,sys
from pathlib import Path
import cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from scipy.ndimage import median_filter
from rgbd_known_geometry import raster
from rgbd_tapir_motion import pose_from_affine,select_motion
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'known_scene/pipette_depth_03_hand_milestone'
def main():
 config=json.loads((OUT/'milestone_config.json').read_text());base=Path(config['source_run']);C=json.loads((base/'calibration.json').read_text());K=np.array(C['K']);N=C['frames'];ids=np.arange(config['start'],config['stop'],config['stride']);hands=np.load(base/'hands_mano.npz');display=np.load(base/'hands_visible.npz');tr=np.load(base/'tapir/tracks.npz');ref=json.loads((base/'segmentation_prompts.json').read_text())['frame'];rng=np.random.default_rng(12)
 reliability={};assets={};SZ=(160,120)
 for obj,index in [('pipette',0),('tube',1)]:
  mesh=trimesh.load(base/obj/'mesh_local.ply',process=False);T=np.load(base/obj/'poses.npz')['T_camera_object'];quality=json.loads((base/obj/'quality.json').read_text())['quality'];rows=[];meshdepth=[];masks=[]
  for i in range(N):
   m=cv2.imread(str(base/f'masks/{obj}/{i:06d}.png'),0)>0;d=cv2.imread(str(base/f'depth/{i:06d}.png'),-1)*C['depth_scale_m'];md=raster(mesh.vertices@T[i,:3,:3].T+T[i,:3,3],mesh.faces,K,(C['height'],C['width']));support=m&(d>.15)&(d<2)&np.isfinite(md)&(np.abs(d-md)<.015);coverage=support.sum()/max(1,m.sum());err=float(np.median(np.abs(d[support]-md[support]))) if support.any() else .1
   qI=float(np.clip((quality[i]['iou']-.4)/.4,0,1));qD=float(coverage*np.exp(-.5*(err/.01)**2));qT=1.;qP=1.;innovation_mm=0.;innovation_deg=0.
   if i!=ref:
    prev=i-1 if i>ref else i+1;j=min(i,prev);old=cv2.imread(str(base/f'masks/{obj}/{prev:06d}.png'),0)>0;points=tr['tracks'][j,index];confidence=tr['confidence'][j,index];A,a,b=select_motion(points,confidence,old,m,i<prev);qT=0.
    if A is not None:
     source=points[:,1 if i<prev else 0];keep=np.array([np.any(np.linalg.norm(a-v,axis=1)<1e-3) for v in source]);qT=float(len(a)/len(points)*confidence[keep].min(1).mean())
    motion=quality[i].get('motion_prediction',{});pred=T[prev].copy()
    if motion.get('status')=='tapir':
     x=pose_from_affine(np.r_[Rotation.from_matrix(pred[:3,:3]).as_rotvec(),pred[:3,3]],np.array(motion['affine']),K,motion['depth_delta_m']);pred[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix();pred[:3,3]=x[3:]
    delta=np.linalg.norm(T[i,:3,3]-pred[:3,3]);angle=np.arccos(np.clip(T[i,:3,2]@pred[:3,2],-1,1)) if obj=='tube' else (Rotation.from_matrix(pred[:3,:3]).inv()*Rotation.from_matrix(T[i,:3,:3])).magnitude();qP=float(np.exp(-.5*(delta/.02)**2-.5*(angle/np.deg2rad(20))**2));innovation_mm=float(delta*1000);innovation_deg=float(np.degrees(angle))
   raw=float(config['object_weights'][obj]*qI*qT*qP*np.sqrt(qD));rows.append(dict(frame=i,q_I=qI,q_T=qT,q_P=qP,q_D=qD,depth_coverage=float(coverage),depth_residual_m=err,innovation_mm=innovation_mm,innovation_deg=innovation_deg,raw=raw))
   if i in ids:meshdepth.append(cv2.resize(np.where(np.isfinite(md),md,0),(160,120),interpolation=cv2.INTER_NEAREST));masks.append(cv2.resize(m.astype(np.uint8),(160,120),interpolation=cv2.INTER_NEAREST))
  raw=np.array([r['raw'] for r in rows]);r=np.minimum(raw,median_filter(raw,size=5,mode='nearest'));r[r<.1]=0
  for i,row in enumerate(rows):row['weight']=float(r[i])
  reliability[obj]=rows;assets[obj]=dict(T=T[ids],depth=np.array(meshdepth),mask=np.array(masks),weight=r[ids])
  print(obj,'active reliability frames',int((r>0).sum()),'median',float(np.median(r)),flush=True)
 (OUT/'object_reliability.json').write_text(json.dumps(dict(formula='r=c_object*q_I*q_T*q_P*sqrt(q_D); min with centered five-frame median; disable below 0.1',thresholds=dict(iou=[.4,.8],innovation_translation_m=.02,innovation_rotation_deg=20,depth_consistency_m=.015,depth_residual_scale_m=.01),calibrated_probability=False,rows=reliability),indent=2))
 data=dict(frame_ids=ids,K=K,object_depth=np.minimum(np.where(assets['pipette']['depth']>0,assets['pipette']['depth'],10),np.where(assets['tube']['depth']>0,assets['tube']['depth'],10)),object_mask=assets['pipette']['mask']|assets['tube']['mask'])
 for obj in assets:
  for key,val in assets[obj].items():data[obj+'_'+key]=val
 for side,obj in [('left','tube'),('right','pipette')]:
  masks=[];depths=[];contours=[];valid=[];grasp=[]
  for i in ids:
   m=cv2.imread(str(OUT/f'masks/{side}/{i:06d}.png'),0)>0
   for o in ['pipette','tube']:m&=~(cv2.imread(str(base/f'masks/{o}/{i:06d}.png'),0)>0)
   finite=np.isfinite(hands[side+'_joints'][i]).all();observed=bool(display[side+'_visible'][i]);j=hands[side+'_joints'][i]
   if finite and (j[:,2]>.1).all():
    uv=j@K.T;uv=uv[:,:2]/uv[:,2:];wrist=uv[0];palm=uv[[5,9,13,17]].mean(0)
    # SAM may merge the two hands. Keep the component associated with this
    # hand's projected palm, then remove the forearm behind its wrist plane.
    count,labels,stats,centers=cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if count>1:
     candidates=np.arange(1,count);score=np.linalg.norm(centers[1:]-palm,axis=1);score[stats[1:,cv2.CC_STAT_AREA]<100]+=1000;m=labels==candidates[np.argmin(score)]
    axis=palm-wrist;axis/=max(np.linalg.norm(axis),1)
    yy,xx=np.indices(m.shape);m&=((xx-wrist[0])*axis[0]+(yy-wrist[1])*axis[1]>-8)
   dep=cv2.imread(str(base/f'depth/{i:06d}.png'),-1)*C['depth_scale_m'];dm=m&(dep>.15)&(dep<1.5)
   if dm.sum()>20:
    anchor=np.quantile(dep[dm],.2);dm&=(dep>anchor-.06)&(dep<anchor+.15)
   small=cv2.resize(m.astype(np.uint8),SZ,interpolation=cv2.INTER_NEAREST);depth=cv2.resize(np.where(dm,dep,0).astype(np.float32),SZ,interpolation=cv2.INTER_NEAREST);cont,_=cv2.findContours(small,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);pts=np.concatenate(cont)[:,0,:] if cont else np.zeros((1,2));sel=np.linspace(0,len(pts)-1,120,dtype=int);contours.append(pts[sel]);masks.append(small);depths.append(depth);valid.append(finite and observed and small.sum()>40 and (depth>0).sum()>20)
   gate=0.
   for lo,hi in config['grasp_intervals'][side]:gate=max(gate,min(np.clip((i-lo)/10,0,1),np.clip((hi-i)/10,0,1)))
   grasp.append(gate)
  data[side+'_mask']=np.array(masks);data[side+'_depth']=np.array(depths);data[side+'_contour']=np.array(contours,np.float32);data[side+'_valid']=np.array(valid);data[side+'_grasp']=np.array(grasp,np.float32)
  print(side,'fittable',int(sum(valid)),'/',len(ids),flush=True)
 np.savez_compressed(OUT/'evidence.npz',**data)
 # Signed collision fields require closed geometry. Use actual pipette and an
 # explicitly conservative convex envelope for hand/tube collisions only.
 import open3d as o3d
 for obj in ['pipette','tube']:
  mesh=trimesh.load(base/obj/'mesh_local.ply',process=False);proxy=mesh if mesh.is_volume else mesh.convex_hull;proxy.export(OUT/f'{obj}_collision_proxy.ply');lo=mesh.bounds[0]-.07;hi=mesh.bounds[1]+.07;resolution=96;z,y,x=np.meshgrid(np.linspace(lo[2],hi[2],resolution),np.linspace(lo[1],hi[1],resolution),np.linspace(lo[0],hi[0],resolution),indexing='ij');pts=np.stack([x,y,z],-1).astype(np.float32)
  scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(proxy.vertices,np.float32)),o3d.core.Tensor(np.asarray(proxy.faces,np.uint32))));sdf=scene.compute_signed_distance(o3d.core.Tensor(pts)).numpy()
  surface=o3d.t.geometry.RaycastingScene();surface.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(mesh.vertices,np.float32)),o3d.core.Tensor(np.asarray(mesh.faces,np.uint32))));distance=surface.compute_distance(o3d.core.Tensor(pts)).numpy();np.savez_compressed(OUT/f'{obj}_fields.npz',sdf=sdf,distance=distance,lo=lo,hi=hi,source_watertight=mesh.is_watertight,proxy_type='original' if mesh.is_volume else 'conservative_convex_envelope')
 print('Prepared evidence and distance fields',flush=True)
if __name__=='__main__':main()
