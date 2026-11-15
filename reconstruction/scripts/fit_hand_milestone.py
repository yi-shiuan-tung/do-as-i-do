"""Four-way MANO hand fitting experiment against recorded RGB-D and fixed objects."""
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--side',choices=['left','right'],required=True);p.add_argument('--gpu',default='6');a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'known_scene/pipette_depth_03_hand_milestone';os.chdir(ROOT/'modules/HaWoR');sys.path.insert(0,str(ROOT/'modules/HaWoR'))
import cv2,numpy as np,torch
import torch.nn.functional as F
from lib.models.mano_wrapper import MANO
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRasterizer,RasterizationSettings
from pytorch3d.utils import cameras_from_opencv_projection
D=torch.device('cuda');torch.set_num_threads(4);torch.manual_seed(0);side=a.side;obj='tube' if side=='left' else 'pipette'
e=np.load(OUT/'evidence.npz');raw=np.load(OUT/'hands_mano.npz');allids=e['frame_ids'];valid=e[side+'_valid'];ix=np.where(valid)[0];ids=allids[ix];B=len(ids);K=e['K'].copy();K[:2]*=.25
if B<10:raise RuntimeError('Insufficient valid hand observations')
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=D)
model=MANO(model_path=str(ROOT/'modules/HaWoR/_DATA'/('data_left/mano_left' if side=='left' else 'data/mano')),is_rhand=side=='right',use_pca=False).to(D)
if side=='left':model.shapedirs[:,0,:]*=-1
faces=torch.as_tensor(raw[side+'_faces'],dtype=torch.long,device=D)[None].expand(B,-1,-1)
base_trans=tensor(raw[side+'_trans'][ids]);base_rot=tensor(raw[side+'_rot'][ids]);base_pose=tensor(raw[side+'_hand_pose'][ids]);base_beta=tensor(raw[side+'_betas'][ids]);shared_beta=base_beta.median(0).values[None].expand(B,-1)
def mano(t,r,p,b=shared_beta):return model(global_orient=axis_angle_to_matrix(r)[:,None],hand_pose=axis_angle_to_matrix(p.reshape(-1,3)).reshape(B,15,3,3),betas=b,transl=t,pose2rot=False)
with torch.no_grad():
 rebuilt=mano(base_trans,base_rot,base_pose,base_beta).vertices;roundtrip=float((rebuilt-tensor(raw[side+'_vertices'][ids])).abs().max());assert roundtrip<1e-5
cam=cameras_from_opencv_projection(torch.eye(3,device=D)[None].expand(B,-1,-1),torch.zeros(B,3,device=D),tensor(K)[None].expand(B,-1,-1),tensor([[120,160]]).expand(B,-1))
raster=MeshRasterizer(cameras=cam,raster_settings=RasterizationSettings(image_size=(120,160),blur_radius=0,faces_per_pixel=1,bin_size=16,max_faces_per_bin=2000))
mask=tensor(e[side+'_mask'][ix]);depth=tensor(e[side+'_depth'][ix]);contour=tensor(e[side+'_contour'][ix]);obj_depth=tensor(e['object_depth'][ix]);obj_mask=tensor(e['object_mask'][ix]);objT=tensor(e[obj+'_T'][ix]);reliability=tensor(e[obj+'_weight'][ix]);grasp=tensor(e[side+'_grasp'][ix]);contact_weight=reliability*grasp
# Distance maps permit hand geometry behind either object but not in background.
dt=tensor(np.stack([cv2.distanceTransform((1-np.maximum(m,o)).astype(np.uint8),cv2.DIST_L2,5) for m,o in zip(e[side+'_mask'][ix],e['object_mask'][ix])]))
fields=np.load(OUT/f'{obj}_fields.npz');sdf=tensor(fields['sdf'])[None,None];distance_field=tensor(fields['distance'])[None,None];lo=tensor(fields['lo']);hi=tensor(fields['hi'])
def local_vertices(v):return torch.bmm(v-objT[:,None,:3,3],objT[:,:3,:3])
def field(v,values):
 xyz=local_vertices(v);g=(xyz-lo)/(hi-lo)*2-1
 # Each frame samples the same local object field; one batch avoids field copies.
 return F.grid_sample(values,g.reshape(1,1,B,-1,3),mode='bilinear',padding_mode='border',align_corners=True).reshape(B,-1)
def project(v):return torch.stack([v[:,:,0]/v[:,:,2].clamp(min=.05)*K[0,0]+K[0,2],v[:,:,1]/v[:,:,2].clamp(min=.05)*K[1,1]+K[1,2]],-1)
def robust(x):return F.smooth_l1_loss(x,torch.zeros_like(x),beta=.5,reduction='none')
triple=torch.tensor((np.diff(ids)[:-1]==3)&(np.diff(ids)[1:]==3),device=D)
pair=torch.tensor(np.diff(ids)==3,device=D)
contact_ids=None
logs={};outputs={};started=time.time()
def metrics(v,joints):
 with torch.no_grad():
  z=raster(Meshes(verts=v,faces=faces)).zbuf[...,0];present=z>0;intersection=present&(mask>.5)&(depth>.15);errors=(z-depth).abs();med=[];iou=[]
  for i in range(B):
   vals=errors[i][intersection[i]];med.append(float(vals.median()*1000) if len(vals) else None)
   occ=(obj_mask[i]>.5)&(z[i]>obj_depth[i]-.003)&(mask[i]<.5);visible=present[i]&~occ;truth=mask[i]>.5;iou.append(float((visible&truth).sum()/((visible|truth).sum().clamp(min=1))))
  s=field(v,sdf);d=field(v,distance_field);return dict(depth_median_mm=med,mask_iou=iou,proxy_penetrating_vertices=(s<-.002).sum(1).cpu().tolist(),proxy_max_penetration_mm=((-s).clamp(min=0).max(1).values*1000).cpu().tolist(),nearest_surface_gap_mm=(d.min(1).values*1000).cpu().tolist())
def store(name,t,r,p,b=shared_beta):
 with torch.no_grad():
  out=mano(t,r,p,b);v=out.vertices;j=out.joints;V=np.zeros((len(allids),778,3),np.float32);J=np.zeros((len(allids),21,3),np.float32);V[ix]=v.cpu().numpy();J[ix]=j.cpu().numpy();outputs[name]=dict(vertices=V,joints=J,trans=t.detach().cpu().numpy(),root=r.detach().cpu().numpy(),pose=p.detach().cpu().numpy(),betas=b.detach().cpu().numpy());logs[name]=metrics(v,j)
  return out
store('original',base_trans,base_rot,base_pose,base_beta)
t=base_trans.clone();r=base_rot.clone();pose=base_pose.clone()
# Initialize global depth translation from measured hand pixels. XY follows the
# same viewing ray; optimization then solves alignment against all observations.
with torch.no_grad():
 v0=mano(t,r,pose).vertices;z0=raster(Meshes(verts=v0,faces=faces)).zbuf[...,0]
 for i in range(B):
  good=(z0[i]>.1)&(depth[i]>.1)&(mask[i]>.5)
  if good.sum()>20:
   dz=(depth[i][good]-z0[i][good]).median().clamp(-.15,.15);ratio=(t[i,2]+dz)/t[i,2];t[i,:2]*=ratio;t[i,2]+=dz
for name,iterations in [('global',100),('articulated',150),('contact',150)]:
 if name=='contact':
  with torch.no_grad():
   v=mano(t,r,pose).vertices;gaps=field(v,distance_field);weights=model.lbs_weights;groups=[list(range(1,4)),list(range(4,7)),list(range(7,10)),list(range(10,13)),list(range(13,16))];contact_ids=[]
   active=contact_weight>.1
   for k,group in enumerate(groups):
    if side=='right' and k==4:continue # thumb/plunger articulation is not a fixed grasp contact
    candidates=torch.where(weights[:,group].sum(1)>.45)[0]
    if active.sum()<5:continue
    g=gaps[active][:,candidates];score=g.median(0).values;order=torch.argsort(score)
    chosen=[int(candidates[x]) for x in order[:8] if score[x]<.025];contact_ids.extend(chosen)
   (OUT/f'{side}_contact_candidates.json').write_text(json.dumps(dict(vertex_ids=contact_ids,method='Up to eight nearest persistent vertices per finger from articulated fit; median gap <25mm; right thumb excluded; hypotheses not measured contact',active_frames=int(active.sum())),indent=2))
 t=torch.nn.Parameter(t.detach());r=torch.nn.Parameter(r.detach());pose=torch.nn.Parameter(pose.detach(),requires_grad=name!='global')
 params=[{'params':[t],'lr':.0015},{'params':[r],'lr':.006}]
 if name!='global':params.append({'params':[pose],'lr':.006})
 optimizer=torch.optim.Adam(params);history=[]
 for it in range(iterations):
  optimizer.zero_grad();out=mano(t,r,pose);v=out.vertices;j=out.joints;uv=project(v);grid=uv.clone();grid[:,:,0]=grid[:,:,0]/159*2-1;grid[:,:,1]=grid[:,:,1]/119*2-1;outside=F.grid_sample(dt[:,None],grid[:,None],align_corners=True,padding_mode='border')[:,0,0]
  silhouette=robust(outside/2).mean();coverage=robust(torch.cdist(contour,uv).min(2).values/2).mean()
  z=raster(Meshes(verts=v,faces=faces)).zbuf[...,0];hit=(z>0)&(mask>.5)&(depth>.15);dep=(robust((z-depth)/.01)*hit).sum()/hit.sum().clamp(min=1)
  anatomical=((pose-base_pose)/.3).square().mean();rootprior=((r-base_rot)/.4).square().mean();transprior=((t-base_trans)/.10).square().mean()
  temporal=robust((j[2:]-2*j[1:-1]+j[:-2])/.003)[triple].mean() if triple.any() else j.sum()*0
  loss=4*dep+.5*silhouette+.5*coverage+.6*anatomical+.1*rootprior+.03*transprior+.2*temporal
  collision=v.sum()*0;contact=v.sum()*0;slip=v.sum()*0
  if name=='contact':
   sd=field(v,sdf);collision=((F.relu(-sd-.002)/.004).square().mean(1)*reliability).mean();loss+=3*collision
   if contact_ids:
    d=field(v,distance_field)[:,contact_ids];contact=(robust(d/.005).mean(1)*contact_weight).mean();loss+=2*contact
    local=local_vertices(v)[:,contact_ids]
    if side=='left':local=torch.stack([local[:,:,2],torch.linalg.vector_norm(local[:,:,:2],dim=-1)],-1)
    stable=((ids[:-1]>=420)&(ids[1:]<=610));sw=torch.minimum(contact_weight[:-1],contact_weight[1:])*tensor(stable)*pair
    slip=(robust((local[1:]-local[:-1])/.005).mean((1,2))*sw).mean();loss+=.2*slip
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite fitting objective')
  loss.backward();torch.nn.utils.clip_grad_norm_([t,r,pose],10);optimizer.step()
  with torch.no_grad():
   delta=pose-base_pose;norm=delta.reshape(B,15,3).norm(dim=-1,keepdim=True);delta=(delta.reshape(B,15,3)*(.6/norm.clamp(min=.6))).reshape(B,45);pose.copy_(base_pose+delta)
   t[:,2].clamp_(.15,1.5)
  if it%25==0 or it==iterations-1:
   row=dict(iteration=it,loss=float(loss.detach()),depth=float(dep.detach()),silhouette=float(silhouette.detach()),coverage=float(coverage.detach()),collision=float(collision.detach()),contact=float(contact.detach()),slip=float(slip.detach()));history.append(row);print(side,name,row,flush=True)
 logs[name+'_history']=history;store(name,t,r,pose)
arrays=dict(frame_ids=allids,valid=valid,fit_indices=ix,faces=raw[side+'_faces'],reliability=e[obj+'_weight'],grasp=e[side+'_grasp'])
for name,record in outputs.items():
 for key,value in record.items():arrays[name+'_'+key]=value
np.savez_compressed(OUT/f'{side}_variants.npz',**arrays)
logs['settings']=dict(side=side,frames=len(allids),fit_frames=B,mano_roundtrip_max_error_m=roundtrip,shape='Original keeps HaWoR per-frame shape. Global/articulated/contact share a fixed per-hand median shape.',joint_constraint='MANO parameterization; initializer pose prior and max 0.6-rad per-joint correction; not a learned anatomical prior',optimizer='Adam; entire sampled sequence jointly with temporal second differences; independent left/right fits',collision_proxy=str(fields['proxy_type']),elapsed_s=time.time()-started)
(OUT/f'{side}_metrics.json').write_text(json.dumps(logs,indent=2,allow_nan=False));print('DONE',side,logs['settings'],flush=True)
