"""Joint RGB-D refinement: two MANO hands and two rigid known objects.
No grasp gate, contact vertices, attraction, or slip terms. Collision constraints
use an augmented-Lagrangian sampled-surface approximation, not a guarantee of
triangle-level or continuous-time nonintersection.
"""
import os,sys,json,time,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',default='6');p.add_argument('--iterations',type=int,default=600);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
import cv2,numpy as np,torch,trimesh
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRasterizer,RasterizationSettings
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.ops import knn_points
R=Path(__file__).resolve().parents[1];O=R/'known_scene/pipette_depth_03_hand_milestone';J=O/'joint';BAS=R/'known_scene/pipette_depth_03_tapir';os.chdir(R/'modules/HaWoR');sys.path.insert(0,str(R/'modules/HaWoR'))
from lib.models.mano_wrapper import MANO
D='cuda';torch.set_num_threads(4);torch.manual_seed(12);np.random.seed(12);archive=np.load(J/'evidence.npz');e={k:archive[k] for k in archive.files};raw=np.load(O/'hands_mano.npz');land=np.load(J/'rgb_landmarks.npz');ids=e['frame_ids'];N=len(ids);K=e['K'].copy();K[:2]*=.25;names=['left','right','pipette','tube'];entities={};params=[];history=[];start=time.time()
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=D)
def parameter(x):return torch.nn.Parameter(tensor(x).detach().clone())
def robust(x):return F.smooth_l1_loss(x,torch.zeros_like(x),beta=.5,reduction='none')
def project(v):return torch.stack([v[...,0]/v[...,2].clamp(min=.05)*K[0,0]+K[0,2],v[...,1]/v[...,2].clamp(min=.05)*K[1,1]+K[1,2]],-1)
def sample_map(im,uv):
 grid=uv.clone();grid[...,0]=grid[...,0]/159*2-1;grid[...,1]=grid[...,1]/119*2-1
 return F.grid_sample(im[:,None],grid[:,None],align_corners=True,padding_mode='border')[:,0,0]
for name in names:
 ent={};entities[name]=ent
 if name in ['left','right']:
  valid=e[name+'_valid'];ent['valid']=tensor(valid);ent['mask']=tensor(e[name+'_mask']);ent['faces']=torch.as_tensor(raw[name+'_faces'],device=D,dtype=torch.long);model=MANO(model_path=str(R/'modules/HaWoR/_DATA'/('data_left/mano_left' if name=='left' else 'data/mano')),is_rhand=name=='right',use_pca=False).to(D)
  if name=='left':model.shapedirs[:,0,:]*=-1
  for v in model.parameters():v.requires_grad_(False)
  ent['model']=model
  def clean(key):
   x=raw[name+'_'+key][ids].copy();x[~valid]=x[valid][0];return x
  ent['t0']=tensor(clean('trans'));ent['r0']=tensor(clean('rot'));ent['p0']=tensor(clean('hand_pose'));beta=np.median(clean('betas')[valid],axis=0);ent['beta']=tensor(np.repeat(beta[None],N,axis=0))
  ent['t']=parameter(ent['t0']);ent['r']=parameter(ent['r0']);ent['p']=parameter(ent['p0']);ent['p'].requires_grad_(False)
  assert ent['t'].data_ptr()!=ent['t0'].data_ptr() and ent['p'].data_ptr()!=ent['p0'].data_ptr()
  params.extend([{'params':[ent['t']],'lr':.0006},{'params':[ent['r']],'lr':.003},{'params':[ent['p']],'lr':.003}])
  ent['landmarks']=tensor(land[name+'_uv']*.25);lw=land[name+'_weight'].copy();maskdist=np.array([cv2.distanceTransform((1-m).astype('uint8'),cv2.DIST_L2,5) for m in e[name+'_mask']]);support=sample_map(tensor(maskdist),ent['landmarks']).detach();ent['landmark_weight']=tensor(lw)*(support<4)*ent['valid'][:,None]
 else:
  mesh=trimesh.load(BAS/name/'mesh_local.ply',process=False);ent['local']=tensor(mesh.vertices);ent['faces']=torch.as_tensor(mesh.faces,device=D,dtype=torch.long);ent['T0']=tensor(e[name+'_T']);ent['r']=parameter(np.zeros((N,3)));ent['t']=parameter(np.zeros((N,3)));ent['valid']=tensor(np.ones(N));ent['mask']=tensor(e[name+'_mask']);ent['reliability']=tensor(e[name+'_weight']);params.extend([{'params':[ent['t']],'lr':.00015},{'params':[ent['r']],'lr':.001}]);f=np.load(J/f'{name}_fields.npz');ent['sdf']=tensor(f['sdf'])[None,None];ent['lo']=tensor(f['lo']);ent['hi']=tensor(f['hi'])
 ent['dt']=tensor(np.array([cv2.distanceTransform((1-m).astype('uint8'),cv2.DIST_L2,5) for m in ent['mask'].cpu().numpy()])).clamp(max=200);contours=[];cloud=[];weights=[]
 for i in range(N):
  m=ent['mask'][i].cpu().numpy().astype('uint8');cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);pts=np.concatenate(cs)[:,0] if cs else np.zeros((1,2));contours.append(pts[np.linspace(0,len(pts)-1,120,dtype=int)])
  dep=e['scene_depth'][i];good=(m>0)&(dep>.15)&(dep<1.5)
  if name in ['pipette','tube']:good &= (e[name+'_depth'][i]>0)&(np.abs(dep-e[name+'_depth'][i])<.025)
  yy,xx=np.where(good)
  if len(xx):
   choose=np.linspace(0,len(xx)-1,160,dtype=int);xx=xx[choose];yy=yy[choose];z=dep[yy,xx];cloud.append(np.stack([(xx-K[0,2])/K[0,0]*z,(yy-K[1,2])/K[1,1]*z,z],-1));weights.append(1.)
  else:cloud.append(np.zeros((160,3)));weights.append(0.)
 ent['contour']=tensor(np.array(contours));ent['cloud']=tensor(np.array(cloud));ent['cloud_valid']=tensor(weights)*ent['valid']
cam=cameras_from_opencv_projection(torch.eye(3,device=D)[None].expand(N,-1,-1),torch.zeros(N,3,device=D),tensor(K)[None].expand(N,-1,-1),tensor([[120,160]]).expand(N,-1))
raster=MeshRasterizer(cameras=cam,raster_settings=RasterizationSettings(image_size=(120,160),blur_radius=0,faces_per_pixel=1,bin_size=16,max_faces_per_bin=16000))
scene_depth=tensor(e['scene_depth']);face_labels=[];allfaces=[];offset=0
for k,name in enumerate(names):
 ent=entities[name];allfaces.append(ent['faces']+offset);face_labels.extend([k]*len(ent['faces']));offset+=778 if name in ['left','right'] else len(ent['local'])
faces=torch.cat(allfaces)[None].expand(N,-1,-1);labels=torch.as_tensor(face_labels,device=D);nf=len(face_labels)
def geometry():
 vertices={};joints={};transforms={}
 for name,ent in entities.items():
  if name in ['left','right']:
   out=ent['model'](global_orient=axis_angle_to_matrix(ent['r'])[:,None],hand_pose=axis_angle_to_matrix(ent['p'].reshape(-1,3)).reshape(N,15,3,3),betas=ent['beta'],transl=ent['t'],pose2rot=False);vertices[name]=out.vertices;joints[name]=out.joints
  else:
   rot=axis_angle_to_matrix(ent['r'])@ent['T0'][:,:3,:3];t=ent['T0'][:,:3,3]+ent['t'];vertices[name]=ent['local'][None]@rot.transpose(1,2)+t[:,None];transforms[name]=(rot,t)
 return vertices,joints,transforms
def render(vertices):
 vv=[torch.where(entities[n]['valid'][:,None,None]>0,vertices[n],torch.full_like(vertices[n],100)) for n in names];fr=raster(Meshes(verts=torch.cat(vv,1),faces=faces));z=fr.zbuf[...,0];fi=fr.pix_to_face[...,0];owner=labels[fi.clamp(min=0)%nf];owner=torch.where(fi>=0,owner,-torch.ones_like(owner));return z,owner

def sdf_query(v,obj,transforms):
 ent=entities[obj];rot,t=transforms[obj];local=(v-t[:,None])@rot;g=(local-ent['lo'])/(ent['hi']-ent['lo'])*2-1
 values=F.grid_sample(ent['sdf'],g.reshape(1,1,N,-1,3),align_corners=True,padding_mode='border').reshape(N,-1)
 # Outside the grid is free space; fields enclose the entire object with margin.
 return torch.where((g.abs()<=1).all(-1),values,torch.full_like(values,.05))
def surface_samples(v,faces):return torch.cat([v,v[:,faces].mean(2)],1)
def hand_normals(v,faces):
 tri=v[:,faces];n=torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1);out=torch.zeros_like(v)
 for k in range(3):out.scatter_add_(1,faces[None,:,k,None].expand(N,-1,3),n)
 return F.normalize(out,dim=-1)
def constraints(vertices,transforms):
 penalties=[];raws={}
 for side in ['left','right']:
  samples=surface_samples(vertices[side],entities[side]['faces'])
  for obj in ['pipette','tube']:
   sd=sdf_query(samples,obj,transforms);violation=F.relu(-sd-.00025)*entities[side]['valid'][:,None];raws[side+'_'+obj]=violation;penalties.append(violation.topk(16,dim=1).values.mean(1)/.005)
 # Both object-to-object directions; hollow tube field preserves insertion.
 for src,dst in [('pipette','tube'),('tube','pipette')]:
  v=vertices[src][:,::3];violation=F.relu(-sdf_query(v,dst,transforms)-.00025);raws[src+'_'+dst]=violation;penalties.append(violation.topk(16,dim=1).values.mean(1)/.005)
 # Moving hand-hand surfaces: nearest outward normal signed distance surrogate.
 for src,dst in [('left','right'),('right','left')]:
  v=vertices[src];target=vertices[dst];norm=hand_normals(target,entities[dst]['faces']);nn=knn_points(v,target,K=1);idx=nn.idx[...,0];nearest=target.gather(1,idx[...,None].expand(-1,-1,3));normal=norm.gather(1,idx[...,None].expand(-1,-1,3));signed=((v-nearest)*normal).sum(-1);violation=F.relu(-signed-.00025)*(nn.dists[...,0]<.02**2)*entities[src]['valid'][:,None]*entities[dst]['valid'][:,None];raws[src+'_'+dst]=violation;penalties.append(violation.topk(16,dim=1).values.mean(1)/.005)
 return torch.stack(penalties,1),raws

def evaluate(vertices,transforms):
 with torch.no_grad():
  z,owner=render(vertices);_,viol=constraints(vertices,transforms);result={}
  for k,name in enumerate(names):
   ent=entities[name];ms=[];de=[];coverage=[]
   for i in range(N):
    if not ent['valid'][i]:ms.append(None);de.append(None);coverage.append(None);continue
    pred=owner[i]==k;truth=ent['mask'][i]>.5;intersection=pred&truth;ms.append(float(intersection.sum()/(pred|truth).sum().clamp(min=1)));coverage.append(float(intersection.sum()/truth.sum().clamp(min=1)));good=intersection&(scene_depth[i]>.15)&(scene_depth[i]<1.5)
    if name in ['pipette','tube']:good &= (scene_depth[i]-tensor(e[name+'_depth'][i])).abs()<.025
    de.append(float((z[i][good]-scene_depth[i][good]).abs().median()*1000) if good.any() else None)
   result[name]=dict(mask_iou=ms,depth_median_mm=de,mask_coverage=coverage)
  result['collision']={key:dict(max_violation_mm=(v.max(1).values*1000).cpu().tolist(),violating_samples=(v>.00025).sum(1).cpu().tolist()) for key,v in viol.items()}
  return result
# Evaluate the true original geometry against corrected evidence.
with torch.no_grad():
 vv,jj,tt=geometry();original={n:(tensor(np.nan_to_num(raw[n+'_vertices'][ids])) if n in ['left','right'] else vv[n]) for n in names};baseline=evaluate(original,tt)
 print('Baseline collisions', {k: float(np.max(v['max_violation_mm'])) for k,v in baseline['collision'].items()}, flush=True)
 # Conservative depth seed while preserving image-space wrist projection.
 z,owner=render(vv)
 for k,s in enumerate(['left','right']):
  ent=entities[s]
  for i in range(N):
   good=(owner[i]==k)&(ent['mask'][i]>.5)&(scene_depth[i]>.15)&(scene_depth[i]<1.5)
   if good.sum()>15:
    dz=(scene_depth[i][good]-z[i][good]).median().clamp(-.06,.06);old=ent['t'][i,2].clone();ent['t'][i,:2]*=(old+dz)/old;ent['t'][i,2]+=dz
optimizer=torch.optim.Adam(params);dual=torch.zeros((N,8),device=D);rho=.5
for it in range(a.iterations):
 if it==150:
  for s in ['left','right']:entities[s]['p'].requires_grad_(True)
 if it in [300,450]:
  rho*=2
  for group in optimizer.param_groups:group['lr']*=.6
 optimizer.zero_grad();verts,joints,transforms=geometry();z,owner=render(verts);scene_render=torch.where(z>0,z,torch.full_like(z,10));total=z.sum()*0;terms={}
 for k,name in enumerate(names):
  ent=entities[name];v=verts[name];uv=project(v);vdepth=sample_map(scene_render.detach(),uv);visible=(v[:,:,2]<=vdepth+.004).detach();valid=ent['valid'];outside=sample_map(ent['dt'],uv)
  silhouette=(robust(outside/2)*visible*valid[:,None]).sum()/(visible*valid[:,None]).sum().clamp(min=1)
  # Mask-to-model term cannot improve by shrinking the overlap region.
  dist=torch.cdist(ent['contour'],uv);dist=dist+(~visible)[:,None]*200;coverage=(robust(dist.min(2).values/2).mean(1)*valid).sum()/valid.sum().clamp(min=1)
  hit=(owner==k)&(ent['mask']>.5)&(scene_depth>.15)&(scene_depth<1.5)
  if name in ['pipette','tube']:hit &= (scene_depth-tensor(e[name+'_depth'])).abs()<.025
  dep=(robust((z-scene_depth)/.01)*hit).sum()/hit.sum().clamp(min=1)
  dd=torch.cdist(ent['cloud'],v);dd=dd+(~visible)[:,None]*2;cloud=(robust(dd.min(2).values/.01).mean(1)*ent['cloud_valid']).sum()/ent['cloud_valid'].sum().clamp(min=1)
  loss=2*dep+1.5*silhouette+1.5*coverage+.4*cloud
  if name in ['left','right']:
   keypoint=robust((project(joints[name])-ent['landmarks'])/3).sum(-1);kw=ent['landmark_weight'];kp=(keypoint*kw).sum()/kw.sum().clamp(min=1);loss+=2*kp
   loss+=.8*((ent['p']-ent['p0'])/.3).square().mean()+.15*((ent['r']-ent['r0'])/.4).square().mean()+.05*((ent['t']-ent['t0'])/.08).square().mean()
   good=(valid[2:]*valid[1:-1]*valid[:-2]);temporal=robust((joints[name][2:]-2*joints[name][1:-1]+joints[name][:-2])/.004).mean((1,2));loss+=.2*(temporal*good).sum()/good.sum().clamp(min=1)
  else:
   weight=.25+ent['reliability'];loss+=2*(weight*((ent['t']/.008).square().mean(1)+(ent['r']/.10).square().mean(1))).mean();loss+=.2*robust((ent['t'][2:]-2*ent['t'][1:-1]+ent['t'][:-2])/.003).mean()
  total+=loss;terms[name]=dict(depth=float(dep.detach()),silhouette=float(silhouette.detach()),coverage=float(coverage.detach()))
 g,viols=constraints(verts,transforms);collision=(dual*g+.5*rho*g.square()).mean();total+=collision*4
 if not torch.isfinite(total):
  print('NONFINITE',it,terms,'collision',collision.detach().cpu(),'g',g.detach().cpu().numpy(),flush=True);raise RuntimeError('Nonfinite objective')
 total.backward();torch.nn.utils.clip_grad_norm_([x for group in params for x in group['params']],20);optimizer.step()
 with torch.no_grad():
  for name,ent in entities.items():
   if name in ['left','right']:
    delta=(ent['p']-ent['p0']).reshape(N,15,3);delta*=.7/delta.norm(dim=-1,keepdim=True).clamp(min=.7);ent['p'].copy_(ent['p0']+delta.reshape(N,45));ent['t'][:,2].clamp_(.15,1.5)
   else:
    for key,cap in [('t',.015),('r',.15)]:ent[key].mul_(cap/ent[key].norm(dim=-1,keepdim=True).clamp(min=cap))
  if (it+1)%50==0:dual.add_(rho*g.detach()).clamp_(max=40)
 if it%50==0 or it==a.iterations-1:
  row=dict(iteration=it,loss=float(total.detach()),collision=float(collision.detach()),max_sample_violation_mm=float(torch.stack([v.max() for v in viols.values()]).max().detach()*1000),terms=terms);history.append(row);print(json.dumps(row),flush=True)
 if it==149:
  with torch.no_grad():v,j,t=geometry();np.savez_compressed(J/'rigid_stage.npz',**{n+'_vertices':x.cpu().numpy() for n,x in v.items()})
with torch.no_grad():
 vertices,joints,transforms=geometry();final=evaluate(vertices,transforms);output=dict(frame_ids=ids)
 for s in ['left','right']:
  ent=entities[s]
  for key,value in [('vertices',vertices[s]),('joints',joints[s]),('trans',ent['t']),('rot',ent['r']),('pose',ent['p']),('betas',ent['beta'])]:output[s+'_'+key]=value.cpu().numpy()
  output[s+'_valid']=e[s+'_valid'];output[s+'_faces']=raw[s+'_faces']
 for obj in ['pipette','tube']:
  rot,t=transforms[obj];T=np.repeat(np.eye(4)[None],N,axis=0);T[:,:3,:3]=rot.cpu().numpy();T[:,:3,3]=t.cpu().numpy();output[obj+'_T']=T;output[obj+'_vertices']=vertices[obj].cpu().numpy()
  final[obj]['translation_correction_mm']=(entities[obj]['t'].norm(dim=-1)*1000).cpu().tolist();final[obj]['rotation_correction_deg']=(entities[obj]['r'].norm(dim=-1)*180/np.pi).cpu().tolist()
 np.savez_compressed(J/'joint_fit.npz',**output)
report=dict(baseline=baseline,final=final,history=history,settings=dict(iterations=a.iterations,elapsed_s=time.time()-start,frames=N,contact_attraction=False,contact_persistence=False,manual_grasp_gates=False,joint_variables='Both wrists, both finger poses after iteration 150, both object poses; fixed shared hand shape and camera',collision='Augmented Lagrangian: both hands against both objects, both object-object directions, approximate signed nearest-surface hand-hand; vertices and face centroids for hand-object. 0.25mm tolerance. No exact triangle or continuous-time guarantee; hand self-collision not constrained.',depth='Shared scene z-buffer and entity ownership; valid measured RGB-D pixels, robust depth and mask-to-surface coverage',landmarks='Independent WiLoR RGB predictions; HaWoR-associated crops; geometric confidence gating',tube='Hollow material shell assembled from original walls, preserves opening',object_bounds='15mm translation and 0.15 rad rotation correction'))
(J/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False));print('DONE joint refinement',flush=True)
