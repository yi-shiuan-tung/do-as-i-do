"""TAPIR-guided framewise prediction and joint RGB-D correction of hands/objects.
Soft bidirectional collision penalties; no contact or window-optimization terms.
"""
import os,sys,json,time,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',default='6');p.add_argument('--iterations',type=int,default=35);p.add_argument('--collision-weight',type=float,default=8.);p.add_argument('--predict-iterations',type=int,default=12);p.add_argument('--limit',type=int);p.add_argument('--reference-frame',type=int,default=549);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
import cv2,numpy as np,torch,trimesh
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRasterizer,RasterizationSettings
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.ops import knn_points
R=Path(__file__).resolve().parents[1];O=R/'known_scene/pipette_depth_03_hand_milestone';J=O/'joint_tapir';SOURCE=O/'joint';J.mkdir(exist_ok=True);BAS=R/'known_scene/pipette_depth_03_tapir';os.chdir(R/'modules/HaWoR');sys.path.insert(0,str(R/'modules/HaWoR'))
from lib.models.mano_wrapper import MANO
sys.path.insert(0,str(R/'scripts'))
from mesh_collision import signed_mesh_distance
from scipy.spatial.transform import Rotation
D='cuda';torch.set_num_threads(4);torch.manual_seed(12);np.random.seed(12);archive=np.load(SOURCE/'evidence.npz');e={k:archive[k] for k in archive.files};raw=np.load(O/'hands_mano.npz');land=np.load(SOURCE/'rgb_landmarks.npz');ids=e['frame_ids'];N=len(ids);K=e['K'].copy();K[:2]*=.25;names=['left','right','pipette','tube'];entities={};params=[];history=[];start=time.time()
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
  mesh=trimesh.load(BAS/name/'mesh_local.ply',process=False);ent['local']=tensor(mesh.vertices);ent['faces']=torch.as_tensor(mesh.faces,device=D,dtype=torch.long);ent['T0']=tensor(e[name+'_T']);ent['r']=parameter(np.zeros((N,3)));ent['t']=parameter(np.zeros((N,3)));ent['valid']=tensor(np.ones(N));ent['mask']=tensor(e[name+'_mask']);ent['reliability']=tensor(e[name+'_weight']);params.extend([{'params':[ent['t']],'lr':.00015},{'params':[ent['r']],'lr':.001}]);f=np.load(SOURCE/f'{name}_fields.npz');ent['sdf']=tensor(f['sdf'])[None,None];ent['lo']=tensor(f['lo']);ent['hi']=tensor(f['hi'])
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
# Every object uses the same area-weighted sampling rule, including thin surfaces.
for obj in ['pipette','tube']:
 ent=entities[obj];tri=ent['local'][ent['faces']];areas=torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1).norm(dim=-1).cpu().numpy();chosen=np.random.choice(len(areas),1536,p=areas/areas.sum());uv=np.random.rand(1536,2);uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1];bary=tensor(np.c_[1-uv.sum(1),uv]);ent['collision_points']=(tri[torch.as_tensor(chosen,device=D)]*bary[:,:,None]).sum(1)
def constraints(vertices,transforms):
 penalties=[];raws={}
 samples={s:surface_samples(vertices[s],entities[s]['faces']) for s in ['left','right']}
 for obj in ['pipette','tube']:
  rot,t=transforms[obj];samples[obj]=entities[obj]['collision_points'][None]@rot.transpose(1,2)+t[:,None]
 # Bidirectional tests for every unordered pair, without hand-object assignments.
 # Object surface inside a hand detects a thin wall traversing that hand even
 # when hand vertices have reached the hollow object's empty interior.
 for i,src in enumerate(names):
  for dst in names[i+1:]:
   for source,target in [(src,dst),(dst,src)]:
    p=samples[source];valid=entities[source]['valid']*entities[target]['valid']
    if target in ['pipette','tube']:sd=sdf_query(p,target,transforms)
    else:sd=signed_mesh_distance(p,vertices[target],entities[target]['faces'])
    violation=F.relu(-sd-.00025)*valid[:,None];raws[source+'_'+target]=violation
    penalties.append(violation.topk(16,dim=1).values.mean(1)/.002)
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
# Immutable full-sequence evidence. The optimizer below contains one frame only.
full_entities=entities;full_e=e;all_ids=ids.copy();frame_count=len(ids);tracking=np.load(J/'tracks.npz');assert np.array_equal(tracking['frame_ids'],all_ids)
track_xy=tracking['tracks'];track_conf=tracking['confidence'];track_ids=tracking['track_ids'];warm=np.load(O/'joint_collision/joint_fit.npz');old_report=json.loads((O/'joint_collision/metrics.json').read_text());reference=int(np.argmin(abs(all_ids-a.reference_frame)));N=1
cam=cameras_from_opencv_projection(torch.eye(3,device=D)[None],torch.zeros(1,3,device=D),tensor(K)[None],tensor([[120,160]]))
raster=MeshRasterizer(cameras=cam,raster_settings=RasterizationSettings(image_size=(120,160),blur_radius=0,faces_per_pixel=1,bin_size=16,max_faces_per_bin=16000));faces=torch.cat(allfaces)[None]
face_offsets={};count=0
for name in names:face_offsets[name]=count;count+=len(full_entities[name]['faces'])
dynamic={'valid','mask','t0','r0','p0','beta','t','r','p','landmarks','landmark_weight','T0','reliability','dt','contour','cloud','cloud_valid'}
records={};frame_logs=[];anchor_cache={};active={}
def select_frame(i,source=None):
 global entities,e,scene_depth,params
 entities={};e={k:(x[i:i+1] if x.ndim and len(x)==frame_count else x) for k,x in full_e.items()};scene_depth=tensor(e['scene_depth']);params=[]
 for name,full in full_entities.items():
  ent={k:(v[i:i+1].detach().clone() if k in dynamic else v) for k,v in full.items()};entities[name]=ent
  if name in ['left','right']:
   continuing=source is not None and bool(e[name+'_valid'][0]) and bool(full_e[name+'_valid'][source])
   for key,field in [('t','trans'),('r','rot'),('p','pose')]:
    value=records[source][name+'_'+field] if continuing else warm[name+'_'+field][i];ent[key]=parameter(value[None])
   params.extend([{'params':[ent['t']],'lr':.0007},{'params':[ent['r']],'lr':.003},{'params':[ent['p']],'lr':.003}])
  else:
   T=records[source][name+'_T'] if source is not None else warm[name+'_T'][i];ent['t']=parameter((T[:3,3]-full_e[name+'_T'][i,:3,3])[None]);ent['r']=parameter(Rotation.from_matrix(T[:3,:3]@full_e[name+'_T'][i,:3,:3].T).as_rotvec()[None]);params.extend([{'params':[ent['t']],'lr':.0003},{'params':[ent['r']],'lr':.002}])
def source_maps(source):
 vv=[]
 for n in names:
  v=tensor(records[source][n+'_vertices'])[None];valid=full_e[n+'_valid'][source] if n in ['left','right'] else True;vv.append(v if valid else torch.full_like(v,100))
 with torch.no_grad():fr=raster(Meshes(verts=torch.cat(vv,1),faces=faces))
 fi=fr.pix_to_face[0,:,:,0].cpu().numpy();owner=np.where(fi>=0,np.array(face_labels)[np.maximum(fi,0)%nf],-1);return fi,owner,fr.bary_coords[0,:,:,0].cpu().numpy()
def prepare_tracks(i,source):
 global active
 active={};counts={n:0 for n in names}
 if source is None:return counts
 fi,owner,bary=source_maps(source)
 for en,name in enumerate(names):
  if name in ['left','right'] and not (full_e[name+'_valid'][source] and full_e[name+'_valid'][i]):continue
  source_slots={int(tid):k for k,tid in enumerate(track_ids[source,en]) if tid>=0 and track_conf[source,en,k]>.35}
  face_list=[];bary_list=[];targets=[];weights=[];source_points=[];source_uv=[];target_uv=[];meshfaces=full_entities[name]['faces'].cpu().numpy();sourcev=records[source][name+'_vertices']
  for slot,tid in enumerate(track_ids[i,en]):
   tid=int(tid);conf=float(track_conf[i,en,slot])
   if tid<0 or conf<.35 or tid not in source_slots:continue
   q=source_slots[tid];uv=track_xy[source,en,q]/4
   if tid not in anchor_cache:
    x,y=np.rint(uv).astype(int);x0=max(0,x-3);x1=min(160,x+4);y0=max(0,y-3);y1=min(120,y+4)
    if x0>=x1 or y0>=y1:continue
    yy,xx=np.where(owner[y0:y1,x0:x1]==en)
    if not len(xx):continue
    xx+=x0;yy+=y0;distance=(xx-uv[0])**2+(yy-uv[1])**2;k=int(distance.argmin())
    if distance[k]>9:continue
    px,py=int(xx[k]),int(yy[k]);face=int(fi[py,px]-face_offsets[name]);anchor_cache[tid]=(name,face,bary[py,px].copy())
   n,face,b=anchor_cache[tid]
   if n!=name:raise RuntimeError('Track identity changed entity')
   point=(sourcev[meshfaces[face]]*b[:,None]).sum(0);projected=np.array([K[0,0]*point[0]/point[2]+K[0,2],K[1,1]*point[1]/point[2]+K[1,2]])
   residual=np.linalg.norm(projected-uv);weight=min(conf,float(track_conf[source,en,q]))*np.exp(-.5*(residual/5)**2)
   if weight<.1:continue
   face_list.append(face);bary_list.append(b);targets.append(track_xy[i,en,slot]/4);weights.append(weight);source_points.append(point);source_uv.append(track_xy[source,en,q]/4);target_uv.append(track_xy[i,en,slot]/4)
  counts[name]=len(face_list)
  if not face_list:continue
  active[name]=dict(face=torch.tensor(face_list,device=D,dtype=torch.long),bary=tensor(np.array(bary_list)),target=tensor(np.array(targets)),weight=tensor(weights),source_points=np.array(source_points),source_uv=np.array(source_uv),target_uv=np.array(target_uv))
 return counts
def motion_seed(i,source):
 if source is None:return
 with torch.no_grad():
  for name,track in active.items():
   if len(track['source_points'])<6:continue
   points=track['source_points'];duv=track['target_uv']-track['source_uv'];motion=np.median(duv,axis=0);depth=float(np.median(points[:,2]));dz=[]
   for p,suv,tuv in zip(points,track['source_uv'],track['target_uv']):
    sx,sy=np.rint(suv).astype(int);tx,ty=np.rint(tuv).astype(int)
    if not(0<=sx<160 and 0<=sy<120 and 0<=tx<160 and 0<=ty<120):continue
    zs=full_e['scene_depth'][source,sy,sx];zt=full_e['scene_depth'][i,ty,tx]
    if .15<zs<1.5 and .15<zt<1.5 and abs(zs-p[2])<.025 and abs(zt-zs)<.04:dz.append(zt-zs)
   zmove=float(np.median(dz)) if len(dz)>=6 else 0.;delta=np.array([motion[0]*depth/K[0,0],motion[1]*depth/K[1,1],zmove]);delta=np.clip(delta,-.025,.025);entities[name]['t'].add_(tensor(delta)[None])
def track_loss(vertices):
 loss=vertices['pipette'].sum()*0;used=0
 for name,t in active.items():
  v=vertices[name][:,entities[name]['faces'][t['face']]];points=(v*t['bary'][None,:,:,None]).sum(2);res=(project(points)[0]-t['target'])/2
  loss+=(robust(res).sum(-1)*t['weight']).sum()/t['weight'].sum().clamp(min=1);used+=1
 return loss/max(used,1)
def clip_parameters():
 with torch.no_grad():
  for name,ent in entities.items():
   if name in ['left','right']:
    delta=(ent['p']-ent['p0']).reshape(1,15,3);delta*=.7/delta.norm(dim=-1,keepdim=True).clamp(min=.7);ent['p'].copy_(ent['p0']+delta.reshape(1,45));ent['t'][:,2].clamp_(.15,1.5)
   else:
    for key,cap in [('t',.015),('r',.15)]:ent[key].mul_(cap/ent[key].norm(dim=-1,keepdim=True).clamp(min=cap))
def snapshot_parameters():return {n:{k:ent[k].detach().clone() for k in (['t','r','p'] if n in ['left','right'] else ['t','r'])} for n,ent in entities.items()}
def observation_loss(vertices,joints):
 z,owner=render(vertices);scene_render=torch.where(z>0,z,torch.full_like(z,10));loss=z.sum()*0
 for k,name in enumerate(names):
  ent=entities[name];v=vertices[name];uv=project(v);visible=(v[:,:,2]<=sample_map(scene_render.detach(),uv)+.004).detach();valid=ent['valid'];outside=sample_map(ent['dt'],uv)
  silhouette=(robust(outside/2)*visible*valid[:,None]).sum()/(visible*valid[:,None]).sum().clamp(min=1)
  distance=torch.cdist(ent['contour'],uv)+(~visible)[:,None]*200;coverage=(robust(distance.min(2).values/2).mean(1)*valid).sum()/valid.sum().clamp(min=1)
  hit=(owner==k)&(ent['mask']>.5)&(scene_depth>.15)&(scene_depth<1.5)
  if name in ['pipette','tube']:hit &= (scene_depth-tensor(e[name+'_depth'])).abs()<.025
  dep=(robust((z-scene_depth)/.01)*hit).sum()/hit.sum().clamp(min=1);dd=torch.cdist(ent['cloud'],v)+(~visible)[:,None]*2;cloud=(robust(dd.min(2).values/.01).mean(1)*ent['cloud_valid']).sum()/ent['cloud_valid'].sum().clamp(min=1)
  term=2*dep+1.5*silhouette+1.5*coverage+.4*cloud
  if name in ['left','right']:
   kp=robust((project(joints[name])-ent['landmarks'])/3).sum(-1);kw=ent['landmark_weight'];term+=2*(kp*kw).sum()/kw.sum().clamp(min=1)
   term+=(.8*((ent['p']-ent['p0'])/.3).square().mean()+.15*((ent['r']-ent['r0'])/.4).square().mean()+.05*((ent['t']-ent['t0'])/.08).square().mean())*valid.mean()
  else:
   w=.25+ent['reliability'];term+=2*(w*((ent['t']/.008).square().mean(1)+(ent['r']/.10).square().mean(1))).mean()
  loss+=term
 return loss
def solve_frame(i,source):
 select_frame(i,source);counts=prepare_tracks(i,source);motion_seed(i,source);clip_parameters();initial=snapshot_parameters()
 if active:
  opt=torch.optim.Adam(params)
  for it in range(a.predict_iterations):
   opt.zero_grad();v,j,t=geometry();loss=4*track_loss(v)
   for n,ent in entities.items():
    loss+=.1*((ent['t']-initial[n]['t'])/.01).square().mean()+.05*((ent['r']-initial[n]['r'])/.15).square().mean()
    if n in ['left','right']:loss+=.1*((ent['p']-initial[n]['p'])/.15).square().mean()
   loss.backward();torch.nn.utils.clip_grad_norm_([p for g in params for p in g['params']],20);opt.step();clip_parameters()
 with torch.no_grad():pv,pj,pt=geometry();predicted={n:v[0].cpu().numpy().copy() for n,v in pv.items()}
 prediction=snapshot_parameters();opt=torch.optim.Adam(params)
 for it in range(a.iterations):
  opt.zero_grad();v,j,t=geometry();obs=observation_loss(v,j);tl=track_loss(v);g,viol=constraints(v,t);collision=g.square().mean();loss=obs+3*tl+a.collision_weight*collision
  for n,ent in entities.items():
   trust=min(1.,counts[n]/20)
   if trust:
    loss+=trust*(.1*((ent['t']-prediction[n]['t'])/.01).square().mean()+.05*((ent['r']-prediction[n]['r'])/.15).square().mean())
    if n in ['left','right']:loss+=trust*.05*((ent['p']-prediction[n]['p'])/.2).square().mean()
  if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite frame {i}')
  loss.backward();torch.nn.utils.clip_grad_norm_([p for g in params for p in g['params']],20);opt.step();clip_parameters()
 with torch.no_grad():
  v,j,t=geometry();metrics=evaluate(v,t);record={};residuals={}
  for n in names:
   record[n+'_vertices']=v[n][0].cpu().numpy().copy();record[n+'_predicted_vertices']=predicted[n]
   if n in ['left','right']:
    record[n+'_joints']=j[n][0].cpu().numpy().copy()
    for key,field in [('t','trans'),('r','rot'),('p','pose'),('beta','betas')]:record[n+'_'+field]=entities[n][key][0].cpu().numpy().copy()
   else:
    rot,pos=t[n];T=np.eye(4);T[:3,:3]=rot[0].cpu().numpy();T[:3,3]=pos[0].cpu().numpy();record[n+'_T']=T
   if n in active:
    tr=active[n];point=(v[n][:,entities[n]['faces'][tr['face']]]*tr['bary'][None,:,:,None]).sum(2);residuals[n]=float((project(point)[0]-tr['target']).norm(dim=-1).median()*4)
  record['metrics']=metrics;records[i]=record;row=dict(sample=int(i),frame=int(all_ids[i]),source_frame=int(all_ids[source]) if source is not None else None,direction='reference' if source is None else 'forward' if i>source else 'backward',anchors=counts,track_residual_median_px=residuals,observation_loss=float(obs),track_loss=float(tl),collision_loss=float(collision));frame_logs.append(row)
 return row
order=[reference]+list(range(reference+1,frame_count))+list(range(reference-1,-1,-1))
if a.limit:order=order[:a.limit]
for step,i in enumerate(order):
 if i==reference-1:anchor_cache={}
 source=None if i==reference else i-1 if i>reference else i+1
 row=solve_frame(i,source)
 if step%10==0 or step==len(order)-1:print(json.dumps(dict(progress=f'{step+1}/{len(order)}',**row)),flush=True)
# Publish only a complete sequence; never present uncomputed frames as fitted.
if len(records)!=frame_count:
 (J/'smoke_report.json').write_text(json.dumps(frame_logs,indent=2));print('Smoke test completed; no full reconstruction published',flush=True);sys.exit(0)
output=dict(frame_ids=all_ids)
for key in records[0]:
 if key!='metrics':output[key]=np.stack([records[i][key] for i in range(frame_count)])
for s in ['left','right']:output[s+'_valid']=full_e[s+'_valid'];output[s+'_faces']=full_entities[s]['faces'].cpu().numpy()
np.savez_compressed(J/'joint_fit.npz',**output)
final={n:{key:[records[i]['metrics'][n][key][0] for i in range(frame_count)] for key in ['depth_median_mm','mask_iou','mask_coverage']} for n in names}
final['collision']={key:{metric:[records[i]['metrics']['collision'][key][metric][0] for i in range(frame_count)] for metric in ['max_violation_mm','violating_samples']} for key in records[0]['metrics']['collision']}
for obj in ['pipette','tube']:
 final[obj]['translation_correction_mm']=(np.linalg.norm(output[obj+'_T'][:,:3,3]-full_e[obj+'_T'][:,:3,3],axis=1)*1000).tolist();rot=output[obj+'_T'][:,:3,:3]@full_e[obj+'_T'][:,:3,:3].transpose(0,2,1);final[obj]['rotation_correction_deg']=np.rad2deg(np.linalg.norm(Rotation.from_matrix(rot).as_rotvec(),axis=1)).tolist()
report=dict(baseline=old_report['baseline'],previous=old_report['final'],final=final,history=frame_logs,settings=dict(frames=frame_count,reference_frame=int(all_ids[reference]),predict_iterations=a.predict_iterations,correction_iterations=a.iterations,elapsed_s=time.time()-start,contact_attraction=False,contact_persistence=False,manual_grasp_gates=False,hard_bound=False,collision_weight=a.collision_weight,overlapping_window_refinement=False,joint_variables='Sequential prediction and joint RGB-D correction of both hands and both objects, forward/backward from reference; no joint window optimization',tracking='Persistent BootsTAPIR IDs for four entities; fixed triangle/barycentric anchors, confidence/mask/association gating; articulated MANO point reprojection for hands',collision='Uniform bidirectional soft collision penalty across all six pairs; 0.25mm tolerance, no strict bound; all four entities remain free',initialization='Reference and reappearing hands use the previous soft RGB-D fit; other frames propagate the previous corrected state',camera='Fixed calibrated RealSense intrinsics',hand_shape='Fixed per-hand median shape'))
(J/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False));print('DONE TAPIR-guided joint reconstruction',flush=True)
