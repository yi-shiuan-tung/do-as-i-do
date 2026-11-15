"""Joint RGB-D refinement: two MANO hands and two rigid known objects.
No grasp gate, contact vertices, attraction, or slip terms. Collision constraints
use an augmented-Lagrangian sampled-surface approximation, not a guarantee of
triangle-level or continuous-time nonintersection.
"""
import os,sys,json,time,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',default='6');p.add_argument('--iterations',type=int,default=100);p.add_argument('--restore-iterations',type=int,default=200);p.add_argument('--max-penetration-mm',type=float,default=1.);p.add_argument('--warm-start',type=Path);p.add_argument('--collision-weight',type=float,default=16.);a=p.parse_args();a.warm_start=a.warm_start.resolve() if a.warm_start else None;os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
import cv2,numpy as np,torch,trimesh
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRasterizer,RasterizationSettings
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.ops import knn_points
R=Path(__file__).resolve().parents[1];O=R/'known_scene/pipette_depth_03_hand_milestone';J=O/'joint_hard';SOURCE=O/'joint';J.mkdir(exist_ok=True);BAS=R/'known_scene/pipette_depth_03_tapir';os.chdir(R/'modules/HaWoR');sys.path.insert(0,str(R/'modules/HaWoR'))
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
 ent=entities[obj];tri=ent['local'][ent['faces']];areas=torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1).norm(dim=-1).cpu().numpy();chosen=np.random.choice(len(areas),2048,p=areas/areas.sum());uv=np.random.rand(2048,2);uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1];bary=tensor(np.c_[1-uv.sum(1),uv]);ent['collision_points']=(tri[torch.as_tensor(chosen,device=D)]*bary[:,:,None]).sum(1)
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
    violation=F.relu(-sd)*valid[:,None];raws[source+'_'+target]=violation
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
  result['collision']={key:dict(max_violation_mm=(F.relu(v.max(1).values-a.max_penetration_mm/1000)*1000).cpu().tolist(),max_penetration_mm=(v.max(1).values*1000).cpu().tolist(),violating_samples=(v>a.max_penetration_mm/1000).sum(1).cpu().tolist()) for key,v in viol.items()}
  return result
# Evaluate the true original geometry against corrected evidence.
with torch.no_grad():
 vv,jj,tt=geometry();original={n:(tensor(np.nan_to_num(raw[n+'_vertices'][ids])) if n in ['left','right'] else vv[n]) for n in names};baseline=evaluate(original,tt)
 print('Baseline collisions', {k: float(np.max(v['max_violation_mm'])) for k,v in baseline['collision'].items()}, flush=True)
 # Warm start all four entities from the previous joint result; none is frozen.
 previous=np.load(a.warm_start.resolve() if a.warm_start else O/'joint_collision/joint_fit.npz')
 for name,ent in entities.items():
  if name in ['left','right']:
   for key,old in [('t','trans'),('r','rot'),('p','pose')]:ent[key].copy_(tensor(previous[name+'_'+old]))
  else:
   ent['t'].copy_(tensor(previous[name+'_T'][:,:3,3])-ent['T0'][:,:3,3]);delta=previous[name+'_T'][:,:3,:3]@e[name+'_T'][:,:3,:3].transpose(0,2,1);ent['r'].copy_(tensor(Rotation.from_matrix(delta).as_rotvec()))
 pv,pj,pt=geometry();previous_metrics=evaluate(pv,pt)
for side in ['left','right']:entities[side]['p'].requires_grad_(True)
def observation_loss():
 verts,joints,transforms=geometry();z,owner=render(verts);scene_render=torch.where(z>0,z,torch.full_like(z,10));total=z.sum()*0;terms={}
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
 return total,terms

limit=a.max_penetration_mm/1000
if limit<=0:raise ValueError('Use a positive numerical penetration tolerance')
allparams=[p for group in params for p in group['params']]
seed=[p.detach().clone() for p in allparams]
restore_history=[];accept_history=[]
def snapshot():return [p.detach().clone() for p in allparams]
def restore_values(values):
 with torch.no_grad():
  for p,x in zip(allparams,values):p.copy_(x)
def blend(before,after,alpha):
 with torch.no_grad():
  for p,x,y in zip(allparams,before,after):p.copy_(x+(y-x)*alpha.reshape(N,*([1]*(p.ndim-1))))
def bounds():
 with torch.no_grad():
  for name,ent in entities.items():
   if name in ['left','right']:
    delta=(ent['p']-ent['p0']).reshape(N,15,3);delta*=.7/delta.norm(dim=-1,keepdim=True).clamp(min=.7);ent['p'].copy_(ent['p0']+delta.reshape(N,45));ent['t'][:,2].clamp_(.15,1.5)
   else:
    for key,cap in [('t',.015),('r',.15)]:ent[key].mul_(cap/ent[key].norm(dim=-1,keepdim=True).clamp(min=cap))
def collision_state():
 v,j,t=geometry();g,raw=constraints(v,t);depth=torch.stack([p.amax(1) for p in raw.values()],1).amax(1)
 # Auxiliary violation norm resolves ties when several points share the maximum.
 score=sum((F.relu(p-limit*.8)/.002).topk(16,dim=1).values.square().mean(1) for p in raw.values())
 return depth,score,raw
@torch.no_grad()
def check():
 d,q,_=collision_state();return d,q
@torch.no_grad()
def improving(old_d,old_q,new_d,new_q):
 return (old_d>limit)&((new_d<=limit)|(new_q<old_q-1e-7))
def reset_momentum(optimizer,rejected):
 for p in allparams:
  state=optimizer.state.get(p,{})
  for key in ['exp_avg','exp_avg_sq']:
   if key in state:state[key][rejected]=0
@torch.no_grad()
def pattern_restore(rounds=1,step=.002):
 accepted_total=0
 for _ in range(rounds):
  for name in names:
   for axis in range(3):
    for sign in [-1,1]:
     old_d,old_q=check()
     if bool((old_d<=limit).all()):return accepted_total
     before=snapshot();entities[name]['t'][:,axis].add_(sign*step*(old_d>limit));bounds();new_d,new_q=check();accept=improving(old_d,old_q,new_d,new_q);candidate=snapshot();blend(before,candidate,accept.float());accepted_total+=int(accept.sum())
 return accepted_total
restore_params=[]
for group in params:
 restore_params.append({'params':group['params'],'lr':group['lr']})
optimizer=torch.optim.Adam(restore_params)
initial_depth,initial_score=check();print('Initial hard-bound feasibility',int((initial_depth<=limit).sum()),'/',N,'max mm',float(initial_depth.max()*1000),flush=True)
for it in range(a.restore_iterations):
 optimizer.zero_grad();depth,score,contact_raw=collision_state();active=depth>limit
 if not active.any():break
 before=snapshot();old_depth=depth.detach();old_score=score.detach();objective=(score*active).sum()/active.sum().clamp(min=1)
 objective.backward();torch.nn.utils.clip_grad_norm_(allparams,50);optimizer.step();bounds();candidate=snapshot();alpha=torch.ones(N,device=D);accepted=torch.zeros(N,dtype=torch.bool,device=D)
 for trial in range(4):
  blend(before,candidate,alpha);new_d,new_q=check();accepted|=improving(old_depth,old_score,new_d,new_q);alpha=torch.where(accepted,alpha,alpha*.5)
 # accepted alphas are unchanged; unaccepted values are explicitly restored.
 blend(before,candidate,torch.where(accepted,alpha,torch.zeros_like(alpha)));reset_momentum(optimizer,~accepted)
 if (it+1)%50==0:pattern_restore(rounds=1,step=.002)
 if it%20==0 or it==a.restore_iterations-1:
  d,q=check();row=dict(stage='restore',iteration=it,feasible_frames=int((d<=limit).sum()),max_penetration_mm=float(d.max()*1000),accepted=int(accepted.sum()));restore_history.append(row);print(json.dumps(row),flush=True)
# Same deterministic translation search for every entity, only where restoration
# remains infeasible. This is a feasibility phase, not a contact assumption.
for round_id in range(4):
 d,q=check()
 if (d<=limit).all():break
 count=pattern_restore(rounds=1,step=.002 if round_id<2 else .005);d,q=check();row=dict(stage='pattern_restore',iteration=round_id,feasible_frames=int((d<=limit).sum()),max_penetration_mm=float(d.max()*1000),accepted=count);restore_history.append(row);print(json.dumps(row),flush=True)
# Larger uniform candidates escape nested-surface local minima. Only infeasible
# frames are considered, all entities receive the same candidate translations.
for step in [.01,.02,.04,.08]:
 d,q=check()
 if (d<=limit).all():break
 count=pattern_restore(rounds=1,step=step);d,q=check();row=dict(stage='global_restore',step_mm=step*1000,feasible_frames=int((d<=limit).sum()),max_penetration_mm=float(d.max()*1000),accepted=count);restore_history.append(row);print(json.dumps(row),flush=True)
restored_depth,_=check();eligible=restored_depth<=limit
restored_parameters=snapshot();print('Restored feasibility',int(eligible.sum()),'/',N,flush=True)
# Constrained RGB-D refinement: a frame's update is accepted only when EVERY
# checked directional penetration stays within the bound. Infeasible frames are
# held and explicitly reported; they are never labeled successful constrained fits.
optimizer=torch.optim.Adam(params)
for it in range(a.iterations):
 optimizer.zero_grad();total,terms=observation_loss()
 if not torch.isfinite(total):raise RuntimeError('Nonfinite observation objective')
 before=snapshot();total.backward();torch.nn.utils.clip_grad_norm_(allparams,20);optimizer.step();bounds();candidate=snapshot()
 # Project the proposed step against the local normal of the most active
 # constraint in each frame, then verify all constraints nonlinearly below.
 restore_values(before);current_depth,_,_=collision_state();normal=torch.autograd.grad(current_depth.sum(),allparams,allow_unused=True);normal=[torch.zeros_like(p) if g is None else g for p,g in zip(allparams,normal)]
 with torch.no_grad():
  step=[y-x for x,y in zip(before,candidate)];metrics=[float(g['lr'])**2 for g in params for _ in g['params']];dot=sum((n*d).reshape(N,-1).sum(1) for n,d in zip(normal,step));den=sum(w*n.square().reshape(N,-1).sum(1) for w,n in zip(metrics,normal));correction=F.relu(dot-(limit*.98-current_depth.detach()))/den.clamp(min=1e-20);correction*=eligible
  for p,x,d,n,w in zip(allparams,before,step,normal,metrics):p.copy_(x+d-w*n*correction.reshape(N,*([1]*(p.ndim-1))))
 bounds();candidate=snapshot();alpha=eligible.float();accepted=~eligible
 for trial in range(8):
  blend(before,candidate,alpha);d,q=check();accepted|=(d<=limit)&eligible;alpha=torch.where(accepted,alpha,alpha*.5)
 accepted &= eligible;blend(before,candidate,torch.where(accepted,alpha,torch.zeros_like(alpha)));reset_momentum(optimizer,~accepted)
 verified,_=check();assert bool((verified[eligible]<=limit).all()),'Hard bound was violated by an accepted step'
 row=dict(iteration=it,accepted_frames=int(accepted.sum()),rejected_frames=int((eligible&~accepted).sum()),eligible_frames=int(eligible.sum()),max_accepted_penetration_mm=float(verified[eligible].max()*1000) if eligible.any() else None,observation_loss=float(total.detach()))
 accept_history.append(row)
 if it%10==0 or it==a.iterations-1:print(json.dumps(dict(stage='constrained',**row)),flush=True)
final_depth,_=check();hard_feasible=final_depth<=limit
(J/'hard_status.json').write_text(json.dumps(dict(limit_mm=a.max_penetration_mm,frame_ids=ids.tolist(),initial_penetration_mm=(initial_depth*1000).cpu().tolist(),restored_penetration_mm=(restored_depth*1000).cpu().tolist(),final_penetration_mm=(final_depth*1000).cpu().tolist(),feasible=hard_feasible.cpu().tolist(),restoration=restore_history,accepted_steps=accept_history,scope='Hard acceptance bound on the fixed bidirectional sampled collision queries at recorded frames; not an exact continuous mesh or inter-frame guarantee. Unresolved initial frames explicitly marked infeasible.'),indent=2))
with torch.no_grad():
 vertices,joints,transforms=geometry();final=evaluate(vertices,transforms);output=dict(frame_ids=ids)
 for s in ['left','right']:
  ent=entities[s]
  for key,value in [('vertices',vertices[s]),('joints',joints[s]),('trans',ent['t']),('rot',ent['r']),('pose',ent['p']),('betas',ent['beta'])]:output[s+'_'+key]=value.cpu().numpy()
  output[s+'_valid']=e[s+'_valid'];output[s+'_faces']=entities[s]['faces'].cpu().numpy()
 for obj in ['pipette','tube']:
  rot,t=transforms[obj];T=np.repeat(np.eye(4)[None],N,axis=0);T[:,:3,:3]=rot.cpu().numpy();T[:,:3,3]=t.cpu().numpy();output[obj+'_T']=T;output[obj+'_vertices']=vertices[obj].cpu().numpy()
  final[obj]['translation_correction_mm']=(entities[obj]['t'].norm(dim=-1)*1000).cpu().tolist();final[obj]['rotation_correction_deg']=(entities[obj]['r'].norm(dim=-1)*180/np.pi).cpu().tolist()
 np.savez_compressed(J/'joint_fit.npz',**output)

report=dict(baseline=baseline,previous=previous_metrics,final=final,history=restore_history+accept_history,settings=dict(iterations=a.iterations,restore_iterations=a.restore_iterations,elapsed_s=time.time()-start,frames=N,contact_attraction=False,contact_persistence=False,manual_grasp_gates=False,max_penetration_mm=a.max_penetration_mm,hard_feasible_frames=int(hard_feasible.sum()),joint_variables='Both wrists, finger poses, and object poses. Same feasibility-restoration and step rejection for every pair. Camera and shape fixed.',collision='Hard per-frame step acceptance: every directional sampled penetration <= specified bound. Feasibility restoration precedes RGB-D descent. Fixed 2048 area-weighted object samples, hand vertices and face centroids; object SDF and local triangle signed queries. No proof for all triangle intersections or between frames; self collision not constrained.',depth='Shared scene z-buffer; measured RGB-D, masks, WiLoR landmarks',landmarks='Independent WiLoR RGB predictions with geometric weights',tube='Hollow material shell',object_bounds='15mm translation and 0.15rad rotation from initial object fit'))
(J/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False));print('DONE hard-constrained refinement',flush=True)
