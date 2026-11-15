"""Known metric pipette tracking and MANO interaction diagnostics for the demo.

Image-fit reconstruction, not a physical contact solver. All output is separate
from the legacy reconstruction. Run with the sam3d-objects environment.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,trimesh
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import known_scene_milestone as base

OUT=base.OUT/'interactions';OUT.mkdir(exist_ok=True)
N=817

def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def load_mask(frame,name):
    p=base.RUN/f'video_segmentation/masks/frame_{frame:06d}_masks/{name}.png'
    im=cv2.imread(str(p),0)
    return np.zeros((360,640),bool) if im is None else cv2.resize(im,(640,360),interpolation=cv2.INTER_NEAREST)>128

def background():
    path=OUT/'background_affine.npz'
    if path.exists():return np.load(path)['affine']
    orb=cv2.ORB_create(2000);ref=cv2.imread(str(base.RUN/'all_frames/000800.png'),0);ka,da=orb.detectAndCompute(ref,None);bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    records=[];indices=sorted(set(list(range(0,N,10))+[350,800,816]));mats=[]
    for i in indices:
        im=cv2.imread(str(base.RUN/f'all_frames/{i:06d}.png'),0);kb,db=orb.detectAndCompute(im,None)
        matches=[m for m,n in bf.knnMatch(da,db,k=2) if m.distance<.7*n.distance]
        pa=np.float32([ka[m.queryIdx].pt for m in matches]);pb=np.float32([kb[m.trainIdx].pt for m in matches]);A,keep=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=2)
        valid=A is not None and int(keep.sum())>=30
        if not valid:A=np.array([[1.,0,0],[0,1.,0]])
        mats.append(A);records.append(dict(frame=i,matches=len(matches),inliers=int(keep.sum()) if keep is not None else 0,valid=bool(valid)))
    mats=np.array(mats);affine=np.empty((N,2,3))
    for a in range(2):
      for b in range(3):affine[:,a,b]=np.interp(np.arange(N),indices,mats[:,a,b])
    np.savez(path,affine=affine);save('background_affine.json',dict(method='ORB partial-affine background correction every 10 frames, interpolated; used for stand occlusion only, not physical world-camera extrinsics',samples=records))
    return affine

class Tracker:
    def __init__(self):
        self.camera=json.loads((base.OUT/'joint_fit/fit.json').read_text());self.f=self.camera['focal_px'];self.mesh=trimesh.load(base.OUT/'assets/pipette_procedural.ply',process=False)
        self.center=self.mesh.bounds.mean(0);self.mesh.apply_translation(-self.center)
        self.samples=trimesh.sample.sample_surface(self.mesh,1200,seed=2026)[0]
        stand=trimesh.load(base.OUT/'assets/env_06_pipette_stand.ply',process=False);p=self.camera['parameters'];self.stand=base.silhouette(stand,np.array(p[:3]),np.array(p[3:6]),self.f)
        self.affine=background()
        self.motion=json.loads((base.RUN/'perframe_tracking_pipette/motion_stats.json').read_text())
    def masks(self,i):
        gt=load_mask(i,'pipette');occ=load_mask(i,'right_hand_0')
        A=self.affine[i].copy();A[:,2]*=.5
        occ|=cv2.warpAffine(self.stand,A,(640,360),flags=cv2.INTER_NEAREST)>0
        return gt,occ&~gt
    def optimize(self,i,seed,prior=None,max_nfev=45):
        gt,occ=self.masks(i);distance=cv2.distanceTransform((~(gt|occ)).astype(np.uint8),cv2.DIST_L2,5)
        contours,_=cv2.findContours(gt.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not contours:return seed,0.,0.
        edge=np.concatenate(contours)[:,0,:];edge=edge[::max(1,len(edge)//140)]
        priorR=Rotation.from_rotvec(prior[:3]) if prior is not None else None
        def residual(x):
            uv=base.project(self.samples,x[:3],x[3:],self.f)*.5
            a=map_coordinates(distance,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=100)
            b=cKDTree(uv).query(edge)[0]
            res=np.r_[a/np.sqrt(len(a)),b/np.sqrt(len(b))]*10
            if prior is not None:
                angle=(priorR.inv()*Rotation.from_rotvec(x[:3])).as_rotvec()
                res=np.r_[res,angle*5,(x[3:]-prior[3:])*35]
            return res
        result=least_squares(residual,seed,diff_step=1e-4,loss='soft_l1',f_scale=3,max_nfev=max_nfev,bounds=(np.r_[np.full(3,-np.inf),-2,-2,.15],np.r_[np.full(3,np.inf),2,2,3]))
        pred=base.silhouette(self.mesh,result.x[:3],result.x[3:],self.f).astype(bool)&~occ
        iou=float((pred&gt).sum()/max(1,(pred|gt).sum()))
        return result.x,iou,float(result.cost)
    def initialize(self,frame,reference=None):
        old=json.loads((base.OUT/f'poses/procedural_{frame:06d}.json').read_text())['candidates']
        oldf=json.loads((base.OUT/'camera.json').read_text())['focal_px'];candidates=[]
        for c in old:
            seed=np.r_[c['rvec'],np.array(c['translation_m'])*self.f/oldf]
            x,iou,cost=self.optimize(frame,seed,None,100)
            angle=0 if reference is None else float((Rotation.from_rotvec(reference[:3]).inv()*Rotation.from_rotvec(x[:3])).magnitude())
            candidates.append(dict(pose=x.tolist(),iou=iou,cost=cost,angle_to_reference_rad=angle,selection_score=iou-.015*angle/np.pi))
        save(f'initialization_{frame:06d}.json',candidates)
        return np.array(max(candidates,key=lambda c:c['selection_score'])['pose'])

def track():
    tracker=Tracker();start=time.time();rows=[];poses=[]
    cache=OUT/'tracking_checkpoint.npz'
    if cache.exists():
        old=np.load(cache);poses=list(old['poses']);rows=json.loads((OUT/'tracking_checkpoint.json').read_text());previous=poses[-1];begin=len(poses);print('resume at',begin,flush=True)
    else:previous=tracker.initialize(0);begin=0
    for i in range(begin,N):
        seed=previous.copy()
        if i:
            m=tracker.motion[i-1];dx=m.get('translation_dx') or 0.;dy=m.get('translation_dy') or 0.;theta=np.deg2rad(m.get('rotation_deg') or 0.)
            seed[:3]=(Rotation.from_euler('z',theta)*Rotation.from_rotvec(previous[:3])).as_rotvec();seed[3]+=dx*seed[5]/tracker.f;seed[4]+=dy*seed[5]/tracker.f
        if i in [350,800]:seed=tracker.initialize(i,previous)
        x,iou,cost=tracker.optimize(i,seed,previous if i else None)
        # A local retry broadens the basin when image agreement is poor. No scale change.
        retried=False
        if iou<.5 and i>0:
            candidates=[(x,iou,cost)]
            for roll in [-.6,.6]:
                other=seed.copy();other[:3]=(Rotation.from_euler('z',roll)*Rotation.from_rotvec(seed[:3])).as_rotvec();candidates.append(tracker.optimize(i,other,previous,65))
            x,iou,cost=max(candidates,key=lambda c:c[1]);retried=True
        previous=x;poses.append(x);rows.append(dict(frame=i,timestamp_s=i/30,iou=iou,status='image_fit' if iou>=.5 else 'low_image_agreement',retried=retried,cost=cost))
        if i%25==0 or i==N-1:
            np.savez(cache,poses=np.array(poses));save('tracking_checkpoint.json',rows);print(f'frame {i}/{N-1}, IoU {iou:.3f}, elapsed {time.time()-start:.1f}s',flush=True)
    poses=np.array(poses);R=Rotation.from_rotvec(poses[:,:3]).as_matrix();t=poses[:,3:]-np.einsum('nij,j->ni',R,tracker.center);T=np.tile(np.eye(4),(N,1,1));T[:,:3,:3]=R;T[:,:3,3]=t
    world=np.array(tracker.camera['T_world_camera'])@T
    np.savez(OUT/'object_poses.npz',T_camera_object=T,T_world_object_approx=world,timestamps=np.arange(N)/30,fit_center=tracker.center,image_iou=np.array([r['iou'] for r in rows]))
    qs=Rotation.from_matrix(R).as_quat()[:,[3,0,1,2]]
    objects=[dict(frame_idx=i,mesh_obj='pipette_procedural.obj',local_to_scene=dict(translation_camera_frame=t[i].tolist(),quat_wxyz_camera_frame=qs[i].tolist(),scale=[1.,1.,1.]),tracking=rows[i]) for i in range(N)]
    save('layout_camera_frame.json',dict(frame='camera_frame',units='metres',mesh_scale=1.,camera=tracker.camera['K'],note='Known fixed metric mesh; independent image fitting with temporal prior. Approximate fixed camera. Not contact constrained.',objects=objects))
    save('tracking_quality.json',dict(frames=N,elapsed_seconds=time.time()-start,per_frame=rows,method='Known mesh, bidirectional-distance image residual, prior on rotation/translation, TAPIR initialization, multistart reinitialization at frames 0/350/800. Sequential forward tracking; scale never optimized.',mesh_sha256=hashlib.sha256((base.OUT/'assets/pipette_procedural.obj').read_bytes()).hexdigest()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['track']);args=p.parse_args();track()
