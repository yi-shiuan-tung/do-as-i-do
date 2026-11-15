"""Metric rigid fitting against RGB silhouettes and robust aligned depth samples."""
import json,hashlib,shutil,subprocess
from pathlib import Path
import cv2,numpy as np,trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
ROOT=Path(__file__).resolve().parents[1]
def uv(v,K):
 q=v@K.T;return q[:,:2]/np.maximum(q[:,2:],.01)
def raster(v,faces,K,shape):
 z=np.full(shape,np.inf,np.float32);tri=np.rint(np.clip(uv(v,K)[faces],-10000,10000)).astype(np.int32);d=v[faces,2].mean(1)
 for j in np.argsort(d)[::-1]:
    if np.all(v[faces[j],2]>.02):cv2.fillConvexPoly(z,tri[j],float(d[j]))
 return z
def fit(run,C,name):
 assert name in ['pipette','tube'];K=np.array(C['K']);N=C['frames'];H=C['height'];W=C['width']
 out=run/name;out.mkdir(exist_ok=True);source=ROOT/'known_scene/pipette_g1_01/assets'/('pipette_procedural.ply' if name=='pipette' else 'env_02_cylinder.ply')
 mesh=trimesh.load(source,process=False);center=mesh.bounds.mean(0);mesh.apply_translation(-center);mesh.export(out/'mesh_local.ply')
 samples=trimesh.sample.sample_surface(mesh,1400,seed=42)[0];localtree=cKDTree(samples)
 hands=np.load(run/'hands_mano.npz') if (run/'hands_mano.npz').exists() else None
 config=json.loads((run/'tracking_config.json').read_text()) if (run/'tracking_config.json').exists() else {}
 use_tapir=config.get('mode')=='tapir';trackdata=None
 if use_tapir:
    from rgbd_tapir_motion import predict,jump_residual
    trackdata=np.load(run/'tapir/tracks.npz')
    if len(trackdata['complete'])!=N-1 or not trackdata['complete'].all():raise ValueError('TAPIR tracks are incomplete')

 def data(i):
    mask=cv2.imread(str(run/f'masks/{name}/{i:06d}.png'),0)>0;dep=cv2.imread(str(run/f'depth/{i:06d}.png'),-1)*C['depth_scale_m']
    y,x=np.where(cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0);z=dep[y,x];ok=(z>.15)&(z<2);x=x[ok];y=y[ok];z=z[ok]
    # Select the nearest sufficiently populated depth mode, excluding background
    # seen through thin/transparent object regions. Retain depth-support metrics.
    if len(z):
        hist,bins=np.histogram(z,bins=np.arange(.15,2.01,.02));peaks=np.where(hist>=max(12,hist.max()*.35))[0];anchor=(bins[peaks[0]]+.01) if len(peaks) else np.median(z)
        ok=(z>anchor-.05)&(z<anchor+.12);x=x[ok];y=y[ok];z=z[ok]
    stride=max(1,len(z)//250);x=x[::stride];y=y[::stride];z=z[::stride]
    pts=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]
    occ=np.zeros((H,W),bool)
    if hands is not None:
      for side in ['left','right']:
        v=hands[f'{side}_vertices'][i]
        if hands[f'{side}_valid'][i] and np.mean(v[:,2]>.1)>.9:
            occ|=np.isfinite(raster(v,hands[f'{side}_faces'],K,(H,W)))
    dist=cv2.distanceTransform((~(mask|occ)).astype(np.uint8),cv2.DIST_L2,5)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    edge=np.concatenate(contours)[:,0,:] if contours else np.empty((0,2));edge=edge[::max(1,len(edge)//180)]
    return mask,dep,pts,occ,dist,edge
 def optimize(i,seed,prior=None,max_nfev=45,dat=None,anchors=None):
    mask,dep,pts,occ,dist,edge=data(i) if dat is None else dat
    if len(edge)<10:return seed,dict(iou=0.,depth_points=len(pts),depth_error_m=None,present=False)
    # Depth beyond 12 cm from prediction is not used to pull a transparent object
    # onto the background. Its support is reported rather than invented.
    goodpts=pts[np.abs(pts[:,2]-seed[5])<.16]
    def residual(x):
        R=Rotation.from_rotvec(x[:3]).as_matrix();v=samples@R.T+x[3:];u=uv(v,K)
        ds=map_coordinates(dep,[u[:,1],u[:,0]],order=0,mode='constant',cval=0)
        dm=map_coordinates(mask.astype(float),[u[:,1],u[:,0]],order=0,mode='constant',cval=0)
        visible=~((ds>.1)&(v[:,2]>ds+.025)&(dm<.5))
        inside=(u[:,0]>=0)&(u[:,0]<W)&(u[:,1]>=0)&(u[:,1]<H)
        r1=map_coordinates(dist,[u[:,1],u[:,0]],order=1,mode='constant',cval=0)*visible*inside
        r2=cKDTree(u).query(edge)[0]
        res=[r1/np.sqrt(len(r1)),r2/np.sqrt(len(r2))]
        if len(goodpts)>=15:
            d=localtree.query((goodpts-x[3:])@R)[0];res.append(np.minimum(d,.05)/.01*15/np.sqrt(len(d)))
        if use_tapir and anchors is not None:
            local,target=anchors;reprojected=uv(local@R.T+x[3:],K)
            res.append(((reprojected-target)*config.get('track_reprojection_weight',1.0)/np.sqrt(len(local))).ravel())
        if use_tapir and prior is not None:
            innovation=(Rotation.from_rotvec(seed[:3]).inv()*Rotation.from_rotvec(x[:3])).as_rotvec()
            res.extend([innovation*config['innovation_rotation_weight'],(x[3:]-seed[3:])*config['innovation_translation_weight'],jump_residual(x,prior,config)])
        if prior is not None:
            res.extend([(Rotation.from_rotvec(prior[:3]).inv()*Rotation.from_rotvec(x[:3])).as_rotvec()*3,(x[3:]-prior[3:])*60])
        return np.concatenate(res)
    loss='soft_l1'
    if use_tapir and prior is not None and name in config.get('quadratic_jump_objects',[]):
        from rgbd_tapir_motion import robust_data_quadratic_jump_loss
        loss=robust_data_quadratic_jump_loss
    opt=least_squares(residual,seed,bounds=([-np.inf]*3+[-2,-2,.15],[np.inf]*3+[2,2,2]),loss=loss,f_scale=2,max_nfev=max_nfev,diff_step=1e-4)
    x=opt.x;R=Rotation.from_rotvec(x[:3]).as_matrix();z=raster(mesh.vertices@R.T+x[3:],mesh.faces,K,(H,W));pred=np.isfinite(z)&~(occ&~mask)&~((dep>.1)&(z>dep+.025)&~mask)
    iou=float((pred&mask).sum()/max(1,(pred|mask).sum()));err=localtree.query((goodpts-x[3:])@R)[0] if len(goodpts) else []
    return x,dict(iou=iou,depth_points=len(goodpts),depth_error_m=float(np.median(err)) if len(err) else None,present=True,cost=float(opt.cost))
 ref=int(json.loads((run/'segmentation_prompts.json').read_text())['frame']);dat=data(ref);mask,dep,pts,*_=dat;y,x=np.where(mask);pixel=np.array([x.mean(),y.mean()]);z0=float(np.median(pts[:,2]));seeds=[]
 # Both objects' local +Z points toward their upper end. Initial image axis from
 # this recording: bottom-left toward top-right; optimize tilt and roll.
 if use_tapir:
    baseline=Path(config['baseline_run']);d=np.load(baseline/name/'poses.npz');T0=d['T_camera_object'][ref]
    initial=np.r_[Rotation.from_matrix(T0[:3,:3]).as_rotvec(),T0[:3,3]]
    best=(initial,json.loads((baseline/name/'quality.json').read_text())['quality'][ref]);seeds=[best]
 else:
  for tilt in [-.6,-.2,.2,.6]:
     axis=np.array([.28,-.96,tilt]);axis/=np.linalg.norm(axis);R0=Rotation.align_vectors([axis],[[0,0,1]])[0]
     for roll in np.linspace(0,2*np.pi,8,endpoint=False):
         R=R0*Rotation.from_rotvec(np.array([0,0,roll]));t=np.r_[(pixel-K[:2,2])*z0/np.diag(K)[:2],z0]
         pose,q=optimize(ref,np.r_[R.as_rotvec(),t],max_nfev=90,dat=dat);seeds.append((pose,q))
  best=min(seeds,key=lambda t:t[1]['cost'])
 print(name,'initial',best[1],flush=True)
 (out/'initialization.json').write_text(json.dumps([dict(pose=x.tolist(),**q) for x,q in seeds],indent=2))
 poses=np.zeros((N,6));quality=[None]*N;poses[ref]=best[0];quality[ref]=best[1]
 for ids in [range(ref+1,N),range(ref-1,-1,-1)]:
    prev=ref;oldmask=mask;olddep=dep
    for i in ids:
        dat=data(i);newmask=dat[0];seed=poses[prev].copy()
        anchors=None;motion=None
        if use_tapir:
            j=min(i,prev);objidx=0 if name=='pipette' else 1;Rprev=Rotation.from_rotvec(poses[prev,:3]).as_matrix()
            md=raster(mesh.vertices@Rprev.T+poses[prev,3:],mesh.faces,K,(H,W))
            seed,anchors,motion=predict(poses[prev],trackdata['tracks'][j,objidx],trackdata['confidence'][j,objidx],oldmask,newmask,olddep,dat[1],md,K,config,reverse=i<prev)
        elif oldmask.sum()>20 and newmask.sum()>20:
            yy,xx=np.where(oldmask);y2,x2=np.where(newmask);delta=np.array([x2.mean()-xx.mean(),y2.mean()-yy.mean()]);seed[3:5]+=delta*seed[5]/np.diag(K)[:2]
        poses[i],quality[i]=optimize(i,seed,poses[prev],dat=dat,anchors=anchors)
        if motion is not None:quality[i]['motion_prediction']=motion
        prev=i;oldmask=newmask;olddep=dat[1]
        if i%50==0:print(name,i,quality[i],flush=True)
        if i%100==0:np.savez(out/'checkpoint.npz',parameters=poses,completed=np.array([q is not None for q in quality]))
 T=np.tile(np.eye(4),(N,1,1));T[:,:3,:3]=Rotation.from_rotvec(poses[:,:3]).as_matrix();T[:,:3,3]=poses[:,3:]
 original=T.copy();original[:,:3,3]-=np.einsum('nij,j->ni',T[:,:3,:3],center)
 valid=np.array([q['present'] and q['iou']>=.4 for q in quality]);ts=np.array([v['color_timestamp_ms'] for v in C['timestamps']]);ts=(ts-ts[0])/1000
 np.savez_compressed(out/'poses.npz',T_camera_object=T,T_camera_original_mesh=original,valid=valid,timestamps=ts,image_iou=[q['iou'] for q in quality],source_center=center)
 summary=dict(source_mesh=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),mesh_dimensions_m=mesh.extents.tolist(),mesh_scale=1,frames=N,valid_frames=int(valid.sum()),iou_median=float(np.median([q['iou'] for q in quality])),coordinate_convention='camera x right y down z forward, metres; T_camera_object maps mesh_local.ply; T_camera_original_mesh maps source mesh',method='RGB silhouette and robust measured depth-to-mesh distance; temporal pose regularization; HaWoR occlusion masks when available',limitations=['transparent depth may measure background','rotation about long axis weakly observed','no hand-object contact optimization','quality validity is silhouette threshold, not ground truth'],quality=quality)
 if use_tapir:
    summary['method']='BootsTAPIR motion initialization and mesh-anchored point reprojection; measured depth and silhouette refinement; innovation and soft jump penalties'
    summary['tracking_config']=config
 (out/'quality.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(name,{k:v for k,v in summary.items() if k!='quality'},flush=True)
def package(run,C,name=None):
 K=np.array(C['K']);N=C['frames'];H=C['height'];W=C['width'];hands=dict(np.load(run/'hands_mano.npz'));objects={}
 for side,idx in [('left',0),('right',1)]:
    observed=np.zeros(N,bool)
    for path in (run/f'hands/cam_space/{idx}').glob('*.json'):
        start,end=map(int,path.stem.split('_'));observed[max(0,start):min(N,end+1)]=True
    finite=np.isfinite(hands[side+'_vertices']).all(axis=(1,2))&np.isfinite(hands[side+'_joints']).all(axis=(1,2))
    hands[side+'_finite']=finite
    hands[side+'_visible']=observed&finite&hands[side+'_valid'].astype(bool)&(np.mean(hands[side+'_vertices'][:,:,2]>.1,axis=1)>.9)
    # Preserve raw model output separately; absent/invalid display geometry must
    # be finite even when hidden, since browser viewers still upload vertices.
    for key in ['vertices','joints']:
        hands[side+'_'+key]=hands[side+'_'+key].copy()
        hands[side+'_'+key][~hands[side+'_visible']]=0
 np.savez_compressed(run/'hands_visible.npz',**hands)
 for name,color in [('pipette',(30,190,245)),('tube',(50,205,70))]:
    objects[name]=(trimesh.load(run/name/'mesh_local.ply',process=False),np.load(run/name/'poses.npz'),color)
 def paint(im,parts,side=False):
    tris=[];depth=[];colors=[];R=Rotation.from_euler('y',-45,degrees=True).as_matrix()
    for v,f,c in parts:
        if side:
            q=(v-np.array([.04,.02,.55]))@R.T;u=q[:,:2]*650+[W/2,H/2];z=q[:,2];good=np.ones(len(f),bool)
        else:u=uv(v,K);z=v[:,2];good=np.all(z[f]>.02,axis=1)
        good&=np.all(np.isfinite(u[f]),axis=(1,2));tris.extend(np.rint(np.clip(u[f][good],-10000,10000)).astype(np.int32));depth.extend(z[f][good].mean(1));colors.extend([c]*int(good.sum()))
    layer=im.copy()
    for j in np.argsort(depth)[::-1]:cv2.fillConvexPoly(layer,tris[j],colors[j])
    return layer if side else cv2.addWeighted(im,.45,layer,.55,0)
 encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{2*W}x{H}','-r',str(C['fps']),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(run/'reconstruction.mp4')],stdin=subprocess.PIPE)
 snaps=[];snapids=set(np.linspace(0,N-1,8,dtype=int));(run/'previews').mkdir(exist_ok=True)
 for i in range(N):
    parts=[]
    for name,(m,d,color) in objects.items():
        if d['valid'][i]:
            T=d['T_camera_object'][i];parts.append((m.vertices@T[:3,:3].T+T[:3,3],m.faces,color))
    for side,color in [('left',(180,110,210)),('right',(235,175,80))]:
        if hands[side+'_visible'][i]:parts.append((hands[side+'_vertices'][i],hands[side+'_faces'],color))
    im=cv2.imread(str(run/f'rgb/{i:06d}.jpg'));overlay=paint(im,parts);oblique=paint(np.full_like(im,240),parts,True)
    cv2.putText(overlay,f'RGB-D known meshes | {i}/{N-1}',(12,24),0,.6,(255,255,255),2)
    cv2.putText(oblique,'Camera-space oblique view',(12,24),0,.6,(30,30,30),1)
    cv2.putText(oblique,'Yellow: pipette | green: tube',(12,H-16),0,.5,(30,30,30),1)
    combo=np.concatenate([overlay,oblique],1);encoder.stdin.write(combo.tobytes())
    if i in snapids:cv2.imwrite(str(run/f'previews/{i:06d}.jpg'),combo);snaps.append(cv2.resize(combo,(640,240)))
    if i%100==0:print('packaged',i,flush=True)
 encoder.stdin.close();assert encoder.wait()==0
 cv2.imwrite(str(run/'contact_sheet.jpg'),np.concatenate([np.concatenate(snaps[j:j+2],1) for j in range(0,8,2)]))
 config=dict(frames=N,fps=C['fps'],width=W,height=H,focal_px=float(K[1,1]),hands='hands_visible.npz',video='rgb.mp4',objects=[dict(name=name,mesh=f'{name}/mesh_local.ply',poses=f'{name}/poses.npz',transform_key='T_camera_object',color=list(color[::-1])) for name,(m,d,color) in objects.items()])
 (run/'scene.json').write_text(json.dumps(config,indent=2))
 viewer=(ROOT/'scripts/view_known_interactions.py').read_text().replace("360/config['focal_px']","(config.get('height',720)/2)/config['focal_px']").replace('aspect=1280/720',"aspect=config.get('width',1280)/config.get('height',720)")
 viewer=viewer.replace('Axial tube roll is assumed. Contact is not enforced.','Tube axial rotation is weakly constrained. Contact is not enforced.').replace('Show low-confidence tube poses','Show low-confidence object poses')
 (run/'view.py').write_text(viewer)
 ts=np.array([r['color_timestamp_ms'] for r in C['timestamps']]);delta=np.diff(ts);validation=dict(frames=N,duration_s=float((ts[-1]-ts[0])/1000),timestamp_step_ms_quantiles=np.quantile(delta,[0,.5,1]).tolist(),depth_color_offset_ms_max=float(max(abs(r['color_timestamp_ms']-r['depth_timestamp_ms']) for r in C['timestamps'])),hands_visible={s:int(hands[s+'_visible'].sum()) for s in ['left','right']},raw_nonfinite_hand_frames={s:int((~hands[s+'_finite']).sum()) for s in ['left','right']},objects={})
 for name,(m,d,color) in objects.items():
    T=d['T_camera_object'];assert np.isfinite(T).all();assert np.allclose(np.linalg.det(T[:,:3,:3]),1,atol=1e-5)
    source=trimesh.load(json.loads((run/name/'quality.json').read_text())['source_mesh'],process=False);orig=d['T_camera_original_mesh'];i=int(json.loads((run/'segmentation_prompts.json').read_text())['frame'])
    assert np.allclose(source.vertices@orig[i,:3,:3].T+orig[i,:3,3],m.vertices@T[i,:3,:3].T+T[i,:3,3],atol=1e-6)
    steps=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);rots=np.degrees((Rotation.from_matrix(T[:-1,:3,:3]).inv()*Rotation.from_matrix(T[1:,:3,:3])).magnitude())
    validation['objects'][name]=dict(valid_frames=int(d['valid'].sum()),median_iou=float(np.median(d['image_iou'])),translation_step_m_quantiles=np.quantile(steps,[.5,.95,1]).tolist(),rotation_step_deg_quantiles=np.quantile(rots,[.5,.95,1]).tolist(),large_translation_jump_frames=(np.where(steps>.05)[0]+1).tolist(),large_rotation_jump_frames=(np.where(rots>30)[0]+1).tolist())
 (run/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))

 from report_rgbd_known import report
 report(run,C,validation)
