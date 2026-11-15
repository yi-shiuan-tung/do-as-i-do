"""Reproducible, CPU-only known-scene feasibility experiments; not production tracking."""
import argparse, gzip, hashlib, io, json, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'known_scene/pipette_g1_01'
SIM=ROOT.parent.parent/'g1-pipette-sim'
RUN=ROOT/'pipette_g1_01'

def save(name,data):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nums(x,default):return np.fromstring(x if x is not None else default,sep=' ')
def transform(node):
    T=np.eye(4);T[:3,3]=nums(node.get('pos'),'0 0 0')
    if node.get('quat'):
        q=nums(node.get('quat'),'');T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    elif node.get('euler'):T[:3,:3]=Rotation.from_euler('xyz',nums(node.get('euler'),'')).as_matrix()
    return T

def extract(root,base,body,include_group,world_transform=None):
    assets={a.get('name'):a for a in root.findall('./asset/mesh')}
    mats={a.get('name'):nums(a.get('rgba'),'0.7 0.7 0.7 1') for a in root.findall('./asset/material')}
    pieces=[]
    def walk(b,T):
        for g in b.findall('geom'):
            if int(g.get('group','0')) not in include_group:continue
            typ=g.get('type','sphere');size=nums(g.get('size'),'1')
            if typ=='mesh':
                a=assets[g.get('mesh')];p=Path(a.get('file'));p=p if p.is_absolute() else base/p
                if p.exists():m=trimesh.load(p,force='mesh',process=False)
                else:
                    with gzip.open(str(p)+'.gz','rb') as f:m=trimesh.load(io.BytesIO(f.read()),file_type='obj',force='mesh',process=False)
                m.apply_scale(nums(a.get('scale'),'1 1 1'))
            elif typ=='box':m=trimesh.creation.box(extents=2*size)
            elif typ=='cylinder':m=trimesh.creation.cylinder(radius=size[0],height=2*size[1],sections=32)
            elif typ=='sphere':m=trimesh.creation.icosphere(subdivisions=2,radius=size[0])
            else:raise ValueError(typ)
            m.apply_transform(T@transform(g));color=nums(g.get('rgba'),'') if g.get('rgba') else mats.get(g.get('material'),np.array([.7,.7,.7,1]))
            m.visual.face_colors=np.tile((color*255).astype(np.uint8),(len(m.faces),1));pieces.append(m)
        for child in b.findall('body'):walk(child,T@transform(child))
    walk(body,np.eye(4) if world_transform is None else world_transform)
    return trimesh.util.concatenate(pieces)

def export_assets():
    src=SIM/'assets/environments/lidar';root=ET.parse(src/'scene.xml').getroot()
    raw=json.loads((src/'manifest.json').read_text());cfg=json.loads((SIM/'config/environments/lidar-scene.json').read_text())
    scale=cfg['uniform_scale'];offset=np.array([0,0,-raw['floor']['z_m']*scale]);info=[]
    for b in root.findall('./worldbody/body'):
        m=extract(root,src/'meshes',b,{1});m.vertices=m.vertices*scale+offset
        m.export(OUT/'assets'/f'{b.get("name")}.ply')
        center=m.bounds.mean(0);T=np.eye(4);T[:3,3]=center
        item=dict(body=b.get('name'),bounds_m=m.bounds.tolist(),dimensions_m=m.extents.tolist(),vertices=len(m.vertices),faces=len(m.faces))
        if b.get('name')=='env_01_pipette':
            canonical=m.copy();canonical.apply_translation(-center);canonical.export(OUT/'assets/pipette_lidar.obj');canonical.export(OUT/'assets/pipette_lidar.ply')
            item['T_world_object_at_import']=T.tolist();item['canonical_origin']='calibrated axis-aligned bounding-box center; axes same as world'
        info.append(item)
    # Procedural candidate: redirect generated cache into this run before building XML.
    sys.path.insert(0,str(SIM))
    from g1_pipette_sim import scene,object_meshes
    object_meshes.ASSETS = OUT/'assets/generator_cache'
    config=scene.load_config(SIM/'config/hil194.json')
    # Resolve raw configuration directly; do not apply unrelated dataset-session overrides.
    proot=ET.fromstring(scene.build_xml(config));body=proot.find('.//body[@name="pipette"]')
    m=extract(proot,SIM/'assets/robot/meshes',body,{2});m.export(OUT/'assets/pipette_procedural.obj');m.export(OUT/'assets/pipette_procedural.ply')
    save('scene_manifest.json',dict(source=str(SIM),source_scene_sha256=sha(src/'scene.xml'),raw_to_metric_scale=scale,raw_to_metric_translation=offset.tolist(),units='metres',axes='x,y as imported; z up',objects=info,procedural_candidate=dict(config=str(SIM/'config/hil194.json'),bounds_m=m.bounds.tolist(),dimensions_m=m.extents.tolist(),frame='original tool-local frame; at rest; no detached tip; no session overrides'),note='Visual geometry only. Collision proxies remain in source scene; not used in calibration.'))
    # orthographic asset contact sheet via matplotlib
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(13,9))
    for row,key in enumerate(['lidar','procedural']):
        m=trimesh.load(OUT/f'assets/pipette_{key}.ply',process=False);v=m.vertices
        for col,(a,b) in enumerate([(0,2),(1,2),(0,1)]):
            ax=axs[row,col]
            from matplotlib.collections import PolyCollection
            depth_axis=next(i for i in range(3) if i not in (a,b))
            order=np.argsort(v[m.faces,depth_axis].mean(1))
            ax.add_collection(PolyCollection(v[m.faces[order]][:,:,[a,b]],facecolors=np.asarray(m.visual.face_colors)[order]/255.,edgecolors='none'))
            ax.autoscale_view();ax.set_aspect('equal');ax.set_title(f'{key} {"XYZ"[a]}/{"XYZ"[b]}');ax.set_xlabel('m');ax.set_ylabel('m')
    fig.tight_layout();fig.savefig(OUT/'diagnostics/asset_comparison.png',dpi=130);plt.close(fig)


def Kmatrix(f):return np.array([[f,0,640],[0,f,360],[0,0,1.]])
def project(points,rv,t,f):return cv2.projectPoints(np.asarray(points,dtype=float),np.asarray(rv,dtype=float),np.asarray(t,dtype=float),Kmatrix(f),None)[0].reshape(-1,2)

def landmarks():
    # Approximate visual annotations by the reviewing assistant, not user measurements.
    # x increases toward near table edge; y increases toward image right in this hypothesis.
    z=.72;top=.835
    return [
      ('base_far_left',[.15363,.13980,z],[927,581],3),
      ('base_near_left',[.23785,.13980,z],[947,691],3),
      ('base_near_right',[.23785,.22055,z],[1064,692],3),
      ('rim_far_left',[.16898,.15863,top],[985,518],4),
      ('rim_far_right',[.16898,.20510,top],[1058,521],7),
      ('rim_near_left',[.22244,.15863,top],[999,603],4),
      ('rim_near_right',[.22244,.20510,top],[1078,591],4),
    ]

def fit_camera(xyz,uv,fixed_f=None,seed_f=1000):
    f=seed_f if fixed_f is None else fixed_f
    ok,rv,t=cv2.solvePnP(xyz,uv,Kmatrix(f),None,flags=cv2.SOLVEPNP_EPNP)
    if not ok:raise RuntimeError('PnP initialization failed')
    x=np.r_[rv.ravel(),t.ravel(),np.log(f)]
    def residual(x):
        focal=np.exp(x[6]) if fixed_f is None else fixed_f
        return (project(xyz,x[:3],x[3:6],focal)-uv).ravel()
    lo=np.r_[np.full(6,-np.inf),np.log(300)];hi=np.r_[np.full(6,np.inf),np.log(4000)]
    result=least_squares(residual,x,bounds=(lo,hi),loss='soft_l1',f_scale=4,max_nfev=500)
    x=result.x;f=float(np.exp(x[6])) if fixed_f is None else fixed_f
    R=Rotation.from_rotvec(x[:3]).as_matrix();T=np.eye(4);T[:3,:3]=R;T[:3,3]=x[3:6]
    depth=(xyz@R.T+x[3:6])[:,2]
    e=np.linalg.norm(project(xyz,x[:3],x[3:6],f)-uv,axis=1)
    return dict(focal_px=f,K=Kmatrix(f).tolist(),T_camera_world=T.tolist(),T_world_camera=np.linalg.inv(T).tolist(),rvec=x[:3].tolist(),tvec=x[3:6].tolist(),rms_px=float(np.sqrt(np.mean(e**2))),errors_px=e.tolist(),positive_depth=bool(np.all(depth>0)),camera_position_world=np.linalg.inv(T)[:3,3].tolist())

def static_check():
    ref=cv2.imread(str(RUN/'all_frames/000800.png'),0);orb=cv2.ORB_create(4000)
    a,da=orb.detectAndCompute(ref,None);bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    results=[]
    for frame in [0,125,350,650]:
        im=cv2.imread(str(RUN/f'all_frames/{frame:06d}.png'),0);b,db=orb.detectAndCompute(im,None)
        matches=[m for m,n in bf.knnMatch(da,db,k=2) if m.distance<.7*n.distance]
        pa=np.float32([a[m.queryIdx].pt for m in matches]);pb=np.float32([b[m.trainIdx].pt for m in matches])
        A,inl=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=2)
        keep=inl.ravel().astype(bool);err=np.linalg.norm(pa[keep]-pb[keep],axis=1)
        results.append(dict(frame=frame,matches=len(matches),inliers=int(keep.sum()),median_raw_displacement_px=float(np.median(err)),affine=A.tolist()))
    save('static_camera_check.json',dict(reference_frame=800,method='ORB ratio matches + robust partial affine; dominant background motion, not proof of exact calibration',results=results))

def camera_stage():
    lm=landmarks();xyz=np.array([x[1] for x in lm],float);uv=np.array([x[2] for x in lm],float)
    save('landmarks.json',dict(frame=800,provenance='Assistant approximate manual 2D annotations and corresponding mesh bounding-slice corners. Rim occlusion and rounded/beveled edges introduce systematic error. Not surveyed points.',landmarks=[dict(name=x[0],world_m=x[1],pixel=x[2],estimated_annotation_uncertainty_px=x[3]) for x in lm]))
    seeds=[400,600,800,1000,1400,2000,3000];solutions=[fit_camera(xyz,uv,seed_f=f) for f in seeds];valid=[s for s in solutions if s['positive_depth']];best=min(valid,key=lambda x:x['rms_px'])
    sweep=[fit_camera(xyz,uv,fixed_f=f) for f in [400,500,600,800,1000,1200,1600,2000,3000,4000]]
    loo=[]
    for i in range(len(lm)):
        keep=np.arange(len(lm))!=i;s=fit_camera(xyz[keep],uv[keep],seed_f=1000)
        s['held_out']=lm[i][0];s['held_out_error_px']=float(np.linalg.norm(project(xyz[i:i+1],s['rvec'],s['tvec'],s['focal_px'])[0]-uv[i]));loo.append(s)
    best.update(status='provisional_local_stand_fit_not_accepted_global_calibration',resolution=[1280,720],distortion=[0,0,0,0,0],assumptions=['static camera','square pixels','principal point at image center','stand dimensions and orientation match scene','approximate hand-annotated corners'],reason='Only one small stand anchors fit; validate global layout and focal sensitivity before use.')
    save('camera.json',best);save('camera_sensitivity.json',dict(multistart=solutions,fixed_focal_sweep=sweep,leave_one_out=loo))
    im=cv2.imread(str(RUN/'all_frames/000800.png'));pred=project(xyz,best['rvec'],best['tvec'],best['focal_px'])
    for i,(p,q) in enumerate(zip(uv,pred)):
        p=tuple(p.astype(int));q=tuple(q.astype(int));cv2.circle(im,p,5,(0,255,0),2);cv2.drawMarker(im,q,(0,0,255),cv2.MARKER_CROSS,12,2);cv2.putText(im,str(i),p,0,.6,(255,255,255),2)
    cv2.putText(im,'STAND ONLY: green annotated / red fitted. Provisional camera.',(25,35),0,.7,(0,0,255),2)
    cv2.imwrite(str(OUT/'diagnostics/camera_landmarks.png'),im)
    static_check()
    print('camera',best);print('sweep',[(s['focal_px'],round(s['rms_px'],2)) for s in sweep]);print('LOO',[(s['held_out'],round(s['focal_px']),round(s['held_out_error_px'],1)) for s in loo])


def silhouette(mesh,rv,t,f,scale=.5):
    uv=project(mesh.vertices,rv,t,f)*scale
    cam=mesh.vertices@Rotation.from_rotvec(rv).as_matrix().T+t
    mask=np.zeros((int(720*scale),int(1280*scale)),np.uint8)
    tri=np.rint(np.clip(uv[mesh.faces],-100000,100000)).astype(np.int32)
    valid=np.all(cam[mesh.faces,2]>.01,axis=1)
    for poly in tri[valid]:cv2.fillConvexPoly(mask,poly,1)
    return mask

def draw_outline(im,mask,color,text=None):
    full=cv2.resize(mask,(1280,720),interpolation=cv2.INTER_NEAREST)
    contours,_=cv2.findContours(full,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im,contours,-1,color,2)
    if text and len(contours):
        pts=max(contours,key=cv2.contourArea);x,y,w,h=cv2.boundingRect(pts);cv2.putText(im,text,(max(0,x),max(60,y)),0,.5,color,1)

def scene_stage():
    cam=json.loads((OUT/'camera.json').read_text());rv=np.array(cam['rvec']);t=np.array(cam['tvec']);f=cam['focal_px']
    im=cv2.imread(str(RUN/'all_frames/000800.png'));original=im.copy();meshes={}
    for key,color,label in [('env_06_pipette_stand',(0,255,255),'stand'),('env_05_pin_rack',(0,255,0),'rack'),('env_03_dome_object_open_assembly',(255,0,255),'instrument'),('env_01_pipette',(255,255,0),'pipette')]:
        m=trimesh.load(OUT/f'assets/{key}.ply',process=False);meshes[key]=m
        mask=silhouette(m,rv,t,f);draw_outline(im,mask,color,label)
        if key=='env_06_pipette_stand':cv2.imwrite(str(OUT/'diagnostics/stand_occlusion.png'),mask*255)
    cv2.putText(im,'PROVISIONAL camera / UNADJUSTED scene: outline mismatch is diagnostic',(20,30),0,.65,(0,0,255),2)
    cv2.imwrite(str(OUT/'diagnostics/scene_unadjusted.png'),im)
    # Held-out static landmarks: identifiable outer white rack corners, with explicit approximate associations.
    # Source rack footprint is along Y; video rack is along depth. Test as-is and constrained XY/yaw registration.
    rack=meshes['env_05_pin_rack'];bounds=rack.bounds;center=bounds.mean(0)
    pts=np.array([[bounds[i,0],bounds[j,1],.727] for i,j in [(0,0),(0,1),(1,1),(1,0)]])
    # Annotated image footprint, clockwise; bottom edge is partly outside image.
    obs=np.array([[445,420],[615,428],[590,711],[384,680]],float)
    # Enumerate corner assignment rather than assume raw ordering.
    best=None
    for reverse in [False,True]:
      for shift in range(4):
        target=np.roll(obs[::-1] if reverse else obs,shift,axis=0)
        for yaw in [0,np.pi/2,np.pi,3*np.pi/2]:
          def res(x):
            R=Rotation.from_euler('z',x[2]).as_matrix();q=(pts-center)@R.T+center+np.r_[x[:2],0]
            return (project(q,rv,t,f)-target).ravel()
          fit=least_squares(res,[0,0,yaw],loss='soft_l1',f_scale=5,max_nfev=200)
          err=np.sqrt(np.mean(res(fit.x)**2))
          if best is None or err<best[0]:best=(err,fit.x,target,res(np.zeros(3)).reshape(-1,2))
    err,x,target,before=best;R=Rotation.from_euler('z',x[2]).as_matrix();adjusted=rack.copy();adjusted.vertices=(rack.vertices-center)@R.T+center+np.r_[x[:2],0]
    adjusted.export(OUT/'assets/rack_registered.ply')
    summary=dict(status='exploratory_placement_fit_not_global_validation',rack_translation_xy_m=x[:2].tolist(),rack_yaw_deg=float(np.rad2deg(x[2])%360),rack_landmarks_world=pts.tolist(),rack_image_annotations=target.tolist(),before_rms_euclidean_px=float(np.sqrt(np.mean(np.sum(before**2,axis=1)))),after_rms_euclidean_px=float(err*np.sqrt(2)),note='No dimensions optimized. Approximate footprint corners; near edge partly cropped. Fitting these points removes them from held-out validation. Instrument open/closed state differs; not optimized.')
    save('scene_registration.json',summary)
    overlay=original.copy()
    for m,color,label in [(meshes['env_06_pipette_stand'],(0,255,255),'stand anchor'),(adjusted,(0,255,0),'rack XY/yaw fitted'),(meshes['env_03_dome_object_open_assembly'],(255,0,255),'instrument unchanged')]:draw_outline(overlay,silhouette(m,rv,t,f),color,label)
    for p in target.astype(int):cv2.circle(overlay,tuple(p),5,(255,255,255),2)
    cv2.putText(overlay,'Exploratory prop registration; geometry and camera remain uncertain',(20,30),0,.65,(0,0,255),2)
    cv2.imwrite(str(OUT/'diagnostics/scene_registered.png'),overlay)
    print(summary)


def fit_object(mesh,frame,cam,occlusion,initial_t,initial_R,starts=8):
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree
    mask=cv2.imread(str(RUN/f'video_segmentation/masks/frame_{frame:06d}_masks/pipette.png'),0)>128
    mask=cv2.resize(mask.astype(np.uint8),(640,360),interpolation=cv2.INTER_NEAREST).astype(bool)
    allowed=mask|occlusion
    dist=cv2.distanceTransform((~allowed).astype(np.uint8),cv2.DIST_L2,5)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=np.concatenate(contours)[:,0,:];contour=contour[::max(1,len(contour)//180)]
    samples,_=trimesh.sample.sample_surface(mesh,2200,seed=19)
    f=cam['focal_px'];candidates=[]
    for yaw in np.linspace(0,2*np.pi,starts,endpoint=False):
        r0=Rotation.from_matrix(initial_R@Rotation.from_euler('z',yaw).as_matrix()).as_rotvec()
        def residual(x):
            q=project(samples,x[:3],x[3:],f)*.5
            forward=map_coordinates(dist,[q[:,1],q[:,0]],order=1,mode='constant',cval=100)
            reverse=cKDTree(q).query(contour)[0]
            # Correct for unequal sample counts; both terms are image distances, not metric depth.
            return np.r_[forward/np.sqrt(len(forward)),reverse/np.sqrt(len(reverse))]*10
        fit=least_squares(residual,np.r_[r0,initial_t],diff_step=1e-4,loss='soft_l1',f_scale=3,max_nfev=100,bounds=(np.r_[np.full(3,-np.inf),-2,-2,.15],np.r_[np.full(3,np.inf),2,2,3]))
        rv=fit.x[:3];t=fit.x[3:];pred=silhouette(mesh,rv,t,f);vis=pred.astype(bool)&~occlusion
        iou=float(np.sum(vis&mask)/max(1,np.sum(vis|mask)))
        candidates.append(dict(rvec=rv.tolist(),translation_m=t.tolist(),quaternion_xyzw=Rotation.from_rotvec(rv).as_quat().tolist(),visible_silhouette_iou=iou,cost=float(fit.cost),nfev=fit.nfev,seed_yaw_deg=float(np.rad2deg(yaw))))
    return sorted(candidates,key=lambda c:c['visible_silhouette_iou'],reverse=True)

def object_stage():
    cam=json.loads((OUT/'camera.json').read_text());T=np.array(cam['T_camera_world']);f=cam['focal_px'];summary=[]
    sm=json.loads((OUT/'scene_manifest.json').read_text());center=np.array(next(x for x in sm['objects'] if x['body']=='env_01_pipette')['T_world_object_at_import'])[:3,3]
    init_t=T[:3,:3]@center+T[:3,3]
    stand=cv2.imread(str(OUT/'diagnostics/stand_occlusion.png'),0)>0
    for frame in [800,0,350]:
      # Stand mask is a frame-800 approximation. Hand mask applies per frame.
      hand=cv2.imread(str(RUN/f'video_segmentation/masks/frame_{frame:06d}_masks/right_hand_0.png'),0)>128
      occ=cv2.resize(hand.astype(np.uint8),(640,360),interpolation=cv2.INTER_NEAREST).astype(bool)|stand
      for kind in ['lidar','procedural']:
        mesh=trimesh.load(OUT/f'assets/pipette_{kind}.ply',process=False)
        source_center=mesh.bounds.mean(0);mesh.apply_translation(-source_center)
        t=init_t.copy();R=T[:3,:3].copy()
        if kind=='procedural':R=R@Rotation.from_euler('z',np.pi/2).as_matrix()
        if frame==350:
            mask=cv2.imread(str(RUN/f'video_segmentation/masks/frame_{frame:06d}_masks/pipette.png'),0)>128
            yy,xx=np.where(mask);z=.43;t=np.array([(np.mean(xx)-640)*z/f,(np.mean(yy)-360)*z/f,z])
            # Image shows the tool tilted toward image right at its top.
            R=Rotation.from_euler('z',.4).as_matrix()@R
        candidates=fit_object(mesh,frame,cam,occ,t,R)
        best=candidates[0];record=dict(frame=frame,mesh=kind,source_mesh_center_m=source_center.tolist(),status='provisional_pose_under_unvalidated_camera',scale=1.0,units='metres',candidates=candidates,occlusion_note='Projected stand silhouette plus SAM3 right-hand mask; stand ordering is approximate, no measured depth and no left-hand mask.')
        save(f'poses/{kind}_{frame:06d}.json',record)
        raw=silhouette(mesh,np.array(best['rvec']),np.array(best['translation_m']),f)
        visible=raw.astype(bool)&~occ
        im=cv2.imread(str(RUN/f'all_frames/{frame:06d}.png'))
        gt=cv2.imread(str(RUN/f'video_segmentation/masks/frame_{frame:06d}_masks/pipette.png'),0)>128
        draw_outline(im,gt.astype(np.uint8),(0,255,0));draw_outline(im,visible.astype(np.uint8),(0,0,255))
        cv2.putText(im,f'{kind} frame {frame} visible IoU={best["visible_silhouette_iou"]:.3f} PROVISIONAL',(20,30),0,.7,(0,0,255),2)
        cv2.putText(im,'green: SAM3 mask / red: known metric mesh, approximate occlusion',(20,58),0,.6,(255,255,255),2)
        cv2.imwrite(str(OUT/f'diagnostics/object_{kind}_{frame:06d}.png'),im)
        summary.append(dict(frame=frame,mesh=kind,best_iou=best['visible_silhouette_iou'],depth_m=best['translation_m'][2],candidate_ious=[x['visible_silhouette_iou'] for x in candidates]))
        print(summary[-1],flush=True)
    save('object_fit_summary.json',summary)


def sensitivity_stage():
    cam=json.loads((OUT/'camera.json').read_text());record=json.loads((OUT/'poses/procedural_000800.json').read_text());best=record['candidates'][0]
    mesh=trimesh.load(OUT/'assets/pipette_procedural.ply',process=False);mesh.apply_translation(-mesh.bounds.mean(0))
    occ=cv2.imread(str(OUT/'diagnostics/stand_occlusion.png'),0)>0
    data=[]
    for f in [800.,1200.,1600.]:
        candidate_cam=dict(cam,focal_px=f)
        # Scale initial camera translation with f to keep the initial projection near the object.
        t=np.array(best['translation_m'])*f/cam['focal_px']
        candidates=fit_object(mesh,800,candidate_cam,occ,t,Rotation.from_rotvec(best['rvec']).as_matrix(),starts=4)
        data.append(dict(focal_px=f,depth_m=candidates[0]['translation_m'][2],iou=candidates[0]['visible_silhouette_iou'],candidates=candidates))
    save('object_focal_sensitivity.json',dict(note='Same fixed metric mesh; stand image occluder held fixed while camera focal and object pose change. Not independent depth validation.',results=data))
    # Held-out table edge: line placement/orientation, independent of stand fitting.
    corners=np.array([[-.384,-.768,.72],[-.384,.768,.72]])
    pred=project(corners,cam['rvec'],cam['tvec'],cam['focal_px']);observed=np.array([[150.,163.],[1279.,176.]])
    line=np.cross(np.r_[pred[0],1],np.r_[pred[1],1]);err=np.abs(np.c_[observed,np.ones(2)]@line)/np.linalg.norm(line[:2])
    save('table_validation.json',dict(provenance='Assistant approximate annotation of visible far table edge on frame 800; assumes modeled far edge is the same physical edge.',world_endpoints_m=corners.tolist(),projected_endpoints_px=pred.tolist(),observed_segment_px=observed.tolist(),endpoint_to_model_line_distance_px=err.tolist(),mean_distance_px=float(err.mean())))
    im=cv2.imread(str(RUN/'all_frames/000800.png'));cv2.line(im,tuple(pred[0].astype(int)),tuple(pred[1].astype(int)),(0,0,255),3);cv2.line(im,tuple(observed[0].astype(int)),tuple(observed[1].astype(int)),(0,255,0),3)
    cv2.putText(im,'Held-out table edge: green observed / red model under stand-fit camera',(20,35),0,.65,(0,0,255),2)
    cv2.imwrite(str(OUT/'diagnostics/table_validation.png'),im)
    # Plots and compact comparison sheet.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ss=json.loads((OUT/'camera_sensitivity.json').read_text())['fixed_focal_sweep'];fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot([x['focal_px'] for x in ss],[x['rms_px'] for x in ss],'o-');axs[0].set(xlabel='Fixed focal length (px)',ylabel='Stand landmark RMS (px)',title='Local camera fit is weakly constrained')
    axs[1].plot([x['focal_px'] for x in data],[x['depth_m'] for x in data],'o-');axs[1].set(xlabel='Focal length (px)',ylabel='Object center depth (m)',title='Similar silhouettes permit different depth')
    fig.tight_layout();fig.savefig(OUT/'diagnostics/calibration_sensitivity.png',dpi=140);plt.close(fig)
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(1280,3*390),'white');draw=ImageDraw.Draw(canvas)
    for row,frame in enumerate([0,350,800]):
      for col,key in enumerate(['lidar','procedural']):
        im=Image.open(OUT/f'diagnostics/object_{key}_{frame:06d}.png');im=im.resize((640,360));canvas.paste(im,(col*640,row*390+30));draw.text((col*640+10,row*390+8),f'{key}, frame {frame}: provisional camera, fixed metric scale',fill='black')
    canvas.save(OUT/'diagnostics/object_comparison.jpg')
    # Explicit provisional canonical transform contract.
    c=record['source_mesh_center_m'];R=Rotation.from_rotvec(best['rvec']).as_matrix();T=np.eye(4);T[:3,:3]=R;T[:3,3]=np.array(best['translation_m'])-R@c
    save('reference_pose.json',dict(status='provisional_not_validated_for_tracking',frame=800,mesh='assets/pipette_procedural.obj',mesh_sha256=sha(OUT/'assets/pipette_procedural.obj'),mesh_scale=1.,T_camera_object=T.tolist(),T_world_object=(np.array(cam['T_world_camera'])@T).tolist(),camera='camera.json',quaternion_order='wxyz',quat_wxyz_camera=Rotation.from_matrix(R).as_quat()[[3,0,1,2]].tolist(),translation_camera_m=T[:3,3].tolist(),image_fit_iou=best['visible_silhouette_iou'],reason='Provisional camera; alternative rotations fit nearly as well. Source object frame retained, not bounding-box-centered fitting frame.',candidate_file='poses/procedural_000800.json'))
    print('focal sensitivity',[(x['focal_px'],x['depth_m'],x['iou']) for x in data]);print('table mean edge error',err.mean())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assets','camera','scene','object','sensitivity']);a=p.parse_args()
    {'assets':export_assets,'camera':camera_stage,'scene':scene_stage,'object':object_stage,'sensitivity':sensitivity_stage}[a.stage]()
