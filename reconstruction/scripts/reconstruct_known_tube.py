"""Track the handled tube using the fixed metric lidar mesh and calibrated camera."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'known_scene/pipette_g1_01'
OUT = SCENE / 'interactions/tube'
RUN = ROOT / 'pipette_g1_01'
N = 817


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def segment():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
    sys.path.insert(0, str(ROOT / 'modules/sam3'))
    import torch
    from sam3.model_builder import build_sam3_video_predictor
    (OUT / 'masks').mkdir(parents=True, exist_ok=True)
    predictor = build_sam3_video_predictor()
    sid = predictor.handle_request(dict(type='start_session', resource_path=str(RUN / 'pipette_g1_01_trimmed.mp4')))['session_id']
    # Assistant annotations on frame 500: body, rim, and lower tube; exclude hand/pipette.
    points = [[646,440],[640,326],[648,505],[555,375],[925,310]]
    labels = [1,1,1,0,0]
    predictor.handle_request(dict(type='add_prompt', session_id=sid, frame_index=500,
        points=torch.tensor(np.array(points)/[1280,720],dtype=torch.float32),
        point_labels=torch.tensor(labels,dtype=torch.int32),obj_id=1))
    rows = {}
    for response in predictor.handle_stream_request(dict(type='propagate_in_video', session_id=sid,
            propagation_direction='both', start_frame_index=500)):
        i = response['frame_index']; output = response['outputs']
        ids = list(output['out_obj_ids']); mask = np.zeros((720,1280),np.uint8)
        if 1 in ids:
            idx = ids.index(1); mask = output['out_binary_masks'][idx].astype(np.uint8)*255
        cv2.imwrite(str(OUT / f'masks/{i:06d}.png'),mask)
        rows[i] = dict(frame=i,pixels=int((mask>0).sum()))
        if i % 100 == 0: print('segmented',i,flush=True)
    save('segmentation.json',dict(prompt_frame=500,points=points,labels=labels,
        provenance='Assistant point annotations; SAM3 same instance propagated both directions',
        frames=[rows[i] for i in sorted(rows)]))
    predictor.handle_request(dict(type='close_session',session_id=sid));predictor.shutdown()


def rotation(x):
    from scipy.spatial.transform import Rotation
    return (Rotation.from_euler('z', x[1]) * Rotation.from_euler('x', x[0])).as_matrix()


def depth_image(vertices,faces,K):
    """Triangle mean-depth raster, approximate visibility rather than exact ray tracing."""
    uv=vertices@K.T;uv=uv[:,:2]/np.maximum(uv[:,2:],.01)*.5
    triangles=np.rint(np.clip(uv[faces],-10000,10000)).astype(np.int32)
    z=vertices[faces,2].mean(1);valid=np.all(vertices[faces,2]>.01,axis=1)
    depth=np.full((360,640),10000.,np.float32)
    for j in np.argsort(z)[::-1]:
        if valid[j]:cv2.fillConvexPoly(depth,triangles[j],float(z[j]))
    return depth


def fit():
    import trimesh
    from scipy.ndimage import map_coordinates
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    import known_scene_milestone as base
    from reconstruct_known_interactions import load_mask
    source = SCENE / 'assets/env_02_cylinder.ply'
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    mesh = trimesh.load(source,process=False)
    center = mesh.bounds.mean(0)
    mesh.apply_translation(-center)
    mesh.export(OUT / 'tube_local.ply');mesh.export(OUT / 'tube_local.obj')
    camera = json.loads((SCENE / 'joint_fit/fit.json').read_text())
    focal = camera['focal_px'];K = np.array(camera['K'])
    samples = trimesh.sample.sample_surface(mesh,1800,seed=129)[0]
    hands = np.load(OUT.parent / 'hands_mano_visible.npz')
    from scipy.spatial.transform import Rotation
    rack=trimesh.load(SCENE/'assets/env_00_cylinder_base.ply',process=False)
    rack.apply_transform(np.array(camera['rack_T_world_from_imported_world']))
    cw=np.array(camera['T_camera_world'])
    rack_depth=depth_image(rack.vertices@cw[:3,:3].T+cw[:3,3],rack.faces,K)
    np.save(OUT/'rack_depth.npy',rack_depth)
    affine=np.load(OUT.parent/'background_affine.npz')['affine']
    masks = [];occluders = [];centroids = [];rack_depths=[]
    for i in range(N):
        path = OUT / f'masks/{i:06d}.png'
        if not path.exists(): raise RuntimeError(f'Missing segmentation: {path}')
        gt = cv2.resize(cv2.imread(str(path),0),(640,360),interpolation=cv2.INTER_NEAREST)>128
        masks.append(gt)
        y,x = np.where(gt);centroids.append(np.array([x.mean(),y.mean()])*2 if len(x) else None)
        occ = load_mask(i,'pipette') | load_mask(i,'right_hand_0')
        a=affine[i].copy();a[:,2]*=.5
        rack_depths.append(cv2.warpAffine(rack_depth,a,(640,360),flags=cv2.INTER_NEAREST,borderValue=10000.))
        if hands['left_visible'][i]:
            hv = hands['left_vertices'][i]
            hm = trimesh.Trimesh(vertices=hv,faces=hands['left_faces'],process=False)
            occ |= base.silhouette(hm,np.zeros(3),np.zeros(3),focal)>0
        occluders.append(occ & ~gt)

    def silhouette(x):
        from scipy.spatial.transform import Rotation
        return base.silhouette(mesh,Rotation.from_matrix(rotation(x)).as_rotvec(),x[2:],focal)>0

    def optimize(i,seed,prior=None):
        gt=masks[i];occ=occluders[i]
        dist=cv2.distanceTransform((~(gt|occ)).astype(np.uint8),cv2.DIST_L2,5)
        contours,_=cv2.findContours(gt.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not contours or gt.sum()<40:return seed,0.
        edge=np.concatenate(contours)[:,0,:];edge=edge[::max(1,len(edge)//180)]
        def residual(x):
            v=samples@rotation(x).T+x[2:];uv=v@K.T;uv=uv[:,:2]/np.maximum(uv[:,2:],.01)*.5
            a=map_coordinates(dist,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=100)
            rd=map_coordinates(rack_depths[i],[uv[:,1],uv[:,0]],order=0,mode='constant',cval=10000.)
            # Rack occludes only geometry behind its rendered surface. A plain rack
            # silhouette wrongly hides tube errors in front of the rack.
            a*=1/(1+np.exp(np.clip((v[:,2]-rd)/.002,-50,50)))
            b=cKDTree(uv).query(edge)[0]
            res=np.r_[a/np.sqrt(len(a)),b/np.sqrt(len(b))]*10
            if prior is not None:res=np.r_[res,(x[:2]-prior[:2])*8,(x[2:]-prior[2:])*75]
            return res
        # Restrict the top toward the camera (visible mouth); axial roll is a gauge,
        # not measured from this nearly symmetric transparent tube.
        lower=[np.pi/2,-4,-2,-2,.2];upper=[np.pi-.15,4,2,2,2.]
        result=least_squares(residual,np.clip(seed,np.array(lower)+1e-7,np.array(upper)-1e-7),
            bounds=(lower,upper),diff_step=1e-4,loss='soft_l1',f_scale=3,max_nfev=65)
        x=result.x
        td=depth_image(mesh.vertices@rotation(x).T+x[2:],mesh.faces,K)
        pred=(td<9999)&(td<rack_depths[i]+.002)&~occ
        iou=float((pred&gt).sum()/max(1,(pred|gt).sum()))
        return result.x,iou

    candidates=[]
    # Frame 500 is open and nearly vertical, with the complete lower tube visible.
    for angle in [95,110,125,140]:
        for depth in [.45,.6,.8]:
            xy=(centroids[500]-[640,360])*depth/focal
            x,iou=optimize(500,np.r_[np.deg2rad(angle),0,xy,depth])
            candidates.append(dict(pose=x.tolist(),iou=iou))
    save('initialization.json',candidates)
    initial=max(candidates,key=lambda r:r['iou']);poses=np.zeros((N,5));ious=np.zeros(N)
    poses[500]=initial['pose'];ious[500]=initial['iou']
    for indices in [range(501,N),range(499,-1,-1)]:
        prev=500
        for i in indices:
            seed=poses[prev].copy()
            if centroids[i] is not None and centroids[prev] is not None:
                seed[2:4]+=(centroids[i]-centroids[prev])*seed[4]/focal
            poses[i],ious[i]=optimize(i,seed,poses[prev]);prev=i
            if i%100==0:print('fit tube',i,'IoU',round(ious[i],3),flush=True)
    T=np.tile(np.eye(4),(N,1,1))
    for i,x in enumerate(poses):T[i,:3,:3]=rotation(x);T[i,:3,3]=x[2:]
    present=np.array([m.sum()>=40 for m in masks])
    valid=present & (ious>=.5)
    np.savez(OUT/'tube_poses.npz',T_camera_tube=T,T_world_tube_approx=np.array(camera['T_world_camera'])@T,
        image_iou=ious,mask_present=present,valid=valid,timestamps=np.arange(N)/30,
        source_center=center,parameters=poses)
    from scipy.spatial.transform import Rotation
    q=Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]
    save('layout_camera_frame.json',dict(frame='camera_frame',units='metres',mesh_scale=1.,objects=[
        dict(frame_idx=i,visible=bool(valid[i]),local_to_scene=dict(translation_camera_frame=T[i,:3,3].tolist(),
        quat_wxyz_camera_frame=q[i].tolist(),scale=[1,1,1])) for i in range(N)]))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==fingerprint
    save('tracking_quality.json',dict(source_mesh=str(source),source_sha256=fingerprint,
        mesh_dimensions_m=mesh.extents.tolist(),original_bounds_center=center.tolist(),
        coordinate_note='tube_local mesh equals source mesh minus bounds center, no scaling/deformation; T maps tube_local to camera metres',
        method='SAM3 single instance, two temporal passes from frame 500, fixed metric mesh silhouette fitting with temporal prior',
        assumptions=['visible mouth faces camera','axial roll fixed by Rz(phi)Rx(theta) gauge, not observed',
        'left MANO projection used as image occluder; fitted rack uses approximate triangle-depth occlusion',
        'constant provisional camera','no contact constraints'],
        frames=N,mask_present_frames=int(present.sum()),valid_frames=int(valid.sum()),
        iou_quantiles=np.quantile(ious,[0,.1,.5,.9,1]).tolist(),
        per_frame=[dict(frame=i,iou=float(ious[i]),mask_pixels=int(masks[i].sum()),
        status='image_fit' if valid[i] else 'low_image_agreement' if present[i] else 'unobserved_carried_pose') for i in range(N)]))
    print('fit complete',np.quantile(ious,[0,.1,.5,.9,1]),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('stage',choices=['segment','fit'])
    args = parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage == 'segment': segment()
    elif args.stage == 'fit': fit()
