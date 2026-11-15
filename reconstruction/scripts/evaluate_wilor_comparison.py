"""Compare matched-input hand reconstructions; metrics are proxies, not 3D GT."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from compare_wilor import ROOT,RUN,SCENE,OUT


def stats(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,p90=float(np.percentile(a,90)) if len(a) else None)


def project(v,K):
    a=v@K.T;return a[:,:2]/np.maximum(a[:,2:],.001)


def mask_mesh(v,faces,K):
    mask=np.zeros((180,320),np.uint8)
    uv=np.rint(np.clip(project(v,K)/4,-2000,2000)).astype(np.int32)
    valid=v[faces][:,:,2].min(1)>.02
    for tri in uv[faces[valid]]:cv2.fillConvexPoly(mask,tri,1)
    return mask>0


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--own-detector',action='store_true');parser.add_argument('--metrics-only',action='store_true');args=parser.parse_args()
    source=OUT;tag='wilor_own_detector' if args.own_detector else 'wilor'
    if args.own_detector: OUT=source/'own_detector';OUT.mkdir(exist_ok=True)
    w=np.load(source/f'{tag}_hands.npz');h=np.load(SCENE/'interactions/hands_mano.npz')
    visibility=json.loads((SCENE/'interactions/visibility.json').read_text())
    K=np.array(json.loads((SCENE/'joint_fit/fit.json').read_text())['K']);N=len(w['left_valid'])
    models={'hawor':h,'wilor':w};valid={};common={};results={};perframe={}
    for side in ['left','right']:
        valid[side]={ 'hawor':np.array(visibility[side],bool), 'wilor':w[f'{side}_valid'].astype(bool)}
        common[side]=valid[side]['hawor']&valid[side]['wilor']
        for model,data in models.items():
            j=data[f'{side}_joints'];v=data[f'{side}_vertices'];c=common[side]
            assert j.shape==(N,21,3) and v.shape==(N,778,3) and np.isfinite(v).all()
            adjacent=c[1:]&c[:-1];triples=c[:-2]&c[1:-1]&c[2:]
            wriststep=np.linalg.norm(np.diff(j[:,0],axis=0),axis=1)*1000
            wristacc=np.linalg.norm(j[2:,0]-2*j[1:-1,0]+j[:-2,0],axis=1)*1000
            rel=j-j[:,:1];fingeracc=np.linalg.norm(rel[2:]-2*rel[1:-1]+rel[:-2],axis=2).mean(1)*1000
            depth=j[c,0,2]
            results[f'{model}_{side}']=dict(valid_frames=int(valid[side][model].sum()),common_frames=int(c.sum()),wrist_depth_m=stats(depth),wrist_step_mm_per_frame=stats(wriststep[adjacent]),wrist_second_difference_mm_per_frame2=stats(wristacc[triples]),root_relative_joint_second_difference_mm_per_frame2=stats(fingeracc[triples]))
        delta=np.linalg.norm(w[f'{side}_joints'][common[side]]-h[f'{side}_joints'][common[side]],axis=2)*1000
        results[f'{side}_between_methods_joint_distance_mm']=stats(delta)
    # Right-hand mask is the only available segmentation label. Same frames,
    # same uncapped MANO topology, and no model-dependent occlusion correction.
    shared=common['right'];ious={m:[] for m in models};iouframes=[]
    faces=w['right_faces']
    for i in np.flatnonzero(shared):
        im=cv2.imread(str(RUN/f'video_segmentation/masks/frame_{i:06d}_masks/right_hand_0.png'),0)
        if im is None:continue
        gt=cv2.resize(im,(320,180),interpolation=cv2.INTER_NEAREST)>128
        if gt.sum()<25:continue
        iouframes.append(int(i))
        for m,d in models.items():
            pred=mask_mesh(d['right_vertices'][i],faces,K)
            ious[m].append(float((pred&gt).sum()/max(1,(pred|gt).sum())))
    for m in models:results[f'{m}_right_silhouette_iou']=stats(ious[m])
    perframe['mask_iou']={'frames':iouframes,**ious}
    mesh=trimesh.load(SCENE/'assets/pipette_procedural.ply',process=False)
    samples=trimesh.sample.sample_surface(mesh,15000,seed=32)[0]
    tree=cKDTree(np.r_[samples,mesh.vertices])
    T=np.load(SCENE/'interactions/object_poses.npz')['T_camera_object']
    for side in ['left','right']:
        ids=np.flatnonzero(common[side]);perframe[side]={'frames':ids.tolist()}
        for m,d in models.items():
            mins=[];p05=[]
            for i in ids:
                local=(d[f'{side}_vertices'][i]-T[i,:3,3])@T[i,:3,:3]
                distance=tree.query(local)[0]*1000;mins.append(float(distance.min()));p05.append(float(np.quantile(distance,.05)))
            perframe[side][m]={'nearest_object_distance_mm':mins,'vertex_distance_p05_mm':p05}
            results[f'{m}_{side}']['object_nearest_distance_mm']=stats(mins)
            results[f'{m}_{side}']['object_vertex_distance_p05_mm']=stats(p05)
    prov=json.loads((source/f'{tag}_provenance.json').read_text());prov.pop('records')
    summary=dict(sequence_frames=N,results=results,wilor_runtime=prov,
        limitations=['No measured 3D hand ground truth or synchronized sensor depth. All scores are diagnostics, not absolute accuracy.',
        'HaWoR baseline is the fresh calibrated known-scene reconstruction; WiLoR uses observed cached detections without interpolation or temporal inference.',
        'HaWoR visibility uses observed chunks, archive validity, positive depth and right-hand SAM3 coverage. WiLoR validity uses detections and positive depth. Geometric metrics use their intersection.',
        'SAM3 right-hand masks are predictions and omit occluded regions. Raw full-mesh silhouette IoU penalizes hidden geometry; no left-hand masks available.',
        'Object distance is unsigned and may be small for intersecting meshes; it does not measure contact correctness or penetration. Includes visible frames whether or not grasping.',
        'Lower temporal second differences can reflect smoothing or under-motion; inspect the video.',
        'No HaWoR timing was recorded for this cached baseline, so no speedup ratio is reported. WiLoR runtime excludes detection and visualization.',
        'Calibrated full-frame camera conversion is an approximation for monocular WiLoR; internal network focal settings are unchanged.'])
    (OUT/'comparison_metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (OUT/'per_frame_metrics.json').write_text(json.dumps(perframe,allow_nan=False))
    if args.metrics_only:
        print(json.dumps(results,indent=2));return
    # Reuse the existing comparison renderer, but keep all output in this directory.
    from package_known_interactions import paint
    writer=cv2.VideoWriter(str(OUT/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
    assert writer.isOpened()
    sheets=[];sample_ids=[0,125,350,500,650,800]
    for i in range(N):
        image=cv2.imread(str(RUN/f'all_frames/{i:06d}.png'));columns=[]
        for m,d in models.items():
            parts=[]
            for side,color in [('left',(185,90,190)),('right',(230,150,60))]:
                if valid[side][m][i]: parts.append((d[f'{side}_vertices'][i],w[f'{side}_faces'],color))
            # Fixed object shown in oblique view as a sampled cloud to avoid
            # mesh simplification and expensive dense rasterization.
            top=cv2.resize(paint(image.copy(),parts,K),(640,360))
            bottom=np.full((720,640,3),245,np.uint8)
            bottom=paint(bottom,parts,side=True)
            from scipy.spatial.transform import Rotation
            ob=samples[::8]@T[i,:3,:3].T+T[i,:3,3]
            view=Rotation.from_euler('yx',[65,-12],degrees=True).as_matrix()
            uv=((ob-[.10,.04,.55])@view.T)[:,:2]*1100+[320,360]
            for x,y in np.rint(uv).astype(int):
                if 0<=x<640 and 0<=y<720:cv2.circle(bottom,(x,y),1,(0,170,180),-1)
            bottom=cv2.resize(bottom,(640,360))
            column=np.vstack([top,bottom]);cv2.putText(column,f'{m.upper()} | frame {i}',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
            cv2.putText(column,'Camera overlay / oblique view (fixed pipette)',(10,350),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            columns.append(column)
        combined=np.hstack(columns);writer.write(combined)
        if i in sample_ids:
            cv2.imwrite(str(OUT/f'comparison_{i:06d}.jpg'),combined);sheets.append(cv2.resize(combined,(640,360)))
        if i%100==0:print('Rendered',i,flush=True)
    writer.release();cv2.imwrite(str(OUT/'contact_sheet.jpg'),np.vstack([np.hstack(sheets[i:i+2]) for i in range(0,len(sheets),2)]))
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
