"""Create combined tube/pipette/MANO preview and a relocatable viewer bundle."""
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from reconstruct_known_tube import OUT,SCENE,RUN,ROOT,N,save,depth_image
from package_known_interactions import paint


def main():
    tube=trimesh.load(OUT/'tube_local.ply',process=False)
    pipette=trimesh.load(SCENE/'assets/pipette_procedural.ply',process=False)
    ts=np.load(OUT/'tube_poses.npz');ps=np.load(OUT.parent/'object_poses.npz')
    hands=np.load(OUT.parent/'hands_mano_visible.npz')
    camera=json.loads((SCENE/'joint_fit/fit.json').read_text());K=np.array(camera['K'])
    rack_depth=np.load(OUT/'rack_depth.npy');affine=np.load(OUT.parent/'background_affine.npz')['affine']
    tree=cKDTree(np.r_[trimesh.sample.sample_surface(tube,20000,seed=131)[0],tube.vertices])
    snapshots=[0,125,250,350,500,650,800];(OUT/'previews').mkdir(exist_ok=True)
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','bgr24','-s','1920x720','-r','30','-i','-','-an','-c:v','libx264',
        '-preset','fast','-crf','20','-pix_fmt','yuv420p',str(OUT/'hands_pipette_tube.mp4')],stdin=subprocess.PIPE)
    rows=[]
    try:
        for i in range(N):
            tt=ts['T_camera_tube'][i];pt=ps['T_camera_object'][i]
            pv=pipette.vertices@pt[:3,:3].T+pt[:3,3]
            parts=[(pv,pipette.faces,(30,190,245))]
            valid=bool(ts['valid'][i]);iou=float(ts['image_iou'][i])
            if valid:
                tv=tube.vertices@tt[:3,:3].T+tt[:3,3];parts.append((tv,tube.faces,(90,220,80)))
            record=dict(frame=i,tube_valid=valid,tube_iou=iou)
            for side,color in [('left',(180,110,210)),('right',(235,175,80))]:
                visible=bool(hands[side+'_visible'][i]);record[side+'_minimum_distance_m']=None
                if visible:
                    hv=hands[side+'_vertices'][i];parts.append((hv,hands[side+'_faces'],color))
                    if valid:record[side+'_minimum_distance_m']=float(tree.query((hv-tt[:3,3])@tt[:3,:3])[0].min())
            record['pipette_tube_minimum_distance_m']=float(tree.query((pv-tt[:3,3])@tt[:3,:3])[0].min()) if valid else None
            rows.append(record)
            im=cv2.imread(str(RUN/f'all_frames/{i:06d}.png'))
            if valid:
                a=affine[i].copy();a[:,2]*=.5
                rd=cv2.warpAffine(rack_depth,a,(640,360),flags=cv2.INTER_NEAREST,borderValue=10000.)
                td=depth_image(tv,tube.faces,K)
                visible=cv2.resize(((td<9999)&(td<rd+.002)).astype(np.uint8),(1280,720),interpolation=cv2.INTER_NEAREST)>0
                layer=paint(im,[(tv,tube.faces,(90,220,80))],K)
                im[visible]=layer[visible]
            overlay=paint(im,[p for p in parts if p[2] != (90,220,80)],K)
            sideview=paint(np.full((720,640,3),242,np.uint8),parts,side=True)
            cv2.putText(overlay,f'Frame {i}/{N-1} | tube IoU {iou:.2f} | '+('image fit' if valid else 'TUBE FIT UNCERTAIN: hidden'),(15,30),0,.65,(255,255,255),2)
            cv2.putText(overlay,'green: tube | yellow: pipette | purple/blue: MANO hands',(15,57),0,.6,(255,255,255),2)
            cv2.putText(sideview,'Fixed meshes; no contact correction',(15,30),0,.55,(40,40,40),1)
            cv2.putText(sideview,'Tube axial roll assumed, not measured',(15,55),0,.5,(40,40,40),1)
            combined=np.concatenate([overlay,sideview],axis=1);encoder.stdin.write(combined.tobytes())
            if i in snapshots:cv2.imwrite(str(OUT/f'previews/frame_{i:06d}.jpg'),combined)
            if i%100==0:print('packaged',i,flush=True)
    finally:encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('ffmpeg failed')
    save('interaction_diagnostics.json',dict(method='Unsigned nearest sampled tube surface distances; not contact or penetration proof',per_frame=rows))
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(1440,870),'white');draw=ImageDraw.Draw(canvas)
    for n,i in enumerate([0,125,350,500,650,800]):
        xy=((n%2)*720,(n//2)*290)
        canvas.paste(Image.open(OUT/f'previews/frame_{i:06d}.jpg').resize((720,270)),(xy[0],xy[1]+20))
        draw.text((xy[0]+5,xy[1]+4),f'Frame {i}',fill='black')
    canvas.save(OUT/'contact_sheet.jpg')
    segmentation=Image.new('RGB',(1280,4*380),'white');label=ImageDraw.Draw(segmentation)
    for n,i in enumerate(snapshots):
        im=cv2.imread(str(RUN/f'all_frames/{i:06d}.png'))
        mask=cv2.imread(str(OUT/f'masks/{i:06d}.png'),0)>128
        im[mask]=im[mask]*.5+np.array([0,127,0])
        xy=((n%2)*640,(n//2)*380)
        segmentation.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((640,360)),(xy[0],xy[1]+20))
        label.text((xy[0]+5,xy[1]+4),f'Frame {i}: SAM3 tube instance',fill='black')
    segmentation.save(OUT/'segmentation_contact_sheet.jpg')
    bundle=OUT/'viewer_bundle';bundle.mkdir(exist_ok=True)
    files={OUT/'tube_local.ply':'tube.ply',OUT/'tube_poses.npz':'tube_poses.npz',
        SCENE/'assets/pipette_procedural.ply':'pipette.ply',OUT.parent/'object_poses.npz':'pipette_poses.npz',
        OUT.parent/'hands_mano_visible.npz':'hands.npz',RUN/'pipette_g1_01_trimmed.mp4':'demo.mp4',
        ROOT/'scripts/view_known_interactions.py':'view.py',OUT/'tracking_quality.json':'tube_tracking_quality.json',
        OUT/'interaction_diagnostics.json':'tube_interaction_diagnostics.json'}
    for src,name in files.items():shutil.copy2(src,bundle/name)
    config=dict(frames=N,fps=30,focal_px=camera['focal_px'],hands='hands.npz',video='demo.mp4',objects=[
        dict(name='pipette',mesh='pipette.ply',poses='pipette_poses.npz',transform_key='T_camera_object',color=[245,190,30]),
        dict(name='tube',mesh='tube.ply',poses='tube_poses.npz',transform_key='T_camera_tube',color=[80,220,90])])
    (bundle/'scene.json').write_text(json.dumps(config,indent=2)+'\n')
    (bundle/'README.md').write_text('# Portable hand/pipette/tube viewer\n\nInstall dependencies in a local Python environment:\n\n```sh\npython -m pip install numpy scipy opencv-python trimesh viser\npython view.py\n```\n\nOpen http://localhost:8082 on your own machine. No CUDA, source repository, or model checkpoints are needed. Stop with Ctrl+C. Use --port to change the port.\n\nGreen: tube; yellow: pipette; purple/blue: hands. Low-confidence tube frames are hidden by default; enable their checkbox to inspect candidate poses. Tube axial roll is assumed. Hand/object contact is not enforced; pipette tracking has known jumps at frames 160 and 350. See the accompanying reconstruction report before using these estimates.\n')
    with tarfile.open(OUT/'viewer_bundle.tar.gz','w:gz') as tar:tar.add(bundle,arcname='viewer_bundle')
    print('bundle saved',flush=True)


if __name__=='__main__':main()
