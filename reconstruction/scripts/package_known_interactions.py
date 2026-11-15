"""Package calibrated MANO + known-object sequence, visibility, and diagnostics."""
import json,subprocess,hashlib
from pathlib import Path
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import known_scene_milestone as base
from reconstruct_known_interactions import OUT,N,load_mask

def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def camera_uv(v,K):
    uv=v@K.T;return uv[:,:2]/np.maximum(uv[:,2:],.001)

def paint(image,parts,K=None,side=False):
    triangles=[];depths=[];colors=[]
    view=Rotation.from_euler('yx',[65,-12],degrees=True).as_matrix()
    for v,faces,color in parts:
        if side:
            q=(v-np.array([.10,.04,.55]))@view.T;uv=q[:,:2]*1100+[320,360];z=q[:,2];good=np.ones(len(faces),bool)
        else:
            uv=camera_uv(v,K);z=v[:,2];good=np.all(z[faces]>.02,axis=1)
        tri=uv[faces];good&=np.all(np.isfinite(tri),axis=(1,2))
        tri=np.rint(np.clip(tri,-10000,10000)).astype(np.int32);tri=tri[good]
        if not len(tri):continue
        triangles.append(tri);depths.append(z[faces[good]].mean(1));colors.append(np.tile(np.array(color,dtype=np.uint8),(len(tri),1)))
    if not triangles:return image
    tris=np.concatenate(triangles);z=np.concatenate(depths);cs=np.concatenate(colors);layer=image.copy()
    for i in np.argsort(z)[::-1]:cv2.fillConvexPoly(layer,tris[i],tuple(int(x) for x in cs[i]))
    return layer if side else cv2.addWeighted(image,.4,layer,.6,0)

def main():
    camera=json.loads((base.OUT/'joint_fit/fit.json').read_text());K=np.array(camera['K']);poses=np.load(OUT/'object_poses.npz');Ts=poses['T_camera_object'];ious=poses['image_iou']
    mesh=trimesh.load(base.OUT/'assets/pipette_procedural.ply',process=False);hands={k:v.copy() for k,v in np.load(OUT/'hands_mano.npz').items()}
    samples=trimesh.sample.sample_surface(mesh,15000,seed=32)[0];tree=cKDTree(np.r_[samples,mesh.vertices]);observed={}
    for side,index in [('left',0),('right',1)]:
        flags=np.zeros(N,bool)
        for file in (OUT/f'input_demo/cam_space/{index}').glob('*.json'):
            start,end=map(int,file.stem.split('_'));flags[start:end+1]=True
        if side=='right':flags&=np.array([load_mask(i,'right_hand_0').sum()>100 for i in range(N)])
        v=hands[f'{side}_vertices'];flags&=np.mean(v[:,:,2]>.02,axis=1)>.9
        flags&=hands[f'{side}_valid'].astype(bool)
        observed[side]=flags
    display={k:v.copy() for k,v in hands.items()}
    for side in ['left','right']:
        display[f'{side}_visible']=observed[side];display[f'{side}_valid']=observed[side].astype(np.float32)
        for key in ['vertices','joints']:display[f'{side}_{key}'][~observed[side]]=0
    np.savez_compressed(OUT/'hands_mano_visible.npz',**display)
    save('visibility.json',dict(method='Raw HaWoR observed chunk coverage, archive validity, positive depth; right side additionally requires SAM3 mask >100 pixels at half resolution. Infilled-only hands omitted from display/metrics.',left_visible_frames=int(observed['left'].sum()),right_visible_frames=int(observed['right'].sum()),left=observed['left'].tolist(),right=observed['right'].tolist()))
    rows=[];snapshots=[0,125,250,350,500,650,800];(OUT/'previews').mkdir(exist_ok=True)
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s','1920x720','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(OUT/'hand_object_reconstruction.mp4')]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
      for i in range(N):
        T=Ts[i];v=mesh.vertices@T[:3,:3].T+T[:3,3];parts=[(v,mesh.faces,(30,190,245))];record={'frame':i,'timestamp_s':i/30,'object_iou':float(ious[i])}
        gt=load_mask(i,'pipette');hm=load_mask(i,'right_hand_0');adjacent=bool(np.any(cv2.dilate(gt.astype(np.uint8),np.ones((9,9),np.uint8))&hm.astype(np.uint8)))
        record['right_hand_adjacent_in_image']=adjacent
        for side,color in [('left',(180,110,210)),('right',(235,175,80))]:
            record[f'{side}_observed']=bool(observed[side][i])
            if not observed[side][i]:
                record[f'{side}_minimum_distance_m']=None;record[f'{side}_nearby_vertex_count_5mm']=None;continue
            hv=hands[f'{side}_vertices'][i];local=(hv-T[:3,3])@T[:3,:3];distance=tree.query(local)[0]
            record[f'{side}_minimum_distance_m']=float(distance.min());record[f'{side}_distance_p05_m']=float(np.quantile(distance,.05));record[f'{side}_nearby_vertex_count_5mm']=int((distance<.005).sum())
            parts.append((hv,hands[f'{side}_faces'],color))
        rows.append(record)
        image=cv2.imread(str(base.RUN/f'all_frames/{i:06d}.png'));overlay=paint(image,parts,K);sideview=paint(np.full((720,640,3),242,np.uint8),parts,side=True)
        cv2.putText(overlay,f'Known mesh + fresh MANO | frame {i} / {N-1} | IoU {ious[i]:.2f}',(20,30),0,.7,(255,255,255),2)
        cv2.putText(overlay,'yellow: pipette | purple: left hand | blue: right hand',(20,58),0,.6,(255,255,255),2)
        d=record.get('right_minimum_distance_m');distance_text='right hand not observed' if d is None else f'nearest hand/tool samples: {d*1000:.1f} mm (not contact proof)'
        cv2.putText(overlay,distance_text,(20,86),0,.6,(255,255,255),2)
        cv2.putText(sideview,'Oblique 3D view (camera-space)',(15,30),0,.6,(40,40,40),2)
        cv2.putText(sideview,'Fixed metric mesh; no contact correction',(15,58),0,.5,(40,40,40),1)
        combined=np.concatenate([overlay,sideview],axis=1);encoder.stdin.write(combined.tobytes())
        if i in snapshots:cv2.imwrite(str(OUT/f'previews/frame_{i:06d}.jpg'),combined)
        if i%100==0:print('packaged frame',i,flush=True)
    finally:encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('Video encoding failed')
    R=Rotation.from_matrix(Ts[:,:3,:3]);angles=np.degrees((R[:-1].inv()*R[1:]).magnitude());steps=np.linalg.norm(np.diff(Ts[:,:3,3],axis=0),axis=1)
    adjacent_rows=[r for r in rows if r['right_hand_adjacent_in_image'] and r['right_observed']]
    d=np.array([r['right_minimum_distance_m'] for r in adjacent_rows]);near=int((d<.005).sum())
    summary=dict(frames=N,fps=30,object_iou_quantiles=np.quantile(ious,[0,.1,.5,.9,1]).tolist(),low_iou_frames=np.where(ious<.5)[0].tolist(),translation_step_m_quantiles=np.quantile(steps,[.5,.95,1]).tolist(),rotation_step_deg_quantiles=np.quantile(angles,[.5,.95,1]).tolist(),right_observed_image_adjacent_frames=len(d),right_min_distance_m_quantiles=np.quantile(d,[0,.1,.5,.9,1]).tolist() if len(d) else [],image_adjacent_frames_with_distance_under_5mm=near,interpretation='Unsigned nearest sampled object-surface distance from MANO vertices. Does not establish contact or detect penetration. Hand/object poses fitted independently, no physical contact constraint.',per_frame=rows)
    save('interaction_diagnostics.json',summary)
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(1440,3*290),'white');draw=ImageDraw.Draw(canvas)
    for n,frame in enumerate([0,125,350,500,650,800]):
        im=Image.open(OUT/f'previews/frame_{frame:06d}.jpg').resize((720,270));canvas.paste(im,((n%2)*720,(n//2)*290+20));draw.text(((n%2)*720+5,(n//2)*290+4),f'Frame {frame}: calibrated image overlay and oblique 3D',fill='black')
    canvas.save(OUT/'interaction_contact_sheet.jpg')
    print({k:v for k,v in summary.items() if k!='per_frame'},flush=True)

if __name__=='__main__':main()
