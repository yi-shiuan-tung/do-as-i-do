"""Reconstruct saved prediction states for visualization; no model inference or fit."""
import json,shutil
from pathlib import Path
import cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from rgbd_tapir_motion import pose_from_affine,select_motion
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'known_scene/fitting_process'
def main():
 OUT.mkdir(exist_ok=True);checks={}
 shutil.copy2(ROOT/'scripts/templates/fitting_process.html',OUT/'index.html')
 for number in ['01','02','03']:
  run=ROOT/f'known_scene/pipette_depth_{number}_tapir';C=json.loads((run/'calibration.json').read_text());K=np.array(C['K']);N=C['frames'];ref=json.loads((run/'segmentation_prompts.json').read_text())['frame'];tracks=np.load(run/'tapir/tracks.npz');objects={};count=0
  for name,idx in [('pipette',0),('tube',1)]:
   mesh=trimesh.load(run/name/'mesh_local.ply',process=False);poses=np.load(run/name/'poses.npz')['T_camera_object'];quality=json.loads((run/name/'quality.json').read_text())['quality'];states=[]
   masks=[cv2.imread(str(run/f'masks/{name}/{i:06d}.png'),0)>0 for i in range(N)]
   for i in range(N):
    prev=i if i==ref else (i-1 if i>ref else i+1);T=poses[prev];pred=T.copy();motion=quality[i].get('motion_prediction',{});points=[]
    if i!=ref:
     previous=np.r_[Rotation.from_matrix(T[:3,:3]).as_rotvec(),T[:3,3]]
     if motion['status']=='tapir':
      x=pose_from_affine(previous,np.array(motion['affine']),K,motion['depth_delta_m']);pred[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix();pred[:3,3]=x[3:]
     j=min(i,prev);tr=tracks['tracks'][j,idx];cf=tracks['confidence'][j,idx];A,aa,bb=select_motion(tr,cf,masks[prev],masks[i],i<prev)
     assert len(aa)==motion['visible_points'],(number,name,i,len(aa),motion)
     if motion['status']=='tapir':assert A is not None and np.allclose(A,motion['affine'],atol=1e-8)
     elif A is not None:raise AssertionError('Saved tracking rejection mismatch')
     for k in range(len(tr)):
      a=tr[k,1 if i<prev else 0];b=tr[k,0 if i<prev else 1];used=A is not None and bool(np.any(np.linalg.norm(aa-a,axis=1)<1e-3));points.append([*a.tolist(),*b.tolist(),float(cf[k].min()),used])
     count+=1
    final=poses[i];correction_mm=float(np.linalg.norm(final[:3,3]-pred[:3,3])*1000);correction_deg=float(np.degrees((Rotation.from_matrix(pred[:3,:3]).inv()*Rotation.from_matrix(final[:3,:3])).magnitude()))
    states.append(dict(previous=prev,predicted=pred.tolist(),points=points,motion=motion,correction_mm=correction_mm,correction_deg=correction_deg,quality={k:v for k,v in quality[i].items() if k!='motion_prediction'}))
   objects[name]=dict(vertices=np.round(mesh.vertices,7).tolist(),faces=mesh.faces.tolist(),poses=poses.tolist(),states=states)
  data=dict(recording=number,frames=N,reference=ref,width=C['width'],height=C['height'],K=K.tolist(),rgb_base=f'../pipette_depth_{number}_tapir/rgb/',mask_base=f'../pipette_depth_{number}_tapir/masks/',depth_base=f'depth_{number}/',timestamps_s=((np.array([t['color_timestamp_ms'] for t in C['timestamps']])-C['timestamps'][0]['color_timestamp_ms'])/1000).tolist(),objects=objects)
  (OUT/f'data_{number}.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
  dest=OUT/f'depth_{number}';dest.mkdir(exist_ok=True)
  for i in range(N):
   path=dest/f'{i:06d}.jpg'
   if path.exists():continue
   d=cv2.imread(str(run/f'depth/{i:06d}.png'),-1)*C['depth_scale_m'];gray=np.uint8(np.clip((d-.2)/1.0,0,1)*255);im=cv2.applyColorMap(gray,cv2.COLORMAP_TURBO);im[d==0]=0;cv2.imwrite(str(path),cv2.resize(im,(320,240),interpolation=cv2.INTER_NEAREST),[cv2.IMWRITE_JPEG_QUALITY,90])
  checks[number]=dict(frames=N,reference=ref,verified_saved_motion_transitions=count,point_filter_reproduced=True,source='saved final poses + saved affine/depth delta; models and optimization not rerun')
  print(number,checks[number],flush=True)
 (OUT/'validation.json').write_text(json.dumps(checks,indent=2))
if __name__=='__main__':main()
