"""Known metric meshes with recorded RealSense intrinsics and aligned depth.
Stages run in sam3 (segment), hawor (hands), sam3d-objects (fit/package).
"""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('stage',choices=['segment','hands','fit','package']);p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',default='6');p.add_argument('--object',choices=['pipette','tube']);a=p.parse_args();a.run=a.run.resolve()
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
import cv2,numpy as np
C=json.loads((a.run/'calibration.json').read_text());K=np.array(C['K']);N=C['frames'];W=C['width'];H=C['height']
def save(name,data): (a.run/name).write_text(json.dumps(data,indent=2,allow_nan=False))
def segment():
 sys.path.insert(0,str(ROOT/'modules/sam3'))
 import torch
 from sam3.model_builder import build_sam3_video_predictor
 predictor=build_sam3_video_predictor()
 prompt_path=a.run/'segmentation_prompts.json'
 if not prompt_path.exists():raise FileNotFoundError(f'Create recording-specific prompts first: {prompt_path}')
 config=json.loads(prompt_path.read_text());prompts=config['points'];ref=int(config['frame']);labels=config['labels']
 assert 0<=ref<N
 for name,points in prompts.items():
    dest=a.run/'masks'/name;dest.mkdir(parents=True,exist_ok=True)
    if len(list(dest.glob('*.png')))==N:continue
    sid=predictor.handle_request(dict(type='start_session',resource_path=str(a.run/'rgb')))['session_id']
    predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=ref,points=torch.tensor(np.array(points)/[W,H],dtype=torch.float32),point_labels=torch.tensor(labels,dtype=torch.int32),obj_id=1))
    for r in predictor.handle_stream_request(dict(type='propagate_in_video',session_id=sid,propagation_direction='both',start_frame_index=ref)):
        out=r['outputs'];ids=list(out['out_obj_ids']);m=np.zeros((H,W),np.uint8)
        if 1 in ids:m=np.asarray(out['out_binary_masks'][ids.index(1)],np.uint8)*255
        cv2.imwrite(str(dest/f"{r['frame_index']:06d}.png"),m)
        if r['frame_index']%100==0:print(name,r['frame_index'],flush=True)
    predictor.handle_request(dict(type='close_session',session_id=sid))
 predictor.shutdown()
def hands():
 import shutil
 seq=a.run/'hands';frames=seq/'extracted_images';frames.mkdir(parents=True,exist_ok=True)
 # HaWoR assumes a centered, square-pixel pinhole: resample RGB to that virtual
 # camera, retaining the same 3D camera axes and metric output coordinates.
 f=float((K[0,0]+K[1,1])/2);kv=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
 mx,my=cv2.initUndistortRectifyMap(K,np.zeros(5),np.eye(3),kv,(W,H),cv2.CV_32FC1)
 for i in range(N):
    dest=frames/f'{i:06d}.png'
    if not dest.exists():cv2.imwrite(str(dest),cv2.remap(cv2.imread(str(a.run/f'rgb/{i:06d}.jpg')),mx,my,cv2.INTER_LINEAR))
 save('hand_camera.json',dict(K=kv.tolist(),method='RGB resampled to centered square-pixel pinhole for HaWoR; same camera axes as recorded RGB',static_camera=True))
 module=ROOT/'modules/HaWoR';source=module/'demo.py';code=source.read_text();marker='    # Here we use aitviewer';assert marker in code
 os.chdir(module);sys.path.insert(0,str(module));sys.argv=[str(source),'--video_path',str(frames),'--img_focal',str(f),'--static_camera']
 exec(compile(code.split(marker)[0],str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
 shutil.copy2(seq/'all_hand_meshes.npz',a.run/'hands_mano.npz')
if a.stage=='segment':segment()
elif a.stage=='hands':hands()
else:
 from rgbd_known_geometry import fit,package
 {'fit':fit,'package':package}[a.stage](a.run,C,a.object)
