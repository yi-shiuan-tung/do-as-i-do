"""Independent WiLoR RGB landmarks, with HaWoR-projected crops for association.
Weights are geometric support heuristics, not calibrated keypoint confidence.
"""
import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='7';os.environ['PYOPENGL_PLATFORM']='egl'
import cv2,numpy as np,torch
from torch.utils.data import default_collate
R=Path(__file__).resolve().parents[1];O=R/'known_scene/pipette_depth_03_hand_milestone';J=O/'joint';J.mkdir(exist_ok=True);repo=R/'modules/WiLoR';os.chdir(repo);sys.path.insert(0,str(repo))
from wilor.configs import get_config
from wilor.models import WiLoR
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils import recursive_to
cfg=get_config(str(repo/'pretrained_models/model_config.yaml'),update_cachedir=True);cfg.defrost();cfg.MODEL.BBOX_SHAPE=[192,256];cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None);cfg.MANO.DATA_DIR=str(repo/'mano_data');cfg.MANO.MODEL_PATH=str(repo/'mano_data');cfg.MANO.MEAN_PARAMS=str(repo/'mano_data/mano_mean_params.npz');cfg.freeze()
model=WiLoR(cfg,init_renderer=False);w=torch.load(repo/'pretrained_models/wilor_final.ckpt',map_location='cpu',weights_only=True);inc=model.load_state_dict(w.get('state_dict',w),strict=False);assert not (set(inc.missing_keys)&set(dict(model.named_parameters())));del w;model.cuda().eval();torch.set_num_threads(4)
e=np.load(O/'evidence.npz');raw=np.load(O/'hands_mano.npz');K=e['K'];ids=e['frame_ids'];records=[];out={s+'_uv':np.zeros((len(ids),21,2),np.float32) for s in ['left','right']};out.update({s+'_weight':np.zeros((len(ids),21),np.float32) for s in ['left','right']})
for k,i in enumerate(ids):
 for side in ['left','right']:
  if not e[side+'_valid'][k]:continue
  v=raw[side+'_vertices'][i];uv=v@K.T;uv=uv[:,:2]/uv[:,2:];box=np.r_[np.clip(uv.min(0)-8,[0,0],[639,479]),np.clip(uv.max(0)+8,[0,0],[639,479])];records.append((k,int(i),side,box))
for offset in range(0,len(records),8):
 chunk=records[offset:offset+8];items=[]
 for k,i,s,box in chunk:items.append(ViTDetDataset(cfg,cv2.imread(str(O/f'rgb/{i:06d}.jpg')),box[None],np.array([int(s=='right')]),rescale_factor=2.)[0])
 batch=recursive_to(default_collate(items),'cuda')
 with torch.inference_mode():pred=model(batch)
 uv=pred['pred_keypoints_2d'].cpu().numpy();centers=batch['box_center'].cpu().numpy();sizes=batch['box_size'].cpu().numpy()
 for n,(k,i,s,box) in enumerate(chunk):
  uv[n,:,0]*=1 if s=='right' else -1
  points=uv[n]*sizes[n]+centers[n];out[s+'_uv'][k]=points
  j=raw[s+'_joints'][i]@K.T;j=j[:,:2]/j[:,2:];difference=np.linalg.norm(points-j,axis=1)
  # Downweight disagreements and points outside the recorded image. No visibility claim.
  weight=np.exp(-.5*(difference/35)**2);weight[(points[:,0]<0)|(points[:,0]>=640)|(points[:,1]<0)|(points[:,1]>=480)]=0;out[s+'_weight'][k]=weight
 if offset%80==0:print(offset+len(chunk),'/',len(records),flush=True)
np.savez_compressed(J/'rgb_landmarks.npz',**out)
(J/'landmark_provenance.json').write_text(json.dumps({'model':'WiLoR pretrained checkpoint','input':'RGB only; crops associated from original HaWoR projection','coordinates':'WiLoR normalized crop prediction, mirrored for left, scaled by crop size, translated by crop center','weights':'exp(-0.5*(distance to HaWoR projection / 35 pixels)^2); out-of-image zero; further depth/mask gating during fitting','not_ground_truth':True},indent=2));print('DONE landmarks',flush=True)
