"""BootsTAPIR adjacent-frame correspondences, both objects in one inference pass."""
import argparse,json,os,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',default='6');p.add_argument('--limit',type=int);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
import cv2,numpy as np,torch
from tapnet.torch import tapir_model
ROOT=Path(__file__).resolve().parents[1];C=json.loads((a.run/'calibration.json').read_text());N=C['frames'];W=C['width'];H=C['height'];P=40
checkpoint=ROOT/'weights/tapnet/bootstapir_checkpoint_v2.pt';model=tapir_model.TAPIR(pyramid_level=1);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model=model.cuda().eval();torch.set_num_threads(4);torch.manual_seed(0)
out=a.run/'tapir';out.mkdir(exist_ok=True);dest=out/'tracks.npz';coords=np.zeros((N-1,2,P,2,2),np.float32);confidence=np.zeros((N-1,2,P,2),np.float32);complete=np.zeros(N-1,bool)
if dest.exists():
 old=np.load(dest);coords[:]=old['tracks'];confidence[:]=old['confidence'];complete[:]=old['complete']
def sample(mask):
 y,x=np.where(cv2.erode(mask,np.ones((3,3),np.uint8))>0)
 if len(x)<P:y,x=np.where(mask>0)
 if not len(x):return np.zeros((P,2),np.float32)
 pts=np.c_[x,y].astype(np.float32);pts=pts[::max(1,len(pts)//1500)]
 # Deterministic farthest-point sampling covers the visible surface, not only its axis.
 chosen=[int(np.argmin(np.linalg.norm(pts-pts.mean(0),axis=1)))];d=np.full(len(pts),np.inf)
 for _ in range(P-1):
  d=np.minimum(d,np.sum((pts-pts[chosen[-1]])**2,axis=1));chosen.append(int(d.argmax()))
 return pts[chosen]
def load(i):return cv2.cvtColor(cv2.resize(cv2.imread(str(a.run/f'rgb/{i:06d}.jpg')),(256,256)),cv2.COLOR_BGR2RGB)
started=time.time()
for t in range(min(N-1,a.limit or N-1)):
 if complete[t]:continue
 pts=np.concatenate([sample(cv2.imread(str(a.run/f'masks/{name}/{t:06d}.png'),0)) for name in ['pipette','tube']]);queries=np.c_[np.zeros(len(pts)),pts[:,1]*256/H,pts[:,0]*256/W].astype(np.float32)
 frames=np.stack([load(t),load(t+1)]);video=torch.from_numpy(frames).cuda().float()[None]/127.5-1
 with torch.inference_mode():res=model(video,torch.from_numpy(queries).cuda()[None])
 tr=res['tracks'][0].cpu().numpy()*np.array([W/256,H/256]);conf=((1-torch.sigmoid(res['occlusion'][0]))*(1-torch.sigmoid(res['expected_dist'][0]))).cpu().numpy()
 coords[t]=tr.reshape(2,P,2,2);confidence[t]=conf.reshape(2,P,2);complete[t]=True
 if t%100==0:
  np.savez_compressed(dest,tracks=coords,confidence=confidence,complete=complete);print(t,'/',N-1,'elapsed',round(time.time()-started,1),flush=True)
np.savez_compressed(dest,tracks=coords,confidence=confidence,complete=complete)
(out/'provenance.json').write_text(json.dumps(dict(model='BootsTAPIR v2',checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),resolution=[256,256],points_per_object=P,objects=['pipette','tube'],method='Adjacent frame pairs; newly sampled visible mask points each pair; raw model confidence retained; no interpolated tracks',completed_pairs=int(complete.sum()),elapsed_s=time.time()-started),indent=2))
