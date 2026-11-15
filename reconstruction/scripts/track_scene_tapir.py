"""Persistent BootsTAPIR tracklets for both hands and both known objects.
Tracking is processed in bounded-memory segments with a shared boundary frame.
This is not overlapping-window pose optimization. Surviving IDs are carried;
low-confidence or mask-inconsistent tracks are replaced with new IDs.
"""
import os,argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',default='7');p.add_argument('--segment',type=int,default=12);p.add_argument('--points',type=int,default=64);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
import cv2,numpy as np,torch
from tapnet.torch import tapir_model
R=Path(__file__).resolve().parents[1];O=R/'known_scene/pipette_depth_03_hand_milestone';J=O/'joint_tapir';J.mkdir(exist_ok=True);e=np.load(O/'joint/evidence.npz');ids=e['frame_ids'];names=['left','right','pipette','tube'];T=len(ids);P=a.points;K=e['K'];torch.set_num_threads(4);torch.manual_seed(12)
model=tapir_model.TAPIR(pyramid_level=1);ckpt=R/'weights/tapnet/bootstapir_checkpoint_v2.pt';model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));model.cuda().eval()
masks={n:e[n+'_mask'].astype('uint8') for n in names};valid={n:(e[n+'_valid'] if n in ['left','right'] else np.ones(T,bool)) for n in names}
frames=np.array([cv2.cvtColor(cv2.resize(cv2.imread(str(O/f'rgb/{i:06d}.jpg')),(256,256)),cv2.COLOR_BGR2RGB) for i in ids]);tracks=np.zeros((T,4,P,2),np.float32);confidence=np.zeros((T,4,P),np.float32);track_ids=np.full((T,4,P),-1,np.int32);nextid=0;carry={n:[] for n in names};segments=[];start_time=time.time()
def inside(n,t,xy):
 if not valid[n][t]:return False
 x,y=np.rint(np.asarray(xy)/4).astype(int)
 if x<0 or x>=160 or y<0 or y>=120:return False
 dil=cv2.dilate(masks[n][t],np.ones((3,3),np.uint8));return bool(dil[y,x])
def replenish(n,t,existing,count):
 m=cv2.erode(masks[n][t],np.ones((3,3),np.uint8));yy,xx=np.where(m>0)
 if len(xx)<count:yy,xx=np.where(masks[n][t]>0)
 if not len(xx):return []
 pool=np.c_[xx*4+2,yy*4+2].astype('float32');pool=pool[::max(1,len(pool)//2000)];d=np.full(len(pool),np.inf)
 for point in existing:d=np.minimum(d,((pool-point)**2).sum(1))
 result=[]
 for k in range(count):
  ix=int(np.argmin(np.linalg.norm(pool-pool.mean(0),axis=1))) if not existing and k==0 else int(d.argmax())
  if k or existing:
   if d[ix]<16:break
  result.append(pool[ix]);d=np.minimum(d,((pool-pool[ix])**2).sum(1))
 return result
for start in range(0,T-1,a.segment-1):
 end=min(T,start+a.segment);query=[];records=[]
 for entity,n in enumerate(names):
  carried=[(tid,xy) for tid,xy in carry[n] if inside(n,start,xy)][:P];available=np.where(valid[n][start:end])[0]
  if not len(available):continue
  birth=int(available[0]);seed_frame=start+birth
  if birth:carried=[]
  seed=[x for _,x in carried];new=replenish(n,seed_frame,seed,P-len(seed));items=[(tid,xy,0) for tid,xy in carried]
  for xy in new:items.append((nextid,xy,birth));nextid+=1
  for slot,(tid,xy,qtime) in enumerate(items):query.append([qtime,xy[1]*256/480,xy[0]*256/640]);records.append((entity,slot,tid,qtime))
 video=torch.from_numpy(frames[start:end]).cuda().float()[None]/127.5-1
 with torch.inference_mode():res=model(video,torch.tensor(np.array(query),device='cuda',dtype=torch.float32)[None])
 xy=res['tracks'][0].cpu().numpy()*[640/256,480/256];conf=((1-torch.sigmoid(res['occlusion'][0]))*(1-torch.sigmoid(res['expected_dist'][0]))).cpu().numpy();carry={n:[] for n in names}
 tracks[start:end]=0;confidence[start:end]=0;track_ids[start:end]=-1
 for q,(entity,slot,tid,birth) in enumerate(records):
  n=names[entity]
  for local in range(birth,end-start):
   t=start+local;ok=np.isfinite(xy[q,local]).all() and inside(n,t,xy[q,local]);c=float(conf[q,local]) if ok else 0.
   tracks[t,entity,slot]=xy[q,local];confidence[t,entity,slot]=c;track_ids[t,entity,slot]=tid
  if confidence[end-1,entity,slot]>.55:carry[n].append((tid,xy[q,-1]))
 segments.append(dict(start=int(start),end=int(end-1),query_points=len(records),carried={n:len(carry[n]) for n in names}));print(start,end,'carried',segments[-1]['carried'],flush=True)
 # Save a progress snapshot after each complete tracking segment.
 np.savez_compressed(J/'tracks.partial.npz',frame_ids=ids,tracks=tracks,confidence=confidence,track_ids=track_ids)
np.savez_compressed(J/'tracks.npz',frame_ids=ids,tracks=tracks,confidence=confidence,track_ids=track_ids)
(J/'tracking_provenance.json').write_text(json.dumps(dict(model='BootsTAPIR v2',entities=names,points_per_entity=P,segment_length=a.segment,method='Persistent IDs across tracking segments; one boundary frame; mask-gated confidence; replenished IDs for lost tracks; all selected RGB frames',frames=ids.tolist(),segments=segments,total_track_ids=nextid,elapsed_s=time.time()-start_time),indent=2));print('DONE four-entity tracking',flush=True)
