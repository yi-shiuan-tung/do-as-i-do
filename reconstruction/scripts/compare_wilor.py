"""Matched-detection WiLoR baseline for the calibrated known-scene video.

No interpolation or smoothing. Detector boxes and track-majority handedness are
reused from HaWoR. Outputs preserve original frame numbers and camera coordinates.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'pipette_g1_01'
SCENE=ROOT/'known_scene/pipette_g1_01'
OUT=SCENE/'comparison_wilor'


def records():
    tracks=np.load(RUN/'pipette_g1_01_trimmed/tracks_0_817/model_tracks.npy',allow_pickle=True).item()
    chosen={}
    for tid,track in tracks.items():
        observed=[t for t in track if t['det']]
        if not observed: continue
        side=int(np.mean([float(np.asarray(t['det_handedness']).reshape(-1)[0]) for t in observed])>=.5)
        for t in observed:
            b=np.asarray(t['det_box']).reshape(-1)
            if len(b)<4 or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]: continue
            rec={'frame':int(t['frame']),'side':side,'box':b[:4].tolist(),'confidence':float(b[4]) if len(b)>4 else 1.,'track':str(tid)}
            key=(rec['frame'],side)
            if key not in chosen or rec['confidence']>chosen[key]['confidence']: chosen[key]=rec
    return [chosen[k] for k in sorted(chosen)]


def infer(args):
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import torch,cv2
    from torch.utils.data import default_collate
    repo=ROOT/'modules/WiLoR';sys.path.insert(0,str(repo));os.chdir(repo)
    from wilor.configs import get_config
    from wilor.models import WiLoR
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full
    cfg=get_config(str(repo/'pretrained_models/model_config.yaml'),update_cachedir=True)
    cfg.defrost();cfg.MODEL.BBOX_SHAPE=[192,256]
    cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None)
    cfg.MANO.DATA_DIR=str(repo/'mano_data');cfg.MANO.MODEL_PATH=str(repo/'mano_data');cfg.MANO.MEAN_PARAMS=str(repo/'mano_data/mano_mean_params.npz');cfg.freeze()
    started=time.perf_counter()
    model=WiLoR(cfg,init_renderer=False)
    checkpoint=repo/'pretrained_models/wilor_final.ckpt'
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state=weights.get('state_dict',weights)
    incompat=model.load_state_dict(state,strict=False)
    missing=set(incompat.missing_keys)&set(dict(model.named_parameters()))
    if missing or incompat.unexpected_keys: raise RuntimeError(f'Checkpoint mismatch: {missing}, {incompat.unexpected_keys}')
    del weights,state
    model=model.cuda().eval();torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    K=np.array(json.loads((SCENE/'joint_fit/fit.json').read_text())['K'])
    assert abs(K[0,0]-K[1,1])<1e-5 and np.allclose(K[:2,2],[640,360])
    frames=sorted((RUN/'all_frames').glob('*.png'));N=len(frames)
    data={}
    for side in ['left','right']:
        for key,shape in [('vertices',(N,778,3)),('joints',(N,21,3)),('cam_t',(N,3)),('betas',(N,10)),('canonical_global_orient',(N,1,3,3)),('canonical_hand_pose',(N,15,3,3))]:data[f'{side}_{key}']=np.zeros(shape,np.float32)
        data[f'{side}_valid']=np.zeros(N,bool)
        data[f'{side}_confidence']=np.zeros(N,np.float32)
        data[f'{side}_faces']=np.asarray(model.mano.faces)[:,[0,2,1]] if side=='left' else np.asarray(model.mano.faces)
    recs=json.loads(args.records.read_text()) if args.records else records()
    if args.limit: recs=[r for r in recs if r['frame']<args.limit]
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();forward=0
    for offset in range(0,len(recs),args.batch_size):
        chunk=recs[offset:offset+args.batch_size];items=[]
        for r in chunk:
            im=cv2.imread(str(frames[r['frame']]))
            if im is None or im.shape[:2]!=(720,1280): raise ValueError('Missing/unexpected frame')
            items.append(ViTDetDataset(cfg,im,np.array([r['box']]),np.array([r['side']]),rescale_factor=2.0)[0])
        batch=recursive_to(default_collate(items),'cuda')
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode():pred=model(batch)
        torch.cuda.synchronize();forward+=time.perf_counter()-t
        cam=pred['pred_cam'].clone();cam[:,1]*=(2*batch['right']-1)
        trans=cam_crop_to_full(cam,batch['box_center'].float(),batch['box_size'].float(),batch['img_size'].float(),float(K[0,0])).cpu().numpy()
        for n,r in enumerate(chunk):
            i=r['frame'];side=['left','right'][r['side']];sign=2*r['side']-1
            v=pred['pred_vertices'][n].cpu().numpy().copy();j=pred['pred_keypoints_3d'][n].cpu().numpy().copy()
            v[:,0]*=sign;j[:,0]*=sign;v+=trans[n];j+=trans[n]
            data[f'{side}_vertices'][i]=v;data[f'{side}_joints'][i]=j;data[f'{side}_cam_t'][i]=trans[n]
            data[f'{side}_valid'][i]=np.isfinite(v).all() and (v[:,2]>.02).mean()>.9
            data[f'{side}_confidence'][i]=r['confidence']
            for dest,src in [('betas','betas'),('canonical_global_orient','global_orient'),('canonical_hand_pose','hand_pose')]:data[f'{side}_{dest}'][i]=pred['pred_mano_params'][src][n].cpu().numpy()
        if offset%80==0: print(f'{offset+len(chunk)}/{len(recs)} hand observations, {time.perf_counter()-start:.1f}s',flush=True)
    elapsed=time.perf_counter()-start
    name='smoke' if args.limit else ('wilor_own_detector' if args.records else 'wilor')
    np.savez_compressed(OUT/f'{name}_hands.npz',**data)
    report=dict(frames=N,observations=len(recs),load_seconds=load_seconds,inference_and_preprocessing_seconds=elapsed,forward_seconds=forward,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,torch=torch.__version__,gpu=torch.cuda.get_device_name(),batch_size=args.batch_size,K=K.tolist(),checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,missing_buffers=incompat.missing_keys,boxes=(str(args.records) if args.records else 'Cached HaWoR observed boxes; track-majority handedness; highest confidence if duplicate. No interpolation.'),geometry='Camera x-right y-down z-forward; left-hand x mirroring; translation from cam_crop_to_full with fitted focal. Internal model focal unchanged. Native MANO scale, no fitted scaling.',parameter_convention='canonical_* fields remain right-hand network coordinates, including for mirrored left output.',fast=False,records=recs)
    (OUT/f'{name}_provenance.json').write_text(json.dumps(report,indent=2))
    print('Saved',OUT/f'{name}_hands.npz',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,default=8);args=p.parse_args();OUT.mkdir(exist_ok=True);infer(args)
