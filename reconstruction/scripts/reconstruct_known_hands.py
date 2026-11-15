"""Run HaWoR/MANO afresh at the joint-fit focal length, preserving the baseline."""
import argparse,json,os,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'known_scene/pipette_g1_01/interactions'

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='3');a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    old=ROOT/'pipette_g1_01/pipette_g1_01_trimmed';seq=OUT/'input_demo';seq.mkdir(parents=True,exist_ok=True)
    video=OUT/'input_demo.mp4'
    if not video.exists():video.symlink_to(ROOT/'pipette_g1_01/pipette_g1_01_trimmed.mp4')
    frames=seq/'extracted_images'
    if not frames.exists():frames.symlink_to(old/'extracted_images',target_is_directory=True)
    tracks=seq/'tracks_0_817';tracks.mkdir(exist_ok=True)
    for name in ['model_boxes.npy','model_tracks.npy']:
        if not (tracks/name).exists():shutil.copy2(old/'tracks_0_817'/name,tracks/name)
    cam=json.loads((OUT.parent/'joint_fit/fit.json').read_text());f=cam['focal_px']
    module=ROOT/'modules/HaWoR';source=module/'demo.py'
    # Execute the existing demo's exact reconstruction/export block, omitting its
    # optional window/video rendering section. No vendor source is modified.
    code=source.read_text();marker='    # Here we use aitviewer'
    if marker not in code:raise RuntimeError('HaWoR demo changed; review export boundary')
    provenance=dict(camera=str(OUT.parent/'joint_fit/fit.json'),focal_px=f,gpu=a.gpu,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reuse='Detection tracks and extracted frames only; fresh motion estimation and infilling in separate output directory',camera_assumption='fixed centered pinhole; per-frame camera drift not modeled by HaWoR')
    (OUT/'hands_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    os.chdir(module);sys.path.insert(0,str(module));sys.argv=[str(source),'--video_path',str(video),'--img_focal',str(f),'--static_camera']
    exec(compile(code.split(marker)[0],str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
    shutil.copy2(seq/'all_hand_meshes.npz',OUT/'hands_mano.npz')
    print('Saved calibrated MANO hands:',OUT/'hands_mano.npz',flush=True)

if __name__=='__main__':main()
