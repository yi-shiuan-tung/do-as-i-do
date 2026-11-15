"""Export RealSense RGB and color-aligned uint16 depth, retaining calibration/timestamps."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,pyrealsense2 as rs
p=argparse.ArgumentParser();p.add_argument('bag',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
for d in ['rgb','depth']: (a.out/d).mkdir(exist_ok=True)
pipe=rs.pipeline();cfg=rs.config();cfg.enable_device_from_file(str(a.bag.resolve()),repeat_playback=False)
profile=pipe.start(cfg);play=profile.get_device().as_playback();play.set_real_time(False)
streams=[]
for s in profile.get_streams():
    v=s.as_video_stream_profile();intr=v.get_intrinsics()
    streams.append(dict(stream=str(s.stream_type()),format=str(s.format()),fps=s.fps(),width=intr.width,height=intr.height,fx=intr.fx,fy=intr.fy,ppx=intr.ppx,ppy=intr.ppy,model=str(intr.model),coeffs=intr.coeffs))
scale=profile.get_device().first_depth_sensor().get_depth_scale();align=rs.align(rs.stream.color)
rows=[];last=None;writer=None
try:
 while True:
    try: fs=pipe.wait_for_frames(5000)
    except RuntimeError:
        if play.current_status()==rs.playback_status.stopped: break
        raise
    c0=fs.get_color_frame();d0=fs.get_depth_frame()
    if not c0 or not d0 or c0.get_frame_number()==last:continue
    last=c0.get_frame_number();fs=align.process(fs);c=fs.get_color_frame();d=fs.get_depth_frame()
    rgb=np.asanyarray(c.get_data());dep=np.asanyarray(d.get_data())
    fmt=c.profile.format()
    if fmt==rs.format.rgb8: rgb=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    elif fmt!=rs.format.bgr8:raise RuntimeError(str(fmt))
    intr=c.profile.as_video_stream_profile().get_intrinsics()
    if writer is None:
        fps=c.profile.fps();writer=cv2.VideoWriter(str(a.out/'rgb.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(intr.width,intr.height))
        assert writer.isOpened()
        meta=dict(source=str(a.bag.resolve()),streams=streams,depth_scale_m=scale,K=[[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]],distortion_model=str(intr.model),distortion_coefficients=intr.coeffs,width=intr.width,height=intr.height,fps=fps,depth_alignment='rs.align(color); zero means invalid',depth_to_color=dict(rotation=d0.profile.get_extrinsics_to(c0.profile).rotation,translation=d0.profile.get_extrinsics_to(c0.profile).translation))
    i=len(rows);cv2.imwrite(str(a.out/f'rgb/{i:06d}.jpg'),rgb,[cv2.IMWRITE_JPEG_QUALITY,95]);cv2.imwrite(str(a.out/f'depth/{i:06d}.png'),dep);writer.write(rgb)
    rows.append(dict(frame=i,color_timestamp_ms=c0.get_timestamp(),depth_timestamp_ms=d0.get_timestamp(),color_frame=c0.get_frame_number(),depth_frame=d0.get_frame_number()))
    if i%100==0:print('exported',i,flush=True)
finally:
 pipe.stop()
 if writer:writer.release()
meta['frames']=len(rows);meta['timestamps']=rows
(a.out/'calibration.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='timestamps'},indent=2))
