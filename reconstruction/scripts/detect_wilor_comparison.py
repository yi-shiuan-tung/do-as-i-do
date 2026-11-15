"""WiLoR demo detector pass, separate from reconstruction for timing/coverage."""
import os,json,time,hashlib
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
import cv2,torch,ultralytics
from ultralytics import YOLO
from compare_wilor import ROOT,RUN,OUT
checkpoint=ROOT/'modules/WiLoR/pretrained_models/detector.pt'
model=YOLO(str(checkpoint)).to('cuda')
records=[];start=time.perf_counter()
for i,p in enumerate(sorted((RUN/'all_frames').glob('*.png'))):
    prediction=model(cv2.imread(str(p)),conf=.3,verbose=False)[0]
    for side in [0,1]:
        boxes=prediction.boxes
        chosen=(boxes.cls==side).nonzero().reshape(-1)
        if not len(chosen):continue
        ix=int(chosen[boxes.conf[chosen].argmax()])
        records.append(dict(frame=i,side=side,box=boxes.xyxy[ix].cpu().tolist(),confidence=float(boxes.conf[ix]),track='independent-image-detection'))
    if i%100==0:print('Detected frame',i,flush=True)
torch.cuda.synchronize()
(OUT/'own_detector_records.json').write_text(json.dumps(records))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'own_detector_provenance.json').write_text(json.dumps(dict(confidence=.3,ultralytics=ultralytics.__version__,seconds=time.perf_counter()-start,frames=i+1,observations=len(records),checkpoint=str(checkpoint),sha256=sha(checkpoint),hawor_detector_sha256=sha(ROOT/'modules/HaWoR/weights/external/detector.pt'),selection='Highest confidence per side per frame; matches WiLoR image-demo inference, no tracker. Detector executed in existing HaWoR environment for modern Ultralytics checkpoint-loader compatibility.'),indent=2))
