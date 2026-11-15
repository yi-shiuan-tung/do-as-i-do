"""Plot per-frame agreement and motion for baseline versus TAPIR refinement."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def plot(run):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 baseline=Path(json.loads((run/'tracking_config.json').read_text())['baseline_run'])
 fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
 for row,obj in enumerate(['pipette','tube']):
  for label,folder,color in [('Baseline',baseline,'#999999'),('TAPIR + depth + jump penalties',run,'#1863ab')]:
   d=np.load(folder/obj/'poses.npz');T=d['T_camera_object'];step=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)*1000;rot=np.degrees((Rotation.from_matrix(T[:-1,:3,:3]).inv()*Rotation.from_matrix(T[1:,:3,:3])).magnitude())
   for col,data in enumerate([d['image_iou'],step,rot]):axes[row,col].plot(np.arange(len(data))+(col>0),data,label=label,color=color,lw=.9,alpha=.85)
  for col,title in enumerate(['Silhouette IoU','Translation step (mm/frame)','Rotation step (degrees/frame)']):
   axes[row,col].set_title(obj+': '+title);axes[row,col].set_xlabel('Frame');axes[row,col].grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('Same RGB-D observations and reference pose; combined tracking changes')
 fig.savefig(run/'comparison.png',dpi=140);plt.close(fig)
