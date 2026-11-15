"""Confidence-gated 2D motion initialization and soft SE(3) jump penalties."""
import cv2,numpy as np
from scipy.spatial.transform import Rotation

def select_motion(tracks,confidence,source_mask,target_mask,reverse=False):
 a=tracks[:,1 if reverse else 0].copy();b=tracks[:,0 if reverse else 1].copy();h,w=source_mask.shape
 good=np.isfinite(a).all(1)&np.isfinite(b).all(1)&(confidence.min(1)>.5)
 for pts,mask in [(a,source_mask),(b,target_mask)]:
  xy=np.rint(np.nan_to_num(pts)).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);good&=inside&mask[xy[:,1],xy[:,0]]
 a=a[good];b=b[good]
 if len(a)<6:return None,a,b
 cv2.setRNGSeed(0);A,inlier=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=1000,confidence=.99,refineIters=10)
 if A is None:return None,a,b
 keep=inlier.ravel().astype(bool);a=a[keep];b=b[keep];scale=np.linalg.norm(A[:,0]);angle=np.arctan2(A[1,0],A[0,0])
 if len(a)<6 or not .85<scale<1.18 or abs(angle)>np.deg2rad(35) or np.linalg.norm(np.median(b-a,axis=0))>100:return None,a,b
 return A,a,b

def pose_from_affine(previous,A,K,dz=0.):
 """Apply image motion about its fitted pivot, preserving unobserved 3D tilt."""
 x=previous.copy();z=np.clip(previous[5]+dz,.151,1.99);pixel=K[:2,2]+np.diag(K)[:2]*previous[3:5]/previous[5];target=A[:,:2]@pixel+A[:,2]
 x[3:5]=(target-K[:2,2])*z/np.diag(K)[:2];x[5]=z
 angle=np.arctan2(A[1,0],A[0,0]);x[:3]=(Rotation.from_rotvec([0,0,angle])*Rotation.from_rotvec(previous[:3])).as_rotvec()
 return x

def jump_residual(x,previous,config):
 angle=(Rotation.from_rotvec(previous[:3]).inv()*Rotation.from_rotvec(x[:3])).magnitude();shift=np.linalg.norm(x[3:]-previous[3:])
 return np.array([max(0.,angle-np.deg2rad(config['jump_rotation_deg']))/np.deg2rad(5)*5,max(0.,shift-config['jump_translation_m'])/.01*5])

def predict(previous,tracks,confidence,oldmask,newmask,olddepth,newdepth,meshdepth,K,config,reverse=False):
 A,a,b=select_motion(tracks,confidence,oldmask,newmask,reverse)
 if A is None:return previous.copy(),None,dict(status='hold_previous_unreliable_tracks',visible_points=len(a),anchor_points=0)
 xy=np.rint(a).astype(int);xy2=np.rint(b).astype(int);z=meshdepth[xy[:,1],xy[:,0]];za=olddepth[xy[:,1],xy[:,0]];zb=newdepth[xy2[:,1],xy2[:,0]]
 depthok=np.isfinite(z)&(za>.15)&(zb>.15)&(np.abs(za-z)<.025)&(np.abs(zb-za)<.04)
 dz=0.
 if depthok.sum()>=12:
  ds=zb[depthok]-za[depthok]
  if np.quantile(ds,.75)-np.quantile(ds,.25)<.015:dz=float(np.median(ds))
 seed=pose_from_affine(previous,A,K,dz);ok=np.isfinite(z)&(z>.15)&(z<2)
 anchors=None
 if ok.sum()>=6:
  cam=np.c_[(a[ok,0]-K[0,2])*z[ok]/K[0,0],(a[ok,1]-K[1,2])*z[ok]/K[1,1],z[ok]];R=Rotation.from_rotvec(previous[:3]).as_matrix();anchors=((cam-previous[3:])@R,b[ok])
 return seed,anchors,dict(status='tapir',visible_points=len(a),anchor_points=int(ok.sum()),depth_delta_m=dz,image_rotation_deg=float(np.degrees(np.arctan2(A[1,0],A[0,0]))),affine=A.tolist())

def robust_data_quadratic_jump_loss(z):
 """Soft-L1 data loss, quadratic jump terms located before six prior residuals."""
 rho=np.empty((3,len(z)));root=np.sqrt(1+z);rho[0]=2*(root-1);rho[1]=1/root;rho[2]=-.5/(root**3)
 rho[0,-8:-6]=z[-8:-6];rho[1,-8:-6]=1.;rho[2,-8:-6]=0.
 return rho
