import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rgbd_tapir_motion import pose_from_affine,select_motion,jump_residual,predict,robust_data_quadratic_jump_loss
class MotionTest(unittest.TestCase):
 def test_image_rotation_pivot_and_reverse(self):
  K=np.array([[600.,0,320],[0,600,240],[0,0,1]]);old=np.array([.2,-.1,.05,.1,.02,.5]);angle=.04;R=Rotation.from_euler('z',angle).as_matrix()[:2,:2];pivot=np.array([430,200]);A=np.c_[R,pivot-R@pivot+np.array([3.,-2.])]
  new=pose_from_affine(old,A,K);src=K[:2,2]+600*old[3:5]/old[5];dst=K[:2,2]+600*new[3:5]/new[5];np.testing.assert_allclose(dst,A[:,:2]@src+A[:,2])
  inv=np.linalg.inv(np.vstack([A,[0,0,1]]))[:2];back=pose_from_affine(new,inv,K);np.testing.assert_allclose(back,old,atol=1e-12)
 def test_confidence_masks_outliers_and_backward(self):
  a=np.array([[30,30],[60,30],[90,30],[30,60],[60,60],[90,60],[30,90],[60,90],[90,90]],float);b=a+[3,-2];b[-1]=[180,180];tr=np.stack([a,b],axis=1);cf=np.ones((9,2))*.9;mask=np.ones((200,200),bool)
  A,x,y=select_motion(tr,cf,mask,mask);np.testing.assert_allclose(A,np.array([[1,0,3],[0,1,-2]]),atol=1e-5);self.assertEqual(len(x),8)
  B,_,_=select_motion(tr,cf,mask,mask,True);np.testing.assert_allclose(B,np.array([[1,0,-3],[0,1,2]]),atol=1e-5)
  cf[:]=.1;self.assertIsNone(select_motion(tr,cf,mask,mask)[0])
 def test_mesh_anchors_follow_tracked_translation(self):
  K=np.array([[600.,0,100],[0,600,100],[0,0,1]]);old=np.array([0.,0,0,0,0,.5]);a=np.array([[x,y] for y in [60,80,100,120] for x in [60,80,100,120]],float);b=a+[6,0]
  tracks=np.stack([a,b],1);cf=np.full((16,2),.9);mask=np.ones((200,200),bool);depth=np.full((200,200),.5)
  seed,anchors,info=predict(old,tracks,cf,mask,mask,depth,depth,depth,K,{})
  self.assertEqual(info['status'],'tapir');np.testing.assert_allclose(seed[3:],[.005,0,.5],atol=1e-8)
  local,target=anchors;xyz=local+seed[3:];q=xyz@K.T;np.testing.assert_allclose(q[:,:2]/q[:,2:],target,atol=1e-5)
  reverse,_,_=predict(seed,tracks,cf,mask,mask,depth,depth,depth,K,{},True);np.testing.assert_allclose(reverse,old,atol=1e-8)
 def test_jump_terms_keep_quadratic_growth(self):
  z=np.full(30,100.);rho=robust_data_quadratic_jump_loss(z)
  np.testing.assert_equal(rho[0,-8:-6],100);np.testing.assert_equal(rho[1,-8:-6],1);np.testing.assert_equal(rho[2,-8:-6],0)
  self.assertLess(rho[0,0],20)
  eps=1e-4;plus=robust_data_quadratic_jump_loss(z+eps);minus=robust_data_quadratic_jump_loss(z-eps)
  np.testing.assert_allclose((plus[0]-minus[0])/(2*eps),rho[1],rtol=1e-6)
 def test_jump_penalty(self):
  cfg=dict(jump_rotation_deg=12,jump_translation_m=.02);old=np.array([0.,0,0,0,0,.5]);x=old.copy();x[3]=.01;np.testing.assert_equal(jump_residual(x,old,cfg),[0,0]);x[3]=.08;x[0]=.5;self.assertTrue((jump_residual(x,old,cfg)>0).all())
if __name__=='__main__':unittest.main()
