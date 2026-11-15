"""Regression checks for joint camera/rack fitting with fixed metric geometry."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import joint_scene_fit as joint

class JointFitTest(unittest.TestCase):
    def test_recover_synthetic_camera_and_rigid_placement(self):
        sp=np.array([p[1] for p in joint.base.landmarks()],float)
        x=np.array([1.85,1.85,-.75,.15,.51,1.42,np.log(1000),-.02,.015,-.06])
        su,ru=joint.predictions(x,sp)
        seed=x.copy();seed[6]=np.log(850);seed[7:9]+=[.025,-.02];seed[9]+=.15
        result=joint.fit(seed,sp,su,ru)
        self.assertTrue(result['positive_depth']);self.assertLess(result['overall_rms_px'],1e-4)
        self.assertAlmostEqual(result['focal_px'],1000,places=2)
        np.testing.assert_allclose(joint.rack_transform(result['parameters']),joint.rack_transform(x),atol=1e-5)
    def test_rack_transform_preserves_distances(self):
        x=np.zeros(10);x[7:10]=[.15,-.2,1.1]
        transformed=joint.rack_points(x)
        np.testing.assert_allclose(np.linalg.norm(transformed[:,None]-transformed,axis=-1),np.linalg.norm(joint.RACK_POINTS[:,None]-joint.RACK_POINTS,axis=-1),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(joint.rack_transform(x)[:3,:3]),1.)

if __name__=='__main__':unittest.main()
