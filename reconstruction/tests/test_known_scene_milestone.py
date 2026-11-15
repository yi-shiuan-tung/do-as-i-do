"""Geometry checks for the camera fit and original-object-frame export."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

spec=importlib.util.spec_from_file_location('milestone',Path(__file__).parents[1]/'scripts/known_scene_milestone.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class GeometryTest(unittest.TestCase):
    def test_recover_focal_and_camera_from_nonplanar_landmarks(self):
        rng=np.random.default_rng(42);xyz=rng.uniform([-.3,-.3,0],[.3,.3,.4],(30,3))
        rv=np.array([.2,-.3,.1]);t=np.array([.03,-.02,1.4]);f=1100.
        uv=m.project(xyz,rv,t,f);fit=m.fit_camera(xyz,uv,seed_f=750)
        self.assertTrue(fit['positive_depth']);self.assertLess(fit['rms_px'],1e-5)
        self.assertAlmostEqual(fit['focal_px'],f,places=3)
        np.testing.assert_allclose(fit['T_camera_world'][:3],np.c_[Rotation.from_rotvec(rv).as_matrix(),t],atol=1e-6)
    def test_original_mesh_origin_conversion_preserves_projection(self):
        v=np.array([[.01,.02,-.15],[-.02,.01,.10]]);c=np.array([.003,-.01,-.025]);R=Rotation.from_rotvec([.2,.4,-.1]).as_matrix();t=np.array([.1,.05,.7])
        np.testing.assert_allclose((v-c)@R.T+t,v@R.T+(t-R@c),atol=1e-12)

if __name__=='__main__':unittest.main()
