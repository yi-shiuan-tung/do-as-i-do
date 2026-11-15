"""A rack must not hide a tube surface that is in front of it."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reconstruct_known_tube import depth_image


class TubeOcclusionTest(unittest.TestCase):
    def test_same_image_footprint_changes_visibility_with_depth(self):
        K=np.array([[1000.,0,640],[0,1000,360],[0,0,1]])
        rays=np.array([[-.1,-.1,1],[.1,-.1,1],[.1,.1,1],[-.1,.1,1]])
        faces=np.array([[0,1,2],[0,2,3]])
        rack=depth_image(rays,faces,K)
        front=depth_image(rays*.8,faces,K)
        behind=depth_image(rays*1.2,faces,K)
        self.assertTrue(front[180,320]<rack[180,320]+.002)
        self.assertFalse(behind[180,320]<rack[180,320]+.002)
        self.assertEqual(float(rack[0,0]),10000.)


if __name__=='__main__':unittest.main()
