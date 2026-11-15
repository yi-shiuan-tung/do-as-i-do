"""Regression: a skipped frame must share the fitted frame's reference units."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import trimesh

spec=importlib.util.spec_from_file_location('opt',Path(__file__).parents[1]/'scripts/optimize_translation_scale.py')
opt=importlib.util.module_from_spec(spec);spec.loader.exec_module(opt)

class OutputScaleTest(unittest.TestCase):
    def test_main_valid_and_skipped_frames_share_reference_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);mesh=p/'mesh.obj';mesh.touch();layout=p/'layout.json'
            original={'objects':[{'frame_idx':i,'local_to_scene':{'translation_camera_frame':[0.,0.,1.], 'quat_wxyz_camera_frame':[1.,0.,0.,0.], 'scale':[1.,1.,1.]}} for i in range(2)]}
            layout.write_text(json.dumps(original))
            hand=p/'hands.npz';np.savez(hand,right_vertices=np.zeros((2,12,3)),right_faces=np.array([[0,1,2]]))
            args=SimpleNamespace(layout_json=str(layout),mesh=str(mesh),pointmap_dir=tmp,mask_dir=tmp,hand_meshes=str(hand),anchor_hand='right',viz_dir=None,scale=1.,ref_frame=0,mask_name='object',min_mask_pixels=100,output=str(p/'out.json'))
            vertices=np.c_[np.linspace(-.001,.001,12),np.zeros((12,2))]
            tm=trimesh.Trimesh(vertices=vertices,faces=[[0,1,2]],process=False)
            pm=np.zeros((10,10,3));pm[:,:,2]=4.;K=np.array([[100.,0,5],[0,100.,5],[0,0,1.]])
            mask=np.ones((10,10),bool)
            def get_mask(directory,frame,name):return None if frame==1 and name=='object' else mask
            hits=np.tile([0.,0.,2.],(20,1));pixels=np.zeros(20,dtype=int)
            with patch.object(opt,'parse_args',return_value=args),patch.object(opt.trimesh,'load_mesh',return_value=tm),patch.object(opt,'load_pointmap_and_intrinsics',return_value=(pm,K)),patch.object(opt,'load_mask',side_effect=get_mask),patch.object(opt,'raycast_first_hits',return_value=(hits,pixels,pixels)),contextlib.redirect_stdout(io.StringIO()):opt.main()
            result=json.loads((p/'out.json').read_text());poses=[x['local_to_scene'] for x in result['objects']]
            self.assertAlmostEqual(result['translation_scale_optimization']['ref_frame_k'],.5)
            np.testing.assert_allclose(poses[0]['translation_camera_frame'],[0,0,2.])
            self.assertEqual(poses[0]['translation_optimization_status'],'optimized')
            np.testing.assert_allclose(poses[1]['translation_camera_frame'],[0,0,.5])
            self.assertEqual(poses[1]['translation_optimization_reason'],'missing_object_mask')
            self.assertEqual(json.loads(layout.read_text()),original)

if __name__=='__main__':unittest.main()
