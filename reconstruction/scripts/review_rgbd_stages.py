"""Export independent cached SAM3, object-pose and depth review videos (no inference)."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from rgbd_known_geometry import raster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    calibration = json.loads((run / 'calibration.json').read_text())
    width, height = calibration['width'], calibration['height']
    K = np.array(calibration['K'])
    out = run / 'stage_review'
    out.mkdir(exist_ok=True)
    objects = {}
    for name, color in [('pipette', (30, 190, 245)), ('tube', (50, 205, 70))]:
        mesh = trimesh.load(run / name / 'mesh_local.ply', process=False)
        poses = np.load(run / name / 'poses.npz')
        quality = json.loads((run / name / 'quality.json').read_text())['quality']
        objects[name] = (mesh, poses, quality, color)
    encoders = {}
    for name in ['sam3_masks', 'object_poses', 'masked_depth']:
        encoders[name] = subprocess.Popen([
            'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(calibration['fps']), '-i', '-',
            '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '20', '-pix_fmt', 'yuv420p',
            str(out / f'{name}.mp4')], stdin=subprocess.PIPE)
    review_frames = {0, 216, 217, 222, 223, 238, 439, 530, 531, 568, 569, 570, 769}
    rows = []
    try:
        for i in range(calibration['frames']):
            rgb = cv2.imread(str(run / f'rgb/{i:06d}.jpg'))
            masks_view, poses_view = rgb.copy(), rgb.copy()
            depth = cv2.imread(str(run / f'depth/{i:06d}.png'), -1) * calibration['depth_scale_m']
            depth_view = np.zeros_like(rgb)
            for name, (mesh, poses, quality, color) in objects.items():
                mask = cv2.imread(str(run / f'masks/{name}/{i:06d}.png'), 0) > 0
                masks_view[mask] = (.45 * masks_view[mask] + .55 * np.array(color)).astype(np.uint8)
                T = poses['T_camera_object'][i]
                vertices = mesh.vertices @ T[:3, :3].T + T[:3, 3]
                projected = np.isfinite(raster(vertices, mesh.faces, K, (height, width)))
                poses_view[projected] = (.45 * poses_view[projected] + .55 * np.array(color)).astype(np.uint8)
                contours, _ = cv2.findContours(projected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(poses_view, contours, -1, color, 1)
                depth_colors = cv2.applyColorMap(np.uint8(np.clip((depth - .15) / .85, 0, 1) * 255), cv2.COLORMAP_TURBO)
                depth_view[mask & (depth > 0)] = depth_colors[mask & (depth > 0)]
                depth_view[mask & (depth == 0)] = (255, 255, 255)
                previous = poses['T_camera_object'][max(0, i-1)]
                step = np.linalg.norm(T[:3, 3] - previous[:3, 3]) * 1000
                rotation = np.degrees(Rotation.from_matrix(previous[:3, :3].T @ T[:3, :3]).magnitude())
                rows.append(dict(frame=i, object=name, mask_pixels=int(mask.sum()),
                                 translation_step_mm=float(step), rotation_step_deg=float(rotation),
                                 z_m=float(T[2, 3]), **quality[i]))
            views = dict(sam3_masks=masks_view, object_poses=poses_view, masked_depth=depth_view)
            labels = {'sam3_masks': 'SAM3 masks | yellow pipette / green tube',
                      'object_poses': 'Fitted objects only | full silhouettes, no occlusion clipping',
                      'masked_depth': 'Measured depth inside SAM3 | 0.15-1.0m | white=invalid'}
            for name, im in views.items():
                cv2.rectangle(im, (0, 0), (width, 47), (0, 0, 0), -1)
                cv2.putText(im, labels[name], (8, 18), 0, .45, (255, 255, 255), 1)
                cv2.putText(im, f'Frame {i} | {poses["timestamps"][i]:.3f}s', (8, 38), 0, .45, (255, 255, 255), 1)
                encoders[name].stdin.write(im.tobytes())
            if i in review_frames:
                cv2.imwrite(str(out / f'frame_{i:06d}.jpg'), np.concatenate([rgb, masks_view, poses_view, depth_view], axis=1))
            if i % 100 == 0:
                print(f'Reviewed {i}/{calibration["frames"]}', flush=True)
    finally:
        for encoder in encoders.values():
            encoder.stdin.close()
        codes = [encoder.wait() for encoder in encoders.values()]
    if any(codes):
        raise RuntimeError(f'Video encoding failed: {codes}')
    with (out / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out, flush=True)


if __name__ == '__main__':
    main()
