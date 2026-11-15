"""Write per-recording RGB-D reconstruction provenance and an HTML review report."""
import hashlib,html,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def report(run,C,validation):
 bag=Path(C['source']);sha=hashlib.sha256()
 with bag.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block)
 prompts=json.loads((run/'segmentation_prompts.json').read_text());rows=[];issues=[]
 config=json.loads((run/'tracking_config.json').read_text()) if (run/'tracking_config.json').exists() else {}
 tapir=config.get('mode')=='tapir'

 for name in ['pipette','tube']:
  quality=json.loads((run/name/'quality.json').read_text());v=validation['objects'][name]
  errors=[q['depth_error_m'] for q in quality['quality'] if q['depth_error_m'] is not None]
  rows.append(f'<tr><td>{name}</td><td>{quality["frames"]}</td><td>{quality["valid_frames"]}</td><td>{quality["iou_median"]:.3f}</td><td>{1000*np.median(errors):.2f} mm</td></tr>')
  issues.append(f'<li>{name}: translation jumps &gt;5 cm at frames {v["large_translation_jump_frames"]}; rotation jumps &gt;30° at frames {v["large_rotation_jump_frames"]}. Maximum translation step {1000*v["translation_step_m_quantiles"][-1]:.1f} mm.</li>')
 provenance=dict(source=str(bag),source_sha256=sha.hexdigest(),run=run.name,reference_frame=prompts['frame'],right_hand_object='pipette',left_hand_object='tube',scripts={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in ['extract_rgbd_bag.py','reconstruct_rgbd_known.py','rgbd_known_geometry.py','report_rgbd_known.py']},methods=dict(camera='recorded RGB calibration, no scene fit',depth='RealSense color-aligned measured depth; zero invalid; robust foreground-depth selection',objects='fixed metric meshes; rigid silhouette/depth fit with temporal regularization',hands='HaWoR on resampled RGB; same camera axes; no measured-depth or contact correction',motion='camera-space; no world-camera trajectory or TAPIR',contact='not enforced'))
 if tapir:
  provenance['tracking_config']=config
  provenance['tapir']=json.loads((run/'tapir/provenance.json').read_text())
  provenance['methods']['motion']='BootsTAPIR confidence-gated point motion initializes image-plane pose; mesh-anchored track reprojection, depth and silhouette refine full 6DoF pose'
  provenance['methods']['temporal']='Penalize corrections from TAPIR prediction and soft translation/rotation jumps; no hard velocity clamp'
  for filename in ['track_rgbd_tapir.py','rgbd_tapir_motion.py']:
   provenance['scripts'][filename]=hashlib.sha256((ROOT/'scripts'/filename).read_bytes()).hexdigest()
 (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
 title=html.escape(run.name);c=C['K'];summary=f'{C["frames"]} synchronized RGB-D frames, {C["width"]} × {C["height"]}, {validation["duration_s"]:.2f} seconds.'
 body=f'''<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>body{{max-width:1100px;margin:40px auto;font:17px/1.5 system-ui;color:#222}}video,img{{width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;background:#eee;padding:15px}}</style></head><body><h1>{title}: RGB-D known-mesh reconstruction</h1><p>{summary} Right-hand object: fixed 254 mm pipette mesh. Left-hand object: fixed 137.5 mm tube mesh. Both hands estimated with HaWoR.</p><video controls src="reconstruction.mp4"></video><p>Yellow: pipette. Green: tube. Purple: left hand. Blue: right hand. Second panel: oblique camera-space view. Triangle rendering is approximate.</p><h2>Calibration and depth</h2><p>Recorded intrinsics: fx {c[0][0]:.2f}, fy {c[1][1]:.2f}, cx {c[0][2]:.2f}, cy {c[1][2]:.2f}. RealSense depth-to-color alignment; depth unit {C['depth_scale_m']:.6f} metres. Maximum paired timestamp offset: {validation['depth_color_offset_ms_max']:.3f} ms. Original timestamps retained. No scene-based camera fit or generated object geometry.</p><h2>Fit measurements</h2><table><tr><th>Object</th><th>Frames</th><th>Silhouette passes</th><th>Median IoU</th><th>Median selected-depth residual</th></tr>{''.join(rows)}</table><p>The valid flag means silhouette IoU ≥0.4 only. These are agreement measurements, not ground-truth accuracy or physical validation.</p><h2>Review flags</h2><ul>{''.join(issues)}<li>Transparent tube and thin pipette sections can return background depth. Rejected depth does not provide geometric evidence.</li><li>Axial rotation is weakly observed, especially for the tube. Geometry is rigid: no plunger articulation.</li><li>Hands use RGB inference without depth/contact correction. Gaps and intersections may remain. Observed hand visibility: {validation['hands_visible']}. Raw nonfinite hand-frame counts: {validation['raw_nonfinite_hand_frames']}; these are excluded and zeroed only in the display archive.</li><li>No physical contact, collision constraints, or world-camera tracking.</li></ul><h2>Artifacts</h2><ul><li><a href="pipette/poses.npz">Pipette poses</a> and <a href="pipette/mesh_local.ply">mesh</a></li><li><a href="tube/poses.npz">Tube poses</a> and <a href="tube/mesh_local.ply">mesh</a></li><li><a href="hands_mano.npz">Raw hand meshes</a> and <a href="hands_visible.npz">hand meshes with visibility flags</a></li><li><a href="calibration.json">Calibration/timestamps</a>, <a href="validation.json">validation</a>, <a href="provenance.json">provenance</a>, <a href="segmentation_prompts.json">recording-specific prompts</a></li></ul><p>T_camera_object maps mesh_local.ply into camera coordinates. T_camera_original_mesh maps the original source asset. Units: metres; axes x right, y down, z forward. Mesh scale remains 1.</p><h2>Viewer</h2><pre>cd {html.escape(str(run))}
~/miniconda3/envs/sam3d-objects/bin/python view.py --port 8082</pre><p>Forward port 8082 over SSH to view remotely. Run <code>bash run_reconstruction.sh</code> to reproduce this recording. Existing complete segmentation and hand caches are reused; object fitting runs again.</p><h2>Sample frames</h2><img src="contact_sheet.jpg"></body></html>'''
 if tapir:
  from scipy.spatial.transform import Rotation
  comparison={};comparison_rows=[]
  for obj in ['pipette','tube']:
   results={}
   for label,folder in [('baseline',Path(config['baseline_run'])),('tapir',run)]:
    q=json.loads((folder/obj/'quality.json').read_text());T=np.load(folder/obj/'poses.npz')['T_camera_object'];step=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);angles=np.degrees((Rotation.from_matrix(T[:-1,:3,:3]).inv()*Rotation.from_matrix(T[1:,:3,:3])).magnitude())
    errors=[row['depth_error_m'] for row in q['quality'] if row['depth_error_m'] is not None]
    results[label]=dict(median_iou=q['iou_median'],median_depth_residual_mm=float(np.median(errors)*1000),max_translation_step_mm=float(step.max()*1000),p95_translation_step_mm=float(np.quantile(step,.95)*1000),max_rotation_step_deg=float(angles.max()),p95_rotation_step_deg=float(np.quantile(angles,.95)),translation_jumps_over_50mm=int((step>.05).sum()),rotation_jumps_over_30deg=int((angles>30).sum()))
    if label=='tapir':
     statuses=[row.get('motion_prediction',{}).get('status') for row in q['quality']];results[label]['tapir_supported_transitions']=statuses.count('tapir');results[label]['held_previous_transitions']=statuses.count('hold_previous_unreliable_tracks')
    r=results[label];comparison_rows.append(f'<tr><td>{obj}</td><td>{label}</td><td>{r["median_iou"]:.3f}</td><td>{r["median_depth_residual_mm"]:.2f}</td><td>{r["max_translation_step_mm"]:.1f}</td><td>{r["max_rotation_step_deg"]:.1f}</td></tr>')
   comparison[obj]=results
  (run/'comparison.json').write_text(json.dumps(dict(method='Combined TAPIR initialization, correspondence residual, and stronger temporal penalties compared with prior mask-centroid baseline; not an isolated TAPIR ablation. Same meshes, masks, depth, hands, and reference pose.',objects=comparison),indent=2))
  extra='<h2>TAPIR refinement compared with baseline</h2><p>TAPIR predicts 2D point motion. A robust image similarity transform initializes translation and image-plane rotation; reliable paired depth can update depth. Full 3D pose is then refined using measured depth, masks, mesh-anchored point correspondences, and soft jump penalties. Unreliable tracks fall back to the previous pose. Soft jump thresholds: 20 mm and 12 degrees per frame at approximately 30 fps. Tube jump penalties retain quadratic growth; noisy image/depth residuals use a robust loss.</p><table><tr><th>Object</th><th>Method</th><th>Median IoU</th><th>Depth residual mm</th><th>Max step mm</th><th>Max rotation degrees</th></tr>'+''.join(comparison_rows)+'</table><p>These results compare the combined tracking changes, not TAPIR alone. Smoother trajectories do not establish better physical accuracy. RGB, depth, SAM3 masks, HaWoR hands and reference poses were reused unchanged. See <a href="comparison.json">comparison.json</a> and <a href="tracking_config.json">tracking_config.json</a>.</p>'
  from plot_rgbd_comparison import plot
  plot(run)
  extra+='<img src="comparison.png" alt="Per-frame baseline versus TAPIR fit and motion comparison">'
  body=body.replace('<h2>Fit measurements</h2>',extra+'<h2>Fit measurements</h2>')
 (run/'report.html').write_text(body)
 script='''#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECON_DIR="$(cd "$RUN_DIR/../.." && pwd)"
ENV_ROOT="${RECON_ENV_ROOT:-$HOME/miniconda3/envs}"
RECON_GPU="${RECON_GPU:-6}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
"$ENV_ROOT/sam3/bin/python" "$RECON_DIR/scripts/reconstruct_rgbd_known.py" segment --run "$RUN_DIR" --gpu "$RECON_GPU"
"$ENV_ROOT/hawor/bin/python" "$RECON_DIR/scripts/reconstruct_rgbd_known.py" hands --run "$RUN_DIR" --gpu "$RECON_GPU"
for object in pipette tube; do
 "$ENV_ROOT/sam3d-objects/bin/python" "$RECON_DIR/scripts/reconstruct_rgbd_known.py" fit --run "$RUN_DIR" --object "$object"
done
"$ENV_ROOT/sam3d-objects/bin/python" "$RECON_DIR/scripts/reconstruct_rgbd_known.py" package --run "$RUN_DIR"
'''
 if tapir:
  start=script.index('"$ENV_ROOT/sam3/bin/python"');end=script.index('for object in pipette tube; do')
  script=script[:start]+'"$ENV_ROOT/tapnet/bin/python" "$RECON_DIR/scripts/track_rgbd_tapir.py" --run "$RUN_DIR" --gpu "$RECON_GPU"\n'+script[end:]
 (run/'run_reconstruction.sh').write_text(script)
