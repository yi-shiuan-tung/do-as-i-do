"""Build a local HTML report from completed comparison artifacts."""
import html,json
from compare_wilor import OUT

def main():
    a=json.loads((OUT/'comparison_metrics.json').read_text());b=json.loads((OUT/'own_detector/comparison_metrics.json').read_text())
    det=json.loads((OUT/'own_detector_provenance.json').read_text());dc=json.loads((OUT/'detection_comparison.json').read_text())
    rows=[]
    specs=[('Visible left frames','left','valid_frames'),('Visible right frames','right','valid_frames'),
           ('Left finger-motion second difference (mm/frame²)','left','root_relative_joint_second_difference_mm_per_frame2'),
           ('Right finger-motion second difference (mm/frame²)','right','root_relative_joint_second_difference_mm_per_frame2'),
           ('Left wrist second difference (mm/frame²)','left','wrist_second_difference_mm_per_frame2'),
           ('Right wrist second difference (mm/frame²)','right','wrist_second_difference_mm_per_frame2'),
           ('Right hand–pipette nearest distance (mm)','right','object_nearest_distance_mm')]
    for label,side,metric in specs:
        vals=[a['results']['hawor_'+side][metric],a['results']['wilor_'+side][metric],b['results']['wilor_'+side][metric]]
        vals=[v['median'] if isinstance(v,dict) else v for v in vals]
        rows.append('<tr><td>'+label+'</td>'+''.join(f'<td>{v:.3f}</td>' if isinstance(v,float) else f'<td>{v}</td>' for v in vals)+'</tr>')
    vals=[a['results']['hawor_right_silhouette_iou']['median'],a['results']['wilor_right_silhouette_iou']['median'],b['results']['wilor_right_silhouette_iou']['median']]
    rows.insert(2,'<tr><td>Right-hand silhouette IoU (higher is better)</td>'+''.join(f'<td>{v:.3f}</td>' for v in vals)+'</tr>')
    timing=a['wilor_runtime'];own=b['wilor_runtime'];limitations=''.join('<li>'+html.escape(s)+'</li>' for s in a['limitations'])
    page=f'''<!doctype html><meta charset="utf-8"><title>HaWoR versus WiLoR — pipette</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#18212d;background:#fafafa}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}video,img{{width:100%}}li{{margin:10px 0}}code{{background:#eee;padding:2px 4px}}</style>
<h1>HaWoR versus WiLoR</h1><p>817 frames, 1280×720, 30 FPS. Original calibrated HaWoR output preserved. WiLoR uses full-precision inference without smoothing, gap filling, fast mode, or fitted hand scale.</p>
<p><strong>Finding:</strong> HaWoR has stronger right-hand silhouette agreement and lower median finger-motion variation on this sequence. WiLoR's right hand is closer to the fixed pipette surface, which alone does not establish more accurate contact. Neither method is validated against measured 3D hands.</p>
<h2>Comparison video</h2><video controls preload="metadata" src="comparison.mp4"></video><p>Left column: HaWoR. Right: WiLoR with matched detections. Top: camera overlays. Bottom: illustrative oblique views with the same fixed pipette point cloud. Purple is left hand; blue is right hand. Bottom panels are resized for layout and should not be used for geometric measurements. Invisible/unobserved hands are omitted.</p>
<p><a href="comparison.mp4">Download video</a> · <a href="contact_sheet.jpg">Contact sheet</a></p>
<h2>Metrics</h2><p>Geometric metrics use shared valid frames: 689 left and 677 right for the matched-box test. Medians shown. The own-detector variant uses its own shared-frame intersection. Second differences measure motion variation, not ground-truth error; smoother is not always more accurate.</p>
<table><tr><th>Metric</th><th>HaWoR baseline</th><th>WiLoR, matched boxes</th><th>WiLoR, own detections</th></tr>{''.join(rows)}</table>
<h2>What the detector comparison means</h2><p>The two detector checkpoint SHA-256 hashes match. HaWoR's saved detections came from tracking at threshold 0.2; the WiLoR-style fresh pass uses independent images at threshold 0.3 and highest-confidence selection per hand. Both found {dc['matched']} observations; {dc['shared']} frame/hand keys overlap. Median maximum-coordinate box difference is {dc['box_abs_difference_px_median']:.2f} px. The fresh detector used Ultralytics {det['ultralytics']} in the existing HaWoR environment, because WiLoR's installed older loader failed under PyTorch 2.9. WiLoR reconstruction itself ran in the wilor environment.</p>
<h2>Runtime on one B200</h2><ul><li>Matched-box preprocessing plus reconstruction: {timing['inference_and_preprocessing_seconds']:.1f} s; model forward calls alone: {timing['forward_seconds']:.1f} s; loading: {timing['load_seconds']:.1f} s.</li><li>Own-detector pass: {det['seconds']:.1f} s; subsequent preprocessing/reconstruction: {own['inference_and_preprocessing_seconds']:.1f} s (loading and rendering excluded).</li><li>WiLoR peak PyTorch allocated VRAM: {timing['peak_allocated_GiB']:.2f} GiB; reserved: {timing['peak_reserved_GiB']:.2f} GiB. Batch size 8. This excludes the separate detector process.</li><li>No matching HaWoR runtime measurement was available, so this is not a speed benchmark between models.</li></ul>
<h2>Files and reproduction</h2><ul><li><a href="wilor_hands.npz">Matched-box WiLoR archive</a></li><li><a href="wilor_own_detector_hands.npz">Own-detector WiLoR archive</a></li><li><a href="comparison_metrics.json">Matched metrics and limitations</a></li><li><a href="own_detector/comparison_metrics.json">Own-detector metrics</a></li><li><a href="validation.json">Archive validation</a></li></ul>
<p>Scripts under reconstruction/scripts: <code>compare_wilor.py</code>, <code>detect_wilor_comparison.py</code>, <code>evaluate_wilor_comparison.py</code>, <code>report_wilor_comparison.py</code>. The adapter exports camera-frame vertices/joints and valid flags; canonical MANO parameters remain explicitly labeled and are not interchangeable with HaWoR parameter fields.</p>
<h2>Interpretation limits</h2><ul>{limitations}</ul>'''
    (OUT/'report.html').write_text(page)
    print('Report:',OUT/'report.html')

if __name__=='__main__':main()
