"""Joint camera + rigid rack registration, with immutable metric mesh geometry."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np,trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import known_scene_milestone as base

OUT=base.OUT/'joint_fit';OUT.mkdir(exist_ok=True)
ASSETS=base.OUT/'assets'
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

# Blue top plate, measured from the unchanged source geometry. Corner features are
# not the footprint bounding box; the plate is ~60 mm above the support surface.
RACK_POINTS=np.array([[.0668375,-.241366,.77996],[.0668375,-.138634,.77996],[.273162484,-.138634,.77996],[.273162484,-.241366,.77996]])
RACK_UV=np.array([[466.,430.],[590.,431.],[559.,674.],[414.,669.]])
# Visible large-hole centers, held out from fitting. Approximate image ellipse
# centers; source centers measured from an orthographic projection of the mesh.
HOLE_POINTS=np.array([[.097154,-.20945,.77996],[.097154,-.170574,.77996],
    [.13359,-.20945,.77996],[.13359,-.170574,.77996],
    [.170024,-.20945,.77996],[.206467,-.20945,.77996],
    [.242898,-.20945,.77996],[.242898,-.170574,.77996]])
HOLE_UV=np.array([[505.,460.],[550.,461.],[497.,497.],[544.,500.],
    [488.,537.],[478.,580.],[466.,625.],[521.,633.]])
# Held-out white support TOP corners; partly cropped/rounded, hence approximate.
WHITE_POINTS=np.array([[.040860504,-.262470991,.78175],[.040860504,-.117529,.78175],[.2991395,-.117529,.78175],[.2991395,-.262470991,.78175]])
WHITE_UV=np.array([[445.,420.],[615.,428.],[590.,711.],[384.,680.]])
CENTER=np.array([.17,-.19,.752507478])
DZ=.72-.7218999865

def rack_transform(x):
    R=Rotation.from_euler('z',x[9]).as_matrix();T=np.eye(4);T[:3,:3]=R;T[:3,3]=CENTER-R@CENTER+np.r_[x[7:9],DZ]
    return T

def rack_points(x,points=RACK_POINTS):
    T=rack_transform(x);return points@T[:3,:3].T+T[:3,3]

def predictions(x,stand_points,rack_pts=RACK_POINTS):
    f=np.exp(x[6]);return base.project(stand_points,x[:3],x[3:6],f),base.project(rack_points(x,rack_pts),x[:3],x[3:6],f)

def fit(x0,sp,su,ru=RACK_UV,fixed_f=None,max_nfev=800):
    active=np.arange(10) if fixed_f is None else np.array([0,1,2,3,4,5,7,8,9])
    template=x0.copy()
    if fixed_f is not None:template[6]=np.log(fixed_f)
    def unpack(y):
        x=template.copy();x[active]=y;return x
    def residual(y):
        x=unpack(y);a,b=predictions(x,sp)
        return np.r_[(a-su).ravel(),(b-ru).ravel()]
    lo=np.r_[np.full(6,-np.inf),np.log(300),-1,-1,-4*np.pi]
    hi=np.r_[np.full(6,np.inf),np.log(4000),1,1,4*np.pi]
    opt=least_squares(residual,template[active],bounds=(lo[active],hi[active]),loss='soft_l1',f_scale=4,max_nfev=max_nfev)
    x=unpack(opt.x);a,b=predictions(x,sp);se=np.linalg.norm(a-su,axis=1);re=np.linalg.norm(b-ru,axis=1)
    R=Rotation.from_rotvec(x[:3]).as_matrix();points=np.r_[sp,rack_points(x)];depth=(points@R.T+x[3:6])[:,2]
    return dict(parameters=x.tolist(),focal_px=float(np.exp(x[6])),stand_rms_px=float(np.sqrt(np.mean(se**2))),rack_rms_px=float(np.sqrt(np.mean(re**2))),overall_rms_px=float(np.sqrt(np.mean(np.r_[se,re]**2))),stand_errors_px=se.tolist(),rack_errors_px=re.tolist(),cost=float(opt.cost),positive_depth=bool(np.all(depth>0)),nfev=opt.nfev)

def main():
    lm=base.landmarks();sp=np.array([v[1] for v in lm],float);su=np.array([v[2] for v in lm],float)
    files=[ASSETS/'env_00_cylinder_base.ply',ASSETS/'env_06_pipette_stand.ply'];hashes={str(f):sha(f) for f in files}
    write('annotations.json',dict(frame=800,provenance='Assistant approximate manual annotations. Stand points retained from first milestone. Rack correspondences corrected to blue top-plate corners of env_00_cylinder_base; source dimensions unchanged.',stand_landmarks=[dict(name=v[0],world_m=v[1],pixel=v[2]) for v in lm],rack_top_plate_source_m=RACK_POINTS.tolist(),rack_top_plate_image_px=RACK_UV.tolist(),held_out_white_support_top_source_m=WHITE_POINTS.tolist(),held_out_white_support_top_image_px=WHITE_UV.tolist()))
    starts=[]
    for f in [400,600,800,1000,1400,2000,3000]:
        c=base.fit_camera(sp,su,fixed_f=f)
        for yaw in [-np.pi/2,0,np.pi/2,np.pi]:
            x=np.r_[c['rvec'],c['tvec'],np.log(f),0,0,yaw]
            result=fit(x,sp,su);result['initial_focal_px']=f;result['initial_yaw_deg']=float(np.rad2deg(yaw));starts.append(result)
    best=min((s for s in starts if s['positive_depth']),key=lambda s:s['cost']);x=np.array(best['parameters']);f=best['focal_px'];R=Rotation.from_rotvec(x[:3]).as_matrix();T=np.eye(4);T[:3,:3]=R;T[:3,3]=x[3:6]
    rackT=rack_transform(x)
    white=base.project(rack_points(x,WHITE_POINTS),x[:3],x[3:6],f);white_error=np.linalg.norm(white-WHITE_UV,axis=1)
    holes=base.project(rack_points(x,HOLE_POINTS),x[:3],x[3:6],f)
    hole_errors=np.linalg.norm(holes-HOLE_UV,axis=1)
    write('hole_validation.json',dict(source_points_m=HOLE_POINTS.tolist(),observed_pixels=HOLE_UV.tolist(),projected_pixels=holes.tolist(),errors_px=hole_errors.tolist(),rms_px=float(np.sqrt(np.mean(hole_errors**2))),note='Held-out visible large-hole centers, annotated approximately by assistant; not used in optimization. Occluded holes excluded.'))
    write('fit.json',dict(**best,status='provisional_joint_fit_with_held_out_feature_validation',mesh_geometry_modified=False,instrument_used=False,frame=800,rack_body='env_00_cylinder_base',stand_body='env_06_pipette_stand',K=base.Kmatrix(f).tolist(),T_camera_world=T.tolist(),T_world_camera=np.linalg.inv(T).tolist(),rack_T_world_from_imported_world=rackT.tolist(),rack_translation_xy_m=x[7:9].tolist(),rack_yaw_deg=float(np.rad2deg(x[9])%360),rack_support_z_shift_m=DZ,scale=1.0,held_out_white_errors_px=white_error.tolist(),held_out_white_rms_px=float(np.sqrt(np.mean(white_error**2))),mesh_hashes=hashes,assumptions=['stand is fixed world anchor','rack sits on same tabletop at z=0.72m','only rack XY/yaw change; no mesh scaling or deformation','centered square-pixel pinhole camera; distortion fixed to zero','instrument excluded from objective']))
    write('multistart.json',starts)
    sweep=[fit(x,sp,su,fixed_f=f) for f in [400,500,600,800,1000,1200,1600,2000,3000,4000]]
    # Annotation perturbation: sensitivity experiment, not a statistical confidence interval.
    rng=np.random.default_rng(123);perturbed=[]
    for i in range(30):
        a=su+rng.normal(0,3,su.shape);b=RACK_UV+rng.normal(0,3,RACK_UV.shape)
        perturbed.append(fit(x,sp,a,b))
    loo=[]
    for i in range(len(sp)):
        keep=np.arange(len(sp))!=i;r=fit(x,sp[keep],su[keep]);xx=np.array(r['parameters']);p=base.project(sp[i:i+1],xx[:3],xx[3:6],r['focal_px'])[0]
        r.update(held_out=lm[i][0],held_out_error_px=float(np.linalg.norm(p-su[i])));loo.append(r)
    write('sensitivity.json',dict(fixed_focal_sweep=sweep,annotation_perturbation_sigma_px=3,annotation_perturbations=perturbed,stand_leave_one_out=loo))
    # Render unchanged mesh vertices through transforms; never export modified assets.
    stand=trimesh.load(files[1],process=False);rack=trimesh.load(files[0],process=False)
    transformed=rack.copy();transformed.apply_transform(rackT)
    sm=base.silhouette(stand,x[:3],x[3:6],f);rm=base.silhouette(transformed,x[:3],x[3:6],f)
    im=cv2.imread(str(base.RUN/'all_frames/000800.png'));base.draw_outline(im,sm,(0,255,255),'stand');base.draw_outline(im,rm,(0,255,0),'tube rack (correct asset)')
    cv2.putText(im,'JOINT camera + rack placement; unchanged meshes',(20,30),0,.7,(0,0,255),2)
    cv2.putText(im,f'f={f:.0f}px  stand RMS={best["stand_rms_px"]:.1f}px  rack RMS={best["rack_rms_px"]:.1f}px',(20,58),0,.65,(255,255,255),2)
    cv2.imwrite(str(OUT/'scene_joint.png'),im)
    annotated=im.copy();a,b=predictions(x,sp)
    for pts,pred,col in [(su,a,(0,255,255)),(RACK_UV,b,(0,255,0)),(WHITE_UV,white,(255,0,255))]:
        for uv,p in zip(pts,pred):cv2.circle(annotated,tuple(uv.astype(int)),5,col,2);cv2.drawMarker(annotated,tuple(p.astype(int)),(0,0,255),cv2.MARKER_CROSS,12,2)
    cv2.imwrite(str(OUT/'landmark_validation.png'),annotated)
    hole_overlay=im.copy()
    for a,b in zip(HOLE_UV,holes):
        cv2.circle(hole_overlay,tuple(a.astype(int)),6,(255,0,255),2)
        cv2.drawMarker(hole_overlay,tuple(b.astype(int)),(0,0,255),cv2.MARKER_CROSS,12,2)
    cv2.putText(hole_overlay,f'Held-out holes: magenta observed / red projected; RMS={np.sqrt(np.mean(hole_errors**2)):.1f}px',(20,86),0,.6,(255,255,255),2)
    cv2.imwrite(str(OUT/'hole_validation.png'),hole_overlay)
    # Diagnostic background alignment only: no claim of per-frame physical camera calibration.
    motion=json.loads((base.OUT/'static_camera_check.json').read_text())['results']
    for rec in motion:
        if rec['frame'] not in [0,350,650]:continue
        A=np.array(rec['affine']);frame=rec['frame'];img=cv2.imread(str(base.RUN/f'all_frames/{frame:06d}.png'))
        for mask,color in [(sm,(0,255,255)),(rm,(0,255,0))]:
            full=cv2.resize(mask,(1280,720),interpolation=cv2.INTER_NEAREST);warped=cv2.warpAffine(full,A,(1280,720),flags=cv2.INTER_NEAREST);base.draw_outline(img,warped,color)
        cv2.putText(img,f'Frame {frame}: same scene + background affine correction (diagnostic)',(20,30),0,.65,(0,0,255),2)
        cv2.imwrite(str(OUT/f'scene_frame_{frame:06d}.png'),img)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot([s['focal_px'] for s in sweep],[s['overall_rms_px'] for s in sweep],'o-');axs[0].set(xlabel='Fixed focal length (px)',ylabel='Combined RMS (px)',title='Joint-fit focal sensitivity')
    axs[1].hist([s['focal_px'] for s in perturbed],bins=10);axs[1].set(xlabel='Fitted focal length (px)',ylabel='Count',title='30 annotation perturbations (sigma 3px)')
    fig.tight_layout();fig.savefig(OUT/'sensitivity.png',dpi=140);plt.close(fig)
    assert hashes=={str(p):sha(p) for p in files},'Source meshes changed'
    print(json.dumps(dict(best=best,held_out_white_rms_px=float(np.sqrt(np.mean(white_error**2))),perturbed_focal_quantiles=np.quantile([s['focal_px'] for s in perturbed],[0,.5,1]).tolist(),sweep=[(s['focal_px'],s['overall_rms_px']) for s in sweep]),indent=2))

if __name__=='__main__':main()
