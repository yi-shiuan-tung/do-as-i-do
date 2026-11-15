"""Portable CPU viewer. Run with Python, numpy, scipy, opencv-python, trimesh, viser."""
import argparse
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--port',type=int,default=8082)
    args=parser.parse_args();root=args.bundle
    config=json.loads((root/'scene.json').read_text())
    hands=np.load(root/config['hands']);objects=[]
    for entry in config['objects']:
        mesh=trimesh.load(root/entry['mesh'],force='mesh',process=False)
        data=np.load(root/entry['poses'])
        objects.append((entry,mesh,data))
    cap=cv2.VideoCapture(str(root/config['video']))
    server=viser.ViserServer(port=args.port);server.scene.set_up_direction('-y')
    slider=server.gui.add_slider('Frame',min=0,max=config['frames']-1,step=1,initial_value=0)
    play=server.gui.add_checkbox('Play',initial_value=False)
    show_weak=server.gui.add_checkbox('Show low-confidence tube poses',initial_value=False)
    object_visibility={entry['name']: server.gui.add_checkbox('Show '+entry['name'], initial_value=True)
                       for entry, _, _ in objects}
    hand_visibility={side: server.gui.add_checkbox('Show '+side+' hand', initial_value=True)
                     for side in ['left', 'right']}
    status=server.gui.add_markdown('')
    lock=threading.Lock()

    def update():
        with lock:
            i=int(slider.value);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read()
            if ok:
                server.scene.add_camera_frustum('/camera',fov=2*np.arctan(360/config['focal_px']),
                    aspect=1280/720,scale=.15,image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            for entry,mesh,data in objects:
                T=data[entry['transform_key']][i]
                valid=bool(data['valid'][i]) if 'valid' in data else True
                h=server.scene.add_mesh_simple('/'+entry['name'],vertices=mesh.vertices.astype(np.float32),
                    faces=mesh.faces.astype(np.uint32),color=tuple(entry['color']),side='double',
                    wxyz=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]],position=T[:3,3])
                h.visible=object_visibility[entry['name']].value and (valid or show_weak.value)
                if entry['name']=='tube':
                    status.content=f'Frame {i} — tube IoU {data["image_iou"][i]:.3f}; '+('image fit' if valid else '**low confidence / hidden**')+'\n\nAxial tube roll is assumed. Contact is not enforced.'
            for side,color in [('left',(180,110,210)),('right',(80,175,235))]:
                h=server.scene.add_mesh_simple('/hand_'+side,vertices=hands[side+'_vertices'][i],
                    faces=hands[side+'_faces'].astype(np.uint32),color=color,side='double')
                h.visible=hand_visibility[side].value and bool(hands[side+'_visible'][i])

    @server.on_client_connect
    def connected(client):
        client.camera.position=(.65,-.25,-.2);client.camera.look_at=(.08,.04,.55)

    @slider.on_update
    def change(_):update()

    @show_weak.on_update
    def change_visibility(_):update()

    for control in [*object_visibility.values(), *hand_visibility.values()]:
        control.on_update(change_visibility)

    update()
    try:
        while True:
            time.sleep(1/config['fps'])
            if play.value:slider.value=(slider.value+1)%config['frames']
    except KeyboardInterrupt:pass
    finally:cap.release();server.stop()


if __name__=='__main__':main()
