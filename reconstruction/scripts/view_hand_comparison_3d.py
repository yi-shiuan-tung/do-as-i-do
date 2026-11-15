"""CPU-only orbit viewer for an exported hand comparison folder.
Usage: python view_3d.py --port 8085
Dependencies: numpy, pillow, viser
"""
import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np
from PIL import Image
import viser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--port', type=int, default=8085)
    args = parser.parse_args()
    root = args.bundle
    data = json.loads((root/'data.json').read_text())
    ids = data['ids']
    objects = {name: np.fromfile(root/f'{name}.bin', dtype='<f4').reshape(len(ids), obj['count'], 3)
               for name, obj in data['objects'].items()}
    hands = {(side, variant): np.fromfile(root/f'{side}_{variant}.bin', dtype='<f4').reshape(len(ids), 778, 3)
             for side in data['hands'] for variant in data['variants']}
    server = viser.ViserServer(host='127.0.0.1', port=args.port)
    server.scene.set_up_direction('-y')
    slider = server.gui.add_slider('Sample', min=0, max=len(ids)-1, step=1, initial_value=min(range(len(ids)), key=lambda i: abs(ids[i]-550)))
    play = server.gui.add_checkbox('Play (10 fps)', initial_value=False)
    variant = server.gui.add_dropdown('Hand fit', options=tuple(data['variants']), initial_value='contact')
    visibility = {name: server.gui.add_checkbox('Show '+name, initial_value=True) for name in ['left', 'right', 'pipette', 'tube']}
    opacity = server.gui.add_slider('Object opacity', min=0.1, max=1., step=.1, initial_value=1.)
    wireframe = server.gui.add_checkbox('Wireframe hands', initial_value=False)
    reset = server.gui.add_button('Reset camera')
    status = server.gui.add_markdown('')
    server.gui.add_markdown('Drag to orbit; scroll to zoom; right-drag to pan.\n\nContact-aware fitting is experimental: intersections remain. Tube collision used a convex envelope; the displayed tube is the original mesh. Hidden hands lack usable observed evidence. Units: meters.')
    rgb = server.gui.add_image(np.zeros((480,640,3), dtype=np.uint8), label='Recorded RGB')
    colors = {'left': (52,210,237), 'right': (255,120,99), 'pipette': (245,205,73), 'tube': (112,215,117)}
    lock = threading.Lock()

    def camera(client):
        center = objects['pipette'][int(slider.value)].mean(0)
        client.camera.look_at = tuple(center)
        client.camera.position = tuple(center + np.array([.32, -.16, -.48]))
        client.camera.up_direction = (0., -1., 0.)

    @server.on_client_connect
    def connected(client):
        camera(client)

    @reset.on_click
    def reset_camera(_):
        for client in server.get_clients().values():
            camera(client)

    def update(_=None):
        with lock:
            i = int(slider.value)
            with server.atomic():
                for name, vertices in objects.items():
                    handle = server.scene.add_mesh_simple('/'+name, vertices=vertices[i], faces=np.asarray(data['objects'][name]['faces'], dtype=np.uint32), color=colors[name], side='double', opacity=float(opacity.value))
                    handle.visible = visibility[name].value
                lines = [f'**Frame {ids[i]} · {ids[i]/30:.2f} s · {variant.value}**']
                for side, info in data['hands'].items():
                    handle = server.scene.add_mesh_simple('/'+side, vertices=hands[side, variant.value][i], faces=np.asarray(info['faces'], dtype=np.uint32), color=colors[side], side='double', wireframe=wireframe.value)
                    handle.visible = visibility[side].value and bool(info['valid'][i])
                    lines.append(f'{side}: '+('observed' if info['valid'][i] else 'hidden / insufficient evidence')+f"; object reliability {info['reliability'][i]:.2f}")
                status.content = '\n\n'.join(lines)
                with Image.open(root/'rgb'/f'{ids[i]:06d}.jpg') as im:
                    rgb.image = np.asarray(im.convert('RGB'))

    for control in [slider, variant, opacity, wireframe, *visibility.values()]:
        control.on_update(update)
    update()
    try:
        while True:
            time.sleep(.1)
            if play.value:
                slider.value = (int(slider.value)+1) % len(ids)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
