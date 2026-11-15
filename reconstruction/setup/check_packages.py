#!/usr/bin/env python3
"""Check dependencies and validate Decord's legacy wheel when pip flags its tag.

Decord 0.6.0 ships a legacy cp36 tag in installed WHEEL metadata even though
its ctypes Python wrapper works on this node. Never suppress other pip errors.
"""
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    if result.returncode == 0:
        return
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if lines != ['decord 0.6.0 is not supported on this platform']:
        raise SystemExit(result.returncode)
    # Only the known metadata warning may be waived, after actual video decoding.
    import cv2
    import decord
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / 'probe.mp4')
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (32, 32))
        if not writer.isOpened():
            raise RuntimeError('Cannot create Decord validation video')
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()
        reader = decord.VideoReader(path, ctx=decord.cpu(0))
        assert len(reader) == 1 and reader[0].shape == (32, 32, 3)
    print('Decord 0.6.0 legacy wheel-tag warning: actual CPU video decoding passed.')


if __name__ == '__main__':
    main()
