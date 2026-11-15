#!/usr/bin/env python3
"""Reapply only the captured compatibility patches; refuse conflicting changes."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATCHES = [
    ('HaWoR', 'hawor-b200'),
    ('HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch', 'lietorch-torch2'),
    ('sam-3d-objects', 'sam3d-cu128'),
    ('Fast-SAM3D', 'fastsam3d-cuda'),
]


def main():
    for module, name in PATCHES:
        repo = ROOT / 'modules' / module
        patch = ROOT / 'setup/patches' / f'{name}.patch'
        command = ['git', '-C', str(repo), 'apply']
        if subprocess.run(command + ['--reverse', '--check', str(patch)], capture_output=True).returncode == 0:
            print(f'Already applied: {name}')
        else:
            subprocess.run(command + ['--check', str(patch)], check=True)
            subprocess.run(command + [str(patch)], check=True)
            print(f'Applied: {name}')
    for source in (ROOT / 'env').glob('requirements.reconstruction-*.txt'):
        destination = ROOT / 'modules/sam-3d-objects' / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f'Local requirements differ; review before replacing: {destination}')
        shutil.copyfile(source, destination)


if __name__ == '__main__':
    main()
