#!/usr/bin/env python3
"""Fetch inference assets with atomic downloads and shared SAM3D storage.

Uses existing Hugging Face authentication/HF_TOKEN. MANO must be supplied by
its licensed user; this script does not fetch MANO from third-party mirrors.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HAWOR = ROOT / 'modules/HaWoR'
SAM3D_FILES = '''slat_generator.ckpt slat_encoder.ckpt ss_generator.ckpt
ss_encoder.ckpt slat_decoder_mesh.ckpt slat_decoder_mesh.pt slat_decoder_gs.ckpt
slat_decoder_gs_4.ckpt ss_decoder.ckpt'''.split()
ASSETS = [
    ('https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt', 'weights/external/detector.pt'),
    ('https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt', 'weights/hawor/checkpoints/hawor.ckpt'),
    ('https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt', 'weights/hawor/checkpoints/infiller.pt'),
    ('https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml', 'weights/hawor/model_config.yaml'),
]
DRIVE = [
    ('1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh', 'weights/external/droid.pth'),
    ('1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr', 'thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth'),
]


def usable(path):
    if not path.is_file() or path.stat().st_size < (20 if path.suffix == '.yaml' else 1024):
        return False
    with path.open('rb') as stream:
        head = stream.read(200).lstrip().lower()
    return not head.startswith((b'version https://git-lfs', b'<!doctype html', b'<html'))


def fetch(url, destination, drive=False):
    if usable(destination):
        print(f'Reuse {destination}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    print(f'Download {destination}', flush=True)
    if drive:
        import gdown
        if not gdown.download(id=url, output=str(partial), quiet=False, resume=True):
            raise RuntimeError(f'Google Drive download failed: {destination}')
    else:
        with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
    # YAML has a lower minimum size; partial has an extra suffix.
    if partial.stat().st_size < (20 if destination.suffix == '.yaml' else 1024):
        raise RuntimeError(f'Incomplete download: {partial}')
    with partial.open('rb') as stream:
        if stream.read(200).lstrip().lower().startswith((b'<html', b'<!doctype html', b'version https://git-lfs')):
            raise RuntimeError(f'Download returned HTML or an LFS pointer: {partial}')
    os.replace(partial, destination)


def mano_paths():
    return [HAWOR / '_DATA/data/mano/MANO_RIGHT.pkl', HAWOR / '_DATA/data_left/mano_left/MANO_LEFT.pkl']


def install_mano():
    source_root = Path(os.environ['MANO_DIR']).expanduser() if os.environ.get('MANO_DIR') else None
    for destination in mano_paths():
        if usable(destination):
            continue
        candidates = [HAWOR / '_DATA/data/mano' / destination.name]
        if source_root:
            candidates = [source_root / destination.name, source_root / 'models' / destination.name] + candidates
        source = next((p for p in candidates if usable(p)), None)
        if source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    missing = [str(p) for p in mano_paths() if not usable(p)]
    if missing:
        raise RuntimeError('Provide your licensed MANO models via MANO_DIR (both .pkl files). Missing: ' + ', '.join(missing))


def check_local():
    paths = [HAWOR / relative for _, relative in ASSETS + DRIVE] + mano_paths()
    paths += [ROOT / 'weights/tapnet/bootstapir_checkpoint_v2.pt']
    for module in ['sam-3d-objects', 'Fast-SAM3D']:
        paths += [ROOT / 'modules' / module / 'checkpoints/hf' / f for f in SAM3D_FILES]
        paths += list((ROOT / 'setup/model-configs' / module).glob('*.yaml'))
        # Check installed configurations as well as the preserved copies.
        paths += [ROOT / 'modules' / module / 'checkpoints/hf' / f.name
                  for f in (ROOT / 'setup/model-configs' / module).glob('*.yaml')]
    missing = [str(p) for p in paths if not usable(p)]
    if missing:
        raise RuntimeError('Missing/incomplete assets:\n' + '\n'.join(missing))


def hub_assets(local_only=False):
    from huggingface_hub import hf_hub_download
    for repo, files in [('facebook/sam3', ['config.json', 'sam3.pt']), ('Ruicheng/moge-vitl', ['model.pt'])]:
        for name in files:
            hf_hub_download(repo_id=repo, filename=name, local_files_only=local_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Only check existing weights and caches; no downloads')
    args = parser.parse_args()
    if args.check:
        check_local()
        hub_assets(local_only=True)
        import torch
        if not usable(Path(torch.hub.get_dir()) / 'geocalib/pinhole.tar'):
            raise RuntimeError('GeoCalib pinhole weights not cached')
        print('All required local weights and SAM3/MoGe/GeoCalib caches are present.')
        return

    # Fail before downloading gigabytes if the licensed local models are absent.
    install_mano()
    from huggingface_hub import hf_hub_download, HfApi
    shared = ROOT / 'weights/sam3d_shared/hf'
    # Query only when files are missing; supports repos with or without checkpoints/ prefix.
    missing = [f for f in SAM3D_FILES if not usable(shared / f)]
    if missing:
        revision = json.loads((ROOT / 'setup/model-revisions.json').read_text())['facebook/sam-3d-objects']
        info = HfApi().model_info('facebook/sam-3d-objects', revision=revision)
        names = [f.rfilename for f in info.siblings]
        for name in missing:
            matches = [f for f in names if Path(f).name == name]
            if len(matches) != 1:
                raise RuntimeError(f'Expected exactly one SAM3D asset named {name}; found {matches}')
            downloaded = Path(hf_hub_download('facebook/sam-3d-objects', matches[0], revision=info.sha, local_dir=str(shared)))
            destination = shared / name
            if downloaded != destination:
                os.replace(downloaded, destination)
        (shared / 'download-revision.txt').write_text(info.sha + '\n')
    for module in ['sam-3d-objects', 'Fast-SAM3D']:
        target = ROOT / 'modules' / module / 'checkpoints/hf'
        target.mkdir(parents=True, exist_ok=True)
        for name in SAM3D_FILES:
            destination = target / name
            if usable(destination):
                continue
            if destination.is_symlink():
                destination.unlink()
            elif destination.exists():
                raise RuntimeError(f'Incomplete existing file; move it aside and retry: {destination}')
            destination.symlink_to(os.path.relpath(shared / name, target))
        for config in (ROOT / 'setup/model-configs' / module).glob('*.yaml'):
            destination = target / config.name
            if destination.exists() and destination.read_bytes() != config.read_bytes():
                # A pinned fresh submodule may have older configs. Preserve those first.
                backup = destination.with_suffix('.yaml.before-b200')
                if not backup.exists():
                    shutil.copyfile(destination, backup)
            shutil.copyfile(config, destination)
    for url, relative in ASSETS:
        fetch(url, HAWOR / relative)
    for file_id, relative in DRIVE:
        fetch(file_id, HAWOR / relative, drive=True)
    fetch('https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt',
          ROOT / 'weights/tapnet/bootstapir_checkpoint_v2.pt')
    hub_assets()
    import torch
    fetch('https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar',
          Path(torch.hub.get_dir()) / 'geocalib/pinhole.tar')
    check_local()
    print('Model weights installed. SAM3D weights are shared between both forks.')


if __name__ == '__main__':
    main()
