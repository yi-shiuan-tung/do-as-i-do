# Rebuild the B200 reconstruction node

`install_b200.sh` restores the **reconstruction** pipeline used on this node.
It does not install robot retargeting/deployment environments or replace the NVIDIA driver.

## Before the node refresh

Save this modified checkout, especially `reconstruction/setup/`, `env/`,
`config/paths.sh`, and `run_pipeline.sh`, to persistent storage or your own Git
remote. These changes are local: a fresh clone of the public repository alone
will not contain this installer. The small source patches and model configs in
`setup/` reproduce the fixes inside the submodules; you need not preserve their
build directories.

Also save your licensed `MANO_RIGHT.pkl` and `MANO_LEFT.pkl`. Currently they are at:

- `modules/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl`
- `modules/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl`

For a compact restoration archive (run from `reconstruction/`):

```bash
# Choose a real persistent destination outside the node's ephemeral disk.
BACKUP=/path/to/persistent/do-as-i-do-b200.tar.gz
tar -chzf "$BACKUP" setup env config run_pipeline.sh README.md \
  modules/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl \
  modules/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

This archive contains setup changes and MANO, **not videos or reconstruction
results**. Back those up separately if you need them. Optionally preserve
`weights/`, HaWoR's `weights/` and `thirdparty/Metric3D/weights/`, and your
Hugging Face/Torch caches to avoid downloading weights again. The installer
repairs missing checkpoint symlinks using relative paths.

## After the refresh

Start with Linux x86_64, Conda available on PATH, Git, `/usr/bin/gcc` and
`/usr/bin/g++` (GCC 6–14), and a working NVIDIA driver (`nvidia-smi`). The node
image normally provides these. Conda itself and OS packages are prerequisites;
the installer does not invoke sudo. Interactive SAM3 clicking requires X11
forwarding and the host's X11/OpenGL libraries.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/malik-group/do-as-i-do.git
git -C do-as-i-do checkout 824591b808c342b20079c3b4198a8c2bdf88c74e
GIT_LFS_SKIP_SMUDGE=1 git -C do-as-i-do submodule update --init --recursive
cd do-as-i-do/reconstruction
# Restore the saved installer/patches/configs and MANO from persistent storage.
tar -xzf /path/to/persistent/do-as-i-do-b200.tar.gz
bash setup/install_b200.sh --plan
bash setup/install_b200.sh
```

If MANO was saved separately, use:

```bash
MANO_DIR=/path/to/licensed/mano/models bash setup/install_b200.sh
```

Hugging Face access must already be approved for `facebook/sam3` and
`facebook/sam-3d-objects`. Use an existing `hf auth login` session, or set
`HF_TOKEN` privately in your shell before installation. The script never embeds
or prints a token. If downloads stop for authentication, authenticate and rerun
`--from weights`; no environment reinstall is needed.

## Installed stack

| Environment | Python | PyTorch | CUDA |
|---|---|---|---|
| `sam3` | 3.12 | 2.8.0 | 12.8 |
| `sam3d-objects` | 3.11 | 2.8.0 | 12.8 |
| `hawor` | 3.10 | 2.9.0 | 12.8 |
| `tapnet` | 3.10 | 2.7.0 | 12.8 |

Names come from `config/paths.sh`. A separate `cuda128-toolkit` Conda prefix
provides **nvcc 12.8**, regardless of a system CUDA 13.x toolkit. B200 extensions
are built for `sm_100`. The SAM3D and HaWoR environments each build their own
PyTorch3D. DROID-SLAM/lietorch, Kaolin, nvdiffrast, Mip-Splatting, MoGe, GeoCalib,
and inference dependencies are installed automatically. The scoped OmegaConf
checkpoint-loading patch is preserved; the installer does not globally disable
PyTorch's restricted checkpoint loader.

`env/*-cu128-resolved.txt` constrains packages to the versions observed on
2026-10-10. The parent/submodule revisions are recorded in `setup/source-revisions.txt`.
Dependency source revisions and essential constraints are in the adjacent
`*-sources.txt` and `*-constraints.txt` files. These are dependency constraints,
not relocatable environment archives or wheel hashes. Package availability and
the system driver still affect a fresh installation.

Weights include shared SAM3D checkpoints, HaWoR, detector, DROID, Metric3D,
BootsTAPIR, and cached SAM3/MoGe/GeoCalib models. MANO is copied from your licensed
files. Downloads fail visibly and do not report success for missing assets.
Additional upstream model caches may still be populated on first inference.

## Resume and verify

Stages run in this order:

```text
submodules patches toolkit sam3 sam3d hawor tapnet weights verify
```

```bash
bash setup/install_b200.sh --from hawor  # retry this and all later stages
bash setup/install_b200.sh --only sam3d # reinstall just this environment's deps
bash setup/install_b200.sh --only weights
bash setup/install_b200.sh --only verify
```

A rerun reuses environments and completed weights. It does not delete or replace
an existing environment with the wrong Python version. Source patches are
idempotent and stop on conflicting local edits. Dependencies may be rebuilt on a
rerun. Failures stop the installer; logs are in `.setup-logs/`.

The default free-space guard requires 80 GiB available on the checkout, Conda,
and temporary-file filesystems. This is a conservative budget for installation,
not a measured minimum. For a partial restore with existing environments, you
can explicitly lower it, e.g. `MIN_FREE_GB=20 ... --from weights`. Video results
need additional storage. Use `TMPDIR`, `HF_HOME`, and `TORCH_HOME` consistently
at installation and runtime if moving caches to a larger disk.

`B200_CUDA_HOME` selects an existing separate CUDA 12.8 prefix; `MAX_JOBS`
(default 4) controls native compilation concurrency.

Verification checks package consistency, CUDA tensor execution in all four
environments, compiled-extension imports, pipeline entry points, and weight
presence. It does not run an entire reconstruction or test the X11 GUI.
`bash setup/verify_b200.sh --imports-only` skips `pip check` when diagnosing an
older environment; the full installer always uses strict verification.

SAM3 installs its development/training dependency groups plus `einops`,
`matplotlib`, and `decord`, without Jupyter applications. Decord is imported by
SAM3 even though this predictor reads video through OpenCV. Decord 0.6.0 has a
legacy wheel tag that newer pip versions flag as unsupported. The package
checker tolerates **only that exact warning**, and only after Decord successfully
decodes a generated video. All other dependency errors stop installation.
The pre-refresh SAM3 environment also has a missing Jupyter notebook dependency;
reusing it does not remove unrelated extras automatically.
