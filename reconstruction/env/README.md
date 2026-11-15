# Reconstruction environments on B200

Use [`../setup/install_b200.sh`](../setup/install_b200.sh) for a fresh node.
The [restoration guide](../setup/README.md) covers prerequisites, backups,
weights, verification, and resuming individual installation stages.

```bash
bash setup/install_b200.sh
# Or resume after a failure:
bash setup/install_b200.sh --from sam3d
```

| Default environment | Python | Torch / torchvision |
|---|---|---|
| sam3 | 3.12 | 2.8.0+cu128 / 0.23.0+cu128 |
| sam3d-objects | 3.11 | 2.8.0+cu128 / 0.23.0+cu128 |
| hawor | 3.10 | 2.9.0+cu128 / 0.24.0+cu128 |
| tapnet | 3.10 | 2.7.0+cu128 / 0.22.0+cu128 |

The separate `cuda128-toolkit` prefix supplies nvcc 12.8. Do not install a
CUDA 13.x compiler into these Python environments. B200 uses architecture 10.0.

- `*-cu128-resolved.txt`: package constraints captured from the current node.
- `*-cu128-inference.txt`: inference packages to install.
- `*-cu128-sources.txt`: source-built dependencies with pinned Git revisions.
- `*-cu128-constraints.txt`: essential compatibility pins.
- `requirements.reconstruction-*.txt`: SAM3D package metadata restored by the installer.
- `*.yml`: historical exports for reference; they are not the fresh-node recipe.

The completion helpers `03_complete_sam3d_cu128.sh` and
`04_complete_hawor_cu128.sh` are invoked by the installer after the required
Python/Torch/toolkit setup. They are not standalone fresh-environment installers.
SAM3D uses SDPA attention and shares its environment with Fast-SAM3D.
