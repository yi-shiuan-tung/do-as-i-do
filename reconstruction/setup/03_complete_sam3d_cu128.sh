#!/usr/bin/env bash
# Complete the inference environment for B200 (sm_100), using a separate toolkit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/paths.sh"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_SAM3D"
export CUDA_HOME="${SAM3D_CUDA_HOME:-$CONDA_BASE/envs/cuda128-toolkit}"
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.8' || {
    echo "Set SAM3D_CUDA_HOME to a separate CUDA 12.8 toolkit directory." >&2; exit 1;
}
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS="${MAX_JOBS:-4}" FORCE_CUDA=1
export ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa
unset CUDAHOSTCXX NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
python -c 'import torch; assert torch.__version__ == "2.8.0+cu128", torch.__version__'
CONSTRAINTS="$ROOT/env/sam3d-cu128-constraints.txt"
python -m pip install -c "$CONSTRAINTS" -r "$ROOT/env/sam3d-cu128-inference.txt" \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
python -m pip install --no-build-isolation -c "$CONSTRAINTS" -r "$ROOT/env/sam3d-cu128-sources.txt"
if ! python -c 'import torch, pytorch3d._C' 2>/dev/null; then
    python -m pip install --no-build-isolation --no-deps \
        'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
fi
BUILD_ROOT="${SAM3D_BUILD_ROOT:-$CONDA_PREFIX/sam3d-build-deps}"
mkdir -p "$BUILD_ROOT"
if [ ! -d "$BUILD_ROOT/mip-splatting/.git" ]; then
    git clone https://github.com/autonomousvision/mip-splatting.git "$BUILD_ROOT/mip-splatting"
fi
git -C "$BUILD_ROOT/mip-splatting" checkout dda02ab5ecf45d6edb8c540d9bb65c7e451345a9
git -C "$BUILD_ROOT/mip-splatting" submodule update --init --recursive
python -m pip install --no-build-isolation --no-deps \
    "$BUILD_ROOT/mip-splatting/submodules/diff-gaussian-rasterization"
# Preserve the external toolkit when importing either fork's notebook helper.
python - "$SAM3D_DIR" "$FASTSAM3D_DIR" <<'PATCH'
import sys
from pathlib import Path
for root in sys.argv[1:]:
    p = Path(root) / 'notebook/inference.py'
    s = p.read_text().replace('os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]',
                             'os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])')
    p.write_text(s)
PATCH
python -m pip install --no-deps -e "$SAM3D_DIR"
python -m pip uninstall -y notebook
python "$SAM3D_DIR/patching/hydra"
conda env config vars set -n "$ENV_SAM3D" CUDA_HOME="$CUDA_HOME" \
    CC="$CC" CXX="$CXX" TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS="${MAX_JOBS:-4}" \
    ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa
python -m pip check
(cd "$SAM3D_DIR" && python generate_mesh_sam3d.py --help >/dev/null)
echo "Inference dependencies installed. Reactivate $ENV_SAM3D to load persistent settings."
