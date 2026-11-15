#!/usr/bin/env bash
# Complete inference dependencies in the existing B200 HaWoR environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/paths.sh"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_HAWOR"
export CUDA_HOME="${HAWOR_CUDA_HOME:-$CONDA_BASE/envs/cuda128-toolkit}"
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.8'
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS="${MAX_JOBS:-4}" FORCE_CUDA=1
unset CUDAHOSTCXX NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
python -c 'import torch; assert torch.__version__ == "2.9.0+cu128", torch.__version__'
python -m pip install -c "$ROOT/env/hawor-cu128-constraints.txt" \
    -r "$ROOT/env/hawor-cu128-inference.txt"
python -m pip install --no-build-isolation -c "$ROOT/env/hawor-cu128-constraints.txt" \
    -r "$ROOT/env/hawor-cu128-sources.txt" \
    -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
conda env config vars set -n "$ENV_HAWOR" CUDA_HOME="$CUDA_HOME" \
    CC="$CC" CXX="$CXX" TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS="${MAX_JOBS:-4}"
python -m pip check
python -c 'import torch, droid_backends, lietorch_backends'
(cd "$HAWOR_DIR" && python demo.py --help)
echo "HaWoR inference dependencies installed. Reactivate $ENV_HAWOR."
