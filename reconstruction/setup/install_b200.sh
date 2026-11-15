#!/usr/bin/env bash
# Restore the reconstruction stack used on this B200 node. No sudo/driver changes.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/paths.sh"
STAGES=(submodules patches toolkit sam3 sam3d hawor tapnet weights verify)
FROM=submodules
ONLY=
PLAN=0
usage() {
    cat <<'EOF'
Usage: bash setup/install_b200.sh [--plan | --from STAGE | --only STAGE]
Stages: submodules patches toolkit sam3 sam3d hawor tapnet weights verify
Re-running is supported; existing environments and weights are reused.
--from resumes at a stage and runs all later stages; --only runs one stage.
Prerequisites: Linux x86_64, working NVIDIA driver, conda, git, GCC/G++ 6-14.
Provide HF_TOKEN (or hf auth login) with SAM3/SAM3D access and
MANO_DIR=/path/to/models containing MANO_RIGHT.pkl and MANO_LEFT.pkl.
Optional: B200_CUDA_HOME, MAX_JOBS (default 4), MIN_FREE_GB (default 80).
The minimum-free-space check is a conservative installation budget, not an exact requirement.
EOF
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --plan) PLAN=1; shift ;;
        --from|--only)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            if [ "$1" = --from ]; then FROM="$2"; else ONLY="$2"; fi
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
for selected in "$FROM" "${ONLY:-$FROM}"; do
    case " ${STAGES[*]} " in *" $selected "*) ;; *) echo "Unknown stage: $selected" >&2; exit 2 ;; esac
done
if [ "$PLAN" = 1 ]; then
    usage
    printf '\nEnvironment plan:\n  %s: Python 3.12 / Torch 2.8 cu128\n  %s: Python 3.11 / Torch 2.8 cu128\n  %s: Python 3.10 / Torch 2.9 cu128\n  %s: Python 3.10 / Torch 2.7 cu128\n' "$ENV_SAM3" "$ENV_SAM3D" "$ENV_HAWOR" "$ENV_TAPNET"
    exit 0
fi
command -v conda >/dev/null || { echo 'Install/initialize Conda first, then rerun this script.' >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
export B200_CUDA_HOME="${B200_CUDA_HOME:-$CONDA_BASE/envs/cuda128-toolkit}"
export SAM3D_CUDA_HOME="$B200_CUDA_HOME" HAWOR_CUDA_HOME="$B200_CUDA_HOME"
export MAX_JOBS="${MAX_JOBS:-4}" PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$ROOT/.setup-logs"
LOG="$ROOT/.setup-logs/install-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
STAGE=preflight
trap 'echo "FAILED during $STAGE. Log: $LOG"; echo "Fix the error, then rerun the same command (or use --from with a stage listed in --help)."' ERR
for cmd in git /usr/bin/gcc /usr/bin/g++ nvidia-smi; do command -v "$cmd" >/dev/null; done
[ "$(uname -m)" = x86_64 ] || { echo 'This profile requires Linux x86_64.'; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
compiler_major=$(/usr/bin/gcc -dumpversion | cut -d. -f1)
[ "$compiler_major" -ge 6 ] && [ "$compiler_major" -le 14 ] || { echo 'Select a CUDA 12.8 compatible system GCC (6-14).'; exit 1; }
if [ "$ONLY" != verify ] && [ "$FROM" != verify ]; then
    for location in "$ROOT" "$CONDA_BASE" "${TMPDIR:-/tmp}"; do
        available=$(df -Pk "$location" | awk 'END {print $4}')
        [ "$available" -ge "$((${MIN_FREE_GB:-80} * 1024 * 1024))" ] || {
            echo "Insufficient free space at $location for the default 80 GiB installation budget."
            echo 'Use a larger disk, or set MIN_FREE_GB for a partial/reused installation.'; exit 1;
        }
    done
fi
make_env() {
    local name="$1" py="$2"
    # Conda install repairs an incomplete env, but never silently replaces its Python major/minor.
    if conda run -n "$name" python -c 'import sys' >/dev/null 2>&1; then
        conda run -n "$name" python -c "import sys; assert '.'.join(map(str,sys.version_info[:2])) == '$py', 'Wrong Python version in $name'"
        conda install -y -n "$name" --override-channels -c conda-forge "python=$py" pip ffmpeg
    else
        # Existing but broken envs are repaired through install; absent envs are created.
        if conda env list --json | "$CONDA_BASE/bin/python" -c 'import json,sys,pathlib; sys.exit(not any(pathlib.Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$name"; then
            conda install -y -n "$name" --override-channels -c conda-forge "python=$py" pip ffmpeg
        else
            conda create -y -n "$name" --override-channels -c conda-forge "python=$py" pip ffmpeg
        fi
    fi
    conda activate "$name"
}
base_packages() {
    export PIP_CONSTRAINT="$ROOT/env/$1-cu128-resolved.txt"
    python -m pip install pip setuptools wheel ninja packaging
}
torch_stack() {
    python -m pip install "torch==$1" "torchvision==$2" "torchaudio==$1" --index-url https://download.pytorch.org/whl/cu128
}
stage_submodules() { bash "$ROOT/setup/00_init_submodules.sh"; }
stage_patches() { "$CONDA_BASE/bin/python" "$ROOT/setup/apply_b200_patches.py"; }
stage_toolkit() {
    if [ ! -x "$B200_CUDA_HOME/bin/nvcc" ]; then
        conda create -y -p "$B200_CUDA_HOME" --override-channels \
            -c nvidia/label/cuda-12.8.0 -c defaults 'cuda-toolkit=12.8.0' 'cuda-nvcc=12.8.*'
    fi
    "$B200_CUDA_HOME/bin/nvcc" --version | grep 'release 12.8'
}
stage_sam3() {
    make_env "$ENV_SAM3" 3.12
    base_packages sam3
    torch_stack 2.8.0 0.23.0
    # Decord is imported by SAM3 dataset types even when videos use cv2.
    # Install it explicitly without pulling in unrelated notebook applications.
    python -m pip install -e "$SAM3_PKG_DIR[dev,train]" einops matplotlib decord==0.6.0
    python "$ROOT/setup/check_packages.py"
}
stage_sam3d() {
    make_env "$ENV_SAM3D" 3.11
    base_packages sam3d
    torch_stack 2.8.0 0.23.0
    bash "$ROOT/setup/03_complete_sam3d_cu128.sh"
}
stage_hawor() {
    make_env "$ENV_HAWOR" 3.10
    base_packages hawor
    # HaWoR uses no torchaudio; keep its observed Torch/vision pair.
    python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install numpy==1.26.4 pytorch-lightning==2.2.4 torchmetrics==1.4.0
    export CUDA_HOME="$B200_CUDA_HOME" PATH="$B200_CUDA_HOME/bin:$PATH"
    export CC=/usr/bin/gcc CXX=/usr/bin/g++ TORCH_CUDA_ARCH_LIST=10.0 FORCE_CUDA=1
    unset CUDAHOSTCXX NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
    "$CUDA_HOME/bin/nvcc" --version | grep 'release 12.8'
    # Upstream setup.py defines TWO distributions; pip install . would only install one.
    if ! python -c 'import torch, droid_backends, lietorch_backends, lietorch' 2>/dev/null; then
        (cd "$HAWOR_DIR/thirdparty/DROID-SLAM" && python setup.py build_ext --force && python setup.py install)
    fi
    bash "$ROOT/setup/04_complete_hawor_cu128.sh"
}
stage_tapnet() {
    make_env "$ENV_TAPNET" 3.10
    base_packages tapnet
    torch_stack 2.7.0 0.22.0
    python -m pip install -e "$TAPNET_DIR[torch]" einops tqdm mediapy
    python -m pip check
}
stage_weights() {
    conda activate "$ENV_SAM3D"
    export PIP_CONSTRAINT="$ROOT/env/sam3d-cu128-resolved.txt"
    python -m pip install huggingface-hub gdown
    python "$ROOT/setup/fetch_b200_weights.py"
}
stage_verify() { bash "$ROOT/setup/verify_b200.sh"; }
active=0
for STAGE in "${STAGES[@]}"; do
    [ "$STAGE" != "$FROM" ] || active=1
    if [ -n "$ONLY" ]; then [ "$STAGE" = "$ONLY" ] || continue
    else [ "$active" = 1 ] || continue; fi
    echo "=== INSTALL: $STAGE ==="
    # Subshell prevents one env's compiler/PIP settings leaking into another stage.
    ( "stage_$STAGE" )
done
echo "Selected setup stages completed. Log: $LOG"
