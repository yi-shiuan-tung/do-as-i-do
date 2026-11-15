#!/usr/bin/env bash
# Read-only import/GPU/entry-point checks; does not execute the reconstruction pipeline.
set -eo pipefail
CHECK_PACKAGES=1
case "${1:-}" in
    "") ;;
    --imports-only) CHECK_PACKAGES=0 ;;
    *) echo "Usage: $0 [--imports-only]" >&2; exit 2 ;;
esac
check_packages() { if [ "$CHECK_PACKAGES" = 1 ]; then python "$ROOT/setup/check_packages.py"; fi; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
check_torch() {
    python - "$1" <<'PY'
import sys, torch
assert torch.__version__ == sys.argv[1], (torch.__version__, sys.argv[1])
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available(), 'CUDA unavailable; check driver and GPU allocation'
x = torch.ones(4, device='cuda')
assert (x + x).sum().item() == 8
print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
PY
}
conda activate "$ENV_SAM3"
check_torch 2.8.0+cu128
check_packages
python -c 'import cv2; from sam3.model_builder import build_sam3_video_predictor'
python "$SCRIPTS_DIR/run_sam3_video.py" --help >/dev/null
conda activate "$ENV_SAM3D"
check_torch 2.8.0+cu128
check_packages
python -c 'import torch, pytorch3d._C, kaolin, nvdiffrast.torch, diff_gaussian_rasterization, gsplat, spconv.pytorch; from geocalib import GeoCalib; from moge.model.v1 import MoGeModel'
(cd "$SAM3D_DIR" && python generate_mesh_sam3d.py --help >/dev/null)
(cd "$FASTSAM3D_DIR" && python track_object.py --help >/dev/null)
python "$SCRIPTS_DIR/optimize_translation_scale.py" --help >/dev/null
python "$ROOT/setup/fetch_b200_weights.py" --check
conda activate "$ENV_HAWOR"
check_torch 2.9.0+cu128
check_packages
python -c 'import torch, droid_backends, lietorch_backends, lietorch, pytorch3d._C, torch_scatter, joblib'
(cd "$HAWOR_DIR" && python demo.py --help >/dev/null)
conda activate "$ENV_TAPNET"
check_torch 2.7.0+cu128
check_packages
python "$SCRIPTS_DIR/tapir_velocity_tracking.py" --help >/dev/null
if [ "$CHECK_PACKAGES" = 1 ]; then
    echo 'All B200 reconstruction checks passed.'
else
    echo 'B200 GPU/import/entry-point/weight checks passed (package consistency checks skipped).'
fi
echo 'Interactive clicking still requires a working X11 display.'
