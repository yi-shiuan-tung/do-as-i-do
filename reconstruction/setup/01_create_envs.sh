#!/usr/bin/env bash
# Compatibility entry point; B200/CUDA 12.8 is the supported profile for this node.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-all}" in
    all) for stage in submodules patches toolkit sam3 sam3d hawor tapnet; do bash "$HERE/install_b200.sh" --only "$stage"; done ;;
    sam3|sam3d|hawor|tapnet)
        for stage in submodules patches toolkit "$1"; do bash "$HERE/install_b200.sh" --only "$stage"; done ;;
    *) echo "Usage: $0 [all|sam3|sam3d|hawor|tapnet]" >&2; exit 2 ;;
esac
