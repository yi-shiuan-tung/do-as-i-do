#!/bin/bash
# ════════════════════════════ do-as-i-do reconstruction ════════════════════════════
# End-to-end object reconstruction + pose tracking from a hand-object demo video.
# Stages (each runs in its own conda env; see config/paths.sh):
#   0  extract frames (ffmpeg)
#   1  SAM3 video segmentation — objects (click) + anchor hand (text)        [sam3]
#   2  masks -> 3D meshes ; MoGe pointmap (ref frame) ; HaWoR hands ; gravity (GeoCalib)  [sam3d/hawor]
#   2.5 TAPIR velocity tracking                                              [tapnet]
#   3  object tracking using guided pose prediction ; project mesh ; layout -> camera frame        [sam3d]
#   4  optimize translation/scale (+ optional viser viz)                     [sam3d]
#
# Usage:  ./run_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
# Resume final optimization: START_STAGE=4 ./run_pipeline.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND
# Resume after hand reconstruction: START_STAGE=pointmaps ./run_pipeline.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND
# Resume at hand reconstruction: START_STAGE=hawor ./run_pipeline.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND
# Resume after segmentation: START_STAGE=2 ./run_pipeline.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND
# Example: ./run_pipeline.sh /data/pickplan_pan/pickplan_pan.mp4 28 pan right
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config/paths.sh"

START_STAGE="${START_STAGE:-0}"
case "$START_STAGE" in
    0|2|hawor|pointmaps|4) ;;
    *) echo "START_STAGE must be 0 (full run), 2 (reuse segmentation), hawor (reuse meshes and reference pointmap), pointmaps (reuse completed hand reconstruction), or 4 (final optimization only)." >&2; exit 1 ;;
esac

# ──────────────────────────── Per-run inputs (args) ────────────────────────────
VIDEO_PATH="${1:?usage: run_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]}"
# Resolve to an ABSOLUTE path up front: later stages `cd "$SCRIPTS_DIR"` and into the
# module dirs, so a RELATIVE VIDEO_PATH would stop resolving (run_sam3_video.py would
# then load 0 frames and crash with `IndexError: list index out of range`). This also
# makes the derived VIDEO_DIR / FRAME_PATH / MASKS_DIR absolute.
VIDEO_PATH="$(realpath "$VIDEO_PATH")"
n="${2:-28}"
OBJECT_NAMES=("${3:-pan}")
ANCHOR_HAND="${4:-right}"

# ──────────────────────────── Derived paths ────────────────────────────
VIDEO_DIR="$(dirname "$VIDEO_PATH")"
VIDEO_BASENAME="$(basename "$VIDEO_PATH")"
VIDEO_NAME="${VIDEO_BASENAME%%.*}"
FRAME_PATH="$VIDEO_DIR/$(printf "%04d.png" "$n")"
POINTMAP_PATH="$VIDEO_DIR/$(printf "%04d_pointmap.npy" "$n")"
INTRINSICS_PATH="$VIDEO_DIR/$(printf "%04d_intrinsics.txt" "$n")"
MASKS_DIR="$VIDEO_DIR/video_segmentation/masks/frame_$(printf "%06d" "$n")_masks"
VIDEO_MASKS_DIR="$VIDEO_DIR/video_segmentation/masks"
HAND_MESHES_PATH="$VIDEO_DIR/$VIDEO_NAME/all_hand_meshes.npz"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "$START_STAGE" = 0 ]; then
# Frame extraction (Steps 0 & 1) uses ffmpeg from the sam3 env — activate it FIRST so
# no system/base ffmpeg is required and the pipeline runs directly on a clip.
conda activate "$ENV_SAM3"

# ──────────────── Step 0: Extract all frames ───────────────────────────
echo "=== Extracting all frames ==="
mkdir -p "$VIDEO_DIR/all_frames"
ffmpeg -i "$VIDEO_PATH" -vsync 0 -start_number 0 "$VIDEO_DIR/all_frames/%06d.png"

# ──────────────── Step 1: Save config, extract ref frame, run SAM3 ────
echo "=== Saving config and extracting reference frame ==="
OBJ_ARRAY=$(printf ', "%s"' "${OBJECT_NAMES[@]}")
OBJ_ARRAY="[${OBJ_ARRAY:2}]"
cat > "$VIDEO_DIR/config.json" <<EOF
{
    "frame_number": $n,
    "object_names": $OBJ_ARRAY,
    "anchor_hand": "$ANCHOR_HAND"
}
EOF

ffmpeg -y -i "$VIDEO_PATH" -vf "select=eq(n\,${n})" -vsync 0 -vframes 1 "$FRAME_PATH"

# sam3 env already active (from Step 0) — provides both ffmpeg and the SAM3 model
cd "$SCRIPTS_DIR"

echo "=== Running SAM3 video segmentation (objects, click-based) ==="
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJ_ID="${OBJ_NAME// /_}"
    DISPLAY="$SAM3_DISPLAY" python run_sam3_video.py \
        --video "$VIDEO_PATH" \
        --click \
        --obj_id "$OBJ_ID" \
        --frame_idx "$n"
done

echo "=== Running SAM3 video segmentation (hands, text-based) ==="
HAND_NAME="$ANCHOR_HAND hand"
HAND_ID="${ANCHOR_HAND}_hand_0"
python run_sam3_video.py \
    --video "$VIDEO_PATH" \
    --text "$HAND_NAME" \
    --obj_id "$HAND_ID" \
    --frame_idx "$n"

else
    # Stage 2 relies on the reference image, extracted frames, and segmentation masks.
    [ -s "$FRAME_PATH" ] || { echo "Missing reference image: $FRAME_PATH" >&2; exit 1; }
    [ -d "$VIDEO_DIR/all_frames" ] || { echo "Missing extracted frames: $VIDEO_DIR/all_frames" >&2; exit 1; }
    for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
        OBJ_ID="${OBJ_NAME// /_}"
        [ -s "$MASKS_DIR/$OBJ_ID.png" ] || { echo "Missing object mask: $MASKS_DIR/$OBJ_ID.png" >&2; exit 1; }
    done
    [ -s "$MASKS_DIR/${ANCHOR_HAND}_hand_0.png" ] || { echo "Missing anchor-hand mask in $MASKS_DIR" >&2; exit 1; }
    if [ "$START_STAGE" = hawor ] || [ "$START_STAGE" = pointmaps ]; then
        for artifact in "$POINTMAP_PATH" "$INTRINSICS_PATH"; do
            [ -s "$artifact" ] || { echo "Missing reference artifact: $artifact" >&2; exit 1; }
        done
        for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
            OBJ_ID="${OBJ_NAME// /_}"
            [ -s "$MASKS_DIR/$OBJ_ID/$OBJ_ID.obj" ] || { echo "Missing mesh for $OBJ_ID in $MASKS_DIR" >&2; exit 1; }
        done
        if [ "$START_STAGE" = pointmaps ]; then
            [ -s "$HAND_MESHES_PATH" ] || { echo "Missing hand meshes: $HAND_MESHES_PATH" >&2; exit 1; }
            echo "=== Resuming at per-frame pointmaps using completed hand reconstruction ==="
        else
            echo "=== Resuming at HaWoR using existing meshes and reference pointmap ==="
        fi
    elif [ "$START_STAGE" = 4 ]; then
        [ -s "$HAND_MESHES_PATH" ] || { echo "Missing hand meshes: $HAND_MESHES_PATH" >&2; exit 1; }
        for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
            OBJ_ID="${OBJ_NAME// /_}"
            for artifact in "$MASKS_DIR/$OBJ_ID/$OBJ_ID.obj" "$VIDEO_DIR/obj_tracking_out/$OBJ_ID/combined_visualization/layout_camera_frame.json"; do
                [ -s "$artifact" ] || { echo "Missing optimization input: $artifact" >&2; exit 1; }
            done
        done
        echo "=== Resuming at final translation/scale optimization ==="
    else
        echo "=== Resuming at Stage 2 using existing frames and masks ==="
    fi
fi

# ──────────────── Step 2: 3D reconstruction, pointmaps, HaWoR ────────
if [ "$START_STAGE" != 4 ]; then
if [ "$START_STAGE" != hawor ] && [ "$START_STAGE" != pointmaps ]; then
echo "=== Running batch masks to meshes ==="
conda activate "$ENV_SAM3D"
cd "$SAM3D_DIR"                                  # overlay: generate_mesh_sam3d.py
python generate_mesh_sam3d.py \
    --image_path "$FRAME_PATH" \
    --masks_dir "$MASKS_DIR"

cd "$SCRIPTS_DIR"
echo "=== Computing pointmap for reference frame ==="
python get_pointmap_dir.py --image "$FRAME_PATH" --output "$POINTMAP_PATH"

fi

if [ "$START_STAGE" != pointmaps ]; then
echo "=== Running HaWoR ==="
conda activate "$ENV_HAWOR"
cd "$HAWOR_DIR"                                  # patched: demo.py
IMG_FOCAL=$(head -n 1 "$INTRINSICS_PATH")
python demo.py --video_path "$VIDEO_PATH" --vis_mode cam --img_focal "$IMG_FOCAL" --static_camera
fi

echo "=== Computing pointmaps for all frames ==="
cd "$SCRIPTS_DIR"
conda activate "$ENV_SAM3D"
python get_pointmap_dir.py --image_dir "$VIDEO_DIR/all_frames"

echo "=== Estimating gravity (GeoCalib) ==="
python predict_video_gravity.py "$VIDEO_DIR/all_frames" --output_path "$VIDEO_DIR/gravity.json"

# ──────────────── Step 2.5: TAPIR velocity tracking ───────────────────
conda activate "$ENV_TAPNET"
echo "=== Running TAPIR velocity tracking ==="
cd "$SCRIPTS_DIR"
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJECT_ID="${OBJ_NAME// /_}"
    python tapir_velocity_tracking.py \
        --video "$VIDEO_PATH" \
        --mask-dir "$VIDEO_MASKS_DIR" \
        --object "$OBJECT_ID" \
        --checkpoint "$TAPNET_CKPT"
done

# ──────────────── Step 3: Object Tracking using guided pose prediction & projection ─────────
conda activate "$ENV_SAM3D"
echo "=== Running guided pose prediction for object tracking ==="
cd "$FASTSAM3D_DIR"                              # overlay: track_object.py
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJECT_ID="${OBJ_NAME// /_}"
    python track_object.py \
        --config checkpoints/hf/pipeline.yaml \
        --vid_dir "$VIDEO_DIR" \
        --masks_root "$VIDEO_MASKS_DIR" \
        --object_name "$OBJECT_ID" \
        --init_frame "$n" \
        --output_dir "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID" \
        --guidance_strength 1 \
        --save_layout \
        --fix_scale_to_init_frame \
        --pose_guidance_strength 0.5 \
        --num_pose_samples 25 \
        --scoring_metric render_iou \
        --pose_selection cluster \
        --cluster_dist_thresh 0.3 \
        --cluster_min_size 3 \
        --cluster_w_rot 1.5 \
        --chain_poses \
        --post_optimize \
        --no-enable_shape_icp \
        --chain_on_diffusion \
        --enable_ss_cache \
        --torch_compile \
        --euler_steps 25 \
        --rotvel_json "$VIDEO_DIR/perframe_tracking_$OBJECT_ID/motion_stats.json"

    cd "$SCRIPTS_DIR"

    echo "=== Projecting mesh for $OBJ_NAME ==="
    python run_project_mesh_combined.py \
        --video "$VIDEO_PATH" \
        --mesh "$MASKS_DIR/$OBJECT_ID/${OBJECT_ID}.obj" \
        --json "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout.json" \
        --output-base "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/projected"

    echo "=== Converting layout to camera frame for $OBJ_NAME ==="
    python convert_layout_to_camera_frame.py \
        --input "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout.json" \
        --output "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame.json"

    cd "$FASTSAM3D_DIR"
done

fi

# ──────────────── Step 4: Optimize translation/scale & visualize ──────
cd "$SCRIPTS_DIR"
conda activate "$ENV_SAM3D"

for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJECT_ID="${OBJ_NAME// /_}"
    LAYOUT_JSON_CF="$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame.json"
    LAYOUT_JSON_OPT="$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame_optimized.json"

    echo "=== Optimizing translation/scale for $OBJ_NAME ==="
    python optimize_translation_scale.py \
        --video-dir "$VIDEO_DIR" \
        --layout-json "$LAYOUT_JSON_CF" \
        --hand-meshes "$HAND_MESHES_PATH" \
        --anchor-hand "$ANCHOR_HAND" \
        --ref-frame "$n"

    # Optional trajectory smoothing to minimize depth inconsistencies
    # Writes <..._optimized>_smooth.json; point visualize_3d.py at it below to use.
    # LAYOUT_JSON_SMOOTH="${LAYOUT_JSON_OPT%.json}_smooth.json"
    # python smooth_trajectory.py \
    #     --input "$LAYOUT_JSON_OPT" \
    #     --output "$LAYOUT_JSON_SMOOTH"

    # Optional interactive 3D visualization (needs viser in the sam3d env):
    # MESH_SCALE="$(python3 -c "import json; d=json.load(open('$LAYOUT_JSON_OPT')); print(d['translation_scale_optimization']['mesh_scale'])")"
    # python visualize_3d.py \
    #     --frames-dir "$VIDEO_DIR/all_frames" \
    #     --layout-json "$LAYOUT_JSON_OPT" \
    #     --mesh "$VIDEO_DIR/tracking_output_every_frame/$OBJECT_ID/frame_000000/$OBJECT_ID/${OBJECT_ID}.obj" \
    #     --scale "$MESH_SCALE" \
    #     --translation-scale 1.0 \
    #     --hand-meshes "$HAND_MESHES_PATH" \
    #     --port 8080
done

echo "=== Pipeline complete ==="
