#!/usr/bin/env bash
set -euo pipefail

RESULT_PATH="${RESULT_PATH:-./branch_runs/camera_only_eval/camera_only_camera_only_trainval_model_last_vectors.json}"
DATAROOT="${DATAROOT:-../nuScenes}"
PRIOR_MAP_ROOT="${PRIOR_MAP_ROOT:-../AID4AD_ego_referenced}"
VERSION="${VERSION:-v1.0-trainval}"
EVAL_SET="${EVAL_SET:-val}"
BSZ="${BSZ:-4}"
NWORKERS="${NWORKERS:-8}"
MAP_THRESHOLDS="${MAP_THRESHOLDS:-0.2 0.5 1.0}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${RESULT_PATH%_vectors.json}_paper_ap}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_PREFIX}_metrics.json}"
OUTPUT_LOG="${OUTPUT_LOG:-${OUTPUT_PREFIX}.log}"

mkdir -p "$(dirname "${OUTPUT_JSON}")" "$(dirname "${OUTPUT_LOG}")"

python evaluate_json.py \
  --result_path "${RESULT_PATH}" \
  --dataroot "${DATAROOT}" \
  --prior_map_root "${PRIOR_MAP_ROOT}" \
  --version "${VERSION}" \
  --eval_set "${EVAL_SET}" \
  --bsz "${BSZ}" \
  --nworkers "${NWORKERS}" \
  --ap_only \
  --map_thresholds ${MAP_THRESHOLDS} \
  --output_json "${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_LOG}"

echo "Saved metrics JSON to ${OUTPUT_JSON}"
echo "Saved eval log to ${OUTPUT_LOG}"
