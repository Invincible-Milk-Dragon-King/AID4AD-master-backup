#!/usr/bin/env bash
# Host-side launcher: 4 dual-GPU MapTR jobs in one tmux session (GPUs 0-7).
#
# Round A (default) — train probes + refresh baselines:
#   pane0  0,1  Probe(C→C) train
#   pane1  2,3  Probe(F→C) train   (retrain with force_camera_only)
#   pane2  4,5  Full-C baseline test
#   pane3  6,7  Full-F baseline test
#
# After probes finish, in pane2/3 run:
#   bash jobs/maptr_p2_probe_c2c_test.sh
#   bash jobs/maptr_p3_probe_f2c_test.sh
#
# Usage (HOST):
#   bash tmux_4x2gpu_maptr.sh
#   tmux attach -t maptr4

set -euo pipefail

CONTAINER="${CONTAINER:-aid4ad-maptr-exp}"
SESSION="${SESSION:-maptr4}"
JOB_DIR="/workspace/AID4AD-master/111/modality_degradation/bash/jobs"

if ! command -v tmux >/dev/null; then
  echo "[ERROR] tmux not found on host. Install: sudo apt-get install -y tmux"
  exit 1
fi

if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "[ERROR] container not found: ${CONTAINER}"
  exit 1
fi
if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "[INFO] starting container ${CONTAINER}"
  docker start "${CONTAINER}" >/dev/null
fi

chmod +x \
  "$(dirname "$0")/jobs/maptr_p0_probe_c2c.sh" \
  "$(dirname "$0")/jobs/maptr_p1_probe_f2c.sh" \
  "$(dirname "$0")/jobs/maptr_p2_baseline_c.sh" \
  "$(dirname "$0")/jobs/maptr_p3_baseline_f.sh" \
  "$(dirname "$0")/jobs/maptr_p2_probe_c2c_test.sh" \
  "$(dirname "$0")/jobs/maptr_p3_probe_f2c_test.sh" 2>/dev/null || true

tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}"

tmux new-session -d -s "${SESSION}" -n maptr4
tmux split-window -h -t "${SESSION}:0"
tmux select-pane -t "${SESSION}:0.0"
tmux split-window -v -t "${SESSION}:0.0"
tmux select-pane -t "${SESSION}:0.1"
tmux split-window -v -t "${SESSION}:0.1"
tmux select-layout -t "${SESSION}:0" tiled

# Use docker exec without -t (tmux already provides tty).
tmux send-keys -t "${SESSION}:0.0" "docker exec -i ${CONTAINER} bash ${JOB_DIR}/maptr_p0_probe_c2c.sh" C-m
tmux send-keys -t "${SESSION}:0.1" "docker exec -i ${CONTAINER} bash ${JOB_DIR}/maptr_p1_probe_f2c.sh" C-m
tmux send-keys -t "${SESSION}:0.2" "docker exec -i ${CONTAINER} bash ${JOB_DIR}/maptr_p2_baseline_c.sh" C-m
tmux send-keys -t "${SESSION}:0.3" "docker exec -i ${CONTAINER} bash ${JOB_DIR}/maptr_p3_baseline_f.sh" C-m

cat <<EOF
[OK] session=${SESSION}  container=${CONTAINER}

Attach:
  tmux attach -t ${SESSION}

Keys:
  Ctrl-b o / arrows   switch pane
  Ctrl-b z            zoom pane
  Ctrl-b d            detach (keep running)
  tmux kill-session -t ${SESSION}

Layout:
  +------------------+------------------+
  | p0 GPUs0,1       | p2 GPUs4,5       |
  | Probe(C→C) train | Full-C test      |
  +------------------+------------------+
  | p1 GPUs2,3       | p3 GPUs6,7       |
  | Probe(F→C) train | Full-F test      |
  +------------------+------------------+

After probe train ends, in p2/p3:
  docker exec -it ${CONTAINER} bash ${JOB_DIR}/maptr_p2_probe_c2c_test.sh
  docker exec -it ${CONTAINER} bash ${JOB_DIR}/maptr_p3_probe_f2c_test.sh

Logs:
  111/exp_results/maptr/*.log
EOF
