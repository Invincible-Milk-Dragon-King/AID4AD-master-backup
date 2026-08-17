#!/usr/bin/env bash
set -euo pipefail

# Official checkpoints (clear URLs) for models that provide public weights.
# Download target:
#   <111>/checkpoints/{maptr,gemap}/
#
# MapTR (from README)
# - Camera (maptr_tiny_r50_24e):
#   https://drive.google.com/file/d/1n1FUFnRqdskvmpLdnsuX_VK6pET19h95/view?usp=share_link
# - Fusion (maptr_tiny_fusion_24e):
#   https://drive.google.com/file/d/1CFlJrl3ZDj3gIOysf5Cli9bX5LEYSYO4/view?usp=share_link
#
# GeMap (from README) — modality pair uses *simple* objective on both sides:
# - Camera simple (gemap_simple_r50_110ep, official ~62.7 mAP):
#   https://drive.google.com/file/d/1QNmluapTm_hH-ofMi_QsKXLG2bp38wEW/view?usp=drive_link
# - Fusion simple+SEC (gemap_simple_r50_sec_110ep, official ~66.5 mAP):
#   https://drive.google.com/file/d/1fgHcEcCC2EmUOl8wzqY1Y8lBvG6abXID/view?usp=drive_link
# - Optional SOTA camera full (gemap_full_r50_110ep, official ~69.4 mAP; do NOT Δ vs fusion):
#   https://drive.google.com/file/d/1-tSNztiVRXXlofiVStj3KzLTG1sbtKtF/view?usp=drive_link

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CKPT_DIR="${ROOT}/checkpoints"

mkdir -p "${CKPT_DIR}/maptr" "${CKPT_DIR}/gemap"

if ! python3 -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('drive.google.com', 443)); s.close()" 2>/dev/null; then
  cat <<EOF
[ERROR] Cannot reach drive.google.com from this machine/container.

Workaround (recommended on a networked local PC):
  1) Download with a browser / gdown on a machine that can access Google Drive
  2) Place them at:
       ${CKPT_DIR}/maptr/maptr_tiny_r50_24e.pth
       ${CKPT_DIR}/maptr/maptr_tiny_fusion_24e.pth
       ${CKPT_DIR}/gemap/gemap_simple_r50_110ep.pth
       ${CKPT_DIR}/gemap/gemap_simple_r50_sec_110ep.pth
       ${CKPT_DIR}/gemap/gemap_full_r50_110ep.pth   # optional SOTA camera
  3) Because 111/ is bind-mounted into the container, they appear automatically.

Direct Google Drive links:
  MapTR camera        : https://drive.google.com/file/d/1n1FUFnRqdskvmpLdnsuX_VK6pET19h95/view?usp=share_link
  MapTR fusion        : https://drive.google.com/file/d/1CFlJrl3ZDj3gIOysf5Cli9bX5LEYSYO4/view?usp=share_link
  GeMap simple camera : https://drive.google.com/file/d/1QNmluapTm_hH-ofMi_QsKXLG2bp38wEW/view?usp=drive_link
  GeMap simple fusion : https://drive.google.com/file/d/1fgHcEcCC2EmUOl8wzqY1Y8lBvG6abXID/view?usp=drive_link
  GeMap full camera   : https://drive.google.com/file/d/1-tSNztiVRXXlofiVStj3KzLTG1sbtKtF/view?usp=drive_link
EOF
  exit 1
fi

export PATH="${HOME}/.local/bin:${PATH}"
python3 -m pip install -q --user gdown || true

python3 -m gdown --fuzzy "https://drive.google.com/file/d/1n1FUFnRqdskvmpLdnsuX_VK6pET19h95/view?usp=share_link" -O "${CKPT_DIR}/maptr/maptr_tiny_r50_24e.pth"
python3 -m gdown --fuzzy "https://drive.google.com/file/d/1CFlJrl3ZDj3gIOysf5Cli9bX5LEYSYO4/view?usp=share_link" -O "${CKPT_DIR}/maptr/maptr_tiny_fusion_24e.pth"

# Modality-fair GeMap pair (simple objective).
python3 -m gdown --fuzzy "https://drive.google.com/file/d/1QNmluapTm_hH-ofMi_QsKXLG2bp38wEW/view?usp=drive_link" -O "${CKPT_DIR}/gemap/gemap_simple_r50_110ep.pth"
python3 -m gdown --fuzzy "https://drive.google.com/file/d/1fgHcEcCC2EmUOl8wzqY1Y8lBvG6abXID/view?usp=drive_link" -O "${CKPT_DIR}/gemap/gemap_simple_r50_sec_110ep.pth"

# Optional SOTA camera (full objective); not used for C vs F Δ.
if [[ "${DOWNLOAD_GEMAP_FULL:-0}" == "1" ]]; then
  python3 -m gdown --fuzzy "https://drive.google.com/file/d/1-tSNztiVRXXlofiVStj3KzLTG1sbtKtF/view?usp=drive_link" -O "${CKPT_DIR}/gemap/gemap_full_r50_110ep.pth"
fi

# Convenience aliases used by older docs.
ln -sfn gemap_simple_r50_110ep.pth "${CKPT_DIR}/gemap/gemap_simple_r50_110ep_camera.pth"
ln -sfn gemap_simple_r50_sec_110ep.pth "${CKPT_DIR}/gemap/gemap_simple_r50_sec_110ep_fusion.pth"

echo "[DONE] Downloaded checkpoints:"
ls -lh "${CKPT_DIR}/maptr" "${CKPT_DIR}/gemap"
