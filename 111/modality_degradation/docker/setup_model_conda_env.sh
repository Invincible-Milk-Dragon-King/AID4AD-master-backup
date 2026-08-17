#!/usr/bin/env bash
set -euo pipefail

# Run INSIDE a model container.
# Usage:
#   bash modality_degradation/docker/setup_model_conda_env.sh <model>

MODEL="${1:?model required}"
ENV_NAME="${MODEL}"

# Auto-detect code root:
# - /workspace/AID4AD-master/111  (recommended)
# - /workspace/AID4AD-master      (when 111 is mounted directly)
# - allow manual override via AID4AD_111_ROOT
if [[ -n "${AID4AD_111_ROOT:-}" ]]; then
  ROOT="${AID4AD_111_ROOT}"
elif [[ -d "/workspace/AID4AD-master/111/HDMapNet" ]]; then
  ROOT="/workspace/AID4AD-master/111"
elif [[ -d "/workspace/AID4AD-master/HDMapNet" ]]; then
  ROOT="/workspace/AID4AD-master"
else
  echo "[ERROR] Cannot locate code root."
  echo "Tried:"
  echo "  /workspace/AID4AD-master/111"
  echo "  /workspace/AID4AD-master"
  echo "Set AID4AD_111_ROOT manually and retry, e.g.:"
  echo "  export AID4AD_111_ROOT=/workspace/AID4AD-master/111"
  exit 2
fi
echo "[INFO] Using code root: ${ROOT}"

if conda tos --help &>/dev/null; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] conda env ${ENV_NAME} exists"
else
  conda create -y -n "${ENV_NAME}" python=3.8
fi
conda activate "${ENV_NAME}"

pip install --no-cache-dir torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-cache-dir mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install --no-cache-dir mmdet==2.14.0 mmsegmentation==0.14.1 timm
pip install --no-cache-dir gdown prettytable tqdm ipython tensorboardX pyquaternion shapely nuscenes-devkit
# mmcv Config.pretty_text needs yapf with FormatCode(..., verify=True); removed in yapf>=0.40.2
pip install --no-cache-dir "yapf==0.40.1"
# torch1.9 tensorboard uses distutils.version.LooseVersion; broken on setuptools>=60
pip install --no-cache-dir "setuptools==59.5.0"

case "${MODEL}" in
  hdmapnet)
    cd "${ROOT}/HDMapNet"
    if [[ -f requirement.txt ]]; then pip install --no-cache-dir -r requirement.txt; fi
    ;;
  maptr)
    cd "${ROOT}/MapTR/mmdetection3d" && python setup.py develop
    cd "${ROOT}/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn" && python setup.py build install
    cd "${ROOT}/MapTR" && pip install --no-cache-dir -r requirement.txt
    ;;
  admap)
    # ADMap vendors mmdet3d at repo root (no separate mmdetection3d/ like MapTR).
    cd "${ROOT}/ADMap-main" && python setup.py develop
    cd "${ROOT}/ADMap-main/mmdet3d/maptr/modules/ops/geometric_kernel_attn" && python setup.py build install
    cd "${ROOT}/ADMap-main" && pip install --no-cache-dir -r requirement.txt
    ;;
  gemap)
    cd "${ROOT}/GeMap/mmdetection3d" && python setup.py develop
    cd "${ROOT}/GeMap/projects/mmdet3d_plugin/gemap/modules/ops/geometric_kernel_attn" && python setup.py build install
    cd "${ROOT}/GeMap" && pip install --no-cache-dir -r requirement.txt
    ;;
  himap)
    cd "${ROOT}/HIMap/mmdetection3d" && python setup.py develop
    # HIMap has maptr-style ops; install if present.
    if [[ -d "${ROOT}/HIMap/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn" ]]; then
      cd "${ROOT}/HIMap/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn" && python setup.py build install
    fi
    # hdmap_eval.py imports chamferdist (not always listed in older requirement.txt).
    pip install --no-cache-dir chamferdist
    cd "${ROOT}/HIMap" && pip install --no-cache-dir -r requirement.txt
    # Later pip installs may pull yapf/setuptools/protobuf too new for mmcv/torch1.9/py3.8.
    pip install --no-cache-dir "yapf==0.40.1" "setuptools==59.5.0" "protobuf==3.20.3" "tensorboard==2.11.2"
    ;;
  *)
    echo "Unsupported model: ${MODEL}" && exit 2
    ;;
esac

echo "[DONE] Environment ready: ${ENV_NAME}"
