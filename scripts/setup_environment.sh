#!/usr/bin/env bash
# One independent conda prefix for hardware + model. No existing teleop environment is modified.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
lawm_prefix=${LAWM_CONDA_PREFIX:-/data2/xuchuan/conda_envs/lawm}
conda_executable=${LAWM_CONDA_EXE:-/vol/dissolve/xh1126/miniforge3/bin/conda}
model_source=${1:-}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-/data2/xuchuan/conda_pkgs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/data2/xuchuan/lawm_setup/pip_cache}
export PYTHONNOUSERSITE=1
mkdir -p "$PIP_CACHE_DIR"
if [[ ! -x "$lawm_prefix/bin/python" ]]; then
  "$conda_executable" create --prefix "$lawm_prefix" python=3.10 pip -y
fi
"$lawm_prefix/bin/python" -m pip install 'torch==2.5.0' 'torchvision==0.20.0' --index-url https://download.pytorch.org/whl/cu124
"$lawm_prefix/bin/python" -m pip install 'spconv-cu124==2.3.8' 'timm==1.0.27' addict einops \
  'torch-scatter==2.1.2+pt25cu124' -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
"$lawm_prefix/bin/python" -m pip install -e "$repo_dir[real,dev]" 'pyrealsense2==2.56.5.9235'
site_dir=$("$lawm_prefix/bin/python" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "${SHARPA_SDK_PYTHON:-/opt/sharpa-wave-sdk/python}" > "$site_dir/sharpa_vendor_sdk.pth"
if [[ -n "$model_source" ]]; then
  model_commit=$(git -C "$model_source" rev-parse HEAD)
  if [[ $model_commit != 5a2ccb8* ]]; then
    echo "Expected README-pinned LAWM commit 5a2ccb8, got $model_commit. Review version compatibility first." >&2
    exit 1
  fi
  "$lawm_prefix/bin/python" -m pip install -e "$model_source"
else
  echo "Model source is still required: clone latent-action-world-model at 5a2ccb8, then rerun with its directory." >&2
fi
"$lawm_prefix/bin/python" -m pip check
cd "$repo_dir"
"$lawm_prefix/bin/python" -m lawm_deploy.preflight --output reports/environment.json
