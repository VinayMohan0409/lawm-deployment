#!/usr/bin/env bash
# One independent conda prefix for hardware + model. No existing teleop environment is modified.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
lawm_prefix=${LAWM_CONDA_PREFIX:-$repo_dir/.conda/lawm}
conda_executable=${LAWM_CONDA_EXE:-conda}
model_source=${1:-}
export PYTHONNOUSERSITE=1
if [[ ! -x "$lawm_prefix/bin/python" ]]; then
  "$conda_executable" create --prefix "$lawm_prefix" python=3.10 pip -y
fi
"$lawm_prefix/bin/python" -m pip install 'torch==2.5.0' 'torchvision==0.20.0' --index-url https://download.pytorch.org/whl/cu124
"$lawm_prefix/bin/python" -m pip install 'spconv-cu124==2.3.8' 'timm==1.0.27' addict einops \
  'torch-scatter==2.1.2+pt25cu124' -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
"$lawm_prefix/bin/python" -m pip install -e "$repo_dir[real,dev]" 'pyrealsense2==2.56.5.9235'
if [[ -n ${SHARPA_SDK_PYTHON:-} ]]; then
  site_dir=$("$lawm_prefix/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  printf '%s\n' "$SHARPA_SDK_PYTHON" > "$site_dir/sharpa_vendor_sdk.pth"
fi
if [[ -n "$model_source" ]]; then
  "$lawm_prefix/bin/python" -m pip install -e "$model_source"
else
  echo "Model source is still required: clone latent-action-world-model at 5a2ccb8, then rerun with its directory." >&2
fi
"$lawm_prefix/bin/python" -m pip check
cd "$repo_dir"
echo "Environment ready. Activate with: conda activate $lawm_prefix"
