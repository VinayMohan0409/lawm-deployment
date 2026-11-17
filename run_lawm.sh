#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
local_settings=${LAWM_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/lawm-deploy/environment.sh}
if [[ -f "$local_settings" ]]; then
  source "$local_settings"
fi
if [[ -n ${LAWM_CONDA_PREFIX:-} ]]; then
  export PATH="$LAWM_CONDA_PREFIX/bin:$PATH"
  python_executable="$LAWM_CONDA_PREFIX/bin/python"
else
  python_executable=$(command -v python)
fi
config_file=${LAWM_CONFIG:-configs/real.local.yaml}
if [[ -z ${LAWM_CONFIG:-} && ! -f "$repo_dir/$config_file" ]]; then
  config_file=configs/real.yaml
fi
export PYTHONNOUSERSITE=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM="$MUJOCO_GL"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
cd "$repo_dir"
if [[ ${1:-} == --check ]]; then
  shift
  exec "$python_executable" -m lawm_deploy.preflight --config "$config_file" "$@"
fi
exec "$python_executable" -m lawm_deploy --config "$config_file" "$@"
