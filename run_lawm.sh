#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lawm_prefix=${LAWM_CONDA_PREFIX:-/data2/xuchuan/conda_envs/lawm}
if [[ ! -x "$lawm_prefix/bin/python" ]]; then
  echo "LAWM environment missing: $lawm_prefix; run scripts/setup_environment.sh" >&2
  exit 1
fi
export PATH="$lawm_prefix/bin:$PATH"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM="$MUJOCO_GL"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
cd "$repo_dir"
if [[ ${1:-} == --check ]]; then
  shift
  exec "$lawm_prefix/bin/python" -m lawm_deploy.preflight "$@"
fi
exec "$lawm_prefix/bin/python" -m lawm_deploy --config configs/real.yaml "$@"
