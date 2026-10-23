#!/usr/bin/env bash
# Source this file: source scripts/activate_environment.sh
_marl_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MARL_VENV="${MARL_VENV:-${_marl_root}/.venv}"
if [[ ! -f "${MARL_VENV}/bin/activate" ]]; then
    echo "Missing ${MARL_VENV}; run python3.12 scripts/setup_environment.py first." >&2
    return 1
fi
source "${MARL_VENV}/bin/activate"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${_marl_root}/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_marl_root}/.cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1
unset _marl_root
