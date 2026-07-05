#!/usr/bin/env bash
# Export minimal REPL packages from rlms/.rlm (numpy, rdkit + docker runtime deps).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLMS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PIP="${RLMS_ROOT}/.rlm/bin/pip"
OUT_FILE="${SCRIPT_DIR}/rlm-sandbox-requirements.txt"

# dill + requests: required by DockerREPL inside the container.
# numpy + rdkit: tier4 chemistry REPL code (standard library is in the base image).
REPL_PACKAGES=(dill requests numpy rdkit)

if [[ ! -x "${VENV_PIP}" ]]; then
  echo "Missing venv pip: ${VENV_PIP}" >&2
  exit 1
fi

freeze="$("${VENV_PIP}" freeze)"

{
  echo "# Auto-generated from ${RLMS_ROOT}/.rlm via export_rlm_sandbox_requirements.sh"
  echo "# Re-run ./build_rlm_sandbox.sh to refresh before building the image."
  echo "# REPL surface: stdlib + numpy + rdkit (+ dill/requests for DockerREPL runtime)."
  for pkg in "${REPL_PACKAGES[@]}"; do
    line="$(printf '%s\n' "${freeze}" | grep -iE "^${pkg}==" || true)"
    if [[ -z "${line}" ]]; then
      echo "Missing package in ${RLMS_ROOT}/.rlm freeze: ${pkg}" >&2
      exit 1
    fi
    echo "${line}"
  done
} > "${OUT_FILE}"

echo "Wrote $(wc -l < "${OUT_FILE}") lines to ${OUT_FILE}"
