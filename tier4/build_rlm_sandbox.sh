#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${RLM_SANDBOX_IMAGE:-rlm-sandbox}"

"${SCRIPT_DIR}/export_rlm_sandbox_requirements.sh"
docker build -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile.sandbox" "${SCRIPT_DIR}"
echo "Built Docker image: ${IMAGE}"
