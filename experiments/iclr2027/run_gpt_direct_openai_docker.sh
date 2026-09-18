#!/usr/bin/env bash
# Gate the 45 direct-OpenAI GPT-5-mini Docker cells on one real archival pilot.

set -uo pipefail

project_root="${RXNHAYSTACK_GPT_DIRECT_ROOT:-/home/amin/rlm-gpt-direct-openai}"
experiment_file="experiments/iclr2027/gpt-direct-openai-docker-campaign.toml"
data_dir="/home/amin/datasets/rxnhaystack"
openai_key_file="/home/amin/.openai_api_key_liac"
wandb_key_file="/home/amin/.wandb_api_key"
log_file="${project_root}/artifacts/iclr2027-gpt5mini-direct-openai-docker-v1/queue.log"
pilot="direct-openai-gpt-5-mini-tier4-task17-rlm-x100-r01"

cd "${project_root}"
mkdir -p "$(dirname "${log_file}")"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${log_file}"
}

api_key="$(tr -d '\r\n' < "${openai_key_file}")"
model_status="$(
  curl --silent --show-error --max-time 30 --output /dev/null --write-out '%{http_code}' \
    https://api.openai.com/v1/models/gpt-5-mini \
    -H "Authorization: Bearer ${api_key}"
)"
unset api_key
if [[ "${model_status}" != "200" ]]; then
  log "STOP direct OpenAI model-access check returned HTTP ${model_status}"
  exit 2
fi
log "Direct OpenAI model-access check passed; numeric balance requires an organization admin key"

secret_arguments=(
  --secret-file "OPENAI_API_KEY=${openai_key_file}"
  --secret-file "WANDB_API_KEY=${wandb_key_file}"
)
common_arguments=(
  --data-dir "${data_dir}"
  --max-parallel 1
  --max-run-seconds 21600
  "${secret_arguments[@]}"
)

log "START release pilot ${pilot} at Git $(git rev-parse HEAD)"
if ! uv run --frozen rxnhaystack run "${experiment_file}" \
  --select "${pilot}" "${common_arguments[@]}" >> "${log_file}" 2>&1; then
  log "STOP release pilot failed; broad GPT Docker work was not launched"
  exit 1
fi
log "PASS release pilot ${pilot}; releasing remaining direct-OpenAI GPT Docker cells"

if uv run --frozen rxnhaystack run "${experiment_file}" \
  "${common_arguments[@]}" >> "${log_file}" 2>&1; then
  log "FINISH all direct-OpenAI GPT Docker cells"
  exit 0
fi
status=$?
log "GPT Docker queue ended with status ${status}; successes remain resumable"
exit "${status}"
