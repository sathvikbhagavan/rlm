#!/usr/bin/env bash
# Run the Docker-required Claude cells conservatively on liacpc14.
#
# This is intentionally a one-cell-at-a-time queue. It checks the live
# OpenRouter key balance before every paid cell and retains USD 50 for other
# project work. Existing successes and running cells are skipped by the SQLite
# ledger; failed cells are not retried by this script.

set -uo pipefail

project_root="/home/amin/rlm"
experiment_file="experiments/iclr2027/full-campaign.toml"
openrouter_key_file="/home/amin/.openrouter_api_key"
minimum_remaining_usd="50"
log_file="${RXNHAYSTACK_DOCKER_QUEUE_LOG:-${project_root}/artifacts/iclr2027-six-model-full-v34/docker-claude-queue.log}"
dry_run="${RXNHAYSTACK_DOCKER_QUEUE_DRY_RUN:-0}"

secret_arguments=(
  --secret-file "OPENROUTER_API_KEY=${openrouter_key_file}"
  --secret-file "SWISSAI_RESEARCH_API_KEY=/home/amin/.swissai_research_api_key"
  --secret-file "WANDB_API_KEY=/home/amin/.wandb_api_key"
)

mkdir -p "$(dirname "${log_file}")"
cd "${project_root}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${log_file}"
}

openrouter_remaining_usd() {
  local api_key response remaining
  api_key="$(tr -d '\r\n' < "${openrouter_key_file}")"
  response="$(
    curl --fail --silent --show-error https://openrouter.ai/api/v1/auth/key \
      -H "Authorization: Bearer ${api_key}"
  )" || return 1
  remaining="$(
    printf '%s\n' "${response}" \
      | sed -nE 's/.*"limit_remaining":([0-9]+([.][0-9]+)?).*/\1/p'
  )"
  if [[ ! "${remaining}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    return 1
  fi
  printf '%s\n' "${remaining}"
}

balance_allows_paid_run() {
  local remaining
  if ! remaining="$(openrouter_remaining_usd)"; then
    log "STOP paid queue: OpenRouter balance could not be verified"
    return 1
  fi
  log "OpenRouter remaining USD ${remaining}; protected reserve USD ${minimum_remaining_usd}"
  awk -v remaining="${remaining}" -v reserve="${minimum_remaining_usd}" \
    'BEGIN { exit !(remaining > reserve) }'
}

run_one() {
  local run_id="$1"
  if [[ "${dry_run}" == "1" ]]; then
    log "DRY-RUN ${run_id}"
    return 0
  fi
  if ! balance_allows_paid_run; then
    return 2
  fi
  log "START ${run_id}"
  if uv run --frozen rxnhaystack run "${experiment_file}" \
    --select "${run_id}" \
    --max-parallel 1 \
    --max-run-seconds 21600 \
    "${secret_arguments[@]}" >> "${log_file}" 2>&1; then
    log "FINISH ${run_id}"
    return 0
  fi
  log "FAIL ${run_id}; preserving artifacts and continuing without retry"
  return 1
}

log "Claude Docker queue begins at Git $(git rev-parse HEAD)"

# Low-context cells come first so an overnight run yields the largest number
# of completed experimental cells before the slower full-corpus work.
for scale in x100 x500 xfull; do
  for task in task17 task17b task16; do
    for repetition in 1 2 3 4 5; do
      run_id="full-claude-haiku-4.5-tier4-${task}-rlm-${scale}-r0${repetition}"
      run_one "${run_id}"
      result=$?
      if (( result == 2 )); then
        log "Claude Docker queue stopped by the balance guard"
        exit 0
      fi
    done
  done
done

log "Claude Docker queue exhausted its 45 assigned run IDs"
