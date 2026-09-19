#!/usr/bin/env bash
set -euo pipefail

dashboard_owner="${1:?Usage: start_dashboard_reporting.sh OWNER MACHINE [SEARCH_ROOT] [--restart]}"
dashboard_machine="${2:?Usage: start_dashboard_reporting.sh OWNER MACHINE [SEARCH_ROOT] [--restart]}"
dashboard_search_root="${3:-$HOME}"
dashboard_mode="${4:-ensure}"
dashboard_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dashboard_key="$HOME/.wandb_api_key"
dashboard_entity="${RXNHAYSTACK_DASHBOARD_ENTITY:-liac}"
dashboard_project="${RXNHAYSTACK_DASHBOARD_PROJECT:-rxnhaystack-dashboard}"

if [[ ! -f "$dashboard_key" ]]; then
  echo "Missing $dashboard_key" >&2
  exit 2
fi
chmod 600 "$dashboard_key"
if ! command -v tmux >/dev/null; then
  echo "tmux is required" >&2
  exit 2
fi
if [[ "$dashboard_mode" != "ensure" && "$dashboard_mode" != "--restart" ]]; then
  echo "Fourth argument must be --restart when supplied" >&2
  exit 2
fi

declare -A dashboard_seen_inodes=()
dashboard_started=0
dashboard_existing=0
dashboard_skipped=0

if [[ "$dashboard_mode" == "--restart" ]]; then
  dashboard_session_prefix="rxnhaystack-dashboard-${dashboard_machine}-"
  while IFS= read -r dashboard_old_session; do
    [[ -n "$dashboard_old_session" ]] && tmux kill-session -t "$dashboard_old_session"
  done < <(
    tmux list-sessions -F '#S' 2>/dev/null \
      | awk -v prefix="$dashboard_session_prefix" 'index($0, prefix) == 1'
  )
  for dashboard_legacy_session in \
    "rxn-dashboard-${dashboard_machine}" "rxn-control-room-${dashboard_machine}"; do
    tmux has-session -t "$dashboard_legacy_session" 2>/dev/null \
      && tmux kill-session -t "$dashboard_legacy_session"
  done
fi

echo "Scanning $dashboard_search_root for authoritative RxnHaystack ledgers..."

while IFS= read -r -d '' dashboard_campaign_dir; do
  dashboard_ledger="$dashboard_campaign_dir/ledger.sqlite3"
  if [[ ! -f "$dashboard_ledger" ]]; then
    continue
  fi
  dashboard_inode="$(stat -Lc '%d:%i' "$dashboard_ledger")"
  if [[ -n "${dashboard_seen_inodes[$dashboard_inode]:-}" ]]; then
    continue
  fi
  dashboard_seen_inodes[$dashboard_inode]=1

  dashboard_directory="$(basename "$(dirname "$dashboard_ledger")")"
  case "$dashboard_directory" in
    iclr2027-six-model-full-v34)
      dashboard_experiment="experiments/iclr2027/full-campaign.toml"
      dashboard_kind="full-v34"
      ;;
    iclr2027-matched-cardinality-v7)
      dashboard_experiment="experiments/iclr2027/matched-cardinality-campaign.toml"
      dashboard_kind="matched-v7"
      ;;
    iclr2027-oracle-predicate-v1)
      dashboard_experiment="experiments/iclr2027/oracle-predicate-campaign.toml"
      dashboard_kind="oracle-predicate-v1"
      ;;
    iclr2027-oracle-executor-v1)
      dashboard_experiment="experiments/iclr2027/oracle-executor-campaign.toml"
      dashboard_kind="oracle-executor-v1"
      ;;
    iclr2027-task16-prospective-decomposition-v1)
      dashboard_experiment="experiments/iclr2027/prospective-decomposition.toml"
      dashboard_kind="prospective-v1"
      ;;
    iclr2027-gpt5mini-direct-openai-docker-v1)
      dashboard_experiment="experiments/iclr2027/gpt-direct-openai-docker-campaign.toml"
      dashboard_kind="gpt-direct-docker-v1"
      ;;
    iclr2027-gpt5mini-direct-openai-recovery-v1)
      dashboard_experiment="experiments/iclr2027/gpt-direct-openai-recovery-campaign.toml"
      dashboard_kind="gpt-direct-recovery-v1"
      ;;
    iclr2027-deepseek-paid-openrouter-continuation-v1)
      dashboard_experiment="experiments/iclr2027/deepseek-paid-openrouter-continuation.toml"
      dashboard_kind="deepseek-paid-continuation-v1"
      ;;
    *)
      continue
      ;;
  esac

  dashboard_observed="$(uv run --directory "$dashboard_repo" --frozen python -c \
    'import sqlite3,sys; c=sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True); print(c.execute("SELECT COUNT(*) FROM runs WHERE attempts > 0").fetchone()[0])' \
    "$dashboard_ledger")"
  if [[ "$dashboard_observed" == "0" ]]; then
    dashboard_skipped=$((dashboard_skipped + 1))
    continue
  fi

  dashboard_models="$(uv run --directory "$dashboard_repo" --frozen python -c \
    'import sqlite3,sys; c=sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True); print("\n".join(str(row[0]) for row in c.execute("SELECT DISTINCT json_extract(spec_json, \"$.model\") FROM runs WHERE attempts > 0") if row[0]))' \
    "$dashboard_ledger")"
  dashboard_assigned=true
  case "$dashboard_machine:$dashboard_kind" in
    jed:full-v34)
      grep -Eqi 'GLM-5\.2' <<<"$dashboard_models" || dashboard_assigned=false
      ;;
    jed:oracle-predicate-v1 | jed:oracle-executor-v1)
      ;;
    jed:*)
      dashboard_assigned=false
      ;;
    kuma:full-v34)
      grep -Eqi 'claude|gpt-5-mini' <<<"$dashboard_models" || dashboard_assigned=false
      ;;
    kuma:matched-v7)
      grep -Eqi 'gpt-5-mini' <<<"$dashboard_models" || dashboard_assigned=false
      ;;
    kuma:*)
      dashboard_assigned=false
      ;;
    liacpc14:full-v34)
      grep -Eqi 'deepseek|claude|GLM-5\.2' <<<"$dashboard_models" || dashboard_assigned=false
      ;;
    liacpc14:oracle-predicate-v1 | liacpc14:prospective-v1 | liacpc14:gpt-direct-docker-v1 | liacpc14:gpt-direct-recovery-v1 | liacpc14:deepseek-paid-continuation-v1)
      ;;
    liacpc14:*)
      dashboard_assigned=false
      ;;
    liacpc15:full-v34)
      grep -Eqi 'qwen|gemini' <<<"$dashboard_models" || dashboard_assigned=false
      ;;
    liacpc15:matched-v7)
      grep -Eqi 'qwen' <<<"$dashboard_models" || dashboard_assigned=false
      ;;
    liacpc15:*)
      dashboard_assigned=false
      ;;
  esac
  if [[ "$dashboard_assigned" != true ]]; then
    continue
  fi

  dashboard_path_hash="$(printf '%s' "$dashboard_ledger" | sha256sum | cut -c1-8)"
  dashboard_source="${dashboard_machine}-${dashboard_kind}-${dashboard_path_hash}"
  dashboard_session="rxnhaystack-dashboard-${dashboard_source}"

  if tmux has-session -t "$dashboard_session" 2>/dev/null; then
    if [[ "$dashboard_mode" == "--restart" ]]; then
      tmux kill-session -t "$dashboard_session"
      echo "Restarting reporter: $dashboard_ledger ($dashboard_session)"
    else
      echo "Already reporting: $dashboard_ledger ($dashboard_session)"
      dashboard_existing=$((dashboard_existing + 1))
      continue
    fi
  fi

  uv run --directory "$dashboard_repo" --frozen rxnhaystack dashboard update \
    "$dashboard_experiment" \
    --ledger-path "$dashboard_ledger" \
    --source-id "$dashboard_source" \
    --machine "$dashboard_machine" \
    --owner "$dashboard_owner" \
    --entity "$dashboard_entity" \
    --project "$dashboard_project" \
    --local-only

  printf -v dashboard_command \
    'cd %q && uv run --frozen rxnhaystack dashboard update %q --ledger-path %q --source-id %q --machine %q --owner %q --entity %q --project %q --watch-seconds 300 --heartbeat-seconds 1800 --secret-file %q' \
    "$dashboard_repo" "$dashboard_experiment" "$dashboard_ledger" "$dashboard_source" \
    "$dashboard_machine" "$dashboard_owner" "$dashboard_entity" "$dashboard_project" \
    "WANDB_API_KEY=$dashboard_key"
  tmux new-session -d -s "$dashboard_session" "$dashboard_command"
  echo "Started: $dashboard_ledger ($dashboard_session, $dashboard_observed observed jobs)"
  dashboard_started=$((dashboard_started + 1))
done < <(
  find "$dashboard_search_root" \
    \( -type d \( -name .git -o -name .venv -o -name .cache -o -name wandb \
    -o -name node_modules \) -prune \) -o \
    \( -type d \( -name iclr2027-six-model-full-v34 \
    -o -name iclr2027-matched-cardinality-v7 \
    -o -name iclr2027-oracle-predicate-v1 \
    -o -name iclr2027-oracle-executor-v1 \
    -o -name iclr2027-task16-prospective-decomposition-v1 \
    -o -name iclr2027-gpt5mini-direct-openai-docker-v1 \
    -o -name iclr2027-gpt5mini-direct-openai-recovery-v1 \
    -o -name iclr2027-deepseek-paid-openrouter-continuation-v1 \) -print0 -prune \) \
    2>/dev/null
)

if ((dashboard_started == 0 && dashboard_existing == 0)); then
  echo "No populated authoritative ledgers assigned to $dashboard_machine were found under $dashboard_search_root" >&2
  exit 1
fi

echo "Dashboard reporting ready in $dashboard_entity/$dashboard_project: started=$dashboard_started, already_running=$dashboard_existing, empty_skipped=$dashboard_skipped"
