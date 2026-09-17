#!/usr/bin/env bash
set -euo pipefail

dashboard_owner="${1:?Usage: start_dashboard_reporting.sh OWNER MACHINE [SEARCH_ROOT]}"
dashboard_machine="${2:?Usage: start_dashboard_reporting.sh OWNER MACHINE [SEARCH_ROOT]}"
dashboard_search_root="${3:-$HOME}"
dashboard_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dashboard_key="$HOME/.wandb_api_key"

if [[ ! -f "$dashboard_key" ]]; then
  echo "Missing $dashboard_key" >&2
  exit 2
fi
chmod 600 "$dashboard_key"
if ! command -v tmux >/dev/null; then
  echo "tmux is required" >&2
  exit 2
fi

declare -A dashboard_seen_inodes=()
dashboard_started=0
dashboard_existing=0
dashboard_skipped=0

while IFS= read -r -d '' dashboard_ledger; do
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

  dashboard_path_hash="$(printf '%s' "$dashboard_ledger" | sha256sum | cut -c1-8)"
  dashboard_source="${dashboard_machine}-${dashboard_kind}-${dashboard_path_hash}"
  dashboard_session="rxnhaystack-dashboard-${dashboard_source}"

  if tmux has-session -t "$dashboard_session" 2>/dev/null; then
    echo "Already reporting: $dashboard_ledger ($dashboard_session)"
    dashboard_existing=$((dashboard_existing + 1))
    continue
  fi

  uv run --directory "$dashboard_repo" --frozen rxnhaystack dashboard update \
    "$dashboard_experiment" \
    --ledger-path "$dashboard_ledger" \
    --source-id "$dashboard_source" \
    --machine "$dashboard_machine" \
    --owner "$dashboard_owner" \
    --local-only

  printf -v dashboard_command \
    'cd %q && uv run --frozen rxnhaystack dashboard update %q --ledger-path %q --source-id %q --machine %q --owner %q --watch-seconds 300 --heartbeat-seconds 1800 --secret-file %q' \
    "$dashboard_repo" "$dashboard_experiment" "$dashboard_ledger" "$dashboard_source" \
    "$dashboard_machine" "$dashboard_owner" "WANDB_API_KEY=$dashboard_key"
  tmux new-session -d -s "$dashboard_session" "$dashboard_command"
  echo "Started: $dashboard_ledger ($dashboard_session, $dashboard_observed observed jobs)"
  dashboard_started=$((dashboard_started + 1))
done < <(
  find "$dashboard_search_root" -type f \
    \( -path '*/artifacts/iclr2027-six-model-full-v34/ledger.sqlite3' \
    -o -path '*/artifacts/iclr2027-matched-cardinality-v7/ledger.sqlite3' \) \
    -print0 2>/dev/null
)

if ((dashboard_started == 0 && dashboard_existing == 0)); then
  echo "No populated full-v34 or matched-v7 ledgers were found under $dashboard_search_root" >&2
  exit 1
fi

echo "Dashboard reporting ready: started=$dashboard_started, already_running=$dashboard_existing, empty_skipped=$dashboard_skipped"
