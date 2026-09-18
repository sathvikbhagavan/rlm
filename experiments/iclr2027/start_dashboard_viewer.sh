#!/usr/bin/env bash
set -euo pipefail

dashboard_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dashboard_session="rxn-dashboard-view"
dashboard_key="$HOME/.wandb_api_key"

if [[ ! -f "$dashboard_key" ]]; then
  echo "Missing $dashboard_key" >&2
  exit 2
fi
chmod 600 "$dashboard_key"

tmux has-session -t "$dashboard_session" 2>/dev/null && tmux kill-session -t "$dashboard_session"
printf -v dashboard_command \
  'cd %q && uv run --frozen rxnhaystack dashboard view --entity liac --project rxnhaystack-dashboard --source liac/rxnhaystack-control-room --source sathvikbhagavan-epfl/rxnhaystack-dashboard --secret-file %q' \
  "$dashboard_repo" "WANDB_API_KEY=$dashboard_key"
tmux new-session -d -s "$dashboard_session" "$dashboard_command"

for _ in $(seq 1 300); do
  if curl -fsS http://127.0.0.1:8765/index.html >/dev/null 2>&1; then
    echo "RxnHaystack Dashboard is ready on 127.0.0.1:8765"
    exit 0
  fi
  sleep 1
done

echo "Dashboard viewer did not become ready; recent output:" >&2
tmux capture-pane -pt "$dashboard_session":0 -S -30 >&2 || true
exit 1
