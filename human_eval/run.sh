#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bundled_dataset="human_eval/data/reactionSmilesFigShareUSPTO2023_cleaned.txt"
legacy_dataset="$HOME/datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt"
if [[ -n "${RXNHAYSTACK_CLEANED_DATASET:-}" ]]; then
  dataset_path="$RXNHAYSTACK_CLEANED_DATASET"
elif [[ -f "$bundled_dataset" ]]; then
  dataset_path="$bundled_dataset"
else
  dataset_path="$legacy_dataset"
fi
if [[ ! -f human_eval/generated/canonical-v4/manifest.json ]]; then
  uv run --frozen --with-requirements human_eval/requirements.txt python -m human_eval.cli build-bundle --dataset "$dataset_path"
fi
if [[ ! -f human_eval/local_state/dataset_index.sqlite3 ]]; then
  uv run --frozen --with-requirements human_eval/requirements.txt python -m human_eval.cli index-dataset --dataset "$dataset_path"
fi
exec uv run --frozen --with-requirements human_eval/requirements.txt python -m human_eval.cli serve --dataset "$dataset_path" "$@"
