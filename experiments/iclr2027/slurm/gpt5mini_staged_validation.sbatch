#!/usr/bin/env bash
# Staged GPT-5-mini validation. Do not submit without an explicit approval sentinel.
#SBATCH --job-name=rxnh-gpt5m-staged-validation
#SBATCH --account=claire
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/smmansou/slurm-%x-%j.out
#SBATCH --error=/home/smmansou/slurm-%x-%j.err

set -euo pipefail

: "${RXNHAYSTACK_LAUNCH_APPROVED:?Explicit staged-validation approval is required}"
[[ "$RXNHAYSTACK_LAUNCH_APPROVED" == "APPROVED_GPT5MINI_STAGED_VALIDATION" ]] || exit 2
: "${RETRY_COMMIT:?Set RETRY_COMMIT to the reviewed retry-definition commit}"
[[ "$RETRY_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2

readonly SOURCE_REPO=git@github.com:sathvikbhagavan/rlm.git
readonly UV=/home/smmansou/.local/bin/uv
readonly SHARD_ROOT="/home/smmansou/rxnhaystack-staged-validation-${RETRY_COMMIT:0:12}"
readonly REPO="$SHARD_ROOT/repo"
readonly SCORING_MANIFEST=experiments/iclr2027/retry-gpt5mini-scoring-pilots.toml
readonly MEMORY_MANIFEST=experiments/iclr2027/retry-gpt5mini-task13-memory-pilot.toml

[[ "${SLURM_CPUS_PER_TASK:-0}" -ge 12 && "${SLURM_MEM_PER_NODE:-0}" -ge 65536 ]] || exit 2
for secret in .openrouter_api_key .wandb_api_key; do
  path="/home/smmansou/$secret"
  [[ -f "$path" && "$(stat -c %a "$path")" == 600 ]] || exit 2
done

mkdir -p "$SHARD_ROOT"
if [[ ! -d "$REPO/.git" ]]; then
  git clone "$SOURCE_REPO" "$REPO"
fi
git -C "$REPO" checkout --detach "$RETRY_COMMIT"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$RETRY_COMMIT" ]]
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet
cd "$REPO"
"$UV" sync --frozen

child_pid=
forward_interrupt() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -INT "$child_pid"
    wait "$child_pid" || true
  fi
  exit 143
}
trap forward_interrupt TERM INT

run_one() {
  local manifest=$1
  local selector=$2
  "$UV" run --frozen rxnhaystack run "$manifest" \
    --data-dir /home/smmansou/datasets/rxnhaystack \
    --select "$selector" \
    --max-parallel 1 \
    --retry-failed \
    --secret-file OPENROUTER_API_KEY=/home/smmansou/.openrouter_api_key \
    --secret-file WANDB_API_KEY=/home/smmansou/.wandb_api_key &
  child_pid=$!
  wait "$child_pid"
  child_pid=
}

# Each command must succeed before the next stage starts. In particular, r02 is
# unreachable unless the complete r01 job has a successful ledger result.
run_one "$SCORING_MANIFEST" pilot-gpt5mini-tier4-task14-rlm-xfull-r02
run_one "$SCORING_MANIFEST" pilot-gpt5mini-tier4-task15-rlm-x100-r01
run_one "$MEMORY_MANIFEST" pilot-gpt5mini-task13-memory8192-xfull-r01
run_one "$MEMORY_MANIFEST" pilot-gpt5mini-task13-memory8192-xfull-r02
