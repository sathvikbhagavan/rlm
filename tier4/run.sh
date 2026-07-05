#!/usr/bin/env bash

# Build the REPL sandbox image (packages exported from ../../.rlm):
#   ./build_rlm_sandbox.sh

# Kill each task16 invocation after 4 hours (GNU timeout exit code 124).
TASK16_TIMEOUT=4h

run_task16() {
  timeout --foreground --kill-after=30s "${TASK16_TIMEOUT}" python3 rlm_task16.py "$@"
}

# python3 rlm_task17b.py --context-size 100 --environment local
# python3 rlm_task17b.py --context-size 100 --environment local
# python3 rlm_task17b.py --context-size 100 --environment local
# python3 rlm_task17b.py --context-size 100 --environment local
# python3 rlm_task17b.py --context-size 100 --environment local

# python3 rlm_task17b.py --context-size 500 --environment local
# python3 rlm_task17b.py --context-size 500 --environment local
# python3 rlm_task17b.py --context-size 500 --environment local
# python3 rlm_task17b.py --context-size 500 --environment local
# python3 rlm_task17b.py --context-size 500 --environment local

# python3 rlm_task17b.py --context-size -1 --environment docker
# python3 rlm_task17b.py --context-size -1 --environment docker
# python3 rlm_task17b.py --context-size -1 --environment docker
# python3 rlm_task17b.py --context-size -1 --environment docker
# python3 rlm_task17b.py --context-size -1 --environment docker

# run_task16 --context-size 100 --environment local
# run_task16 --context-size 100 --environment local
# run_task16 --context-size 100 --environment local
# run_task16 --context-size 100 --environment local
# run_task16 --context-size 100 --environment local

# run_task16 --context-size 500 --environment local
# run_task16 --context-size 500 --environment local
# run_task16 --context-size 500 --environment local
# run_task16 --context-size 500 --environment local
# run_task16 --context-size 500 --environment local

# run_task16 --context-size -1 --environment docker
# run_task16 --context-size -1 --environment docker
run_task16 --context-size -1 --environment docker
run_task16 --context-size -1 --environment docker
run_task16 --context-size -1 --environment docker
run_task16 --context-size -1 --environment docker
run_task16 --context-size -1 --environment docker