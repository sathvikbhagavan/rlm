#!/usr/bin/env bash

# Arize Phoenix tracing environment options (project-agnostic).
# Usage:
#   source ./arize_exports.sh
#   use_phoenix_local
#   start_phoenix_local_storage
#   # then run your script manually

# ---------------------------------------------------------------------------
# Core tracing toggles
# ---------------------------------------------------------------------------
export TRACING_ENABLE="${TRACING_ENABLE:-1}"
export TRACING_BATCH="${TRACING_BATCH:-0}"   # 1 = batch span processor
export PHOENIX_PROJECT_NAME="${PHOENIX_PROJECT_NAME:-phoenix-tracing-dev}"
export PHOENIX_STORAGE_DIR="${PHOENIX_STORAGE_DIR:-/users/sathvikbhagavan/rlms/phoenix}"
export PHOENIX_WORKING_DIR="${PHOENIX_WORKING_DIR:-${PHOENIX_STORAGE_DIR}}"
export PHOENIX_SQL_DATABASE_URL="${PHOENIX_SQL_DATABASE_URL:-sqlite:///${PHOENIX_STORAGE_DIR}/phoenix.db}"

# Optional span context attributes (if your app uses these)
export TRACE_SESSION_ID="${TRACE_SESSION_ID:-}"
export TRACE_USER_ID="${TRACE_USER_ID:-}"

# ---------------------------------------------------------------------------
# OpenInference masking/privacy options
# Set to true/false as needed.
# ---------------------------------------------------------------------------
export OPENINFERENCE_HIDE_INPUTS="${OPENINFERENCE_HIDE_INPUTS:-false}"
export OPENINFERENCE_HIDE_OUTPUTS="${OPENINFERENCE_HIDE_OUTPUTS:-false}"
export OPENINFERENCE_HIDE_INPUT_MESSAGES="${OPENINFERENCE_HIDE_INPUT_MESSAGES:-false}"
export OPENINFERENCE_HIDE_OUTPUT_MESSAGES="${OPENINFERENCE_HIDE_OUTPUT_MESSAGES:-false}"
export OPENINFERENCE_HIDE_INPUT_IMAGES="${OPENINFERENCE_HIDE_INPUT_IMAGES:-false}"
export OPENINFERENCE_HIDE_INPUT_TEXT="${OPENINFERENCE_HIDE_INPUT_TEXT:-false}"
export OPENINFERENCE_HIDE_OUTPUT_TEXT="${OPENINFERENCE_HIDE_OUTPUT_TEXT:-false}"
export OPENINFERENCE_HIDE_EMBEDDING_VECTORS="${OPENINFERENCE_HIDE_EMBEDDING_VECTORS:-false}"
export OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS="${OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS:-false}"
export OPENINFERENCE_HIDE_LLM_PROMPTS="${OPENINFERENCE_HIDE_LLM_PROMPTS:-false}"
export OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH="${OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH:-32000}"

# ---------------------------------------------------------------------------
# Phoenix endpoint profiles
# Call one of these after sourcing this file.
# ---------------------------------------------------------------------------
use_phoenix_local() {
  export PHOENIX_COLLECTOR_ENDPOINT="${PHOENIX_COLLECTOR_ENDPOINT:-http://localhost:6006}"
  export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
  export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${PHOENIX_COLLECTOR_ENDPOINT%/}/v1/traces"
  unset OTEL_EXPORTER_OTLP_ENDPOINT
  unset PHOENIX_API_KEY
  echo "Phoenix local profile active: PHOENIX_COLLECTOR_ENDPOINT=${PHOENIX_COLLECTOR_ENDPOINT}"
  echo "Local Phoenix storage dir: ${PHOENIX_STORAGE_DIR}"
}


# Local Phoenix process files.
export PHOENIX_PID_FILE="${PHOENIX_PID_FILE:-${PHOENIX_STORAGE_DIR}/phoenix.pid}"
export PHOENIX_LOG_FILE="${PHOENIX_LOG_FILE:-${PHOENIX_STORAGE_DIR}/phoenix.log}"

_phoenix_start_cmd() {
  if command -v phoenix >/dev/null 2>&1; then
    echo "phoenix serve"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3 -m phoenix.server.main serve"
    return 0
  fi
  echo ""
}

_kill_stale_phoenix_processes() {
  pkill -f "phoenix.server.main serve" >/dev/null 2>&1 || true
  pkill -f "phoenix serve" >/dev/null 2>&1 || true
}

_free_phoenix_grpc_port() {
  # Phoenix starts both HTTP (6006) and gRPC (4317) collectors.
  if command -v fuser >/dev/null 2>&1; then
    fuser -k 4317/tcp >/dev/null 2>&1 || true
    return 0
  fi
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -t -iTCP:4317 -sTCP:LISTEN 2>/dev/null || true)"
    if [ -n "${pids}" ]; then
      kill ${pids} >/dev/null 2>&1 || true
    fi
  fi
}

# Start local Phoenix with persistent storage in PHOENIX_STORAGE_DIR (background).
start_phoenix_local_storage() {
  clean_start="${1:-}"
  mkdir -p "${PHOENIX_STORAGE_DIR}"

  if [ "${clean_start}" = "--clean" ]; then
    echo "Performing clean Phoenix startup..."
    _kill_stale_phoenix_processes
    _free_phoenix_grpc_port
    rm -f "${PHOENIX_PID_FILE}"
  fi

  if [ -f "${PHOENIX_PID_FILE}" ]; then
    old_pid="$(cat "${PHOENIX_PID_FILE}" 2>/dev/null)"
    if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "Phoenix is already running (pid=${old_pid})."
      echo "Log: ${PHOENIX_LOG_FILE}"
      return 0
    fi
    rm -f "${PHOENIX_PID_FILE}"
  fi

  start_cmd="$(_phoenix_start_cmd)"
  if [ -z "${start_cmd}" ]; then
    echo "Could not find phoenix CLI or python3."
    echo "Install with: python3 -m pip install --user arize-phoenix"
    return 1
  fi

  echo "Starting Phoenix at http://localhost:6006"
  echo "Persisting traces under: ${PHOENIX_STORAGE_DIR}"
  nohup bash -lc "${start_cmd}" > "${PHOENIX_LOG_FILE}" 2>&1 &
  new_pid=$!
  echo "${new_pid}" > "${PHOENIX_PID_FILE}"
  sleep 2
  if kill -0 "${new_pid}" 2>/dev/null; then
    echo "Phoenix started in background (pid=${new_pid})."
    echo "Log: ${PHOENIX_LOG_FILE}"
    return 0
  fi
  echo "Phoenix failed to start. Check log: ${PHOENIX_LOG_FILE}"
  rm -f "${PHOENIX_PID_FILE}"
  return 1
}

restart_phoenix_local_storage() {
  start_phoenix_local_storage --clean
}

phoenix_status() {
  if [ -f "${PHOENIX_PID_FILE}" ]; then
    pid="$(cat "${PHOENIX_PID_FILE}" 2>/dev/null)"
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      echo "Phoenix is running (pid=${pid})."
      echo "Log: ${PHOENIX_LOG_FILE}"
      return 0
    fi
  fi
  echo "Phoenix is not running."
}

stop_phoenix_local_storage() {
  if [ ! -f "${PHOENIX_PID_FILE}" ]; then
    echo "No PID file found. Phoenix may already be stopped."
    return 0
  fi
  pid="$(cat "${PHOENIX_PID_FILE}" 2>/dev/null)"
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    echo "Stopped Phoenix (pid=${pid})."
  else
    echo "Phoenix process not found for pid=${pid}."
  fi
  rm -f "${PHOENIX_PID_FILE}"
}

# Default profile on source: local.
use_phoenix_local
start_phoenix_local_storage
