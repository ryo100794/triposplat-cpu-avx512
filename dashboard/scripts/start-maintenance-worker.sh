#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRIPOSPLAT_DASHBOARD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${ROOT}/run" "${ROOT}/logs"
PID_FILE="${ROOT}/run/maintenance-worker.pid"
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  printf 'Maintenance worker already running: pid=%s\n' "$(cat "${PID_FILE}")"
  exit 0
fi
nohup "${ROOT}/scripts/run-maintenance-worker.sh" >>"${ROOT}/logs/maintenance-worker.log" 2>&1 &
printf '%s\n' "$!" >"${PID_FILE}"
printf 'Maintenance worker started: pid=%s\n' "$!"
