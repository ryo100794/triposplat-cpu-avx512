#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRIPOSPLAT_DASHBOARD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${ROOT}/run" "${ROOT}/logs"
PID_FILE="${ROOT}/run/dashboard.pid"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  printf 'Dashboard already running: pid=%s\n' "$(cat "${PID_FILE}")"
  exit 0
fi

nohup "${ROOT}/scripts/run-dashboard.sh" >>"${ROOT}/logs/dashboard.log" 2>>"${ROOT}/logs/dashboard-error.log" &
pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"

for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:10101/api/health >/dev/null; then
    printf 'Dashboard started: pid=%s url=http://0.0.0.0:10101\n' "${pid}"
    exit 0
  fi
  sleep 1
done

printf 'Dashboard failed health check; see %s\n' "${ROOT}/logs/dashboard-error.log" >&2
exit 1
