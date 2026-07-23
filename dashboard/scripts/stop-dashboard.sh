#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRIPOSPLAT_DASHBOARD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PID_FILE="${ROOT}/run/dashboard.pid"
[[ -f "${PID_FILE}" ]] || { printf 'Dashboard is not running\n'; exit 0; }
pid="$(cat "${PID_FILE}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}"
  for _ in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.25
  done
fi
rm -f "${PID_FILE}"
printf 'Dashboard stopped\n'
