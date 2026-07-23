#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRIPOSPLAT_DASHBOARD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PID_FILE="${ROOT}/run/maintenance-worker.pid"
[[ -f "${PID_FILE}" ]] || { printf 'Maintenance worker is not running\n'; exit 0; }
pid="$(cat "${PID_FILE}")"
kill "${pid}" 2>/dev/null || true
rm -f "${PID_FILE}"
printf 'Maintenance worker stopped\n'
