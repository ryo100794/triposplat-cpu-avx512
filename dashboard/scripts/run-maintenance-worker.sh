#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRIPOSPLAT_DASHBOARD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONF="${TRIPOSPLAT_DASHBOARD_CONF:-${ROOT}/config/dashboard.env}"
source "${CONF}"
export TRIPOSPLAT_DASHBOARD_DSN PGPASSFILE LD_LIBRARY_PATH
export PYTHONPATH="${ROOT}/src"
exec "${ROOT}/.venv/bin/python" -m triposplat_dashboard.maintenance_queue run --poll-seconds 30 --stale-minutes 30
