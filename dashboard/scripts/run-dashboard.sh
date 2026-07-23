#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRIPOSPLAT_DASHBOARD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONF="${TRIPOSPLAT_DASHBOARD_CONF:-${ROOT}/config/dashboard.env}"
[[ -f "${CONF}" ]] || { printf 'Missing config: %s\n' "${CONF}" >&2; exit 2; }
source "${CONF}"

export TRIPOSPLAT_DASHBOARD_DSN
export PGPASSFILE
export PYTHONPATH="${ROOT}/src"
cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" -m triposplat_dashboard.server --host 0.0.0.0 --port 10101
