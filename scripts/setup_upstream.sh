#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${TRIPOSPLAT_REPO:-${ROOT}/vendor/TripoSplat}"
UPSTREAM_URL="${TRIPOSPLAT_URL:-https://github.com/VAST-AI-Research/TripoSplat.git}"
UPSTREAM_COMMIT="${TRIPOSPLAT_COMMIT:-a78fa12d06dbf1381ca548bfac32bb68cb8c451d}"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${UPSTREAM_DIR}")"
  git clone "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
fi

git -C "${UPSTREAM_DIR}" fetch --tags origin
git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_COMMIT}"
printf 'TripoSplat upstream: %s at %s\n' "${UPSTREAM_DIR}" "$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
