#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

find scripts -maxdepth 1 -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python3 -m py_compile scripts/*.py

if rg -n --glob '!verify_source.sh' '/workspace/3dgs|/root/|~/slam|gdrive:|SUJINASHI|youtube_account|213.173.|BEGIN [A-Z ]*PRIVATE|gho_[A-Za-z0-9]+' README.md README_ja.md docs scripts THIRD_PARTY_NOTICES.md; then
  printf 'Public-source audit found a forbidden path, host, or credential pattern.\n' >&2
  exit 1
fi

printf 'Source verification passed.\n'
