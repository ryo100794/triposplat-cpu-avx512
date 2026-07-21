#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  printf 'Usage: %s RUN_DIR COMMAND [ARG ...]\n' "$0" >&2
  exit 2
fi

RUN_DIR="$1"
shift
mkdir -p "${RUN_DIR}"

if [[ -s "${RUN_DIR}/job.pid" ]]; then
  old_pid="$(sed -n '1p' "${RUN_DIR}/job.pid")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    printf 'A recorded job is still running: pid=%s\n' "${old_pid}" >&2
    exit 3
  fi
fi

nohup bash -c '
  set +e
  run_dir="$1"
  shift
  started="$(date +%s)"
  printf "%s\n" "${started}" >"${run_dir}/started_epoch"
  "$@" >"${run_dir}/run.log" 2>&1
  rc=$?
  finished="$(date +%s)"
  printf "%s\n" "${rc}" >"${run_dir}/exit_code"
  printf "%s\n" "${finished}" >"${run_dir}/finished_epoch"
  exit "${rc}"
' _ "${RUN_DIR}" "$@" </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${RUN_DIR}/job.pid"
printf 'pid=%s log=%s\n' "${pid}" "${RUN_DIR}/run.log"
