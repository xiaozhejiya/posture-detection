#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/runtime.log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}" || exit 1

{
  echo "[$(date --iso-8601=seconds)] Starting posture detection"
  if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found: ${PROJECT_ROOT}/.venv"
    exit 1
  fi
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  exec python main.py --source-type usb --camera-id "${POSTURE_CAMERA_ID:-0}" --calibrate-on-start "$@"
} >> "${LOG_FILE}" 2>&1
