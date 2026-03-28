#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="${1:-cradle}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Please install Miniconda or Anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[INFO] Updating existing conda environment: ${ENV_NAME}"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "[INFO] Creating conda environment: ${ENV_NAME}"
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

echo "[INFO] Running Phase 1 environment validation"
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/scripts/check_env.py"
