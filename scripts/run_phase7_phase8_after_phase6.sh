#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jcz/sure/cradle"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %T')] Waiting for Phase 6 jobs to finish..."
while pgrep -af "python scripts/generate_scenarios.py --type [12]" >/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] Phase 6 finished, starting Phase 7 evaluation"
conda run --no-capture-output -n cradle python "$ROOT/scripts/evaluate.py" \
  > "$LOG_DIR/phase7_evaluate.log" 2>&1

echo "[$(date '+%F %T')] Phase 7 finished, starting Phase 8 experiments"
conda run --no-capture-output -n cradle python - <<'PY' \
  > "$LOG_DIR/phase8_experiment.log" 2>&1
import yaml
from application.risk_predictor import run_experiment

with open('/home/jcz/sure/cradle/configs/config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

run_experiment(acg_type=1, cfg=cfg)
run_experiment(acg_type=2, cfg=cfg)
PY

echo "[$(date '+%F %T')] Phase 7/8 follow-up completed"
