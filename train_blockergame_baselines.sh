#!/bin/sh
# Train IQL, VDN, and QMIX on the blocker game in sequence; each run writes under ./log/<log-dir>/ via main.py.
set -e
cd "$(dirname "$0")"

BASE_LOG="blockergame_baseline"
APP="blocker"
EPOCHS="${BASELINE_EPOCHS:-60000}"
BATCH="${BASELINE_BATCH:-128}"
BUF="${BASELINE_BUFFER:-1000}"
MAXT="${BASELINE_MAX_ENV_T:-24}"

for algo in iql vdn qmix; do
  echo "========== Training ${algo} =========="
  python3 main.py \
    --rl-model "${algo}" \
    --application "${APP}" \
    --log-dir "${BASE_LOG}_${algo}" \
    --batch-size "${BATCH}" \
    --buffer-size "${BUF}" \
    --max-env-t "${MAXT}" \
    --training-epochs "${EPOCHS}" \
    --epsilon-scheduler linear \
    --epsilon-finish 0.05
done

echo "All baselines finished. Logs: ./log/${BASE_LOG}_*/"
