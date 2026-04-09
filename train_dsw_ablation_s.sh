#!/bin/sh
# DSW ablation dsw-qmix-s: constant λ = 0.5 (no cost_weight MLP)
set -e
cd "$(dirname "$0")"

BASE_LOG="${DSW_ABL_LOG_ROOT:-blockergame_ablation}"
APP="${DSW_ABL_APP:-blocker}"
EPOCHS="${DSW_ABL_EPOCHS:-60000}"
BATCH="${DSW_ABL_BATCH:-128}"
BUF="${DSW_ABL_BUFFER:-8000}"
MAXT="${DSW_ABL_MAX_ENV_T:-24}"
SEQLEN="${DSW_ABL_SEQ_LEN:-24}"

COMMON="python3 main.py --rl-model rnn --mixer multi-qmix --policy-disc --hard-mixer-mono --application ${APP} \
  --batch-size ${BATCH} --buffer-size ${BUF} --max-env-t ${MAXT} --seq-len ${SEQLEN} \
  --training-epochs ${EPOCHS} --epsilon-scheduler linear --epsilon-finish 0.1 --blocker-shaping-scale 0.08"

echo "========== dsw-qmix-s (static cost weight 1.0, hard |w| + soft mono) =========="
${COMMON} --log-dir "${BASE_LOG}_dsw_qmix_s" --static-cost-weight 1.0
