#!/usr/bin/env sh
# Blockergame DSW (full): DRQN + MultiQMixer + soft monotonicity.
# - Align seq-len with max-env-t so batches see long-range context (default seq-len=10 is too short for long episodes).
# - Stronger shaping optional via --blocker-shaping-scale (tune 0.05–0.12).
# - --hard-mixer-mono: QMIX-style |w| on both reward/cost mixers; soft mono_loss still on (see lambda_mono_* in DSW learner).
#
# Target / TD (reduces periodic loss spikes from hard target copies every N steps):
# - --use-soft-target: Polyak update targets every train step (see learners/dsw_learner.py).
# - Env: USE_SOFT_TARGET=0 to omit --use-soft-target (hard sync every --target-update-interval).
# - Env: SOFT_TARGET_TAU, TD_LOSS (mse|huber), HUBER_DELTA, TARGET_UPDATE_INTERVAL (hard targets only).
set -e
cd "$(dirname "$0")"

USE_SOFT_TARGET="${USE_SOFT_TARGET:-1}"
SOFT_TARGET_TAU="${SOFT_TARGET_TAU:-0.005}"
TD_LOSS="${TD_LOSS:-huber}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
TARGET_UPDATE_INTERVAL="${TARGET_UPDATE_INTERVAL:-200}"

SOFT_ARGS=""
if [ "$USE_SOFT_TARGET" != "0" ]; then
  SOFT_ARGS="--use-soft-target --soft-target-tau ${SOFT_TARGET_TAU}"
fi

python3 main.py \
  --rl-model rnn \
  --mixer multi-qmix \
  --policy-disc \
  --hard-mixer-mono \
  --log-dir blockergame_-DSW \
  --batch-size 128 \
  --application blocker \
  --training-epochs 60000 \
  --buffer-size 8000 \
  --max-env-t 24 \
  --seq-len 24 \
  --epsilon-scheduler linear \
  --epsilon-finish 0.1 \
  --blocker-shaping-scale 0.08 \
  --target-update-interval "${TARGET_UPDATE_INTERVAL}" \
  --td-loss "${TD_LOSS}" \
  --huber-delta "${HUBER_DELTA}" \
  ${SOFT_ARGS}
