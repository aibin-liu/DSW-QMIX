#!/bin/sh
# Run both DSW ablations in sequence (or invoke train_dsw_ablation_s.sh / train_dsw_ablation_h.sh separately).
set -e
cd "$(dirname "$0")"

./train_dsw_ablation_s.sh
./train_dsw_ablation_h.sh

echo "Done. Logs under ./log/${DSW_ABL_LOG_ROOT:-blockergame_ablation}_dsw_qmix_s/ and ./log/${DSW_ABL_LOG_ROOT:-blockergame_ablation}_dsw_qmix_h/"
