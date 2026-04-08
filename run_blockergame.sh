# example to run CMIX-M on blockergame task
python3 main.py --rl-model rnn --mixer multi-qmix --policy-disc --log-dir blockergame_-DSW --batch-size 128 --application blocker --training-epochs 60000 --buffer-size 1000 --max-env-t 24 --epsilon-scheduler linear --epsilon-finish 0.05
