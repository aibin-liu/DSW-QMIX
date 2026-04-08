# example: DSW + MultiQMixer on vehicular network
python3 main.py --rl-model rnn --mixer multi-qmix --policy-disc --log-dir vn_DSW --application vn --batch-size 32 --training-epochs 100000 --max-env-t 4 --gamma 0.5 --epsilon-scheduler exp

