# example: DSW + MultiQMixer on vehicular network (hard |w| mono on both mixers + soft mono_loss)
python3 main.py --rl-model rnn --mixer multi-qmix --policy-disc --hard-mixer-mono --log-dir vn_DSW --application vn --batch-size 32 --training-epochs 100000 --max-env-t 4 --gamma 0.5 --epsilon-scheduler exp

