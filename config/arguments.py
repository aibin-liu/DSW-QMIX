import argparse
import torch as th

def get_arg():
    parser = argparse.ArgumentParser(description='QMIX-VN')

    # Add config here
    # algorithm config
    # if mixer == None, use IQL with simple_agent; rnn uses RNN agent + DSW learner
    parser.add_argument('--rl-model', default='simple', 
        help='Reinforcement learning model: simple | rnn')
    parser.add_argument('--mixer', default=None, 
        help='Mixer type, if None means only use local rewards. select(vdn|qmix|multi-qmix|None)')
    parser.add_argument('--lr', type=float, default=0.0005,
        help='Learning rate')
    parser.add_argument('--optimizer', default="RMSprop",
        help='Optimizer type, including (RMSprop|Adam)')
    parser.add_argument('--optim-alpha', type=float, default=0.99,
        help='RMSprop optimizer alpha')
    parser.add_argument('--optim-eps', type=float, default=1e-5,
        help='RMSprop optimizer eps')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='Discounted factor for reinforcement learning')
    parser.add_argument('--grad-norm-clip', type=float, default=10,
        help='Reduce magnitude of gradients above this L2 norm')
    parser.add_argument('--target-update-interval', type=float, default=200,
        help='Update target network every target-update-interval updates')
    parser.add_argument('--epsilon-start', type=float, default=1.,
        help='Starting value for epsilon decay schedule')
    parser.add_argument('--epsilon-finish', type=float, default=0.01,
        help='Finishing value for epsilon decay schedule')
    parser.add_argument('--epsilon-scheduler', default='exp', 
        help='Epsilon decay scheduler, select (exp|linear), default is exp')
    parser.add_argument('--policy-disc', action='store_true', default=False,
        help='Use discrete policy in constrained RL algorithm')
    parser.add_argument('--weight-alpha', type=float, default=0.1,
        help='Weighted QMIX alpha parameter')
    parser.add_argument('--cost-weight-mlp-hidden', type=int, default=64,
        help='Hidden size for state-dependent cost-weight MLP in DSW learner')
    parser.add_argument('--policy-rho', type=float, default=1.0,
        help='DSW policy extraction: penalty weight rho on max(0,-Q_p)^2 in joint argmax objective')
    parser.add_argument('--double-q', action='store_true', default=False,
        help='Double DQN-style joint bootstrap (online max, target values)')
    # replay buffer config
    parser.add_argument('--buffer-size', type=int, default=1000,
        help='Capacity: transitions (simple) or max episodes (rnn)')
    parser.add_argument('--seq-len', type=int, default=10,
        help='Subsequence length for rnn episode replay (DRQN-style)')

    # running config
    parser.add_argument('--application', default='vn', 
            help='Application scenario, select (blocker|vn), i.e. blocker game, vehicular network')
    parser.add_argument('--log-dir', default='test',
        help='Directory to save model logs')
    parser.add_argument('--env-path', default=None,
        help='Environment instance path, if None setup a new environment instance')
    parser.add_argument('--max-env-t', type=int, default=32,
        help='Maximum number of iteration step of the experiments in training epoch')
    parser.add_argument('--training-episodes', type=int,default=1,
        help='Number of training episodes')
    parser.add_argument('--training-epochs', type=int,default=10000,
        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
        help='Sample batch size from replay buffer')
    parser.add_argument('--model-load-path', default=None,
        help='Model parameters loading path, if None do not load model from checkpoint')

    args = parser.parse_args()

    return args
