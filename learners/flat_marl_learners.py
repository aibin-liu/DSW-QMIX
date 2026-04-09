"""
Unconstrained flat MARL baselines for tabular-style Q-learning with SimpleAgent:
IQL (no mixer), VDN, QMIX.

Intended for envs that can emit a vector-valued team reward (e.g. blockergame with
non-simple scheme); `main.py` sets `env.set_scheme("simple")` for these models so
only the primary reward signal is used.
"""

import copy

import torch as th

from .q_learner import QLearner


def _move_q_learner_to_device(learner: QLearner) -> None:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    learner.device = device
    for a in learner.agents:
        a.to(device)
    for a in learner.target_agents:
        a.to(device)
    if learner.mixer is not None:
        learner.mixer.to(device)
        learner.target_mixer.to(device)


class IQLLearner(QLearner):
    """Independent Q-learning: no mixing, team reward equals sum of local targets from env."""

    def __init__(self, agents, args):
        a = copy.deepcopy(args)
        a.mixer = None
        super().__init__(agents, a)
        _move_q_learner_to_device(self)


class VDNLearner(QLearner):
    """Value-decomposition network: sum of agent Q-values."""

    def __init__(self, agents, args):
        a = copy.deepcopy(args)
        a.mixer = "vdn"
        super().__init__(agents, a)
        _move_q_learner_to_device(self)


class QMIXLearner(QLearner):
    """QMIX monotonic mixing."""

    def __init__(self, agents, args):
        a = copy.deepcopy(args)
        a.mixer = "qmix"
        super().__init__(agents, a)
        _move_q_learner_to_device(self)
