REGISTRY = {}
from .q_learner import QLearner
from .dsw_learner import DSWLearner
from .flat_marl_learners import IQLLearner, VDNLearner, QMIXLearner

REGISTRY["simple"] = QLearner
REGISTRY["rnn"] = DSWLearner
REGISTRY["iql"] = IQLLearner
REGISTRY["vdn"] = VDNLearner
REGISTRY["qmix"] = QMIXLearner