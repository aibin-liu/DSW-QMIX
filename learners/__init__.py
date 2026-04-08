REGISTRY = {}
from .q_learner import QLearner
from .dsw_learner import DSWLearner
REGISTRY["simple"] = QLearner
REGISTRY["rnn"] = DSWLearner