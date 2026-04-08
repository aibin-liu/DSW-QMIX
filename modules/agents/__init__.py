REGISTRY = {}

from .rnn_agent import RNNAgent
from .simple_agent import SimpleAgent
REGISTRY['simple'] = SimpleAgent
REGISTRY['rnn'] = RNNAgent