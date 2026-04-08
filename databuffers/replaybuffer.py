"""

	Replay Buffer for Deep Reinforcement Learning

"""

from collections import deque
import random
import numpy as np


class ReplayBuffer():
    def __init__(self, size_buffer, random_seed=8):
        self.__size_bf = size_buffer
        self.__length = 0
        self.__buffer = deque()
        random.seed(random_seed)
        np.random.seed(random_seed)


    @property
    def buffer(self):
        return self.__buffer


    def add(self, exp):
        if self.__length < self.__size_bf:
            self.__buffer.append(exp)
            self.__length += 1
        else:
            self.__buffer.popleft()
            self.__buffer.append(exp)

    def __len__(self):
        return self.__length

    def sample_batch(self, size_batch):

        if self.__length < size_batch:
            batch = random.sample(self.__buffer, self.__length)
        else:
            batch = random.sample(self.__buffer, size_batch)
        return batch

    def clear(self):
        self.__buffer.clear()
        self.__length = 0


class EpisodeReplayBuffer:
    """
    Stores complete episodes (each a list of transition dicts). Samples fixed-length
    subsequences for DRQN / recurrent QMIX-style training (i.i.d. over chunks).
    """

    def __init__(self, max_episodes: int, random_seed: int = 8):
        self.max_episodes = max_episodes
        self._episodes = deque()
        self._rng = random.Random(random_seed)
        self._np_rng = np.random.RandomState(random_seed)

    def __len__(self):
        return len(self._episodes)

    def add_episode(self, episode):
        """Append one episode (non-empty list of transitions)."""
        if not episode:
            return
        if len(self._episodes) >= self.max_episodes:
            self._episodes.popleft()
        self._episodes.append(episode)

    def clear(self):
        self._episodes.clear()

    def sample_sequences(self, batch_size: int, seq_len: int, n_agents: int):
        """
        Sample batch_size subsequences of length seq_len.
        Returns a dict of numpy arrays with shapes:
          states, next_states: (B, T, n_agents, obs_dim)
          global_state, next_global_state: (B, T, state_dim)
          actions: (B, T, n_agents)
          global_reward: (B, T, R)  R=2 or 1 from first transition's shape
          done_mask: (B, T)
          mask: (B, T) float32 1 = valid timestep, 0 = padding
        """
        eps_list = list(self._episodes)
        if len(eps_list) == 0:
            raise ValueError("EpisodeReplayBuffer is empty")

        # Infer obs_dim, state_dim, reward width from first transition
        t0 = eps_list[0][0]
        obs_dim = len(np.asarray(t0["states"][0]).flatten())
        state_dim = len(np.asarray(t0["global_state"]).flatten())

        gr0 = t0["global_reward"]
        if isinstance(gr0, (list, np.ndarray)) and np.asarray(gr0).size > 1:
            rdim = int(np.asarray(gr0).reshape(-1).shape[0])
        else:
            rdim = 1

        B = batch_size
        T = seq_len

        states = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        next_states = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        global_state = np.zeros((B, T, state_dim), dtype=np.float32)
        next_global_state = np.zeros((B, T, state_dim), dtype=np.float32)
        actions = np.zeros((B, T, n_agents), dtype=np.int64)
        global_reward = np.zeros((B, T, rdim), dtype=np.float32)
        done_mask = np.zeros((B, T), dtype=np.float32)
        mask = np.zeros((B, T), dtype=np.float32)

        for b in range(B):
            ep = eps_list[self._rng.randint(0, len(eps_list) - 1)]
            L = len(ep)
            if L >= T:
                # Inclusive randint: max start is L - T so slice [start:start+T] has exactly T steps.
                start = self._rng.randint(0, L - T)
                chunk = ep[start : start + T]
                mask[b, :] = 1.0
            else:
                chunk = ep + [None] * (T - L)
                mask[b, :L] = 1.0

            for t in range(T):
                tr = chunk[t]
                if tr is None:
                    continue
                for i in range(n_agents):
                    states[b, t, i] = np.asarray(tr["states"][i], dtype=np.float32).flatten()
                    next_states[b, t, i] = np.asarray(tr["states_new"][i], dtype=np.float32).flatten()
                global_state[b, t] = np.asarray(tr["global_state"], dtype=np.float32).flatten()
                next_global_state[b, t] = np.asarray(tr["global_state_new"], dtype=np.float32).flatten()
                actions[b, t] = np.asarray(tr["actions"], dtype=np.int64).reshape(n_agents)
                gr = tr["global_reward"]
                if rdim == 1:
                    global_reward[b, t, 0] = float(gr)
                else:
                    global_reward[b, t] = np.asarray(gr, dtype=np.float32).reshape(rdim)
                done_mask[b, t] = float(tr["done_mask"])

        return {
            "states": states,
            "next_states": next_states,
            "global_state": global_state,
            "next_global_state": next_global_state,
            "actions": actions,
            "global_reward": global_reward,
            "done_mask": done_mask,
            "mask": mask,
        }
