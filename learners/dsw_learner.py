import copy
import itertools
import json
import math
import time
import numpy as np
from typing import List, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from modules.mixers.multi_qmix import MultiQMixer


class DSWLearner:
    """
    Two TD losses: mixer_rew vs r_0 + gamma * Q_r^target(s', u*), mixer_cost vs r_1 + gamma * Q_p^target(s', u*),
    summed for the TD part of the loss. Bootstrap joint u* maximizes
    Q_r + lambda_p(s') Q_p - (rho/2) * relu(-Q_p)^2 (double-Q: online objective, target values).
    Monotonicity regulariser uses Q_r + lambda_p(s) Q_p (no combined Q_tot in TD).
    """

    def __init__(self, agents, args):
        self.args = args
        self.agents = agents
        self.target_agents = copy.deepcopy(agents)

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        for agent in self.agents:
            agent.to(self.device)
        for agent in self.target_agents:
            agent.to(self.device)
        state_dim = getattr(args, "state_dim", 0)
        n_agents = len(self.agents)

        self.mixer_rew = MultiQMixer(state_dim, n_agents, 1, args).to(self.device)
        self.mixer_cost = MultiQMixer(state_dim, n_agents, 1, args).to(self.device)
        self.target_mixer_rew = copy.deepcopy(self.mixer_rew).to(self.device)
        self.target_mixer_cost = copy.deepcopy(self.mixer_cost).to(self.device)

        sw = getattr(args, "static_cost_weight", None)
        self._static_cost_weight: Optional[float] = float(sw) if sw is not None else None
        if self._static_cost_weight is not None:
            self.cost_weight_net = None
            self.target_cost_weight_net = None
            cw_params = []
        else:
            cw_hid = int(getattr(args, "cost_weight_mlp_hidden", 64))
            self.cost_weight_net = nn.Sequential(
                nn.Linear(state_dim, cw_hid),
                nn.ReLU(),
                nn.Linear(cw_hid, 1),
            ).to(self.device)
            self.target_cost_weight_net = copy.deepcopy(self.cost_weight_net).to(self.device)
            cw_params = list(self.cost_weight_net.parameters())

        agent_params = []
        for agent in self.agents:
            agent_params.extend(list(agent.parameters()))

        mixer_params = list(self.mixer_rew.parameters()) + list(self.mixer_cost.parameters())
        self.params = agent_params + mixer_params + cw_params

        base_lr = float(getattr(args, "lr", 5e-4))
        mixer_lr = float(getattr(args, "lr_mixer", base_lr * 0.5))
        wd = float(getattr(args, "weight_decay", 1e-5))

        optim_groups = [
            {"params": agent_params, "lr": base_lr},
            {"params": mixer_params + cw_params, "lr": mixer_lr, "weight_decay": wd}
        ]

        self.optimiser = Adam(optim_groups, betas=(0.9, 0.999), eps=1e-8)

        self.lr_milestones: List[int] = list(getattr(args, "lr_milestones", [2_000_000, 3_500_000, 5_000_000]))
        self.lr_decay_factor: float = float(getattr(args, "lr_decay_factor", 0.5))
        self._next_lr_idx: int = 0

        self.target_update_interval: int = int(getattr(args, "target_update_interval", 200))
        self.train_step: int = 0

        self.mono_method: str = getattr(args, "mono_method", "autograd")
        self.mono_eps: float = float(getattr(args, "mono_eps", 0.01))
        self.mono_p: int = int(getattr(args, "mono_p", 2))
        self.mono_fd_delta: float = float(getattr(args, "mono_fd_delta", 1e-2))
        self.mono_detach_q: bool = bool(getattr(args, "mono_detach_q", True))

        self.lambda_mono_start: float = float(getattr(args, "lambda_mono_start", 0.3))
        self.lambda_mono_end: float = float(getattr(args, "lambda_mono_end", 0.6))
        self.lambda_mono_warmup_steps: int = int(getattr(args, "lambda_mono_warmup_steps", 0))
        self.lambda_mono_anneal_steps: int = int(getattr(args, "lambda_mono_anneal_steps", 50000))
        self.lambda_mono_schedule: str = getattr(args, "lambda_mono_schedule", "linear")
        if getattr(args, "disable_soft_mono", False):
            self.lambda_mono_start = 0.0
            self.lambda_mono_end = 0.0
        self.cur_lambda_mono: float = self.lambda_mono_start

        self.td_loss: str = str(getattr(args, "td_loss", "mse")).lower()
        self.huber_delta: float = float(getattr(args, "huber_delta", 1.0))

        self.use_soft_target: bool = bool(getattr(args, "use_soft_target", False))
        self.soft_target_tau: float = float(getattr(args, "soft_target_tau", 0.01))

        self.grad_clip = float(getattr(self.args, "grad_norm_clip", 10.0))
        self.policy_rho = float(getattr(args, "policy_rho", 1.0))
        self._joint_idx_cache_key = None
        self._joint_indices_tensor = None

    @staticmethod
    def _td_bootstrap_mask(done_mask: th.Tensor, reward_0: th.Tensor) -> th.Tensor:
        """
        Blockergame keeps done_mask=1 on the winning step so main.py still records the +2 transition.
        The next state is already the reset layout, so TD must not add gamma * Q(s') there.
        Primary reward > 1.5 indicates terminal win (e.g. +2 or +3 in blockergame).
        """
        r0 = reward_0[..., 0:1]
        terminal_win = r0 > 1.5
        return done_mask * (~terminal_win).float()

    @staticmethod
    def _cost_w_from_logits(logits: th.Tensor) -> th.Tensor:
        """Map unconstrained MLP output to strictly positive cost weights (softplus + floor)."""
        return F.softplus(logits).clamp(min=1e-6)

    def _cost_w(self, global_state_flat: th.Tensor) -> th.Tensor:
        """(N, state_dim) -> (N, 1), values > 0."""
        if self._static_cost_weight is not None:
            N = global_state_flat.shape[0]
            return global_state_flat.new_full((N, 1), self._static_cost_weight)
        return self._cost_w_from_logits(self.cost_weight_net(global_state_flat))

    def _target_cost_w(self, global_state_flat: th.Tensor) -> th.Tensor:
        if self._static_cost_weight is not None:
            N = global_state_flat.shape[0]
            return global_state_flat.new_full((N, 1), self._static_cost_weight)
        return self._cost_w_from_logits(self.target_cost_weight_net(global_state_flat))

    def _lambda_mono_at(self, step: int) -> float:
        start = self.lambda_mono_start
        end = self.lambda_mono_end

        if self.lambda_mono_anneal_steps <= 0 or start == end:
            return start
        if step < self.lambda_mono_warmup_steps:
            return start

        t = (step - self.lambda_mono_warmup_steps) / float(self.lambda_mono_anneal_steps)
        t = max(0.0, min(1.0, t))

        sched = (self.lambda_mono_schedule or "linear").lower()
        if sched == "linear":
            return start + (end - start) * t
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))

    def _apply_lr_decay_if_due(self, step: int) -> None:
        while self._next_lr_idx < len(self.lr_milestones) and step >= self.lr_milestones[self._next_lr_idx]:
            for g in self.optimiser.param_groups:
                g["lr"] *= self.lr_decay_factor
            self._next_lr_idx += 1

    @staticmethod
    def _polyak_update_(online_module, target_module, tau: float):
        with th.no_grad():
            for p_t, p in zip(target_module.parameters(), online_module.parameters()):
                p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def _get_joint_indices(self, n_agents: int, n_actions: int) -> th.Tensor:
        key = (n_agents, n_actions, str(self.device))
        if self._joint_idx_cache_key != key:
            self._joint_indices_tensor = th.tensor(
                list(itertools.product(range(n_actions), repeat=n_agents)),
                device=self.device,
                dtype=th.long,
            )
            self._joint_idx_cache_key = key
        return self._joint_indices_tensor

    @staticmethod
    def _batched_joint_gather(mac_out: th.Tensor, joints: th.Tensor) -> th.Tensor:
        """mac_out (N, n_agents, n_actions), joints (J, n_agents) -> (N, J, n_agents)."""
        N, na, _ = mac_out.shape
        J = joints.shape[0]
        dev = mac_out.device
        n_idx = th.arange(N, device=dev).view(N, 1, 1).expand(N, J, na)
        a_idx = th.arange(na, device=dev).view(1, 1, na).expand(N, J, na)
        ac = joints.view(1, J, na).expand(N, J, na)
        return mac_out[n_idx, a_idx, ac]

    def _policy_objective(self, q_rew: th.Tensor, q_cost: th.Tensor, lam: th.Tensor, rho: float) -> th.Tensor:
        """q_rew, q_cost (N, J); lam (N, 1) -> (N, J)."""
        pen = F.relu(-q_cost).pow(2)
        return q_rew + lam * q_cost - 0.5 * rho * pen

    def _td_bootstrap_targets(
        self, mac_next_on: th.Tensor, mac_next_tgt: th.Tensor, ngs_flat: th.Tensor, n_agents: int, n_actions: int
    ):
        """(target_q_rew, target_q_cost) each (N, 1), detached."""
        N = mac_next_on.shape[0]
        joints = self._get_joint_indices(n_agents, n_actions)
        J = joints.shape[0]
        na = n_agents
        rho = self.policy_rho
        ngs_rep = ngs_flat.unsqueeze(1).expand(N, J, -1).reshape(N * J, -1)

        with th.no_grad():
            if getattr(self.args, "double_q", False):
                q_j_on = self._batched_joint_gather(mac_next_on, joints)
                q_in = q_j_on.reshape(N * J, na)
                q_rew_on = self.mixer_rew(q_in, ngs_rep).view(N, J)
                q_cost_on = self.mixer_cost(q_in, ngs_rep).view(N, J)
                lam_on = self._cost_w(ngs_flat).view(N, 1)
                obj = self._policy_objective(q_rew_on, q_cost_on, lam_on, rho)
                best_j = obj.argmax(dim=1)
                q_j_tgt = self._batched_joint_gather(mac_next_tgt, joints)
                q_sel = q_j_tgt[th.arange(N, device=self.device), best_j]
                t_in = q_sel.unsqueeze(1)
                tqw = self.target_mixer_rew(t_in, ngs_flat).view(N, 1)
                tqc = self.target_mixer_cost(t_in, ngs_flat).view(N, 1)
            else:
                q_j_tgt = self._batched_joint_gather(mac_next_tgt, joints)
                q_in = q_j_tgt.reshape(N * J, na)
                q_rew_t = self.target_mixer_rew(q_in, ngs_rep).view(N, J)
                q_cost_t = self.target_mixer_cost(q_in, ngs_rep).view(N, J)
                lam_t = self._target_cost_w(ngs_flat).view(N, 1)
                obj = self._policy_objective(q_rew_t, q_cost_t, lam_t, rho)
                best_j = obj.argmax(dim=1)
                ar = th.arange(N, device=self.device)
                tqw = q_rew_t[ar, best_j].view(N, 1)
                tqc = q_cost_t[ar, best_j].view(N, 1)
        return tqw, tqc

    def _maybe_update_targets(self) -> None:
        """Hard targets: sync every ``target_update_interval`` steps. Soft targets: Polyak step every step."""
        if self.use_soft_target:
            self._update_targets()
        elif self.train_step % self.target_update_interval == 0:
            self._update_targets()

    def joint_greedy_actions(self, mac_out: th.Tensor, global_state: th.Tensor) -> th.Tensor:
        """
        Joint argmax of Q_r + lambda_p(s) Q_p - (rho/2) relu(-Q_p)^2 (training-time policy).
        mac_out: (B, n_agents, n_actions); global_state: (B, state_dim). Returns (B, n_agents) int64.
        """
        with th.no_grad():
            B, na, n_actions = mac_out.shape
            joints = self._get_joint_indices(na, n_actions)
            J = joints.shape[0]
            ngs_rep = global_state.unsqueeze(1).expand(B, J, -1).reshape(B * J, -1)
            q_j = self._batched_joint_gather(mac_out, joints)
            q_in = q_j.reshape(B * J, na)
            q_rew = self.mixer_rew(q_in, ngs_rep).view(B, J)
            q_cost = self.mixer_cost(q_in, ngs_rep).view(B, J)
            lam = self._cost_w(global_state).view(B, 1)
            obj = self._policy_objective(q_rew, q_cost, lam, self.policy_rho)
            best_j = obj.argmax(dim=1)
            return joints[best_j]

    @staticmethod
    def _q_from_agent_forward(out):
        return out[0] if isinstance(out, tuple) else out

    def _rnn_mac_out(self, states_bt: th.Tensor, agents) -> th.Tensor:
        """states_bt: (B, T, n_agents, obs_dim) -> (B, T, n_agents, n_actions). h_0 = 0 each batch row."""
        B, T, n_agents, _ = states_bt.shape
        mac_agents = []
        for i, agent in enumerate(agents):
            h = th.zeros(B, agent.rnn.hidden_size, device=self.device, dtype=states_bt.dtype)
            qs_t = []
            for t in range(T):
                o = states_bt[:, t, i, :]
                out = agent(o, h)
                q, h = out[0], out[1]
                qs_t.append(q.unsqueeze(1))
            mac_agents.append(th.cat(qs_t, dim=1))
        return th.stack(mac_agents, dim=2)

    def _train_sequence(self, batch: dict):
        """DRQN-style: batch from EpisodeReplayBuffer.sample_sequences (numpy arrays + mask)."""
        self.train_step += 1
        self._apply_lr_decay_if_due(self.train_step)

        states = th.as_tensor(batch["states"], dtype=th.float32, device=self.device)
        next_states = th.as_tensor(batch["next_states"], dtype=th.float32, device=self.device)
        global_state = th.as_tensor(batch["global_state"], dtype=th.float32, device=self.device)
        next_global_state = th.as_tensor(batch["next_global_state"], dtype=th.float32, device=self.device)
        actions = th.as_tensor(batch["actions"], dtype=th.long, device=self.device)
        global_reward = th.as_tensor(batch["global_reward"], dtype=th.float32, device=self.device)
        done_mask = th.as_tensor(batch["done_mask"], dtype=th.float32, device=self.device).unsqueeze(-1)
        mask = th.as_tensor(batch["mask"], dtype=th.float32, device=self.device).unsqueeze(-1)

        B, T, n_agents, _ = states.shape
        batch_size = B * T

        if global_reward.dim() == 3 and global_reward.size(-1) >= 2:
            reward_0 = global_reward[..., 0:1]
            reward_1 = global_reward[..., 1:2]
        else:
            r = global_reward.unsqueeze(-1) if global_reward.dim() == 2 else global_reward
            reward_0 = r
            reward_1 = th.zeros_like(reward_0)

        # #region agent log
        _dbg_t0 = time.perf_counter()
        # #endregion
        mac_out = self._rnn_mac_out(states, self.agents)
        n_actions = mac_out.shape[-1]
        with th.no_grad():
            mac_out_next_on = self._rnn_mac_out(next_states, self.agents)
            target_mac_out = self._rnn_mac_out(next_states, self.target_agents)
        # #region agent log
        _dbg_t1 = time.perf_counter()
        # #endregion

        actions_exp = actions.unsqueeze(-1)
        chosen_action_qvals_agents = th.gather(mac_out, dim=3, index=actions_exp).squeeze(3)

        q_in = chosen_action_qvals_agents.reshape(batch_size, 1, n_agents)
        gs = global_state.reshape(batch_size, -1)
        ngs = next_global_state.reshape(batch_size, -1)

        mac_next_on = mac_out_next_on.reshape(batch_size, n_agents, n_actions)
        mac_next_tgt = target_mac_out.reshape(batch_size, n_agents, n_actions)
        target_q_rew, target_q_cost = self._td_bootstrap_targets(
            mac_next_on, mac_next_tgt, ngs, n_agents, n_actions
        )
        # #region agent log
        _dbg_t2 = time.perf_counter()
        n_joint = int(n_actions**n_agents)
        # #endregion
        target_q_rew = target_q_rew.view(B, T, 1)
        target_q_cost = target_q_cost.view(B, T, 1)

        q_tot_rew = self.mixer_rew(q_in, gs).view(B, T, 1)
        q_tot_cost = self.mixer_cost(q_in, gs).view(B, T, 1)

        gamma = getattr(self.args, "gamma", 0.96)
        bootstrap = self._td_bootstrap_mask(done_mask, reward_0)
        targets_rew = reward_0 + gamma * bootstrap * target_q_rew
        targets_cost = reward_1 + gamma * bootstrap * target_q_cost
        targets_rew = targets_rew.detach()
        targets_cost = targets_cost.detach()
        td_error_rew = q_tot_rew - targets_rew
        w_td = self._cost_w(gs).view(B, T, 1)
        td_error_cost = w_td * q_tot_cost - targets_cost

        mask_sum = mask.sum().clamp(min=1.0)
        if self.td_loss == "huber":
            delta = reward_0.new_tensor(self.huber_delta)

            def _huber_masked(td_err):
                abs_err = td_err.abs()
                quad = th.clamp(abs_err, max=delta)
                lin = abs_err - quad
                per = 0.5 * quad ** 2 + delta * lin
                return (mask * per).sum() / mask_sum

            td_loss = _huber_masked(td_error_rew) + _huber_masked(td_error_cost)
        else:
            td_loss = (mask * (td_error_rew ** 2)).sum() / mask_sum + (mask * (td_error_cost ** 2)).sum() / mask_sum

        loss = td_loss
        mono_loss_val = 0.0
        cur_lambda = self._lambda_mono_at(self.train_step)
        self.cur_lambda_mono = cur_lambda
        # #region agent log
        _dbg_t3 = time.perf_counter()
        # #endregion

        if cur_lambda > 0.0:
            q_base = chosen_action_qvals_agents.detach() if self.mono_detach_q else chosen_action_qvals_agents
            q_fm = q_base.clone().requires_grad_(True)

            q_flat = q_fm.reshape(batch_size, n_agents)
            q_mono_in = q_flat.unsqueeze(1)
            gs_flat = gs
            q_m_rew = self.mixer_rew(q_mono_in, gs_flat).view(B, T, 1)
            q_m_cost = self.mixer_cost(q_mono_in, gs_flat).view(B, T, 1)
            w_m = self._cost_w(gs_flat).view(B, T, 1)
            q_mono_tot = (q_m_rew + w_m * q_m_cost).squeeze(-1)

            if self.mono_method == "autograd":
                partials = th.autograd.grad(
                    (mask.squeeze(-1) * q_mono_tot).sum(),
                    q_fm,
                    create_graph=True,
                    retain_graph=True,
                )[0]
            elif self.mono_method == "fd":
                delta_m = self.mono_fd_delta
                q_flat_det = q_fm.detach().reshape(batch_size, n_agents)
                w_fd = self._cost_w(gs_flat).detach()
                with th.no_grad():
                    Q_base_f = (
                        self.mixer_rew(q_flat_det.unsqueeze(1), gs_flat)
                        + w_fd * self.mixer_cost(q_flat_det.unsqueeze(1), gs_flat)
                    ).view(B, T)
                slopes = []
                for i in range(n_agents):
                    q_plus = q_flat_det.clone()
                    q_plus[:, i] += delta_m
                    Q_plus = (
                        self.mixer_rew(q_plus.unsqueeze(1), gs_flat)
                        + w_fd * self.mixer_cost(q_plus.unsqueeze(1), gs_flat)
                    ).view(B, T)
                    slopes.append(((Q_plus - Q_base_f) / delta_m).unsqueeze(-1))
                partials = th.cat(slopes, dim=-1).detach()
            else:
                raise ValueError(f"Unknown mono_method '{self.mono_method}'")

            viol = F.relu(self.mono_eps - partials)
            if self.mono_p == 1:
                mono_per = viol
            else:
                mono_per = viol ** 2
            mono_loss = (mask * mono_per).sum() / (mask_sum * n_agents)
            loss = loss + cur_lambda * mono_loss
            with th.no_grad():
                mono_loss_val = mono_loss.item()

        # #region agent log
        _dbg_t4 = time.perf_counter()
        # #endregion
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimiser.step()
        # #region agent log
        _dbg_t5 = time.perf_counter()
        if self.train_step <= 5 or self.train_step % 200 == 0:
            _pl = {
                "sessionId": "5fa299",
                "timestamp": int(time.time() * 1000),
                "hypothesisId": "H_joint_H_rnn_H_mono",
                "location": "dsw_learner.py:_train_sequence",
                "message": "dsw_train_step_timing",
                "data": {
                    "train_step": int(self.train_step),
                    "B": int(B),
                    "T": int(T),
                    "N_flat": int(batch_size),
                    "n_agents": int(n_agents),
                    "n_actions": int(n_actions),
                    "n_joint": int(n_joint),
                    "nj_times_N": int(n_joint * batch_size),
                    "sec_rnn_three_mac": float(_dbg_t1 - _dbg_t0),
                    "sec_td_bootstrap": float(_dbg_t2 - _dbg_t1),
                    "sec_td_mixers_to_mono_start": float(_dbg_t3 - _dbg_t2),
                    "sec_mono_block": float(_dbg_t4 - _dbg_t3),
                    "sec_backward_optim": float(_dbg_t5 - _dbg_t4),
                    "cur_lambda_mono": float(cur_lambda),
                },
                "runId": "perf-debug",
            }
            with open(
                "/home/kai/Documents/bachelor_dissertation/DSW-QMIX/.cursor/debug-5fa299.log",
                "a",
                encoding="utf-8",
            ) as _df:
                _df.write(json.dumps(_pl) + "\n")
        # #endregion

        self._maybe_update_targets()

        # #region agent log
        if self.train_step % 1000 == 0:
            msum = mask.sum().clamp(min=1.0)
            mean_r0 = float((mask * reward_0).sum() / msum)
            max_r0 = float((mask * reward_0).max()) if mask.any() else 0.0
            td_item = float(td_loss.item())
            scale_mono = float(cur_lambda * mono_loss_val) if cur_lambda > 0.0 else 0.0
            payload = {
                "sessionId": "472a7c",
                "timestamp": int(time.time() * 1000),
                "hypothesisId": "H1-H4",
                "location": "dsw_learner.py:_train_sequence",
                "message": "train_step_stats",
                "data": {
                    "train_step": int(self.train_step),
                    "cur_lambda_mono": float(cur_lambda),
                    "td_loss": td_item,
                    "mono_loss_unweighted": float(mono_loss_val),
                    "lambda_times_mono": scale_mono,
                    "td_vs_lambda_mono_ratio": (td_item / scale_mono) if scale_mono > 1e-12 else None,
                    "mean_r0_masked": mean_r0,
                    "max_r0_masked": max_r0,
                    "grad_norm_clipped": float(grad_norm.item()),
                },
                "runId": "pre-fix-debug",
            }
            with open(
                "/home/kai/Documents/bachelor_dissertation/DSW-QMIX/.cursor/debug-472a7c.log",
                "a",
                encoding="utf-8",
            ) as _df:
                _df.write(json.dumps(payload) + "\n")
        # #endregion

        return loss.item(), td_loss.item(), mono_loss_val

    def train(self, batch):
        if isinstance(batch, dict) and "mask" in batch:
            return self._train_sequence(batch)

        self.train_step += 1
        self._apply_lr_decay_if_due(self.train_step)

        states = th.tensor(np.array([d['states'] for d in batch]), dtype=th.float32).to(self.device)
        next_states = th.tensor(np.array([d['states_new'] for d in batch]), dtype=th.float32).to(self.device)
        global_state = th.tensor(np.array([d['global_state'] for d in batch]), dtype=th.float32).to(self.device)
        next_global_state = th.tensor(np.array([d['global_state_new'] for d in batch]), dtype=th.float32).to(self.device)
        actions = th.tensor(np.array([d['actions'] for d in batch]), dtype=th.long).view(-1, len(self.agents))
        global_reward = th.tensor(np.array([d['global_reward'] for d in batch]), dtype=th.float32).to(self.device)

        done_mask = th.tensor(np.array([d['done_mask'] for d in batch]), dtype=th.float32).unsqueeze(1).to(self.device)

        batch_size = states.shape[0]

        if global_reward.dim() == 2 and global_reward.size(1) >= 2:
            reward_0 = global_reward[:, 0:1] # reward
            reward_1 = global_reward[:, 1:2] # penalty
        else:
            r = global_reward.view(-1, 1) if global_reward.dim() == 1 else global_reward
            reward_0 = r
            reward_1 = th.zeros_like(reward_0)

        agent_outs = []
        for i, agent in enumerate(self.agents):
            out = agent(states[:, i, :])
            agent_outs.append(self._q_from_agent_forward(out))
        mac_out = th.stack(agent_outs, dim=1)
        n_actions = mac_out.shape[-1]
        n_agents = len(self.agents)

        agent_outs_next_on = []
        target_mac_out = []
        with th.no_grad():
            for i, agent in enumerate(self.agents):
                out = agent(next_states[:, i, :])
                agent_outs_next_on.append(self._q_from_agent_forward(out))
            for i, agent in enumerate(self.target_agents):
                out = agent(next_states[:, i, :])
                target_mac_out.append(self._q_from_agent_forward(out))
        mac_next_on = th.stack(agent_outs_next_on, dim=1)
        target_mac_out = th.stack(target_mac_out, dim=1)

        actions_expanded = actions.unsqueeze(2)
        chosen_action_qvals_agents = th.gather(mac_out, dim=2, index=actions_expanded).squeeze(2)

        q_in = chosen_action_qvals_agents.unsqueeze(1)

        target_q_rew, target_q_cost = self._td_bootstrap_targets(
            mac_next_on, target_mac_out, next_global_state, n_agents, n_actions
        )

        q_tot_rew = self.mixer_rew(q_in, global_state).view(batch_size, 1)
        q_tot_cost = self.mixer_cost(q_in, global_state).view(batch_size, 1)

        gamma = getattr(self.args, "gamma", 0.99)
        bootstrap = self._td_bootstrap_mask(done_mask, reward_0)
        targets_rew = reward_0 + gamma * bootstrap * target_q_rew
        targets_cost = reward_1 + gamma * bootstrap * target_q_cost
        targets_rew = targets_rew.detach()
        targets_cost = targets_cost.detach()
        td_error_rew = q_tot_rew - targets_rew
        w_td = self._cost_w(global_state)
        td_error_cost = w_td * q_tot_cost - targets_cost

        if self.td_loss == "huber":
            delta = reward_0.new_tensor(self.huber_delta)

            def _huber(x):
                abs_err = x.abs()
                quad = th.clamp(abs_err, max=delta)
                lin = abs_err - quad
                return (0.5 * quad ** 2 + delta * lin).mean()

            td_loss = _huber(td_error_rew) + _huber(td_error_cost)
        else:
            td_loss = (td_error_rew ** 2).mean() + (td_error_cost ** 2).mean()

        loss = td_loss
        mono_loss_val = 0.0
        cur_lambda = self._lambda_mono_at(self.train_step)
        self.cur_lambda_mono = cur_lambda

        if cur_lambda > 0.0:
            q_for_mono = chosen_action_qvals_agents.detach() if self.mono_detach_q else chosen_action_qvals_agents
            q_for_mono = q_for_mono.requires_grad_(True)
            q_mono_in = q_for_mono.unsqueeze(1)

            q_m_rew = self.mixer_rew(q_mono_in, global_state).view(batch_size, 1)
            q_m_cost = self.mixer_cost(q_mono_in, global_state).view(batch_size, 1)
            w_m = self._cost_w(global_state)
            q_mono_tot = q_m_rew + w_m * q_m_cost

            if self.mono_method == "autograd":
                partials = th.autograd.grad(q_mono_tot.sum(), q_for_mono, create_graph=True, retain_graph=True)[0]
            elif self.mono_method == "fd":
                delta_m = self.mono_fd_delta
                base_q = q_for_mono.detach()
                Q_base = q_mono_tot.detach()
                w_fd = self._cost_w(global_state).detach()
                parts = []
                for i in range(len(self.agents)):
                    q_plus = base_q.clone()
                    q_plus[:, i] += delta_m
                    q_pi = q_plus.unsqueeze(1)
                    Qp = (
                        self.mixer_rew(q_pi, global_state).view(batch_size, 1)
                        + w_fd * self.mixer_cost(q_pi, global_state).view(batch_size, 1)
                    )
                    slope_i = (Qp - Q_base) / delta_m
                    parts.append(slope_i.squeeze(1))
                partials = th.stack(parts, dim=1)
            else:
                raise ValueError(f"Unknown mono_method '{self.mono_method}'")

            viol = F.relu(self.mono_eps - partials)
            if self.mono_p == 1:
                mono_loss = viol.mean()
            else:
                mono_loss = (viol ** 2).mean()

            loss = loss + cur_lambda * mono_loss
            with th.no_grad():
                mono_loss_val = mono_loss.item()

        self.optimiser.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimiser.step()

        self._maybe_update_targets()

        return loss.item(), td_loss.item(), mono_loss_val

    def _update_targets(self):
        if self.use_soft_target:
            for online, target in zip(self.agents, self.target_agents):
                self._polyak_update_(online, target, self.soft_target_tau)
            self._polyak_update_(self.mixer_rew, self.target_mixer_rew, self.soft_target_tau)
            self._polyak_update_(self.mixer_cost, self.target_mixer_cost, self.soft_target_tau)
            if self._static_cost_weight is None:
                self._polyak_update_(self.cost_weight_net, self.target_cost_weight_net, self.soft_target_tau)
        else:
            for online, target in zip(self.agents, self.target_agents):
                target.load_state_dict(online.state_dict())
            self.target_mixer_rew.load_state_dict(self.mixer_rew.state_dict())
            self.target_mixer_cost.load_state_dict(self.mixer_cost.state_dict())
            if self._static_cost_weight is None:
                self.target_cost_weight_net.load_state_dict(self.cost_weight_net.state_dict())

    def cuda(self):
        for agent in self.agents:
            agent.cuda()
        for agent in self.target_agents:
            agent.cuda()
        self.mixer_rew.cuda()
        self.target_mixer_rew.cuda()
        self.mixer_cost.cuda()
        self.target_mixer_cost.cuda()
        if self._static_cost_weight is None:
            self.cost_weight_net.cuda()
            self.target_cost_weight_net.cuda()

    def save_models(self, path):
        for i, agent in enumerate(self.agents):
            th.save(agent.state_dict(), f"{path}/agent_{i}.th")
        th.save(self.mixer_rew.state_dict(), f"{path}/mixer_rew.th")
        th.save(self.mixer_cost.state_dict(), f"{path}/mixer_cost.th")
        if self._static_cost_weight is not None:
            with open(f"{path}/static_cost_weight.txt", "w") as f:
                f.write(str(self._static_cost_weight))
        else:
            th.save(self.cost_weight_net.state_dict(), f"{path}/cost_weight_net.th")
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(th.load(f"{path}/agent_{i}.th", map_location=lambda storage, loc: storage))
            self.target_agents[i].load_state_dict(agent.state_dict())

        self.mixer_rew.load_state_dict(th.load(f"{path}/mixer_rew.th", map_location=lambda storage, loc: storage))
        self.target_mixer_rew.load_state_dict(self.mixer_rew.state_dict())
        self.mixer_cost.load_state_dict(th.load(f"{path}/mixer_cost.th", map_location=lambda storage, loc: storage))
        self.target_mixer_cost.load_state_dict(self.mixer_cost.state_dict())

        if self._static_cost_weight is None:
            self.cost_weight_net.load_state_dict(th.load(f"{path}/cost_weight_net.th", map_location=lambda storage, loc: storage))
            self.target_cost_weight_net.load_state_dict(self.cost_weight_net.state_dict())

        self.optimiser.load_state_dict(th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
