"""
Deterministic DDPG actor for TMRL 0.7.1 LIDAR observations.

Verified against installed tmrl==0.7.1 source:
- tmrl.actor.TorchActorModule.__init__(self, observation_space, action_space, device="cpu")
  (confirmed via inspection of the actual installed package, not guessed)
- act() must return a numpy.array; act_() (called by RolloutWorker) already applies
  torch.no_grad() and collates a single obs into a batch of 1 before calling act().
- RolloutWorker.act() is called with test=not train (networking.py) -> test=False during
  training-data collection, test=True during evaluation episodes. This is the correct,
  TMRL-native signal for "when should exploration noise be applied", confirmed via source.
- Observation space for LIDAR is a gymnasium.spaces.Tuple, NOT a flat vector
  (speed[1], lidar_flat[76], prev_act[3], prev_act[3]) = 83, confirmed via
  tmrl.custom.tm.tm_gym_interfaces.TM2020InterfaceLidar.get_observation_space()
  and rtgym.envs.real_time_env (act_buf_len appends previous actions to the Tuple).
- The tuple-handling pattern (torch.cat(obs, -1)) mirrors
  tmrl.custom.custom_models.SquashedGaussianMLPActor / MLPQFunction.
- MLPQFunction (tmrl.custom.custom_models) is reused directly as the critic -- confirmed
  compatible with tuple observations, no reimplementation needed.
- DDPGActorCritic.__init__(observation_space, action_space) with exactly 2 positional args
  matches the construction contract confirmed in
  tmrl.custom.custom_algorithms.SpinupSacAgent.__post_init__:
      model = self.model_cls(observation_space, action_space)
      self.model = model.to(device)
"""

import numpy as np
import torch
import torch.nn as nn

from tmrl.actor import TorchActorModule
from tmrl.custom.custom_models import mlp, MLPQFunction  # MLPQFunction reused directly, confirmed compatible
from tmrl.util import prod


class DDPGMLPActor(TorchActorModule):
    """
    Deterministic actor: 83 -> 256 -> ReLU -> 256 -> ReLU -> 3 -> Tanh -> [-1, 1] * act_limit
    """

    def __init__(self, observation_space, action_space, device="cpu",
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space, device)  # TorchActorModule API, verified

        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False

        dim_act = action_space.shape[0]
        self.act_limit = float(action_space.high[0])

        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)

        # Exploration noise callable, e.g. ddpg.noise.OrnsteinUhlenbeckNoise, called as noise(action, act_limit).
        # NOT part of state_dict -> does NOT survive TorchActorModule.save()/load() (torch.save(state_dict())).
        # Must therefore be set on the actor instance living in the RolloutWorker/inference script directly,
        # never relied upon to transfer from the TrainingAgent's copy. See train_ddpg.py / Phase 5 notes.
        self.noise = None

    def forward(self, obs):
        """Differentiable deterministic forward pass. obs: tuple of batched tensors (or single tensor)."""
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        pi_action = torch.tanh(mu) * self.act_limit
        return pi_action

    def act(self, obs, test=False):
        """
        Called by TorchActorModule.act_(), which already applies torch.no_grad()
        and has already collated a single obs into a batch of 1 on self.device.
        RolloutWorker passes test=not train, so test=False means "exploring" -> apply noise
        if one is attached; test=True means evaluation -> stay purely deterministic.
        Must return numpy.array (per tmrl.actor.ActorModule.act docstring).
        """
        a = self.forward(obs)
        res = a.squeeze().detach().cpu().numpy()
        if not len(res.shape):
            res = np.expand_dims(res, 0)
        if not test and self.noise is not None:
            res = self.noise(res, act_limit=self.act_limit)
        return res


class DDPGActorCritic(nn.Module):
    """
    Combined actor+critic module, following the exact instantiation contract confirmed in
    tmrl.custom.custom_algorithms.SpinupSacAgent.__post_init__:
        model = self.model_cls(observation_space, action_space)   # 2 positional args only
        self.model = model.to(device)                              # device applied afterwards
    A single critic (no twin-Q as in SAC/TD3) per the DDPG spec: only one Q-network + one target.
    """

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.actor = DDPGMLPActor(observation_space, action_space, hidden_sizes=hidden_sizes, activation=activation)
        self.critic = MLPQFunction(observation_space, action_space, hidden_sizes=hidden_sizes, activation=activation)

    def act(self, obs, test=False):
        return self.actor.act(obs, test=test)