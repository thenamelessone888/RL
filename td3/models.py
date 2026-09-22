"""
Deterministic TD3 actor + twin critics for TMRL 0.7.1 LIDAR observations.

Verified against installed tmrl==0.7.1 source (same inspection performed for ddpg/models.py,
re-checked here since TD3 introduces a *second*, independent critic):
- tmrl.actor.TorchActorModule.__init__(self, observation_space, action_space, device="cpu")
- tmrl.actor.TorchActorModule.load(path, device): `self.load_state_dict(torch.load(path,
  map_location=self.device, weights_only=True)); return self` -- mutates and returns the SAME
  instance. This matters for TD3ActorCritic.act(): the actor object identity never changes
  across a weight reload, so any exploration noise object attached post-hoc to the actor
  instance (see td3/noise.py, train_td3.py) survives every RolloutWorker.update_actor_weights()
  call. Re-verified directly against the installed wheel; not assumed from the DDPG report.
- tmrl.custom.custom_models.MLPQFunction: a generic Q(s,a) regressor with ZERO SAC-specific
  logic (torch.cat((*obs, act), -1) -> mlp -> squeeze(-1)). Confirmed compatible with tuple
  observations. TD3's two critics are two independently-constructed, independently-initialized
  MLPQFunction instances -- imported and reused directly, never subclassed or copy-pasted.
- Actor/critic construction contract, confirmed in tmrl.custom.custom_algorithms.SpinupSacAgent
  .__post_init__: `model = self.model_cls(observation_space, action_space)` (exactly two
  positional args) then `self.model = model.to(device)` (device applied AFTERWARDS, not at
  construction time). TD3ActorCritic.__init__ matches this signature exactly.

TD3-specific design (Fujimoto, Hoof & Meger, 2018 "Addressing Function Approximation Error in
Actor-Critic Methods"), as distinct from the project's existing DDPG implementation:
- TWO independent critics (self.critic1, self.critic2), each with its own target, used for the
  clipped double-Q Bellman target min(Q1_targ, Q2_targ) computed in td3/agent.py. This is the
  central TD3 mechanism that DDPG (single critic) does not have.
- The actor architecture itself (deterministic, tanh-squashed MLP) is UNCHANGED from DDPG --
  TD3 differs from DDPG in how the actor is *trained* (delayed updates, smoothed targets) and
  in the critic (twin-Q), not in the actor's network topology. We therefore keep the actor
  identical in shape/behavior to ddpg.models.DDPGMLPActor (same 83->256->256->3 architecture,
  same tanh*act_limit output scaling, same tuple-observation handling), but it is defined here
  as its own class (not imported from ddpg/) so that TD3's actor and DDPG's actor remain two
  independently-instantiated, independently-checkpointed modules with non-colliding state_dict
  namespaces/paths (see td3/config.py's isolation asserts) -- per the project rule that TD3
  must be a genuinely separate implementation, not a thin wrapper around the DDPG code.
"""

import numpy as np
import torch
import torch.nn as nn

from tmrl.actor import TorchActorModule
from tmrl.custom.custom_models import mlp, MLPQFunction  # MLPQFunction reused directly, confirmed compatible
from tmrl.util import prod


class TD3MLPActor(TorchActorModule):
    """
    Deterministic actor: dim_obs -> 256 -> ReLU -> 256 -> ReLU -> 3 -> Tanh -> [-1, 1] * act_limit

    Identical topology to the project's DDPG actor. TD3's departure from DDPG lives entirely in
    how this actor is trained (td3/agent.py: delayed updates against a *smoothed* target critic
    pair), not in this module.
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

        # Exploration noise callable, e.g. td3.noise.GaussianExplorationNoise, called as
        # noise(action, act_limit). NOT part of state_dict -> does NOT survive
        # TorchActorModule.save()/load() (torch.save(state_dict())). Must therefore be set on
        # the actor instance living in the RolloutWorker/inference script directly, never
        # relied upon to transfer from the TrainingAgent's copy (see train_td3.py).
        self.noise = None

        # Optional runtime-only hook for live rollout forensics.  It is never
        # part of the state_dict and therefore cannot alter a saved policy.
        self.trace_callback = None

    def forward(self, obs):
        """Differentiable deterministic forward pass. obs: tuple of batched tensors (or single tensor)."""
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        pi_action = torch.tanh(mu) * self.act_limit
        return pi_action

    def act(self, obs, test=False):
        """
        Called by TorchActorModule.act_(), which already applies torch.no_grad() and has
        already collated a single obs into a batch of 1 on self.device. RolloutWorker passes
        test=not train, so test=False means "exploring" -> apply noise if one is attached;
        test=True means evaluation -> stay purely deterministic. Must return numpy.array (per
        tmrl.actor.ActorModule.act docstring).

        NOTE: this exploration noise is entirely separate from TD3's *target policy smoothing*
        noise (td3/agent.py), which is added only inside the critic's Bellman-target computation
        during training and never touches the action actually sent to TrackMania.
        """
        a = self.forward(obs)
        raw_action = a.squeeze().detach().cpu().numpy()
        res = raw_action.copy()
        if not len(res.shape):
            res = np.expand_dims(res, 0)
        if not test and self.noise is not None:
            res = self.noise(res, act_limit=self.act_limit)

        if self.trace_callback is not None:
            # `obs` is the exact collated observation used by the actor after
            # RolloutWorker preprocessing.  Keep the trace compact but retain
            # all action-history vectors and LIDAR/speed summary statistics.
            trace = {
                "test": bool(test),
                "speed": float(obs[0].reshape(-1)[0].detach().cpu()),
                "lidar_min": float(obs[1].min().detach().cpu()),
                "lidar_mean": float(obs[1].mean().detach().cpu()),
                "lidar_max": float(obs[1].max().detach().cpu()),
                "previous_actions": [
                    x.reshape(-1).detach().cpu().numpy().tolist()
                    for x in obs[2:]
                ],
                "raw_action": raw_action.reshape(-1).tolist(),
                "final_action": np.asarray(res).reshape(-1).tolist(),
                "applied_noise": (
                    np.asarray(res).reshape(-1)
                    - raw_action.reshape(-1)
                ).tolist(),
            }
            self.trace_callback(trace)
        return res


class TD3ActorCritic(nn.Module):
    """
    Combined actor + twin-critic module, following the exact instantiation contract confirmed
    in tmrl.custom.custom_algorithms.SpinupSacAgent.__post_init__:
        model = self.model_cls(observation_space, action_space)   # 2 positional args only
        self.model = model.to(device)                              # device applied afterwards

    TWO independent critics (critic1, critic2) -- this is TD3's defining structural difference
    from the project's existing single-critic DDPG. Both are plain, independently-initialized
    tmrl.custom.custom_models.MLPQFunction instances (same class SAC's own twin critics use),
    imported and reused directly rather than reimplemented.
    """

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.actor = TD3MLPActor(observation_space, action_space, hidden_sizes=hidden_sizes, activation=activation)
        self.critic1 = MLPQFunction(observation_space, action_space, hidden_sizes=hidden_sizes, activation=activation)
        self.critic2 = MLPQFunction(observation_space, action_space, hidden_sizes=hidden_sizes, activation=activation)

    def act(self, obs, test=False):
        return self.actor.act(obs, test=test)
