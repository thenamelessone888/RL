"""
DDPG TrainingAgent for TMRL 0.7.1.

Verified against installed tmrl==0.7.1 source:
- tmrl.training.TrainingAgent is an ABC with __init__(self, observation_space, action_space, device)
  and abstract methods train(batch) -> dict, get_actor() -> ActorModule.
- tmrl.custom.custom_algorithms.SpinupSacAgent (the actual SAC implementation shipped with tmrl)
  is declared as `@dataclass(eq=0)` with observation_space/action_space/device as the first
  three fields, and does NOT call TrainingAgent.__init__() explicitly -- the dataclass-generated
  __init__ sets self.observation_space/self.action_space/self.device directly, since those field
  names match. We follow the exact same pattern here (not guessed, copied from the installed
  SpinupSacAgent's actual construction, confirmed via inspection).
- tmrl.training_offline.TrainingOffline.__post_init__ instantiates the agent as:
      self.agent = self.training_agent_cls(observation_space=..., action_space=..., device=...)
  i.e. ONLY these three kwargs are supplied at instantiation time. Every other hyperparameter
  (model_cls, gamma, tau, lr_actor, lr_critic, ...) must be pre-bound with tmrl.util.partial in
  ddpg/config.py, exactly mirroring how tmrl.config.config_objects.py builds `AGENT = partial(SAC_Agent, ...)`.
- self.model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model))) and
  get_actor() -> self.model_nograd.actor is the exact idiom used by SpinupSacAgent to broadcast
  the current policy to RolloutWorkers without disturbing the trainable model; reused verbatim
  for consistency (no_grad/copy_shared are general tmrl.custom.utils.nn helpers, not SAC-specific).

DDPG algorithm design choices (ours, not from TMRL -- report was not supplied for this project,
these are standard DDPG choices per the project's PHASE 3 specification):
- Single critic + single target critic (no twin-Q; that would be TD3, not DDPG).
- Target initialization: hard copy (target <- online) via deepcopy, before any soft updates.
- Bellman target: y = r + gamma * (1 - done) * Q_target(s', actor_target(s'))  (no entropy term).
- Critic loss: MSE(Q(s,a), y).
- Actor loss: -mean(Q(s, actor(s))).
- Soft/Polyak update, using the project's own `tau` naming (opposite convention from SAC's
  `polyak`): target = tau * online + (1 - tau) * target, i.e. implemented as
      p_targ.data.mul_(1 - tau); p_targ.data.add_(tau * p.data)
  which is algebraically the same operation as SAC's `mul_(polyak); add_((1-polyak)*p)` with
  tau = 1 - polyak. Typical DDPG tau is small (~0.001-0.005), i.e. the analogous polyak would be
  large (~0.995-0.999), consistent with SAC's default polyak=0.995.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.optim import Adam, AdamW, SGD

from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property

from ddpg.models import DDPGActorCritic


@dataclass(eq=0)
class DDPGAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None                     # device the model lives on (None -> auto cuda/cpu)
    model_cls: type = DDPGActorCritic
    gamma: float = 0.99                     # discount factor
    tau: float = 0.005                      # soft target-update rate (see module docstring)
    lr_actor: float = 1e-4                  # DDPG paper default
    lr_critic: float = 1e-3                 # DDPG paper default
    optimizer_actor: str = "adam"           # one of ["adam", "adamw", "sgd"]
    optimizer_critic: str = "adam"
    l2_critic: float = None                 # optional critic weight decay (DDPG paper uses 1e-2)
    warmstart_actor_path: str = None        # optional behavior-cloned actor .tmod

    # Broadcastable, gradient-detached, storage-shared copy of self.model, used by get_actor().
    # Exact same idiom as SpinupSacAgent.model_nograd.
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        model = self.model_cls(observation_space, action_space)  # 2 positional args, verified contract
        self.model = model.to(device)

        # Optional human-demonstration warm start. Only the actor is initialized;
        # the critic remains freshly initialized and learns from normal DDPG replay.
        if self.warmstart_actor_path:
            import os
            if not os.path.isfile(self.warmstart_actor_path):
                raise FileNotFoundError(
                    f"Human warm-start actor not found: {self.warmstart_actor_path}"
                )
            warm_actor = self.model.actor.load(
                self.warmstart_actor_path,
                device=device,
            )
            self.model.actor.load_state_dict(warm_actor.state_dict())

        # Target initialization: target <- online (hard copy), then detach gradients.
        self.model_target = no_grad(deepcopy(self.model))

        def _optimizer_cls(name):
            name = name.lower()
            if name == "adam":
                return Adam
            elif name == "adamw":
                return AdamW
            else:
                return SGD

        actor_opt_cls = _optimizer_cls(self.optimizer_actor)
        critic_opt_cls = _optimizer_cls(self.optimizer_critic)

        critic_kwargs = {"lr": self.lr_critic}
        if self.l2_critic is not None:
            critic_kwargs["weight_decay"] = self.l2_critic

        self.actor_optimizer = actor_opt_cls(self.model.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = critic_opt_cls(self.model.critic.parameters(), **critic_kwargs)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        o, a, r, o2, d, _ = batch  # (prev_obs, action, reward, new_obs, terminated, truncated); as returned
        # by Memory.__getitem__ / TorchMemory.collate, confirmed via tmrl.memory.Memory docstring.

        # ---- Critic update ----
        with torch.no_grad():
            a2 = self.model_target.actor(o2)
            q_targ = self.model_target.critic(o2, a2)
            backup = r + self.gamma * (1 - d) * q_targ

        q = self.model.critic(o, a)
        loss_critic = ((q - backup) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---- Actor update ----
        # Freeze critic params so actor's backward pass doesn't waste compute on their grads.
        self.model.critic.requires_grad_(False)

        pi = self.model.actor(o)
        q_pi = self.model.critic(o, pi)
        loss_actor = -q_pi.mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        self.model.critic.requires_grad_(True)

        # ---- Soft target update (both actor and critic) ----
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)

        return dict(
            loss_actor=loss_actor.detach().item(),
            loss_critic=loss_critic.detach().item(),
        )