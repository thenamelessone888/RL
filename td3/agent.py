from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD

from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property

from td3.models import TD3ActorCritic


def _grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().data.norm(2).item() ** 2
    return total ** 0.5


@dataclass(eq=0)
class TD3Agent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None

    model_cls: type = TD3ActorCritic

    gamma: float = 0.99
    tau: float = 0.005

    policy_delay: int = 2

    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5

    lr_actor: float = 1e-3
    lr_critic: float = 1e-3

    optimizer_actor: str = "adam"
    optimizer_critic: str = "adam"

    l2_critic: float = None

    # Number of critic-only updates before the actor is allowed to move.
    critic_warmup_steps: int = 1000

    # Optional behavior-cloning actor initialization.
    warmstart_actor_path: str = None

    # BC-anchor regularization coefficient (TD3+BC style, Fujimoto & Gu 2021).
    # Only active when warmstart_actor_path is set. 0.0 disables it entirely,
    # reproducing plain TD3's unregularized actor loss (the default, so a
    # non-warm-started agent is unaffected).
    #
    # See the "candidate fixes" note in this module's train() docstring for
    # why this exists: critic_warmup_steps alone was found (Phase 6, TD3
    # research plan) to be insufficient to prevent the actor from
    # immediately collapsing away from a good BC policy once it starts
    # following a still-inaccurate critic's gradient.
    bc_reg_alpha: float = 0.0

    # The MSE anchor term itself (NOT bc_reg_alpha/lmbda, which scales the
    # Q-maximization side of the loss) is multiplied by a weight that
    # linearly decays from 1.0 down to bc_reg_anchor_min_weight over
    # bc_reg_anchor_decay_steps total critic updates, then stays at the
    # floor. decay_steps=0 disables decay (weight stays 1.0, the original
    # constant-anchor behavior).
    #
    # IMPORTANT (self-correction): a first version of this decayed
    # bc_reg_alpha itself downward, which is backwards -- bc_reg_alpha
    # scales lmbda multiplying the Q term, so a SMALLER bc_reg_alpha means a
    # SMALLER lmbda, which makes the (unscaled) MSE anchor term relatively
    # *stronger*, not weaker. Decaying bc_reg_alpha down would have made the
    # anchor's pull increase over training, the opposite of the intended
    # fix. This weight instead multiplies the anchor term directly, so
    # decaying it toward 0 unambiguously weakens the pull toward the BC
    # actor's prediction, leaving bc_reg_alpha/lmbda's Q-scaling untouched.
    #
    # Added after curriculum stage custom_2_second (2026-09-19) showed a
    # single stuck-at-wall failure persisting for 16+ consecutive rounds --
    # far longer than any prior occurrence. Hypothesis: with the anchor held
    # at full strength forever, the actor's gradient is permanently pulled
    # back toward the ORIGINAL frozen BC actor's prediction for that state --
    # and since the human demonstration never covered "how to escape being
    # stuck" (a competent driver doesn't get stuck), the BC actor's own
    # prediction there is presumably also poor, actively fighting the RL
    # signal that's trying to teach a recovery maneuver. Decaying the
    # anchor's weight over training keeps the original collapse-prevention
    # benefit early (when the critic is least trustworthy) while giving the
    # actor much more freedom later (when the critic has seen far more data)
    # to learn things the BC actor never demonstrated, like recovering from
    # a stuck state.
    bc_reg_anchor_min_weight: float = 1.0
    bc_reg_anchor_decay_steps: int = 0

    def _effective_anchor_weight(self):
        # Defensive: these are dataclass fields, but an agent restored from
        # a pre-decay checkpoint (unpickled directly, bypassing __init__)
        # won't have them set either -- see train()'s similar guard.
        decay_steps = getattr(self, "bc_reg_anchor_decay_steps", 0)
        if decay_steps <= 0:
            return 1.0
        min_weight = getattr(self, "bc_reg_anchor_min_weight", 1.0)
        progress = min(1.0, self._total_it / decay_steps)
        return 1.0 + progress * (min_weight - 1.0)

    model_nograd = cached_property(
        lambda self: no_grad(copy_shared(self.model))
    )

    def __post_init__(self):
        observation_space = self.observation_space
        action_space = self.action_space

        device = self.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = device

        # ---------------------------------------------------------
        # Main TD3 networks
        # ---------------------------------------------------------
        model = self.model_cls(observation_space, action_space)
        self.model = model.to(device)

        self.act_limit = float(action_space.high[0])

        # ---------------------------------------------------------
        # Optional human BC warm-start
        # ---------------------------------------------------------
        self.bc_actor = None

        if self.warmstart_actor_path:
            import os

            if not os.path.isfile(self.warmstart_actor_path):
                raise FileNotFoundError(
                    f"Human warm-start actor not found: "
                    f"{self.warmstart_actor_path}"
                )

            warm_actor = self.model.actor.load(
                self.warmstart_actor_path,
                device=device,
            )

            self.model.actor.load_state_dict(
                warm_actor.state_dict()
            )

            # Frozen reference copy of the BC actor, held for the lifetime
            # of training. Never optimized. Used only to anchor the online
            # actor via bc_reg_alpha (see train()) so that once critic
            # warm-up ends, the actor is not free to drift arbitrarily far
            # from demonstrated behavior based on a still-inaccurate critic.
            self.bc_actor = no_grad(deepcopy(self.model.actor))

        # ---------------------------------------------------------
        # Target networks start from the initialized main networks.
        #
        # If BC warm-start is enabled, this means:
        #
        # main actor  = human policy
        # target actor = human policy
        #
        # main critics = random
        # target critics = same random critics
        # ---------------------------------------------------------
        self.model_target = no_grad(deepcopy(self.model))

        # ---------------------------------------------------------
        # Optimizers
        # ---------------------------------------------------------
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

        critic_kwargs = {
            "lr": self.lr_critic
        }

        if self.l2_critic is not None:
            critic_kwargs["weight_decay"] = self.l2_critic

        self.actor_optimizer = actor_opt_cls(
            self.model.actor.parameters(),
            lr=self.lr_actor,
        )

        self.critic_optimizer = critic_opt_cls(
            list(self.model.critic1.parameters())
            + list(self.model.critic2.parameters()),
            **critic_kwargs,
        )

        # ---------------------------------------------------------
        # TD3 counters / diagnostics
        # ---------------------------------------------------------
        self._total_it = 0
        self._last_loss_actor = float("nan")
        self._last_actor_grad_norm = float("nan")
        self._last_bc_reg_term = float("nan")
        self._last_anchor_weight = float("nan")

    def get_actor(self):
        return self.model_nograd.actor

    def _soft_update_target_critics(self):
        """Polyak-update ONLY the two target critics."""
        with torch.no_grad():
            for p, p_targ in zip(
                self.model.critic1.parameters(),
                self.model_target.critic1.parameters(),
            ):
                p_targ.data.mul_(1.0 - self.tau)
                p_targ.data.add_(self.tau * p.data)

            for p, p_targ in zip(
                self.model.critic2.parameters(),
                self.model_target.critic2.parameters(),
            ):
                p_targ.data.mul_(1.0 - self.tau)
                p_targ.data.add_(self.tau * p.data)

    def _soft_update_targets(self):
        """Polyak-update actor + both critics."""
        with torch.no_grad():
            for p, p_targ in zip(
                self.model.parameters(),
                self.model_target.parameters(),
            ):
                p_targ.data.mul_(1.0 - self.tau)
                p_targ.data.add_(self.tau * p.data)

    def _smoothed_target_action(self, o2):
        """
        TD3 target policy smoothing.

        a' = pi_target(s') + clipped Gaussian noise
        """
        a2 = self.model_target.actor(o2)

        smoothing_noise = (
            torch.randn_like(a2) * self.target_policy_noise
        )

        smoothing_noise = smoothing_noise.clamp(
            -self.target_noise_clip,
            self.target_noise_clip,
        )

        a2 = (
            a2 + smoothing_noise
        ).clamp(
            -self.act_limit,
            self.act_limit,
        )

        return a2

    def train(self, batch):
        o, a, r, o2, d, _ = batch

        # Defensive: an agent restored from a checkpoint saved under an
        # older code version is unpickled directly (bypassing
        # __post_init__ entirely, standard Python dataclass/pickle
        # behavior), so any private state attribute added after that
        # checkpoint was written simply won't exist on the restored
        # object. Backfill defaults rather than crash on the first
        # resumed train() call.
        for _attr, _default in (
            ("_last_loss_actor", float("nan")),
            ("_last_actor_grad_norm", float("nan")),
            ("_last_bc_reg_term", float("nan")),
            ("_last_anchor_weight", float("nan")),
        ):
            if not hasattr(self, _attr):
                setattr(self, _attr, _default)

        self._total_it += 1

        # =========================================================
        # 1. TD3 TARGET
        # =========================================================
        with torch.no_grad():
            a2 = self._smoothed_target_action(o2)

            q1_targ = self.model_target.critic1(
                o2,
                a2,
            )

            q2_targ = self.model_target.critic2(
                o2,
                a2,
            )

            # Clipped Double Q-learning:
            q_targ = torch.min(
                q1_targ,
                q2_targ,
            )

            backup = (
                r
                + self.gamma * (1.0 - d) * q_targ
            )

        # =========================================================
        # 2. UPDATE BOTH CRITICS
        # =========================================================
        q1 = self.model.critic1(o, a)
        q2 = self.model.critic2(o, a)

        loss_critic1 = (
            (q1 - backup) ** 2
        ).mean()

        loss_critic2 = (
            (q2 - backup) ** 2
        ).mean()

        loss_critic = (
            loss_critic1
            + loss_critic2
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        loss_critic.backward()

        critic_grad_norm = (
            _grad_norm(self.model.critic1)
            + _grad_norm(self.model.critic2)
        )

        self.critic_optimizer.step()

        # =========================================================
        # 3. CRITIC WARM-UP
        # =========================================================
        #
        # During warm-up:
        #
        #   critics learn
        #   actor DOES NOT learn
        #
        # This prevents a randomly initialized critic from
        # immediately destroying the human BC policy.
        #
        # Target critics follow the online critics.
        # Target actor remains frozen at the BC policy.
        # =========================================================
        in_warmup = (
            self._total_it
            <= self.critic_warmup_steps
        )

        if in_warmup:
            # During critic warm-up, NEVER update the target actor.
            # Only the target critics track the online critics.
            self._soft_update_target_critics()

            return dict(
                loss_actor=self._last_loss_actor,
                loss_critic=loss_critic.detach().item(),
                loss_critic1=loss_critic1.detach().item(),
                loss_critic2=loss_critic2.detach().item(),
                q1_mean=q1.detach().mean().item(),
                q2_mean=q2.detach().mean().item(),
                target_q_mean=q_targ.detach().mean().item(),
                critic_grad_norm=critic_grad_norm,
                actor_grad_norm=self._last_actor_grad_norm,
                bc_reg_term=self._last_bc_reg_term,
                bc_reg_anchor_weight=self._last_anchor_weight,
                policy_updated=0.0,
                critic_warmup=1.0,
                total_updates=self._total_it,
            )

        # =========================================================
        # 4. NORMAL TD3 DELAYED ACTOR UPDATE
        # =========================================================
        delayed_step = (
            self._total_it % self.policy_delay == 0
        )

        if delayed_step:
            # Freeze critics while calculating actor gradient.
            self.model.critic1.requires_grad_(False)
            self.model.critic2.requires_grad_(False)

            pi = self.model.actor(o)

            q1_pi = self.model.critic1(
                o,
                pi,
            )

            bc_reg_term = float("nan")

            if self.bc_actor is not None and self.bc_reg_alpha > 0:
                # TD3+BC-style anchor (Fujimoto & Gu, 2021), adapted for
                # online fine-tuning from a warm-start actor rather than a
                # fixed offline dataset: the online replay buffer's states
                # are not paired with a recorded human action, so the BC
                # actor's OWN prediction on the current batch's states is
                # used as the regression target instead of a literal
                # dataset action.
                #
                # lmbda normalizes the Q-term's scale against the
                # regression term's scale (mirrors the paper's
                # alpha / mean(|Q|) normalization), so this remains
                # well-behaved as Q-values grow during training instead of
                # requiring per-run manual tuning of a fixed weight.
                with torch.no_grad():
                    bc_pi = self.bc_actor(o)

                lmbda = self.bc_reg_alpha / (
                    q1_pi.detach().abs().mean().clamp(min=1e-6)
                )

                bc_reg_term_t = F.mse_loss(pi, bc_pi)

                anchor_weight = self._effective_anchor_weight()

                loss_actor = (
                    -lmbda * q1_pi.mean()
                    + anchor_weight * bc_reg_term_t
                )

                bc_reg_term = bc_reg_term_t.detach().item()
                self._last_anchor_weight = anchor_weight
            else:
                loss_actor = -q1_pi.mean()

            self._last_bc_reg_term = bc_reg_term

            self.actor_optimizer.zero_grad(
                set_to_none=True
            )

            loss_actor.backward()

            self._last_actor_grad_norm = (
                _grad_norm(self.model.actor)
            )

            self.actor_optimizer.step()

            # Re-enable critic gradients.
            self.model.critic1.requires_grad_(True)
            self.model.critic2.requires_grad_(True)

            self._last_loss_actor = (
                loss_actor.detach().item()
            )

            # TD3 target update occurs only when actor updates.
            self._soft_update_targets()

        return dict(
            loss_actor=self._last_loss_actor,
            loss_critic=loss_critic.detach().item(),
            loss_critic1=loss_critic1.detach().item(),
            loss_critic2=loss_critic2.detach().item(),
            q1_mean=q1.detach().mean().item(),
            q2_mean=q2.detach().mean().item(),
            target_q_mean=q_targ.detach().mean().item(),
            critic_grad_norm=critic_grad_norm,
            actor_grad_norm=self._last_actor_grad_norm,
            bc_reg_term=self._last_bc_reg_term,
            bc_reg_anchor_weight=self._last_anchor_weight,
            policy_updated=float(delayed_step),
            critic_warmup=0.0,
            total_updates=self._total_it,
        )