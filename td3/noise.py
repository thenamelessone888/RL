"""
TD3 exploration noise. Pure Python/NumPy - no TMRL API surface, nothing to verify.

TD3 (Fujimoto et al. 2018) explores with simple, temporally-UNCORRELATED Gaussian noise added
to the deterministic actor's output at rollout time -- this is a deliberate departure from the
project's existing DDPG implementation, which uses Ornstein-Uhlenbeck (OU) noise. We do not
reuse ddpg.noise.OrnsteinUhlenbeckNoise for TD3: the task specification explicitly calls for
"appropriate Gaussian exploration with configurable decay. Do not blindly reuse DDPG OU noise."

This class is entirely separate from TD3's *target policy smoothing* noise (also Gaussian,
also clipped), which is generated fresh inside td3/agent.py's train() and is never applied to
actions sent to TrackMania -- it only perturbs the *target* actor's next-state action used to
compute the critics' Bellman backup. Conflating the two would be a real TD3 implementation bug
(this file implements only the rollout-time exploration noise).
"""

import numpy as np


class GaussianExplorationNoise:
    """
    IID Gaussian exploration noise, added directly to the deterministic actor's action, with
    optional linear or exponential decay of sigma over the number of act() calls. Unlike OU
    noise, samples at consecutive timesteps are independent (no temporal correlation, no
    internal `state` to carry across steps) -- this is what makes it "TD3-style" rather than
    "DDPG-style" exploration, per Fujimoto et al. 2018 and the project's own TD3 specification.
    """

    def __init__(self, action_dim, sigma=0.1, sigma_min=0.0, decay_steps=0, decay_type="linear"):
        """
        Args:
            action_dim (int): dimensionality of the action space (3 for [gas, brake, steer])
            sigma (float): initial noise standard deviation (TD3 paper default: 0.1 * act_limit,
                            i.e. 0.1 for this project's [-1, 1]-bounded action space)
            sigma_min (float): floor value sigma decays towards
            decay_steps (int): number of act() calls over which sigma decays to sigma_min
                                (0 disables decay, sigma stays constant)
            decay_type (str): "linear" or "exponential"
        """
        assert decay_type in ("linear", "exponential")
        self.action_dim = action_dim
        self.sigma_init = sigma
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self._step_count = 0

    def reset(self):
        """No-op: Gaussian exploration noise carries no per-episode state (unlike OU noise's
        mean-reverting `state`). Kept for interface symmetry with ddpg.noise.OrnsteinUhlenbeckNoise
        so callers can treat either noise object uniformly (e.g. call .reset() at episode
        boundaries without needing to know which algorithm is running)."""
        pass

    def _update_sigma(self):
        if self.decay_steps <= 0:
            return
        frac = min(1.0, self._step_count / self.decay_steps)
        if self.decay_type == "linear":
            self.sigma = self.sigma_init + frac * (self.sigma_min - self.sigma_init)
        else:  # exponential
            ratio = self.sigma_min / self.sigma_init if self.sigma_init > 0 else 0.0
            ratio = max(ratio, 1e-8)
            self.sigma = self.sigma_init * (ratio ** frac)

    def sample(self):
        """Draw one IID Gaussian noise sample and advance the decay schedule."""
        self._update_sigma()
        noise = np.random.randn(self.action_dim).astype(np.float32) * self.sigma
        self._step_count += 1
        return noise

    def __call__(self, action, act_limit=1.0):
        """Apply noise to a deterministic action and clip to [-act_limit, act_limit]."""
        noisy = action + self.sample()
        return np.clip(noisy, -act_limit, act_limit)
