"""
DDPG exploration noise. Pure Python/NumPy - no TMRL API surface, nothing to verify.
Our own design choice (report's suggested starting values: theta=0.15, sigma=0.20).
"""

import numpy as np


class OrnsteinUhlenbeckNoise:
    """
    Standard OU process for DDPG exploration, with optional linear or exponential
    decay of sigma over time. The actor itself stays deterministic; this is added
    externally to the actor's action at rollout time.
    """

    def __init__(self, action_dim, theta=0.15, sigma=0.20, dt=1e-2,
                 sigma_min=0.0, decay_steps=0, decay_type="linear"):
        """
        Args:
            action_dim (int): dimensionality of the action space (3 for [gas, brake, steer])
            theta (float): OU mean-reversion rate
            sigma (float): initial OU volatility
            dt (float): time step size for the OU discretization
            sigma_min (float): floor value sigma decays towards
            decay_steps (int): number of act() calls over which sigma decays to sigma_min
                                (0 disables decay, sigma stays constant)
            decay_type (str): "linear" or "exponential"
        """
        assert decay_type in ("linear", "exponential")
        self.action_dim = action_dim
        self.theta = theta
        self.sigma_init = sigma
        self.sigma = sigma
        self.dt = dt
        self.sigma_min = sigma_min
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self._step_count = 0
        self.state = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self):
        """Call at the start of each episode."""
        self.state = np.zeros(self.action_dim, dtype=np.float32)

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
        """Draw one OU noise sample and advance internal state/decay."""
        self._update_sigma()
        dx = self.theta * (-self.state) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state = self.state + dx
        self._step_count += 1
        return self.state.astype(np.float32)

    def __call__(self, action, act_limit=1.0):
        """Apply noise to a deterministic action and clip to [-act_limit, act_limit]."""
        noisy = action + self.sample()
        return np.clip(noisy, -act_limit, act_limit)