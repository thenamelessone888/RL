"""
Stuck-recovery reward shaping (mitigation for docs/known_issues.md's
"deterministic policy gets stuck at a fixed point" finding).

Root cause recap (see docs/known_issues.md for the full trace analysis): the
policy has no learned recovery behavior when it collides with a wall --
lidar_min stays at 0.0 and speed stays near 0 for many consecutive steps,
with the actor just weakly re-committing forward gas instead of reversing or
turning away, until the reward function's own stall-timeout
(MIN_STEPS=70 + FAILURE_COUNTDOWN=10) ends the episode. Since TMRL's reward
is otherwise pure forward-progress-along-trajectory, there is no direct
signal telling the critic that "the action taken while wedged against this
wall was bad" -- reward is just 0 in that state, same as many other
uninteresting states, giving the actor's -Q(s,pi(s)) gradient nothing
specific to push against.

This module adds ONE project-side subclass of TM2020InterfaceLidar (Section
16's "least invasive integration" -- no installed TMRL file is modified)
that applies a small negative reward once the car has been stuck (lidar_min
below a threshold AND speed below a threshold) for several consecutive
steps. This gives the critic an explicit, immediate reason to value escaping
a stuck state, rather than waiting for episode-length statistics to
implicitly discourage it much more weakly.

This is reward shaping, not a guaranteed fix -- Section 12 of the research
plan is explicit that this should be evaluated with real data (see
experiments/ for whichever run first tests this), not just assumed to work
because it sounds reasonable.
"""

import numpy as np

from tmrl.custom.tm.tm_gym_interfaces import TM2020InterfaceLidar


class TM2020InterfaceLidarStuckRecovery(TM2020InterfaceLidar):
    """
    Identical to TM2020InterfaceLidar, except get_obs_rew_terminated_info()
    applies stuck_penalty to the reward once the car has been "stuck"
    (lidar_min <= stuck_lidar_threshold AND speed <= stuck_speed_threshold)
    for stuck_patience consecutive steps.

    stuck_patience defaults to half of TMRL's own reward-config
    FAILURE_COUNTDOWN (10, per TmrlData/config/config.json), so this signal
    arrives well before the stall-timeout would otherwise end the episode
    with no specific penalty attached to the stuck state itself.
    """

    def __init__(self, *args,
                 stuck_lidar_threshold=0.5,
                 stuck_speed_threshold=2.0,
                 stuck_patience=5,
                 stuck_penalty=-0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._stuck_lidar_threshold = stuck_lidar_threshold
        self._stuck_speed_threshold = stuck_speed_threshold
        self._stuck_patience = stuck_patience
        self._stuck_penalty = stuck_penalty
        self._consecutive_stuck_steps = 0

    def reset(self, seed=None, options=None):
        self._consecutive_stuck_steps = 0
        return super().reset(seed=seed, options=options)

    def _apply_stuck_shaping(self, obs, rew, info):
        """
        Pure(ish) shaping step, split out from get_obs_rew_terminated_info()
        so it can be unit-tested offline against synthetic (obs, rew, info)
        without requiring a live TrackMania window/screenshot (see
        tests/test_stuck_recovery.py). Mutates self._consecutive_stuck_steps
        and `info`; returns the (possibly penalized) reward.
        """
        speed = float(obs[0][0])
        lidar_min = float(np.min(obs[1]))

        is_stuck_now = (
            lidar_min <= self._stuck_lidar_threshold
            and speed <= self._stuck_speed_threshold
        )

        if is_stuck_now:
            self._consecutive_stuck_steps += 1
        else:
            self._consecutive_stuck_steps = 0

        info["consecutive_stuck_steps"] = self._consecutive_stuck_steps

        if self._consecutive_stuck_steps >= self._stuck_patience:
            rew = np.float32(rew + self._stuck_penalty)
            info["stuck_penalty_applied"] = True
        else:
            info["stuck_penalty_applied"] = False

        return rew

    def get_obs_rew_terminated_info(self):
        obs, rew, terminated, info = super().get_obs_rew_terminated_info()
        rew = self._apply_stuck_shaping(obs, rew, info)
        return obs, rew, terminated, info
