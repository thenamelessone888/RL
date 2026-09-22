"""
Human-driving TrackMania interface + environment construction for demonstration recording.

Verified against installed tmrl==0.7.1 source:
- tmrl.custom.tm.tm_gym_interfaces.TM2020Interface.send_control(control) docstring:
      "Non-blocking function. Applies the action given by the RL policy.
       If control is None, does nothing (e.g. to record)"
  i.e. TMRL's own interface already anticipates a "recording" mode where no synthetic
  control (keyboard SendInput / vgamepad ViGEm signals) should be emitted.
- control_keyboard.apply_control() / control_gamepad.control_gamepad() (imported by
  tm_gym_interfaces.py) are OUTPUT-only automation: they synthesize keypresses (Windows
  SendInput) or virtual-gamepad axis values (vgamepad/ViGEm) to drive TrackMania as if an
  RL policy were playing. TMRL has NO mechanism to *read* physical human keyboard/gamepad
  state anywhere in tm_gym_interfaces.py, control_keyboard.py, or control_gamepad.py.
- tmrl.tools.record.record_reward_dist() (TMRL's own "--record-reward --use-keyboard" tool)
  is TMRL's own precedent for "human drives while TMRL passively watches": it NEVER touches
  send_control at all, and reads keypresses via the third-party `keyboard` package
  (keyboard.is_pressed('e')), imported lazily and NOT listed in tmrl's own install
  requirements (Requires-Dist) -- an optional dependency the user installs separately,
  exactly as we require here (see human_input.py).
- reset_race()/close_finish_pop_up_tm20() (episode-boundary automation: pressing the
  in-game reset/respawn key or virtual-gamepad B button, and dismissing the finish popup)
  are UNRELATED to continuous driving control and are left untouched (inherited as-is), so
  the environment still auto-resets between recorded episodes exactly as in normal TMRL
  rollouts, using whichever mechanism (keyboard Delete key or virtual gamepad) matches
  cfg.PRAGMA_GAMEPAD / the user's existing in-game keybindings.

Therefore the correct, smallest project-side adapter is: override ONLY send_control() to be
a no-op (TMRL never emits synthetic keyboard/gamepad signals), leaving the human's real
physical input to reach TrackMania directly and unmodified through the OS, while the
observation/reward/reset pipeline (grab_lidar_speed_and_data, RewardFunction, LIDAR/speed
telemetry via TM2020OpenPlanetClient) is reused byte-for-byte from TM2020InterfaceLidar.

No installed tmrl files are modified; this is a pure project-side subclass.
"""

import numpy as np

import tmrl.config.config_constants as cfg
from tmrl.custom.tm.tm_gym_interfaces import TM2020InterfaceLidar
from tmrl.envs import GenericGymEnv
from tmrl.util import partial

from ddpg.config import DDPG_CONFIG_DICT, DDPG_DATASET_PATH


class TM2020InterfaceLidarHuman(TM2020InterfaceLidar):
    """
    Identical to TM2020InterfaceLidar (same observation/reward/reset pipeline), except:

    1. send_control() is a no-op: TMRL never emits synthetic keyboard/gamepad signals, so the
       human's real physical input is the only thing driving the car. Episode-boundary
       automation (reset_race / close_finish_pop_up_tm20, inherited unchanged) still runs.

    2. get_obs_rew_terminated_info() no longer lets RewardFunction.compute_reward()'s own
       `terminated` signal end the episode.

       Root cause (verified against tmrl.custom.tm.utils.compute_reward.RewardFunction,
       installed 0.7.1 source): compute_reward() force-terminates once
       `failure_counter > FAILURE_COUNTDOWN` (default 10) consecutive no-progress steps have
       elapsed past `MIN_STEPS` (default 70) -- i.e. ~4s at the shipped 20Hz time_step_duration
       -- whenever the car isn't advancing along the trajectory stored in reward.pkl. That
       trajectory is either (a) recorded for a DIFFERENT track than the one being driven now,
       or (b) doesn't exist yet for this track at all, in which case RewardFunction.__init__
       silently falls back to a 2-point dummy trajectory ([[0,0,0],[1,1,1]]) that ANY real
       position is automatically "too far" from. Either way, every human-recorded episode was
       being killed after ~4 seconds regardless of driving quality -- this is an artifact of
       reusing an RL-rollout safety mechanism in a context (human demonstration recording)
       it was never designed for, not a bug in the recorder/dataset code.

       We still compute the progress-based reward (useful for logging/inspection), we just
       stop trusting its `terminated` verdict. The ONLY thing allowed to end a human-recorded
       episode now is: an actual finish-line crossing (`data[8]`, unchanged from the parent
       class), rtgym's own ep_max_length/RW_MAX_SAMPLES_PER_EPISODE timeout (unchanged,
       handled upstream by RolloutWorker/GenericGymEnv), or the human's own Q/P key
       (handled in human_recorder.py, unrelated to this interface).
    """
    def send_control(self, control):
        pass  # human drives; TMRL must never emit synthetic input during recording

    def get_obs_rew_terminated_info(self):
        img, speed, data = self.grab_lidar_speed_and_data()
        # compute_reward()'s progress value is still useful to log; its `terminated` verdict
        # is deliberately discarded -- see class docstring above.
        rew, _reward_fn_wanted_terminate = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        end_of_track = bool(data[8])
        info = {"reward_fn_wanted_terminate": _reward_fn_wanted_terminate}
        terminated = False
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info


# Reuse the exact same img_hist_len/gamepad settings as the DDPG worker config.
HUMAN_INT = partial(TM2020InterfaceLidarHuman, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)

# Reuse DDPG's RTGYM_CONFIG (time_step_duration, ep_max_length, act_buf_len, act_in_obs, ...)
# verbatim -- only the interface class differs -- so recorded episodes use identical timing
# and action-history semantics to normal DDPG rollouts.
HUMAN_CONFIG_DICT = DDPG_CONFIG_DICT.copy()
HUMAN_CONFIG_DICT["interface"] = HUMAN_INT

HUMAN_ENV_CLS = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": HUMAN_CONFIG_DICT})

# ISOLATED dataset directory: never the SAC dataset, never the DDPG replay dataset.
HUMAN_DATASET_FOLDER = cfg.TMRL_FOLDER / "dataset_human"
HUMAN_DATASET_FOLDER.mkdir(parents=True, exist_ok=True)
HUMAN_DATASET_PATH = str(HUMAN_DATASET_FOLDER)

assert HUMAN_DATASET_PATH != cfg.DATASET_PATH
assert HUMAN_DATASET_PATH != DDPG_DATASET_PATH