"""
Human demonstration recorder for TMRL 0.7.1.

Reuses, unmodified:
- TM2020InterfaceLidarHuman (human_interface.py) -- itself a thin subclass of
  TM2020InterfaceLidar reusing its observation/reward/reset pipeline verbatim.
- The same GenericGymEnv / rtgym real-time stepping pipeline any algorithm's RolloutWorker uses
  (same time_step_duration, ep_max_length, act_buf_len, act_in_obs -- see human_interface.py).
- tmrl.networking.Buffer (verified: a tiny generic class, self.memory=[], append_sample(),
  clear() -- identical to what RolloutWorker uses internally).
- tmrl.custom.custom_memories.get_local_buffer_sample_lidar (the exact SAC/DDPG sample
  compressor) and tmrl.custom.custom_memories.MemoryTMLidar.append_buffer/get_transition
  (the exact native on-disk transition format/reconstruction logic) -- we do NOT invent a
  second history/compression mechanism (see Problem 4 lesson).

Verified action/observation pairing against tmrl.networking.RolloutWorker.step()/reset()
source (docstring: "act is for the PREVIOUS transition (act, obs(act))"):
- On env.reset(): act = env.unwrapped.default_action (rtgym's own zero action), exactly as
  RolloutWorker.reset() does (`self.env.unwrapped.default_action`), NOT a human-read action
  (no control decision has been "applied" yet at reset time).
- On each step: act = human-read action (stands in for self.actor.act_(obs) in the normal
  pipeline); new_obs, rew, terminated, truncated, info = env.step(act); obs_preprocessor is
  applied to new_obs BEFORE sample compression, exactly matching RolloutWorker.step()'s order
  (get_local_buffer_sample_lidar's `obs[1][-19:]` slice only makes sense on the *flattened*
  76-length LIDAR array produced by the preprocessor, not the raw (4,19) array).

Pausing uses env.unwrapped.wait() (rtgym's own documented pause mechanism -- see
RealTimeGymInterface.step()'s docstring: "If you want to 'pause' the environment ... use the
wait() method"), followed by env.reset() to resume, matching rtgym's own documented contract.

Dataset persistence: Memory.__init__ only ever READS data.pkl (verified: no save() method
exists anywhere in tmrl.memory or tmrl.custom.custom_memories) -- TMRL expects whoever seeds
an offline dataset to write data.pkl themselves. We do so here via plain pickle.dump of
memory.data, in the exact list-of-9-parallel-lists format MemoryTM.__init__ reads back.

NEVER loads or references the pretrained SAC model or any algorithm's actor. NEVER feeds data
into an algorithm's own replay dataset. NEVER sends synthetic control (see
TM2020InterfaceLidarHuman).
"""

import pickle
import time
from pathlib import Path

from tmrl.networking import Buffer
from tmrl.custom.custom_memories import MemoryTMLidar, get_local_buffer_sample_lidar
from tmrl.custom.tm.tm_preprocessors import obs_preprocessor_tm_lidar_act_in_obs

from demonstrations.human_interface import HUMAN_ENV_CLS, HUMAN_DATASET_PATH
from demonstrations.human_input import KeyboardHumanController

BANNER = r"""
============================================================
HUMAN DEMONSTRATION RECORDER
============================================================

TrackMania ready.
Human control enabled (TMRL sends no synthetic input; you drive
with TrackMania's normal keyboard/gamepad controls in-game).

Recorder controls (polled via the `keyboard` package):
  R            = Start recording / resume after pause
  P            = Pause recording (car frozen; call R to resume)
  Q or ESC     = Stop recording and save

Output:
  {dataset_path}
""".strip("\n")


def _load_or_create_memory():
    """
    A real MemoryTMLidar instance used purely as the encoder/decoder for the native
    on-disk transition format (append_buffer / get_transition), never for RL sampling.
    memory_size is set very large so append_buffer never silently trims recorded data.
    """
    import tmrl.config.config_constants as cfg
    return MemoryTMLidar(memory_size=100_000_000,
                          batch_size=1,
                          dataset_path=HUMAN_DATASET_PATH,
                          imgs_obs=cfg.IMG_HIST_LEN,
                          act_buf_len=cfg.ACT_BUF_LEN,
                          nb_steps=1,
                          sample_preprocessor=None,
                          crc_debug=False,
                          device="cpu")


def _save_memory(memory):
    path = Path(HUMAN_DATASET_PATH) / "data.pkl"
    with open(path, "wb") as f:
        pickle.dump(memory.data, f)


def _reset_sample(env, buffer):
    """Mirrors tmrl.networking.RolloutWorker.reset(): act = env.unwrapped.default_action."""
    act = env.unwrapped.default_action
    new_obs, info = env.reset()
    new_obs = obs_preprocessor_tm_lidar_act_in_obs(new_obs)
    sample = get_local_buffer_sample_lidar(act, new_obs, 0.0, False, False, info)
    buffer.append_sample(sample)
    return new_obs, info


def run_recorder(episodes=None, minutes=None):
    print(BANNER.format(dataset_path=HUMAN_DATASET_PATH))
    print()

    controller = KeyboardHumanController()
    env = HUMAN_ENV_CLS()
    memory = _load_or_create_memory()
    print(f"[INFO] Existing transitions in dataset: {len(memory)}")

    print("[WAIT] Press R in this console/game window to start recording...")
    while not controller.start_requested():
        time.sleep(0.05)

    buffer = Buffer()
    episode_count = 0
    start_time = time.time()
    recording = True
    _reset_sample(env, buffer)
    print("[START] Recording episode 1...")

    try:
        while True:
            if controller.stop_requested():
                print("[STOP] Stop requested.")
                break

            if controller.pause_requested():
                recording = not recording
                if not recording:
                    print("[PAUSE] Recording paused. Press R to resume (starts a new episode).")
                    env.unwrapped.wait()  # rtgym's own documented pause mechanism
                else:
                    print(f"[RESUME] Recording episode {episode_count + 1}...")
                    _reset_sample(env, buffer)
                continue

            if not recording:
                time.sleep(0.05)
                continue

            act = controller.read_action()
            new_obs, rew, terminated, truncated, info = env.step(act)
            new_obs = obs_preprocessor_tm_lidar_act_in_obs(new_obs)
            sample = get_local_buffer_sample_lidar(act, new_obs, rew, terminated, truncated, info)
            buffer.append_sample(sample)

            if terminated or truncated:
                print()
                print("=" * 60)
                print("[EPISODE END]")
                print(f"terminated = {terminated}")
                print(f"truncated  = {truncated}")
                print(f"info       = {info}")
                print("=" * 60)
                print()

                memory.append(buffer)
                buffer.clear()
                _save_memory(memory)
                episode_count += 1
                print(f"[EPISODE {episode_count}] done. Total transitions: {len(memory)}")

                stop_now = False
                if episodes is not None and episode_count >= episodes:
                    stop_now = True
                if minutes is not None and (time.time() - start_time) >= minutes * 60:
                    stop_now = True
                if stop_now:
                    break

                _reset_sample(env, buffer)
                print(f"[START] Recording episode {episode_count + 1}...")
    finally:
        if len(buffer.memory) > 0:
            memory.append(buffer)
            buffer.clear()
        _save_memory(memory)
        try:
            env.unwrapped.wait()
        except Exception:
            pass
        print()
        print(f"[DONE] Saved {episode_count} episode(s), {len(memory)} transitions to:")
        print(f"       {HUMAN_DATASET_PATH}")


if __name__ == "__main__":
    run_recorder()
