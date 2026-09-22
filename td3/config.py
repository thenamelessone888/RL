
"""
TD3 configuration for TMRL 0.7.1.

This module reuses TMRL's existing:
    - TrackMania LIDAR environment
    - LIDAR observation preprocessing
    - LIDAR sample compressor
    - MemoryTMLidar replay memory
    - TorchTrainingOffline training infrastructure

But uses our own:
    - TD3Agent          (td3/agent.py)
    - TD3ActorCritic     (td3/models.py)

IMPORTANT:
    The pretrained SAC run is NEVER modified or overwritten.
    The project's existing DDPG run (DDPG_LIDAR_<RUN_NAME>) is NEVER modified or overwritten.

    TD3 uses:
        TD3_LIDAR_<RUN_NAME>.tmod
        TD3_LIDAR_<RUN_NAME>_t.tmod
        TD3_LIDAR_<RUN_NAME>_t.tcpt
        TmrlData/dataset_td3/

    The existing SAC and DDPG files remain untouched.

DDPG vs. TD3 -- summary of the algorithmic differences implemented across this package:

    | Aspect                      | DDPG (ddpg/)                  | TD3 (td3/, this package)            |
    |-----------------------------|-------------------------------|--------------------------------------|
    | Critics                     | 1 (critic)                    | 2, independent (critic1, critic2)   |
    | Bellman target               | Q_target(s', mu_target(s'))   | min(Q1_target, Q2_target)(s', smoothed a') |
    | Target policy smoothing      | none                          | clipped Gaussian noise on target action |
    | Actor/target update frequency| every train() call            | every `policy_delay` critic updates |
    | Exploration noise (rollout) | Ornstein-Uhlenbeck             | decaying IID Gaussian               |

Everything below this point follows the exact same reuse pattern as ddpg/config.py:
infrastructure (interface, memory, sample compressor, obs preprocessor,
TorchTrainingOffline) is imported unmodified; only the algorithm-specific
AGENT/POLICY/TRAINER partials differ.
"""

import rtgym

import tmrl.config.config_constants as cfg

from tmrl.custom.tm.tm_gym_interfaces import TM2020InterfaceLidar
from tmrl.custom.custom_memories import (
    MemoryTMLidar,
    get_local_buffer_sample_lidar,
)
from tmrl.custom.tm.tm_preprocessors import (
    obs_preprocessor_tm_lidar_act_in_obs,
)

from tmrl.envs import GenericGymEnv
from tmrl.training_offline import TorchTrainingOffline
from tmrl.util import partial

from td3.agent import TD3Agent
from td3.models import TD3ActorCritic


# ============================================================================
# CONFIGURATION SAFETY CHECKS
# ============================================================================

# TD3 implementation expects the 4-frame LIDAR observation pipeline (same as DDPG).
assert cfg.PRAGMA_LIDAR, (
    "TD3 configuration requires the LIDAR interface. "
    "The current TmrlData/config/config.json is not configured for LIDAR."
)

# This implementation currently supports the flat LIDAR history only.
assert not cfg.PRAGMA_RNN, (
    "TD3 configuration does not support RNN observations."
)


# ============================================================================
# RUN NAME
# ============================================================================

# Example:
#
#     SAC_4_LIDAR_pretrained
#
# becomes:
#
#     TD3_LIDAR_SAC_4_LIDAR_pretrained
#
# The TD3 prefix guarantees separate filenames from both the official SAC baseline
# AND this project's existing DDPG run (which uses the DDPG_LIDAR_ prefix).

TD3_RUN_NAME = "TD3_LIDAR_" + cfg.RUN_NAME


# ============================================================================
# TD3 WEIGHTS / CHECKPOINT PATHS
# ============================================================================

# We can share TMRL's existing weights/checkpoint directories.
# We NEVER share the filenames.

TD3_WEIGHTS_FOLDER = cfg.WEIGHTS_FOLDER
TD3_CHECKPOINTS_FOLDER = cfg.CHECKPOINTS_FOLDER


# Clean experiment namespace.  No path in this directory existed before this
# reset, so TMRL's checkpoint loader cannot resume the legacy TD3 run.
TD3_CLEAN_HUMAN_EXPERIMENT_NAME = "TD3_CLEAN_HUMAN_WARMSTART"
TD3_CLEAN_HUMAN_EXPERIMENT_FOLDER = (
    cfg.TMRL_FOLDER / "experiments" / TD3_CLEAN_HUMAN_EXPERIMENT_NAME
)
TD3_CLEAN_HUMAN_EXPERIMENT_FOLDER.mkdir(parents=True, exist_ok=True)

TD3_CLEAN_RANDOM_EXPERIMENT_NAME = "TD3_CLEAN_RANDOM"
TD3_CLEAN_RANDOM_EXPERIMENT_FOLDER = (
    cfg.TMRL_FOLDER / "experiments" / TD3_CLEAN_RANDOM_EXPERIMENT_NAME
)
TD3_CLEAN_RANDOM_EXPERIMENT_FOLDER.mkdir(parents=True, exist_ok=True)

# Actor used by the clean rollout worker.
TD3_MODEL_PATH_WORKER = str(
    TD3_CLEAN_RANDOM_EXPERIMENT_FOLDER / "worker.tmod"
)


# Optional worker model history.
#
# TMRL expects model_path_history to contain the filename prefix
# without the .tmod suffix.

TD3_MODEL_PATH_SAVE_HISTORY = str(
    TD3_CLEAN_RANDOM_EXPERIMENT_FOLDER / "history_"
)


# Actor saved by the Trainer.

TD3_MODEL_PATH_TRAINER = str(
    TD3_CLEAN_RANDOM_EXPERIMENT_FOLDER / "trainer.tmod"
)


# Trainer checkpoint.

TD3_CHECKPOINT_PATH = str(
    TD3_CLEAN_RANDOM_EXPERIMENT_FOLDER / "trainer.tcpt"
)


# ============================================================================
# ISOLATION CHECKS (SAC baseline + existing DDPG run)
# ============================================================================

# These assertions prevent accidental reuse of the pretrained SAC paths,
# AND accidental collision with this project's existing DDPG paths.

assert TD3_MODEL_PATH_WORKER != cfg.MODEL_PATH_WORKER, (
    "TD3 worker model path collides with SAC worker model path."
)

assert TD3_MODEL_PATH_TRAINER != cfg.MODEL_PATH_TRAINER, (
    "TD3 trainer model path collides with SAC trainer model path."
)

assert TD3_CHECKPOINT_PATH != cfg.CHECKPOINT_PATH, (
    "TD3 checkpoint path collides with SAC checkpoint path."
)

try:
    from ddpg.config import (
        DDPG_MODEL_PATH_WORKER,
        DDPG_MODEL_PATH_TRAINER,
        DDPG_CHECKPOINT_PATH,
        DDPG_DATASET_PATH,
    )

    assert TD3_MODEL_PATH_WORKER != DDPG_MODEL_PATH_WORKER, (
        "TD3 worker model path collides with DDPG worker model path."
    )

    assert TD3_MODEL_PATH_TRAINER != DDPG_MODEL_PATH_TRAINER, (
        "TD3 trainer model path collides with DDPG trainer model path."
    )

    assert TD3_CHECKPOINT_PATH != DDPG_CHECKPOINT_PATH, (
        "TD3 checkpoint path collides with DDPG checkpoint path."
    )

except ImportError:
    # ddpg/ package not importable in this environment
    # (e.g. a minimal TD3-only checkout).
    #
    # The SAC-isolation asserts above already ran, so TD3 remains
    # safe to configure on its own.
    pass


# ============================================================================
# TD3 DATASET
# ============================================================================

# IMPORTANT:
#
# Do NOT use cfg.DATASET_PATH (SAC/generic) or the DDPG package's dataset_ddpg here.
#
# TD3 gets its own replay dataset directory, isolated from both.

TD3_DATASET_FOLDER = cfg.TMRL_FOLDER / "dataset_td3"

TD3_DATASET_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

TD3_DATASET_PATH = str(
    TD3_DATASET_FOLDER
)

assert TD3_DATASET_PATH != cfg.DATASET_PATH, (
    "TD3 replay dataset path collides with SAC dataset path."
)


# ============================================================================
# TRACKMANIA RTGYM INTERFACE
# ============================================================================

# Reuse TMRL's LIDAR interface, identically to DDPG.
#
# Observation history:
#
#     4 LIDAR frames
#
# and the existing RTGym action-history configuration is preserved.

TD3_INT = partial(
    TM2020InterfaceLidar,
    img_hist_len=cfg.IMG_HIST_LEN,
    gamepad=cfg.PRAGMA_GAMEPAD,
)


# ============================================================================
# RTGYM CONFIGURATION
# ============================================================================

# Start with RTGym's defaults.

TD3_CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()


# Replace the interface with the LIDAR interface.

TD3_CONFIG_DICT["interface"] = TD3_INT


# Preserve the remaining RTGym configuration from the user's existing
# TMRL configuration.
#
# This keeps settings such as:
#     time_step_duration
#     act_in_obs
#     act_buf_len
#     reset_act_buf
#     ep_max_length
#     etc.
#
# exactly aligned with the existing TMRL setup.

for key, value in cfg.ENV_CONFIG["RTGYM_CONFIG"].items():
    TD3_CONFIG_DICT[key] = value


# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================

TD3_ENV_CLS = partial(
    GenericGymEnv,
    id=cfg.RTGYM_VERSION,
    gym_kwargs={
        "config": TD3_CONFIG_DICT,
    },
)


# ============================================================================
# SAMPLE COMPRESSOR
# ============================================================================

# Used by RolloutWorker before samples are sent to the Server.
# Algorithm-agnostic and reused unmodified.

TD3_SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar


# ============================================================================
# OBSERVATION PREPROCESSOR
# ============================================================================

# Converts the raw environment LIDAR observation into the observation format
# consumed by the TD3 actor/critics.
#
# Algorithm-agnostic; reused unmodified.

TD3_OBS_PREPROCESSOR = (
    obs_preprocessor_tm_lidar_act_in_obs
)


# No additional replay-data augmentation.

TD3_SAMPLE_PREPROCESSOR = None


# ============================================================================
# REPLAY MEMORY
# ============================================================================

TD3_MEMORY_SIZE = 1_000_000

TD3_BATCH_SIZE = 256


# TMRL TrainingOffline calls the memory constructor with:
#
#     nb_steps=...
#     device=...
#
# Those are supplied by TMRL.
#
# Everything else is pre-bound here.

TD3_MEMORY = partial(
    MemoryTMLidar,

    memory_size=TD3_MEMORY_SIZE,

    batch_size=TD3_BATCH_SIZE,

    sample_preprocessor=TD3_SAMPLE_PREPROCESSOR,

    # IMPORTANT:
    # Separate from SAC and DDPG replay data.
    dataset_path=TD3_DATASET_PATH,

    imgs_obs=cfg.IMG_HIST_LEN,

    act_buf_len=cfg.ACT_BUF_LEN,

    crc_debug=cfg.CRC_DEBUG,
)


# ============================================================================
# TD3 AGENT
# ============================================================================

# TMRL's TrainingOffline creates the agent with:
#
#     observation_space
#     action_space
#     device
#
# Therefore all TD3-specific hyperparameters are pre-bound here.

TD3_AGENT = partial(
    TD3Agent,

    device=(
        "cuda"
        if cfg.CUDA_TRAINING
        else "cpu"
    ),

    model_cls=TD3ActorCritic,

    # ------------------------------------------------------------------------
    # TD3
    # ------------------------------------------------------------------------

    gamma=0.99,

    # Polyak / target-network update coefficient.
    tau=0.005,

    # Actor update every 2 critic updates.
    policy_delay=2,

    # TD3 target policy smoothing.
    target_policy_noise=0.2,
    target_noise_clip=0.5,

    # ------------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------------

    lr_actor=1e-3,

    lr_critic=1e-3,

    optimizer_actor="adam",

    optimizer_critic="adam",

    # No critic L2 regularization.
    l2_critic=None,

    # ------------------------------------------------------------------------
    # CRITIC WARM-UP
    # ------------------------------------------------------------------------
    #
    # For the first 1000 critic updates:
    #
    #   - critics train
    #   - actor does NOT update
    #   - target critics track online critics
    #   - target actor remains at its initialized policy
    #
    # For the human-warm-start configuration this means the BC actor
    # remains intact while the critics become useful.
    #
    critic_warmup_steps=1000,
)


# ============================================================================
# TD3 TRAINER
# ============================================================================

# Conservative values for the first real TrackMania integration.
# These remain matched with the existing DDPG trainer.

TD3_TRAINER = partial(
    TorchTrainingOffline,

    # Dummy environment used by TrainingOffline to obtain
    # observation_space and action_space.
    env_cls=TD3_ENV_CLS,

    # Our isolated LIDAR replay memory.
    memory_cls=TD3_MEMORY,

    # Our TD3 implementation.
    training_agent_cls=TD3_AGENT,

    # ------------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------------

    epochs=1000,

    rounds=10,

    steps=200,

    # Broadcast updated TD3 actor every 200 training steps.
    update_model_interval=200,

    # Retrieve worker experience every 200 training steps.
    update_buffer_interval=200,

    # Do not allow training to get excessively ahead of environment data.
    max_training_steps_per_env_step=1.0,

    # No profiling for the initial integration.
    profiling=False,

    # No scheduler initially.
    agent_scheduler=None,

    # Begin TD3 updates after at least 200 replay samples.
    start_training=200,

    # Explicit training device.
    device=(
        "cuda"
        if cfg.CUDA_TRAINING
        else "cpu"
    ),
)


# ============================================================================
# HUMAN-DEMONSTRATION WARM START
# ============================================================================

# The recorded human demonstration DATA itself is reused directly.
#
# Only the behavior-cloned actor weights have TD3-specific paths.

TD3_HUMAN_WARMSTART_NAME = TD3_CLEAN_HUMAN_EXPERIMENT_NAME


TD3_HUMAN_WARMSTART_ACTOR_PATH = str(
    TD3_WEIGHTS_FOLDER / (
        TD3_RUN_NAME
        + "_HUMAN_actor.tmod"
    )
)


TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER = str(
    TD3_CLEAN_HUMAN_EXPERIMENT_FOLDER / "worker.tmod"
)


TD3_HUMAN_WARMSTART_MODEL_PATH_TRAINER = str(
    TD3_CLEAN_HUMAN_EXPERIMENT_FOLDER / "trainer.tmod"
)


TD3_HUMAN_WARMSTART_MODEL_PATH_SAVE_HISTORY = str(
    TD3_CLEAN_HUMAN_EXPERIMENT_FOLDER / "history_"
)


TD3_HUMAN_WARMSTART_CHECKPOINT_PATH = str(
    TD3_CLEAN_HUMAN_EXPERIMENT_FOLDER / "trainer.tcpt"
)


# ============================================================================
# HUMAN WARM-START ISOLATION CHECKS
# ============================================================================

assert TD3_HUMAN_WARMSTART_ACTOR_PATH not in {
    cfg.MODEL_PATH_WORKER,
    cfg.MODEL_PATH_TRAINER,
    cfg.CHECKPOINT_PATH,
}

assert TD3_HUMAN_WARMSTART_CHECKPOINT_PATH != (
    TD3_CHECKPOINT_PATH
)

assert TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER != (
    TD3_MODEL_PATH_WORKER
)

assert TD3_HUMAN_WARMSTART_MODEL_PATH_TRAINER != (
    TD3_MODEL_PATH_TRAINER
)


# ============================================================================
# HUMAN WARM-START TD3 AGENT
# ============================================================================

TD3_HUMAN_WARMSTART_AGENT = partial(
    TD3Agent,

    device=(
        "cuda"
        if cfg.CUDA_TRAINING
        else "cpu"
    ),

    model_cls=TD3ActorCritic,

    # ------------------------------------------------------------------------
    # TD3
    # ------------------------------------------------------------------------

    gamma=0.99,

    # Polyak / target-network update coefficient.
    tau=0.005,

    # Actor update every 2 critic updates.
    policy_delay=2,

    # TD3 target policy smoothing.
    target_policy_noise=0.2,
    target_noise_clip=0.5,

    # ------------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------------

    lr_actor=1e-3,

    lr_critic=1e-3,

    optimizer_actor="adam",

    optimizer_critic="adam",

    l2_critic=None,

    # ------------------------------------------------------------------------
    # CRITIC WARM-UP
    # ------------------------------------------------------------------------
    #
    # CRITICAL FOR HUMAN WARM-START:
    #
    # The behavior-cloned actor is loaded first.
    #
    # For the first 1000 critic updates:
    #
    #       HUMAN BC ACTOR
    #             |
    #             v
    #          FROZEN
    #
    # while the twin critics learn.
    #
    # This prevents the initially unreliable critics from immediately
    # pushing the actor away from the demonstrated human behavior.
    #
    critic_warmup_steps=1000,

    # Load the canonical human behavior-cloned actor.
    warmstart_actor_path=TD3_HUMAN_WARMSTART_ACTOR_PATH,
)


# ============================================================================
# HUMAN WARM-START TRAINER
# ============================================================================

TD3_HUMAN_WARMSTART_TRAINER = partial(
    TorchTrainingOffline,

    env_cls=TD3_ENV_CLS,

    memory_cls=TD3_MEMORY,

    training_agent_cls=TD3_HUMAN_WARMSTART_AGENT,

    # ------------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------------

    epochs=1000,

    rounds=10,

    steps=200,

    # Broadcast updated actor every 200 training steps.
    update_model_interval=200,

    # Retrieve worker experience every 200 training steps.
    update_buffer_interval=200,

    max_training_steps_per_env_step=1.0,

    profiling=False,

    agent_scheduler=None,

    # Wait for replay samples before training.
    start_training=200,

    device=(
        "cuda"
        if cfg.CUDA_TRAINING
        else "cpu"
    ),
)
