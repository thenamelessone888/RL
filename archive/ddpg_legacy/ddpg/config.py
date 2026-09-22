
"""
DDPG configuration for TMRL 0.7.1.

This module reuses TMRL's existing:
    - TrackMania LIDAR environment
    - LIDAR observation preprocessing
    - LIDAR sample compressor
    - MemoryTMLidar replay memory
    - TorchTrainingOffline training infrastructure

But uses our own:
    - DDPGAgent
    - DDPGActorCritic

IMPORTANT:
    The pretrained SAC run is NEVER modified or overwritten.

    DDPG uses:
        DDPG_LIDAR_<RUN_NAME>.tmod
        DDPG_LIDAR_<RUN_NAME>_t.tmod
        DDPG_LIDAR_<RUN_NAME>_t.tcpt
        TmrlData/dataset_ddpg/

    The existing SAC files remain untouched.
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

from ddpg.agent import DDPGAgent
from ddpg.models import DDPGActorCritic


# ============================================================================
# CONFIGURATION SAFETY CHECKS
# ============================================================================

# DDPG implementation expects the 4-frame LIDAR observation pipeline.
assert cfg.PRAGMA_LIDAR, (
    "DDPG configuration requires the LIDAR interface. "
    "The current TmrlData/config/config.json is not configured for LIDAR."
)

# This implementation currently supports the flat LIDAR history only.
assert not cfg.PRAGMA_RNN, (
    "DDPG configuration does not support RNN observations."
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
#     DDPG_LIDAR_SAC_4_LIDAR_pretrained
#
# The DDPG prefix guarantees separate filenames.

DDPG_RUN_NAME = "DDPG_LIDAR_" + cfg.RUN_NAME


# ============================================================================
# DDPG WEIGHTS / CHECKPOINT PATHS
# ============================================================================

# We can share TMRL's existing weights/checkpoint directories.
# We NEVER share the filenames.

DDPG_WEIGHTS_FOLDER = cfg.WEIGHTS_FOLDER
DDPG_CHECKPOINTS_FOLDER = cfg.CHECKPOINTS_FOLDER


# Actor used by the RolloutWorker.
DDPG_MODEL_PATH_WORKER = str(
    DDPG_WEIGHTS_FOLDER / (
        DDPG_RUN_NAME + ".tmod"
    )
)


# Optional worker model history.
#
# TMRL expects model_path_history to contain the filename prefix
# without the .tmod suffix.

DDPG_MODEL_PATH_SAVE_HISTORY = str(
    DDPG_WEIGHTS_FOLDER / (
        DDPG_RUN_NAME + "_"
    )
)


# Actor saved by the Trainer.

DDPG_MODEL_PATH_TRAINER = str(
    DDPG_WEIGHTS_FOLDER / (
        DDPG_RUN_NAME + "_t.tmod"
    )
)


# Trainer checkpoint.

DDPG_CHECKPOINT_PATH = str(
    DDPG_CHECKPOINTS_FOLDER / (
        DDPG_RUN_NAME + "_t.tcpt"
    )
)


# ============================================================================
# SAC ISOLATION CHECKS
# ============================================================================

# These assertions prevent accidental reuse of the pretrained SAC paths.

assert DDPG_MODEL_PATH_WORKER != cfg.MODEL_PATH_WORKER, (
    "DDPG worker model path collides with SAC worker model path."
)

assert DDPG_MODEL_PATH_TRAINER != cfg.MODEL_PATH_TRAINER, (
    "DDPG trainer model path collides with SAC trainer model path."
)

assert DDPG_CHECKPOINT_PATH != cfg.CHECKPOINT_PATH, (
    "DDPG checkpoint path collides with SAC checkpoint path."
)


# ============================================================================
# DDPG DATASET
# ============================================================================

# IMPORTANT:
#
# Do NOT use cfg.DATASET_PATH here.
#
# cfg.DATASET_PATH belongs to the existing TMRL/SAC configuration.
#
# DDPG gets its own replay dataset directory.

DDPG_DATASET_FOLDER = cfg.TMRL_FOLDER / "dataset_ddpg"

DDPG_DATASET_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

DDPG_DATASET_PATH = str(
    DDPG_DATASET_FOLDER
)


assert DDPG_DATASET_PATH != cfg.DATASET_PATH, (
    "DDPG replay dataset path collides with SAC dataset path."
)


# ============================================================================
# TRACKMANIA RTGYM INTERFACE
# ============================================================================

# Reuse TMRL's LIDAR interface.
#
# Observation history:
#
#     4 LIDAR frames
#
# and the existing RTGym action-history configuration is preserved.

DDPG_INT = partial(
    TM2020InterfaceLidar,

    img_hist_len=cfg.IMG_HIST_LEN,

    gamepad=cfg.PRAGMA_GAMEPAD,
)


# ============================================================================
# RTGYM CONFIGURATION
# ============================================================================

# Start with RTGym's defaults.

DDPG_CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()


# Replace the interface with the LIDAR interface.

DDPG_CONFIG_DICT["interface"] = DDPG_INT


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
    DDPG_CONFIG_DICT[key] = value


# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================

DDPG_ENV_CLS = partial(
    GenericGymEnv,

    id=cfg.RTGYM_VERSION,

    gym_kwargs={
        "config": DDPG_CONFIG_DICT,
    },
)


# ============================================================================
# SAMPLE COMPRESSOR
# ============================================================================

# Used by RolloutWorker before samples are sent to the Server.

DDPG_SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar


# ============================================================================
# OBSERVATION PREPROCESSOR
# ============================================================================

# Converts the raw environment LIDAR observation into the observation format
# consumed by the DDPG actor/critic.

DDPG_OBS_PREPROCESSOR = (
    obs_preprocessor_tm_lidar_act_in_obs
)


# No additional replay-data augmentation.

DDPG_SAMPLE_PREPROCESSOR = None


# ============================================================================
# REPLAY MEMORY
# ============================================================================

DDPG_MEMORY_SIZE = 1_000_000

DDPG_BATCH_SIZE = 256


# TMRL TrainingOffline calls the memory constructor with:
#
#     nb_steps=...
#     device=...
#
# Those are supplied by TMRL.
#
# Everything else is pre-bound here.

DDPG_MEMORY = partial(
    MemoryTMLidar,

    memory_size=DDPG_MEMORY_SIZE,

    batch_size=DDPG_BATCH_SIZE,

    sample_preprocessor=DDPG_SAMPLE_PREPROCESSOR,

    # IMPORTANT:
    # Separate from SAC replay data.
    dataset_path=DDPG_DATASET_PATH,

    imgs_obs=cfg.IMG_HIST_LEN,

    act_buf_len=cfg.ACT_BUF_LEN,

    crc_debug=cfg.CRC_DEBUG,
)


# ============================================================================
# DDPG AGENT
# ============================================================================

# TMRL's TrainingOffline creates the agent with:
#
#     observation_space
#     action_space
#     device
#
# Therefore all DDPG-specific hyperparameters are pre-bound here.

DDPG_AGENT = partial(
    DDPGAgent,

    device=(
        "cuda"
        if cfg.CUDA_TRAINING
        else "cpu"
    ),

    model_cls=DDPGActorCritic,

    # ------------------------------------------------------------------------
    # DDPG
    # ------------------------------------------------------------------------

    gamma=0.99,

    # Polyak / target-network update coefficient.
    tau=0.005,

    # ------------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------------

    lr_actor=1e-4,

    lr_critic=1e-3,

    optimizer_actor="adam",

    optimizer_critic="adam",

    # No critic L2 regularization initially.
    l2_critic=None,
)


# ============================================================================
# DDPG TRAINER
# ============================================================================

# Conservative values for the first real TrackMania integration.
#
# Once the complete networking pipeline works, these can be increased.

DDPG_TRAINER = partial(
    TorchTrainingOffline,

    # Dummy environment used by TrainingOffline to obtain
    # observation_space and action_space.
    env_cls=DDPG_ENV_CLS,

    # Our isolated LIDAR replay memory.
    memory_cls=DDPG_MEMORY,

    # Our DDPG implementation.
    training_agent_cls=DDPG_AGENT,

    # ------------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------------

    epochs=1000,

    rounds=10,

    steps=200,

    # Broadcast updated DDPG actor every 200 training steps.
    update_model_interval=200,

    # Retrieve worker experience every 200 training steps.
    update_buffer_interval=200,

    # Do not allow training to get excessively ahead of environment data.
    max_training_steps_per_env_step=1.0,

    # No profiling for the initial integration.
    profiling=False,

    # No scheduler initially.
    agent_scheduler=None,

    # Begin DDPG updates after at least 200 replay samples.
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

DDPG_HUMAN_WARMSTART_NAME = DDPG_RUN_NAME + "_HUMAN"

DDPG_HUMAN_WARMSTART_ACTOR_PATH = str(
    DDPG_WEIGHTS_FOLDER / (DDPG_HUMAN_WARMSTART_NAME + "_actor.tmod")
)

DDPG_HUMAN_WARMSTART_MODEL_PATH_WORKER = str(
    DDPG_WEIGHTS_FOLDER / (DDPG_HUMAN_WARMSTART_NAME + ".tmod")
)

DDPG_HUMAN_WARMSTART_MODEL_PATH_TRAINER = str(
    DDPG_WEIGHTS_FOLDER / (DDPG_HUMAN_WARMSTART_NAME + "_t.tmod")
)

DDPG_HUMAN_WARMSTART_MODEL_PATH_SAVE_HISTORY = str(
    DDPG_WEIGHTS_FOLDER / (DDPG_HUMAN_WARMSTART_NAME + "_")
)

DDPG_HUMAN_WARMSTART_CHECKPOINT_PATH = str(
    DDPG_CHECKPOINTS_FOLDER / (DDPG_HUMAN_WARMSTART_NAME + "_t.tcpt")
)

assert DDPG_HUMAN_WARMSTART_ACTOR_PATH not in {
    cfg.MODEL_PATH_WORKER, cfg.MODEL_PATH_TRAINER, cfg.CHECKPOINT_PATH
}
assert DDPG_HUMAN_WARMSTART_CHECKPOINT_PATH != DDPG_CHECKPOINT_PATH
assert DDPG_HUMAN_WARMSTART_MODEL_PATH_WORKER != DDPG_MODEL_PATH_WORKER
assert DDPG_HUMAN_WARMSTART_MODEL_PATH_TRAINER != DDPG_MODEL_PATH_TRAINER

DDPG_HUMAN_WARMSTART_AGENT = partial(
    DDPGAgent,
    device=("cuda" if cfg.CUDA_TRAINING else "cpu"),
    model_cls=DDPGActorCritic,
    gamma=0.99,
    tau=0.005,
    lr_actor=1e-4,
    lr_critic=1e-3,
    optimizer_actor="adam",
    optimizer_critic="adam",
    l2_critic=None,
    warmstart_actor_path=DDPG_HUMAN_WARMSTART_ACTOR_PATH,
)

DDPG_HUMAN_WARMSTART_TRAINER = partial(
    TorchTrainingOffline,
    env_cls=DDPG_ENV_CLS,
    memory_cls=DDPG_MEMORY,
    training_agent_cls=DDPG_HUMAN_WARMSTART_AGENT,
    epochs=1000,
    rounds=10,
    steps=200,
    update_model_interval=200,
    update_buffer_interval=200,
    max_training_steps_per_env_step=1.0,
    profiling=False,
    agent_scheduler=None,
    start_training=200,
    device=("cuda" if cfg.CUDA_TRAINING else "cpu"),
)

