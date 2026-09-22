"""
Per-stage reward trajectory storage and activation (Section 15 of the
research plan: "never accidentally use Track A's reward/reference trajectory
on Track B").

TMRL's RewardFunction always reads from the single fixed path
cfg.REWARD_PATH (TmrlData/reward/reward.pkl) at environment-interface
initialization time -- there is no track parameter in the installed
tmrl==0.7.1 API (verified by reading tmrl/custom/tm/tm_gym_interfaces.py and
tmrl/custom/tm/utils/compute_reward.py). Rather than modify TMRL, this module
is the "smallest external layer necessary" Section 15 calls for: it keeps one
canonical copy of each stage's reward trajectory under
TmrlData/reward/curriculum/<stage>.pkl, and *activates* a stage by copying its
file over cfg.REWARD_PATH immediately before that stage's trainer/worker are
launched -- backing up whatever was active before, so nothing is silently lost.
"""

import pickle
import shutil
import time
from pathlib import Path

import tmrl.config.config_constants as cfg

from curriculum.stages import CURRICULUM_REWARD_DIR, STAGES, stage_by_name

ACTIVE_REWARD_PATH = Path(cfg.REWARD_PATH)
ACTIVE_REWARD_BACKUP_DIR = cfg.TMRL_FOLDER / "reward" / "_pre_curriculum_backup"


def _ensure_dirs():
    CURRICULUM_REWARD_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_REWARD_BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def record_stage_reward(stage_name, use_keyboard=True, force=False):
    """
    Records a new reward trajectory for `stage_name` by driving the track
    that is CURRENTLY LOADED in TrackMania (the caller is responsible for
    having loaded stage.map_file already -- this function cannot do that).

    Thin wrapper around TMRL's own official recording tool
    (tmrl.tools.record.record_reward_dist), pointed at this stage's registry
    path instead of the shared default TmrlData/reward/reward.pkl, so
    recording a new stage's trajectory never touches whatever is currently
    active for training.

    Controls (from TMRL's own tool): press 'e' to start recording, drive the
    track, press 'q' (or reach the finish line) to stop and save.
    """
    _ensure_dirs()

    stage = stage_by_name(stage_name)

    if stage.reward_path.exists() and not force:
        raise FileExistsError(
            f"Reward trajectory already recorded for stage {stage_name!r}: "
            f"{stage.reward_path}\nUse --force to intentionally re-record it."
        )

    print("=" * 60)
    print(f"RECORDING REWARD TRAJECTORY FOR STAGE: {stage_name}")
    print("=" * 60)
    print(f"Map expected to be loaded in TrackMania: {stage.map_file}")
    print(f"Output: {stage.reward_path}")
    print()
    print("Press 'e' in-game to start recording, drive the track, then")
    print("press 'q' or cross the finish line to stop and save.")
    print()

    from tmrl.tools.record import record_reward_dist

    record_reward_dist(path_reward=str(stage.reward_path), use_keyboard=use_keyboard)

    print(f"[DONE] Saved reward trajectory: {stage.reward_path}")


def bootstrap_stage_zero_from_official_baseline(force=False):
    """
    Stage 0 (tmrl_test_baseline) uses the same tmrl-test map already used in
    Phases 0-6, which already has a recorded trajectory: the official
    pretrained-SAC baseline's TmrlData/reward/reward.pkl (Section 28: this
    file is READ here, never modified). This copies it into the curriculum
    registry as stage 0's canonical reward file, so stage 0 is immediately
    "ready" without needing to re-record anything.
    """
    _ensure_dirs()

    stage = stage_by_name("tmrl_test_baseline")

    if stage.reward_path.exists() and not force:
        print(f"Stage {stage.name!r} already has a registered reward file: "
              f"{stage.reward_path}")
        return stage.reward_path

    official_reward_path = Path(cfg.TMRL_FOLDER) / "resources" / "reward.pkl"
    if not official_reward_path.is_file():
        raise FileNotFoundError(
            f"Official baseline reward file not found: {official_reward_path}"
        )

    shutil.copy2(official_reward_path, stage.reward_path)
    print(f"[DONE] Bootstrapped stage 'tmrl_test_baseline' reward from "
          f"{official_reward_path} -> {stage.reward_path}")
    return stage.reward_path


def activate(stage_name):
    """
    Copies stage_name's registered reward trajectory over the live
    cfg.REWARD_PATH, backing up whatever was there first. Must be called
    BEFORE launching the trainer/worker for that stage (the reward file is
    only read once, when TM2020InterfaceLidar.initialize_common() runs).
    """
    _ensure_dirs()

    stage = stage_by_name(stage_name)

    if not stage.ready:
        raise FileNotFoundError(
            f"Stage {stage_name!r} has no recorded reward trajectory yet: "
            f"{stage.reward_path}\n"
            f"Record it first: python train_td3.py record-track-reward {stage_name}"
        )

    if ACTIVE_REWARD_PATH.exists():
        backup_path = ACTIVE_REWARD_BACKUP_DIR / f"reward_{int(time.time())}.pkl"
        shutil.copy2(ACTIVE_REWARD_PATH, backup_path)

    ACTIVE_REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage.reward_path, ACTIVE_REWARD_PATH)

    print(f"[CURRICULUM] Activated reward trajectory for stage {stage_name!r}")
    print(f"[CURRICULUM]   {stage.reward_path} -> {ACTIVE_REWARD_PATH}")


def active_stage_matches(stage_name):
    """
    Best-effort sanity check: True if the currently-active reward.pkl is
    byte-identical to stage_name's registered file. Used to guard against
    starting training with the wrong track's reward file active (Section 15).
    """
    stage = stage_by_name(stage_name)
    if not stage.ready or not ACTIVE_REWARD_PATH.exists():
        return False
    with open(stage.reward_path, "rb") as f:
        stage_bytes = f.read()
    with open(ACTIVE_REWARD_PATH, "rb") as f:
        active_bytes = f.read()
    return stage_bytes == active_bytes


def status():
    print("=" * 60)
    print("CURRICULUM REWARD REGISTRY STATUS")
    print("=" * 60)
    for stage in STAGES:
        ready = "READY" if stage.ready else "NOT RECORDED YET"
        active_marker = " (ACTIVE)" if active_stage_matches(stage.name) else ""
        print(f"  {stage.name:24s} [{ready}]{active_marker}")
        print(f"    map: {stage.map_file}")
        print(f"    reward file: {stage.reward_path}")
    print()
