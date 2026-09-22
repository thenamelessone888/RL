"""
Curriculum stage definitions (Section 10 of the research plan).

A stage is: a track (map) + its recorded reward trajectory + promotion
criteria. The SAME TD3 policy (curriculum/config.py's TD3_CURRICULUM_* paths)
is carried across every stage -- only the active reward file (see
reward_registry.py) and the physically-loaded TrackMania map change between
stages. Map loading cannot be automated (no TMRL/OpenPlanet API for it in
this installation -- verified by inspecting the installed tmrl==0.7.1 source
and its OpenPlanet plugins); it is a manual step, printed by
curriculum/watch_and_promote.py when a stage completes.

Reward trajectories are stored under
    C:\\Users\\sreci\\TmrlData\\reward\\curriculum\\<stage.name>.pkl
(runtime data, per Section 0.1 -- never inside the git-tracked project).
A stage is "ready" only once its reward file has been recorded.
"""

from dataclasses import dataclass
from pathlib import Path

import tmrl.config.config_constants as cfg

CURRICULUM_REWARD_DIR = cfg.TMRL_FOLDER / "reward" / "curriculum"


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    description: str
    # The .Map.Gbx file the user must have loaded in TrackMania for this
    # stage. Informational only -- we cannot load it for them.
    map_file: str
    # Minimum real environment steps to train on this stage before it is
    # even eligible for promotion (Section 10's "fixed-step curriculum").
    min_env_steps: int
    # Number of most-recent completed episodes to average over before
    # evaluating the performance gate (Section 10: "do not promote merely
    # because a single episode succeeds"). None disables the performance
    # gate entirely (pure fixed-step promotion for this stage).
    promotion_window: int = 10
    promotion_return_threshold: float = None

    @property
    def reward_path(self) -> Path:
        return CURRICULUM_REWARD_DIR / f"{self.name}.pkl"

    @property
    def ready(self) -> bool:
        """True once this stage's reward trajectory has been recorded."""
        return self.reward_path.is_file()


# ============================================================================
# Initial curriculum: the two maps TMRL ships out of the box.
#
# tmrl-test:  already has a recorded reward trajectory
#             (TmrlData/resources/reward.pkl, the official pretrained-SAC
#             baseline's trajectory) -- reused here as stage 0's reward via
#             reward_registry.bootstrap_stage_zero_from_official_baseline().
# tmrl-train: no reward trajectory exists yet. Record one with:
#                 python train_td3.py record-track-reward tmrl_train_harder
#             while tmrl-train.Map.Gbx is loaded in TrackMania.
#
# Add more CurriculumStage entries here as more tracks become available
# (Section 10: "the exact tracks must be determined from the actual
# available TrackMania maps" -- never invented).
# ============================================================================

STAGES = [
    CurriculumStage(
        name="tmrl_test_baseline",
        description="tmrl-test: the stock TMRL map already used in Phases 0-6.",
        map_file="tmrl-test.Map.Gbx",
        min_env_steps=20_000,
        promotion_window=10,
        promotion_return_threshold=40.0,
    ),
    CurriculumStage(
        name="tmrl_train_harder",
        description="tmrl-train: TMRL's second stock map, used here as the "
                     "next curriculum stage. Needs its reward trajectory "
                     "recorded before this stage can run.",
        map_file="tmrl-train.Map.Gbx",
        min_env_steps=20_000,
        promotion_window=10,
        promotion_return_threshold=None,  # last stage: fixed-step only
    ),
]


def stage_by_name(name):
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"No curriculum stage named {name!r}. Known stages: "
                    f"{[s.name for s in STAGES]}")


def stage_index(name):
    for i, stage in enumerate(STAGES):
        if stage.name == name:
            return i
    raise KeyError(f"No curriculum stage named {name!r}.")
