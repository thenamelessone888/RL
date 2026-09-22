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
# Curriculum: stage 0 is TMRL's stock tmrl-test map (already has a recorded
# reward trajectory -- TmrlData/resources/reward.pkl, the official
# pretrained-SAC baseline's trajectory -- reused via
# reward_registry.bootstrap_stage_zero_from_official_baseline()), used as a
# quick sanity-check warm-up stage.
#
# Stages 1-6 are the user's own six custom maps, built specifically for this
# curriculum in TrackMania's track editor (Documents\Trackmania\Maps\My Maps\
# first.Map.Gbx .. sixth.Map.Gbx), in increasing difficulty order by design.
# None of their reward trajectories are recorded yet -- record each one
# (while that stage's map is loaded in TrackMania) right before you're ready
# to train it:
#     python train_td3.py record-track-reward custom_1_first
#
# promotion_return_threshold is intentionally left unset (None -- fixed-step
# promotion only) for all six: a performance-based threshold would need to be
# calibrated against real achieved-return data on that specific track, which
# does not exist yet for tracks nobody has trained on (Section 12: don't pick
# a number speculatively). Once you've seen a stage's typical return range in
# practice, you can add a threshold for it, or promote manually at any time:
#     python train_td3.py curriculum-promote
#     python train_td3.py curriculum-jump <stage_name>
# ============================================================================

STAGES = [
    CurriculumStage(
        name="tmrl_test_baseline",
        description="tmrl-test: the stock TMRL map already used in Phases 0-6. "
                     "Quick sanity-check warm-up stage before the custom curriculum.",
        map_file="tmrl-test.Map.Gbx",
        min_env_steps=20_000,
        promotion_window=10,
        promotion_return_threshold=40.0,
    ),
    CurriculumStage(
        name="custom_1_first",
        description="Custom map 'first' -- easiest of the 6-track curriculum.",
        map_file="first.Map.Gbx",
        min_env_steps=20_000,
        promotion_window=10,
        promotion_return_threshold=None,
    ),
    CurriculumStage(
        name="custom_2_second",
        description="Custom map 'second'.",
        map_file="second.Map.Gbx",
        min_env_steps=20_000,
        promotion_window=10,
        promotion_return_threshold=None,
    ),
    CurriculumStage(
        name="custom_3_third",
        description="Custom map 'third'.",
        map_file="third.Map.Gbx",
        min_env_steps=20_000,
        promotion_window=10,
        promotion_return_threshold=None,
    ),
    CurriculumStage(
        name="custom_4_fourth",
        description="Custom map 'fourth'.",
        map_file="fourth.Map.Gbx",
        min_env_steps=20_000,
        promotion_window=10,
        promotion_return_threshold=None,
    ),
    CurriculumStage(
        name="custom_5_fifth",
        description="Custom map 'fifth'.",
        map_file="fifth.Map.Gbx",
        min_env_steps=20_000,
        promotion_window=10,
        promotion_return_threshold=None,
    ),
    CurriculumStage(
        name="custom_6_sixth",
        description="Custom map 'sixth' -- hardest of the 6-track curriculum "
                     "(last stage).",
        map_file="sixth.Map.Gbx",
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
