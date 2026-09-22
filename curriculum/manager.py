"""
Curriculum state machine: which stage is active, how much progress it has
made, and whether it is ready to promote (Section 10/11 of the research
plan). Independent of TD3 (only consumes env-step counts and episode
returns), and independent of TMRL/networking -- consumed by
curriculum/watch_and_promote.py, which is the thing that actually watches a
live trainer and calls into this.

State is persisted to JSON so a promotion decision (and "how much of stage 2
have I already trained") survives process restarts -- resuming the trainer
does not silently reset curriculum progress.
"""

import json
import time
from collections import deque
from pathlib import Path

import tmrl.config.config_constants as cfg

from curriculum.stages import STAGES, stage_index

STATE_PATH = cfg.TMRL_FOLDER / "experiments" / "TD3_CURRICULUM" / "curriculum_state.json"


class CurriculumManager:
    def __init__(self, state_path=STATE_PATH):
        self.state_path = Path(state_path)
        self.stage_idx = 0
        self.env_steps_this_stage = 0
        self.return_window = deque(maxlen=STAGES[0].promotion_window)
        self.promotion_log = []  # list of {"stage": ..., "promoted_at": ...}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if not self.state_path.is_file():
            return
        with open(self.state_path) as f:
            data = json.load(f)
        self.stage_idx = data["stage_idx"]
        self.env_steps_this_stage = data["env_steps_this_stage"]
        window = deque(data["return_window"], maxlen=self.current_stage.promotion_window)
        self.return_window = window
        self.promotion_log = data.get("promotion_log", [])

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stage_idx": self.stage_idx,
            "stage_name": self.current_stage.name,
            "env_steps_this_stage": self.env_steps_this_stage,
            "return_window": list(self.return_window),
            "promotion_log": self.promotion_log,
            "last_updated": time.time(),
        }
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Stage access
    # ------------------------------------------------------------------

    @property
    def current_stage(self):
        return STAGES[self.stage_idx]

    @property
    def is_last_stage(self):
        return self.stage_idx == len(STAGES) - 1

    def jump_to_stage(self, stage_name, reset_progress=True):
        """Manual promotion/override (Section 10 supports this explicitly)."""
        idx = stage_index(stage_name)
        self.stage_idx = idx
        if reset_progress:
            self.env_steps_this_stage = 0
            self.return_window = deque(maxlen=self.current_stage.promotion_window)
        self._save()

    # ------------------------------------------------------------------
    # Progress tracking (called once per completed training round/episode)
    # ------------------------------------------------------------------

    def record_round(self, episode_length_train, return_train):
        """
        Feed one round's worth of stats (as reported by TMRL's own
        TrainingOffline logging -- see watch_and_promote.py). episode_length
        is added to the running env-step count for this stage; return_train
        is pushed into the moving-average window used by the performance
        gate.
        """
        if episode_length_train is not None:
            self.env_steps_this_stage += int(episode_length_train)
        if return_train is not None:
            self.return_window.append(float(return_train))
        self._save()

    # ------------------------------------------------------------------
    # Promotion decision (Section 10: fixed-step AND performance-based;
    # never promote off a single episode)
    # ------------------------------------------------------------------

    def should_promote(self):
        stage = self.current_stage

        if self.env_steps_this_stage < stage.min_env_steps:
            return False

        if stage.promotion_return_threshold is None:
            # Fixed-step-only stage: min_env_steps reached is sufficient.
            return True

        if len(self.return_window) < stage.promotion_window:
            # Window not yet full -- not enough recent episodes to trust
            # an average (Section 10's explicit warning against promoting
            # off too little data).
            return False

        avg_return = sum(self.return_window) / len(self.return_window)
        return avg_return >= stage.promotion_return_threshold

    def promote(self):
        if self.is_last_stage:
            raise RuntimeError(
                f"Already on the last curriculum stage ({self.current_stage.name}); "
                "nothing to promote to."
            )
        completed = self.current_stage
        self.promotion_log.append({
            "stage": completed.name,
            "env_steps": self.env_steps_this_stage,
            "final_avg_return": (
                sum(self.return_window) / len(self.return_window)
                if self.return_window else None
            ),
            "promoted_at": time.time(),
        })
        self.stage_idx += 1
        self.env_steps_this_stage = 0
        self.return_window = deque(maxlen=self.current_stage.promotion_window)
        self._save()
        return self.current_stage

    def status_string(self):
        stage = self.current_stage
        avg_return = (
            sum(self.return_window) / len(self.return_window)
            if self.return_window else float("nan")
        )
        lines = [
            f"Stage {self.stage_idx + 1}/{len(STAGES)}: {stage.name}",
            f"  map: {stage.map_file}",
            f"  env_steps_this_stage: {self.env_steps_this_stage} / {stage.min_env_steps}",
            f"  return window: {len(self.return_window)}/{stage.promotion_window} "
            f"episodes, avg={avg_return:.2f}"
            + (f" (threshold={stage.promotion_return_threshold})"
               if stage.promotion_return_threshold is not None else " (fixed-step stage)"),
            f"  ready to promote: {self.should_promote()}",
        ]
        return "\n".join(lines)
