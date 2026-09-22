"""
Offline unit tests for curriculum/manager.py's promotion logic. Does NOT
require TrackMania, TMRL networking, or any recorded reward file -- pure
state-machine testing against a temp state file.

Run via pytest (`pytest tests/ -v`) or directly:
    python tests/test_curriculum.py
"""

import tempfile
from dataclasses import replace
from pathlib import Path

from curriculum.manager import CurriculumManager
import curriculum.stages as stages_module
from curriculum.stages import CurriculumStage


def _patch_stages(monkeypatch, stage_list):
    """Point curriculum.manager's STAGES at a small, fast, test-only list."""
    monkeypatch.setattr(stages_module, "STAGES", stage_list)
    import curriculum.manager as manager_module
    monkeypatch.setattr(manager_module, "STAGES", stage_list)


def make_test_stages(tmp_path):
    return [
        CurriculumStage(
            name="stage_a",
            description="fixed-step + performance gate",
            map_file="a.Map.Gbx",
            min_env_steps=100,
            promotion_window=3,
            promotion_return_threshold=50.0,
        ),
        CurriculumStage(
            name="stage_b",
            description="fixed-step only (last stage)",
            map_file="b.Map.Gbx",
            min_env_steps=50,
            promotion_window=3,
            promotion_return_threshold=None,
        ),
    ]


def test_main(monkeypatch=None, tmp_path=None):
    print("=" * 60)
    print("CURRICULUM MANAGER TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_stages = make_test_stages(tmp_path)

        import curriculum.stages as stages_mod
        import curriculum.manager as manager_mod
        original_stages = stages_mod.STAGES
        stages_mod.STAGES = test_stages
        manager_mod.STAGES = test_stages

        try:
            state_path = tmp_path / "curriculum_state.json"
            manager = CurriculumManager(state_path=state_path)

            assert manager.stage_idx == 0
            assert manager.current_stage.name == "stage_a"
            print("[PASS] Fresh manager starts at stage 0")

            # --------------------------------------------------------------
            # Below min_env_steps (100): never promote, regardless of return.
            # promotion_window=3, so this also leaves the window unfilled.
            # --------------------------------------------------------------
            manager.record_round(episode_length_train=40, return_train=5.0)
            assert not manager.should_promote(), (
                "must not promote before min_env_steps is reached"
            )
            print("[PASS] No promotion before min_env_steps")

            # --------------------------------------------------------------
            # min_env_steps reached (80 < 100 still, one more call needed),
            # but return window not yet full (2/3 episodes recorded).
            # --------------------------------------------------------------
            manager.record_round(episode_length_train=40, return_train=5.0)
            assert not manager.should_promote(), (
                "must not promote before the return window has enough "
                "episodes, even once min_env_steps is satisfied"
            )
            print("[PASS] No promotion with an unfilled return window "
                  "(Section 10: never promote off too little data)")

            # --------------------------------------------------------------
            # Window now full (3/3) with low returns: below threshold (50).
            # --------------------------------------------------------------
            manager.record_round(episode_length_train=40, return_train=5.0)
            assert manager.env_steps_this_stage >= test_stages[0].min_env_steps
            assert len(manager.return_window) == test_stages[0].promotion_window
            assert not manager.should_promote(), (
                "must not promote while the moving-average return is "
                "below promotion_return_threshold"
            )
            print("[PASS] No promotion while average return is below threshold")

            # --------------------------------------------------------------
            # Push the moving average above threshold: now it should promote.
            # window is maxlen=3, so this evicts the oldest 5.0.
            # --------------------------------------------------------------
            manager.record_round(episode_length_train=10, return_train=100.0)
            manager.record_round(episode_length_train=10, return_train=100.0)
            avg = sum(manager.return_window) / len(manager.return_window)
            assert avg >= test_stages[0].promotion_return_threshold, (
                f"test setup invalid: avg={avg} should exceed threshold"
            )
            assert manager.should_promote()
            print("[PASS] Promotes once min_env_steps AND the moving-average "
                  "return threshold are both satisfied")

            # --------------------------------------------------------------
            # Promote and verify state resets for the new stage.
            # --------------------------------------------------------------
            next_stage = manager.promote()
            assert next_stage.name == "stage_b"
            assert manager.stage_idx == 1
            assert manager.env_steps_this_stage == 0
            assert len(manager.return_window) == 0
            assert len(manager.promotion_log) == 1
            assert manager.promotion_log[0]["stage"] == "stage_a"
            print("[PASS] promote() advances stage and resets progress "
                  "for the new stage, logging the completed one")

            # --------------------------------------------------------------
            # Last stage with no performance gate: fixed-step promotion only.
            # --------------------------------------------------------------
            assert manager.is_last_stage
            manager.record_round(episode_length_train=50, return_train=0.0)
            assert manager.should_promote(), (
                "fixed-step-only stage (promotion_return_threshold=None) "
                "must promote once min_env_steps is reached, regardless of return"
            )
            print("[PASS] Fixed-step-only stage promotes on step count alone")

            try:
                manager.promote()
                raise AssertionError("promote() on the last stage should raise")
            except RuntimeError:
                pass
            print("[PASS] promote() refuses to advance past the last stage")

            # --------------------------------------------------------------
            # Persistence: a fresh manager reloads the same state.
            # --------------------------------------------------------------
            reloaded = CurriculumManager(state_path=state_path)
            assert reloaded.stage_idx == manager.stage_idx
            assert reloaded.env_steps_this_stage == manager.env_steps_this_stage
            assert len(reloaded.promotion_log) == 1
            print("[PASS] Curriculum state persists across manager instances "
                  "(survives a process restart)")

            # --------------------------------------------------------------
            # Manual override.
            # --------------------------------------------------------------
            reloaded.jump_to_stage("stage_a")
            assert reloaded.stage_idx == 0
            assert reloaded.env_steps_this_stage == 0
            print("[PASS] Manual jump_to_stage() override works "
                  "(Section 10: manual promotion must be supported)")

        finally:
            stages_mod.STAGES = original_stages
            manager_mod.STAGES = original_stages

    print()
    print("=" * 60)
    print("CURRICULUM MANAGER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_main()
