# Curriculum learning

Implements Section 10/11 of the research plan: train the same TD3 policy
progressively across tracks of increasing difficulty, from a track you drive
and record yourself through to a harder one, carrying the same weights
forward at each step.

## What's automated vs. what needs you

**Automated** (`curriculum/` package + `train_td3.py`'s `curriculum-*` modes):
- Tracking how many environment steps and what returns the policy has
  achieved on the current stage
- Deciding when a stage's promotion criteria are met (minimum steps trained,
  AND a moving-average return over the last N episodes above a threshold --
  never off a single lucky episode)
- Swapping in the correct recorded reward trajectory for whichever stage is
  active, so training never accidentally uses the wrong track's reference
  path
- Keeping the SAME actor/critic checkpoint across every stage transition
  (Section 11) -- promotion never resets the network

**Cannot be automated** (verified: no such API in installed `tmrl==0.7.1` or
its OpenPlanet plugins):
- **Loading a different map in TrackMania.** When a stage completes, you
  must manually load the next stage's map yourself before resuming training.
- **Recording each new track's reward trajectory.** The very first time a
  track is used, you drive it once (this is exactly the "I drive and save
  reward, it mirrors my path" mechanism you described -- confirmed by
  reading TMRL's own `tmrl/tools/record.py`: it records your car's raw
  position trajectory while you drive, nothing more).

## How a stage works, end to end

1. Load that stage's map in TrackMania (see `curriculum/stages.py` for the
   map file each stage expects, or run `python train_td3.py curriculum-status`).
2. If the stage's reward trajectory hasn't been recorded yet:
   ```bash
   python train_td3.py record-track-reward <stage_name>
   ```
   Press `e` in-game to start recording, drive the track, `q` or the finish
   line to stop and save. One-time per track.
3. Start training (3 terminals, same as any other TD3 run):
   ```bash
   python train_td3.py server
   python train_td3.py curriculum-trainer
   python train_td3.py curriculum-worker
   ```
   `curriculum-trainer`/`curriculum-worker` automatically activate the
   current stage's reward file on startup and print the stage's progress.
4. (Optional but recommended) In a 4th terminal, watch for promotion and get
   a clear "what to do next" banner instead of checking manually:
   ```bash
   python train_td3.py curriculum-watch --trainer-log <path to terminal 2's output>
   ```
   If you're running the trainer directly in a visible terminal rather than
   through a redirected/background process, just watch its own printed
   `[curriculum] stage=... steps=.../N return=... window=.../M` lines
   yourself instead -- `curriculum-watch` is for when the trainer's output
   isn't already something you're staring at.
5. When promotion criteria are met, `curriculum-watch` prints a STAGE
   COMPLETE banner with exact next steps and exits. Stop the trainer/worker
   (Ctrl+C), load the next stage's map, and go back to step 2/3 for the next
   stage. The SAME `trainer.tmod`/`trainer.tcpt` checkpoint is reused
   automatically -- you are not starting over.

Check current progress any time with:
```bash
python train_td3.py curriculum-status
```

## Current stages (`curriculum/stages.py`)

| # | name | map | status |
|---|---|---|---|
| 0 | `tmrl_test_baseline` | `tmrl-test.Map.Gbx` | ready (bootstrapped from the existing official baseline trajectory) |
| 1 | `tmrl_train_harder` | `tmrl-train.Map.Gbx` | needs its reward trajectory recorded |

Only TMRL's two stock maps are configured so far. Per Section 10 ("the exact
tracks must be determined from the actual available TrackMania maps"), adding
more/harder stages (e.g. a custom map built in the track editor) means adding
a `CurriculumStage` entry to `curriculum/stages.py` and recording its reward
trajectory the same way -- no other code changes needed.

## Generalization testing (Section 19)

Once the curriculum has run through all configured stages, evaluate the
resulting policy on a genuinely unseen track (one never trained on) using
the existing `trained-eval` mode:
```bash
python train_td3.py trained-eval --episodes 5 --max-steps 700
```
This uses whichever reward file is currently active in `TmrlData/reward/`,
so for a *quantitative* unseen-track score you'd still want to record that
track's trajectory too (same one-time step as any other track) -- for a
purely qualitative "does it drive around this new track reasonably" check,
no reward recording is needed at all, since you're just watching behavior.
Neither of these has been built as an automated eval pipeline yet; this is
the natural place to extend `trained-eval` further if/when you want it.

## Before starting curriculum training for real

`docs/known_issues.md` documents a diagnosed, unfixed gap: the current
policy has no learned recovery behavior when it gets stuck against a wall.
Curriculum training on top of this will very likely reproduce the same
failure on every stage's own obstacles, making it hard to tell "the
curriculum isn't helping" from "there's an unrelated recovery-behavior gap."
Recommend addressing one of that doc's candidate fixes before or alongside
starting curriculum stage 0's real training run.
