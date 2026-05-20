# DEV LOG

## 2026-05-18

### Done

- Created initial project structure for perception, tracking, policy, robot, configs, data, results, and docs
- Added mock dataset generator for `WAIT`, `ASK`, and `CLEANUP_CANDIDATE`
- Added expert high-level policy
- Added training script for RandomForest and MLP baselines
- Added uncertainty-aware ASK inference
- Added mock robot flow and `main_demo.py`
- Added ROI-based local liquid verification mock path
- Added README, CHANGELOG, and Git-friendly project organization

### Test

- `python policy/generate_mock_dataset.py --out data/processed/dataset_decision.csv --n 1000`
- `python policy/train_policy.py --data data/processed/dataset_decision.csv --model results/decision_model.joblib --algo rf`
- `python main_demo.py --model results/decision_model.joblib --mock --mock-responses y`

### Result

- Dataset generated successfully
- Training script executed successfully
- Validation accuracy: `1.0000` on the synthetic mock split
- Mock demo completed with `WAIT`, `ASK`, `SPILL_SAFE_CLEAR`, and `CLEAR` branches

### Issue

- `mediapipe` compatibility may vary depending on the local Python version
- Current accuracy is from synthetic expert-labeled data, so it should not be treated as a real-world metric

### Next

- Improve `.gitignore` and artifact tracking policy for portfolio use
- Add `results/evaluation_summary.csv`
- Connect real camera inputs after the mock pipeline is stable

### Safety Note

- The decision policy is intentionally conservative
- `WAIT` remains the highest-priority safety action
- Low-confidence non-WAIT predictions should be overridden to `ASK`
- Reducing `wrong_cleanup_rate` is more important than maximizing raw accuracy

## 2026-05-18 - v0.2 Preparation

### Done

- Added `tools/capture_cup_dataset.py` for USB webcam-based cup image collection
- Added keyboard-driven capture flow for green, red, and blue cup datasets
- Added automatic sequential naming and live count overlay
- Added camera index guidance for external USB webcams on Windows
- Updated README with cup dataset capture instructions and collection targets

### Test

- `python -m py_compile tools/capture_cup_dataset.py`
- Hardware validation with a real USB webcam should be done locally:
- `python tools/capture_cup_dataset.py --camera-index 0`
- `python tools/capture_cup_dataset.py --camera-index 1`
- `python tools/capture_cup_dataset.py --camera-index 2`

### Result

- Capture tool implemented for `data/raw/green`, `data/raw/red`, and `data/raw/blue`
- The script is ready for local webcam-based dataset collection

### Issue

- Real webcam access cannot be verified inside this environment
- Camera index may differ depending on whether the built-in laptop camera is also active

### Next

- Collect 100 images each for green, red, and blue cups
- Use captured images to tune HSV thresholds in `perception/detect_cups.py`
- Record representative screenshots in `docs/demo_screenshots/` if useful for the portfolio

### Safety Note

- The project continues to prefer conservative actions over aggressive cleanup
- The final cleanup decision should still wait for local liquid verification even after a global `CLEANUP_CANDIDATE` prediction

## 2026-05-19 - v0.2 Global Cup Detection

### Done

- Updated `perception/detect_cups.py` for real USB webcam execution
- Added `--camera-index`, `--backend`, `--width`, `--height`, and `--show-mask-debug` options
- Added HSV mask cleanup with blur plus morphological open/close
- Added contour scoring to reject small or tape-like regions more aggressively
- Added bbox, center point, cup ID, color, and area visualization
- Added Windows-friendly webcam opening flow for `camera-index 1 + backend dshow`

### Test

- `python -m py_compile perception/detect_cups.py`
- `python perception/detect_cups.py --help`
- Runtime target:
- `python perception/detect_cups.py --config configs/config.yaml --camera-index 1 --backend dshow`
- Runtime debug target:
- `python perception/detect_cups.py --config configs/config.yaml --camera-index 1 --backend dshow --show-mask-debug`

### Result

- Cup detection module is ready for live USB webcam validation
- Real HSV threshold tuning still needs to be done on the actual camera feed

### Issue

- Final green/red/blue HSV thresholds have not been validated in this environment
- Blue tape on the table may still require threshold or area-filter refinement during live testing

### Next

- Run live webcam detection on the collected cup scenes
- Tune `configs/config.yaml` HSV ranges against real frames
- Validate stable `cup_id`, `bbox`, and `center_pixel` output under clutter and partial occlusion

## 2026-05-19 - v0.3 Hand and User Tracking

### Done

- Added backend-aware webcam opening for `detect_hand.py` and `detect_user_presence.py`
- Connected cup detection, hand detection, user presence, and trackers inside `main_demo.py`
- Added `--debug-perception` mode for live visualization
- Updated `InteractionTracker` to compute hand distance from real pixel-space cup centers
- Added debug overlay for hand center, user presence, hand-cup distance, touch count, and last touched time

### Test

- `python -m py_compile main_demo.py perception/detect_hand.py perception/detect_user_presence.py tracking/interaction_tracker.py tracking/user_presence_tracker.py`
- `python main_demo.py --help`
- `python perception/detect_hand.py --help`
- `python perception/detect_user_presence.py --help`
- Runtime target:
- `python main_demo.py --camera-index 1 --backend dshow --debug-perception`

### Result

- The main entrypoint is now ready for live perception debugging with cups, hand, and user presence
- Real runtime validation should be performed on the USB webcam feed

### Issue

- MediaPipe runtime behavior still depends on local installation compatibility
- User presence fallback remains simple when face detection is unavailable
- Current base environment uses Python 3.13 with mediapipe 0.10.35, which does not provide `mp.solutions`
- Existing `MediaPipe Hands` and `Face Detection` code therefore cannot run in the current base environment

### Next

- Run the integrated perception debug view on the real table scene
- Verify stable hand center and user presence under clutter and robot-arm occlusion
- Refine tracker thresholds if touch timing is too sensitive or too slow

### Environment Fix

- Recommended environment for `v0.3`:
- `Python 3.12`
- `mediapipe==0.10.13`
- Verification command:
- `python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"`
- Expected result:
- `True` for `hasattr(mp, 'solutions')`

### Runtime Validation Update

- Created and activated `cup_cleanup_mp312` conda environment with Python 3.12.
- Installed `mediapipe==0.10.13`.
- Verified `mp.solutions` availability:
- `mediapipe version: 0.10.13`
- `hasattr(mp, "solutions") == True`
- Successfully ran:
- `python perception/detect_hand.py --camera-index 1 --backend dshow`
- `python main_demo.py --camera-index 1 --backend dshow --debug-perception`

### Result Update

- Hand center detection works on the real USB webcam feed.
- User presence detection works in the integrated perception debug view.
- `user_present=1`, `hand_visible=1`, `user_absent_time=0.0s` were confirmed.
- Hand-cup distance, `touch_count`, and `last_touched_time` update correctly in the overlay.

### Timing Update

- Added explicit ASK delay handling so a used cup does not trigger `ASK` immediately after release.
- Added a longer delay for never-active cups so they remain `IDLE` while the user is still present unless they stay untouched for a long time.
- Tightened cleanup gating so stationary time alone does not promote a cup to `CLEANUP_CANDIDATE` while the user is still present.

## 2026-05-19 - v0.4 Real Interaction Dataset

### Done

- Added `data_collection/record_interaction_dataset.py`
- Connected cup detection, hand detection, user presence, and interaction tracking into a CSV recorder
- Added interval-based dataset saving to reduce duplicate samples
- Added expert-rule labels during recording for behavior cloning bootstrap
- Added sample count and recorder status overlay
- Protected `data/processed/dataset_decision.csv` from accidental overwrite by real interaction recordings
- Planned scene-specific real interaction files for green, red, blue, and clutter captures
- Added dataset merge and analysis utilities for recorded interaction CSV files

### Test

- `python -m py_compile data_collection/record_interaction_dataset.py`
- `python -m py_compile data_collection/merge_interaction_datasets.py`
- `python -m py_compile data_collection/analyze_interaction_dataset.py`
- Runtime target:
- `python data_collection/record_interaction_dataset.py --camera-index 1 --backend dshow --out data/processed/interaction_green.csv`
- `python data_collection/analyze_interaction_dataset.py --data data/processed/interaction_green.csv`
- `python data_collection/merge_interaction_datasets.py --out data/processed/interaction_dataset_all.csv`

### Result

- Recorder is ready to save real perception-derived interaction features to CSV
- Each saved scene can emit one row per visible cup with an expert-rule label
- Scene-specific CSV workflow is ready for green, red, blue, and clutter collection

### Issue

- Runtime hand and user recording still depends on the correct Python 3.12 plus MediaPipe Solutions environment
- Recorded labels are rule-based bootstrap labels, not human-verified annotations

### Next

- Record real interaction scenes on the robot table into `interaction_green/red/blue/clutter.csv`
- Inspect each scene CSV for feature balance and label distribution
- Merge scene files into `interaction_dataset_all.csv`
- Use the merged real dataset for behavior cloning retraining

### Collection Update

- Completed real interaction dataset collection:
- `data/processed/interaction_green.csv`
- `data/processed/interaction_red.csv`
- `data/processed/interaction_blue.csv`
- `data/processed/interaction_clutter.csv`

### Merge Update

- Merged interaction datasets into `data/processed/interaction_dataset_all.csv`
- Total rows: `1547`
- Source file row counts:
- `interaction_green.csv`: `305`
- `interaction_red.csv`: `266`
- `interaction_blue.csv`: `193`
- `interaction_clutter.csv`: `783`
- Label distribution:
- `ASK`: `717`
- `WAIT`: `558`
- `CLEANUP_CANDIDATE`: `272`

### Analysis Update

- `hand_distance` contains placeholder values up to `999.0`
- `last_touched_time` contains placeholder values above `1000.0`
- `user_absent_time` maximum is `10.630`
- Placeholder clipping is required before training to avoid shortcut learning from sentinel values

### Training Update

- Trained real-data policy with:
- `python policy/train_policy.py --data data/processed/interaction_dataset_all.csv --model results/decision_model_real.joblib --algo rf`
- Saved real-data outputs:
- `results/decision_model_real.joblib`
- `results/classification_report_real.txt`
- `results/confusion_matrix_real.png`
- `results/evaluation_summary_real.csv`
- Validation metrics:
- `accuracy=1.0000`
- `wrong_cleanup_rate=0.0000`
- `ask_override_count=0`
- `cleanup_candidate_precision=1.0000`
- `WAIT recall=1.0000`

### Next Update

- Connect the real trained policy model to live inference paths in `main_demo.py`
- Compare mock-trained and real-trained policy behavior
- Check whether perfect validation metrics reflect true generalization or rule-label shortcut learning

## 2026-05-19 - v0.5 Live Policy Inference

### Done

- Added trajectory-aware active-cup arbitration to the live-policy path in `main_demo.py`
- Updated `InteractionTracker` to expose `active_cup_id`, `is_active_cup`, `time_near_cup`, `time_since_release`, `release_count`, `cup_motion_distance`, `stationary_time`, `was_moved`, and `used_cup_candidate`
- Added `IDLE` suppression for untouched cups when the user is still present
- Updated `record_interaction_dataset.py` to save trajectory-aware interaction features
- Updated `expert_high_level_policy` to support trajectory-aware `WAIT`, `ASK`, `IDLE`, and `CLEANUP_CANDIDATE`

### Test

- `python -m py_compile main_demo.py tracking/interaction_tracker.py policy/infer_policy.py policy/expert_policy.py data_collection/record_interaction_dataset.py`
- Runtime target:
- `python main_demo.py --camera-index 1 --backend dshow --live-policy --model results/decision_model_real.joblib`

### Result

- Live inference now uses a separate arbitration layer instead of directly overlaying independent cup-wise model predictions
- `ASK` is restricted to cups with real trajectory evidence
- Unused cups can remain `IDLE` even in multi-cup scenes where one other cup is active
- The overlay now exposes `ACTIVE`, `USED`, `time_near_cup`, `stationary_time`, and release-related state for debugging
- The current direction is shifting from rule-heavy arbitration toward model-first trajectory-aware behavior cloning with a lightweight safety guard

### Issue

- Existing real interaction CSV files were collected before the new trajectory-aware labels and may need supplemental `IDLE` and stronger release/motion scenes
- The current trained model still uses the older feature set, so arbitration remains the main live-behavior safeguard until a trajectory-aware retraining pass is completed

### Next

- Collect supplemental `IDLE` scenes where the user is present but no cup is touched
- Collect one-active-cup multi-cup scenes so untouched cups remain `IDLE`
- Collect moved-and-released scenes and long stationary abandonment scenes
- Merge the supplemental data and retrain the policy with the expanded feature set and `IDLE` label

### Data Augmentation Plan

- Planned supplemental files:
- `data/processed/interaction_idle.csv`
- `data/processed/interaction_red_active.csv`
- `data/processed/interaction_blue_active.csv`
- `data/processed/interaction_green_active.csv`
- `data/processed/interaction_abandoned.csv`
- Planned merged output:
- `data/processed/interaction_dataset_trajectory_all.csv`

### Trajectory Retraining Update

- Merged trajectory-aware supplemental datasets into `data/processed/interaction_dataset_trajectory_all.csv`
- Total rows: `2346`
- Label distribution:
- `IDLE=1336`
- `ASK=527`
- `WAIT=360`
- `CLEANUP_CANDIDATE=123`
- Trained trajectory-aware policy with:
- `python policy/train_policy.py --data data/processed/interaction_dataset_trajectory_all.csv --model results/decision_model_trajectory.joblib --algo rf`
- Saved outputs:
- `results/decision_model_trajectory.joblib`
- `results/classification_report_trajectory.txt`
- `results/confusion_matrix_trajectory.png`
- `results/evaluation_summary_trajectory.csv`
- Validation metrics:
- `accuracy=1.0000`
- `wrong_cleanup_rate=0.0000`
- `unnecessary_ask_rate=0.0000`
- `idle_precision=1.0000`
- `cleanup_candidate_precision=1.0000`
- `WAIT recall=1.0000`

### Live Evaluation Mode Update

- Added three live policy modes in `main_demo.py`:
- `model_only`
- `safety_guard`
- `arbitration`
- Added live evaluation CSV logging with `--log-live-eval`
- The intended comparison is:
- `model_only` for pure Behavior Cloning inspection
- `safety_guard` for model-first deployment behavior
- `arbitration` for rule-heavier demo stabilization

### Social State Runtime Update

- Added a runtime `OBSERVE` state for post-release waiting before asking
- Added reuse detection so `OBSERVE` or `ASK` can be cancelled when the user grabs the cup again
- Added a soft transition state machine in the live arbitration path
- Preserved `model_only` as the pure BC inspection mode
- Deferred multi-user ownership tracking until dedicated data collection and validation are ready
- Added `ASK_PENDING` so the same cup is not asked repeatedly every frame
- Added `ASK_COOLDOWN` so a rejected or timed-out cup is not immediately re-asked
- Added `READY_TO_CLEAR` so accepted cups can be cleanly handed off to the later local-liquid-verification stage
- Added `y/n` keyboard response handling in live state-machine evaluation

### ASK Priority And Liquid Handoff Update

- Added single-arm ASK priority arbitration so only one cup can become `ASK` or `ASK_PENDING` at a time
- Added `ask_reason`, `ask_priority`, candidate rank, and selection logging for explainable ASK behavior
- Added heuristic `drink_count`, `estimated_consumed_ml`, and `estimated_drink_progress` from hand-cup trajectory
- Tightened the heuristic so a single pick-and-place does not immediately count as a drink or trigger ASK
- ASK after use now requires repeated sip-like events, with the current default milestones set to `drink_count` `5`, `8`, and `10`
- Added release hysteresis, debounce, and cooldown so `release_count` is less sensitive to hand-distance flicker
- Added face-proximity gating so sip-like events are counted only when the cup is brought near the user's face before release
- Added post-accept exclusion so cups that already received `yes` are removed from repeated ASK arbitration and stay only in the downstream cleanup flow
- Kept `model_only` untouched as pure Behavior Cloning inspection while applying the new arbitration only in `state_machine` and `arbitration`
- Changed live cleanup flow so global webcam inference ends at `NEEDS_LIQUID_CHECK` rather than clearing directly
- Recorded the design split explicitly:
- global webcam = cleanup candidate selection
- local or gripper camera = EMPTY/NON_EMPTY verification before `READY_TO_CLEAR` or `SPILL_SAFE_CLEAR`

## Template

Copy this section for future work days.

### Done

- 

### Test

- 

### Result

- 

### Issue

- 

### Next

- 
