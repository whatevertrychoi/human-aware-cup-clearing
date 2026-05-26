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

## 2026-05-20 - v0.6 Social State Machine

### Done

- Added response-aware social state transitions on top of live policy inference
- Added `ASK_PENDING`, `ASK_COOLDOWN`, and `READY_TO_CLEAR` handling for live human-in-the-loop interaction
- Added keyboard `y/n` response flow so one ASK event can transition to acceptance or rejection states
- Added runtime state exclusion so cups accepted for cleanup are removed from repeated global-policy asking
- Added single-arm ASK arbitration so only one cup is asked at a time even when multiple cups are eligible
- Added `ask_priority`, `ask_reason`, candidate rank, and selected-for-ask logging for explainable ASK selection
- Added `NEEDS_LIQUID_CHECK` handoff so global webcam inference stops at candidate selection and local verification remains a downstream step
- Added `drink_count`, `estimated_consumed_ml`, `estimated_drink_progress`, and `last_drink_event_time` as heuristic social-use features
- Added release hysteresis, debounce, and cooldown to stabilize noisy `release_count` spikes from hand-distance flicker
- Added face-proximity gating so sip-like events are counted only when a held cup approaches the user face before release
- Changed ASK eligibility for used cups so single pick-and-place interactions remain in `OBSERVE` and repeated sip-like milestones trigger ASK
- Added ASK milestones at `drink_count` values `5`, `8`, and `10`
- Added overlay support for priority, ask reason, cooldown, readiness for local liquid check, handled cups, and face-aware drink estimates
- Added temporary local-liquid-check mock response handling in live `state_machine` mode so `y/n` can be used to simulate EMPTY vs NON_EMPTY after `NEEDS_LIQUID_CHECK`
- Added overlay/log output for liquid-check mock results so abandoned-cup testing can display whether the cup would be cleared or restored
- Added a ROS2 trigger bridge path for live `state_machine` / `arbitration` execution
- Added one-shot publish logic for:
  - `ASK_TRIGGER`
  - `CANCEL_ASK_TRIGGER`
  - `ROBOT_LIQUID_CHECK_TRIGGER`
- Added a bridge-side cup-id mapping layer so policy-side cup IDs can be translated to the robot YOLO label order without changing the internal policy logic
- Refined `CANCEL_ASK_TRIGGER` semantics so it is no longer emitted for every generic reuse event
- Tightened cancel publishing so `CANCEL_ASK_TRIGGER` is emitted only when:
  - the same cup previously emitted a real `ASK_TRIGGER`
  - that ASK session is still considered active on the bridge side
  - a later reuse event invalidates that active ask session
- Added `CANCEL_ROBOT_LIQUID_CHECK_TRIGGER` so the robot side can be told to stop a pending local liquid-check approach if:
  - a cup was previously selected for `ROBOT_LIQUID_CHECK_TRIGGER`
  - but later drops out of the active `NEEDS_LIQUID_CHECK` selection set
  - and the drop was not caused by a completed mock liquid-check result such as `EMPTY` or `NON_EMPTY`
- Kept the bridge as a transport-layer adapter rather than moving ROS2 publish logic into the policy state machine itself

### Test

- `python -m py_compile main_demo.py policy/state_machine.py tracking/interaction_tracker.py`
- `python -m py_compile main_demo.py integration/ros2_trigger_bridge.py`
- `python main_demo.py --camera-index 1 --backend dshow --live-policy --model results/decision_model_trajectory.joblib --policy-mode state_machine`
- `python main_demo.py --camera-index 1 --backend dshow --live-policy --model results/decision_model_trajectory.joblib --policy-mode state_machine --log-live-eval logs/live_policy_eval_state_machine_priority.csv`
- In abandoned-cup validation, waited for `NEEDS_LIQUID_CHECK` and used:
- `y` to simulate `EMPTY -> clear`
- `n` to simulate `NON_EMPTY -> restore`
- Verified trigger semantics on the ROS2 bridge side conceptually:
  - `ASK_TRIGGER` is only published on the one-shot ASK frame
  - `CANCEL_ASK_TRIGGER` is only published if that cup already published `ASK_TRIGGER`
  - `ROBOT_LIQUID_CHECK_TRIGGER` is only published once per active liquid-check selection interval
  - `CANCEL_ROBOT_LIQUID_CHECK_TRIGGER` is emitted if a previously selected liquid-check cup leaves the active set before completion
- Checked state-machine logs for:
- `OBSERVE -> ASK -> ASK_PENDING`
- `reuse_detected`
- `ask_priority`
- `selected_for_ask`
- `verification_required`
- `exclude_from_policy`

### Result

- The live policy now behaves more like a service robot interaction manager than a framewise classifier
- ASK is generated as a single event and then transitions to `ASK_PENDING` instead of repeating every frame
- Reuse detection works as a cancellation path from `OBSERVE`, `ASK_PENDING`, and cooldown states back to `WAIT`
- Global webcam policy now distinguishes between:
- cups that should be left alone
- cups that can be asked about
- cups that should move to local liquid verification
- Cups that already received a user `yes` response are excluded from repeated ASK arbitration and remain only in the downstream cleanup path
- The current design clearly separates:
- global webcam = social timing and cleanup candidacy
- local or gripper camera = EMPTY/NON_EMPTY verification before physical cleanup
- The abandoned-cup path can now be checked interactively in the live window before real robot/controller integration by using `y/n` at `NEEDS_LIQUID_CHECK`
- The live policy can now publish ROS2-friendly trigger events from the final state-machine outputs without modifying the underlying BC model or tracker logic
- Trigger semantics are now more consistent with downstream robot execution:
  - `ASK_TRIGGER` means "start ask flow for this cup"
  - `CANCEL_ASK_TRIGGER` means "abort a previously started ask flow for this same cup"
  - `ROBOT_LIQUID_CHECK_TRIGGER` means "start local inspection for this abandoned-cup target"
  - `CANCEL_ROBOT_LIQUID_CHECK_TRIGGER` means "abort that pending local inspection because the policy no longer wants it"
- The bridge now tracks active ASK sessions and active liquid-check sessions explicitly so transport-level triggers better match the real intent of the state-machine outputs

### Issue

- `hand_distance` still falls back to `999.0` when the hand is not visible, which can make distance-based ASK priority less informative in some frames
- The drink-count heuristic is still an estimate and depends on face visibility plus camera geometry
- Because `OBSERVE`, `ASK_PENDING`, `READY_TO_CLEAR`, and `NEEDS_LIQUID_CHECK` are runtime state-machine states, they are not yet learned directly by the BC model
- Local liquid verification in the live path is still a mock keyboard-driven branch, not a real robot/gripper camera callback
- Full ROS2 runtime validation still needs to be performed on the Ubuntu robot laptop with `rclpy` and the actual subscriber nodes
- `CANCEL_ROBOT_LIQUID_CHECK_TRIGGER` is currently inferred at the bridge layer from liquid-check set membership changes rather than from a dedicated explicit state-machine cancellation state
- The current bridge relies on in-process memory of active ASK/liquid-check sessions, so process restarts clear that transient trigger history

### Next

- Refine person-distance estimation so ASK priority is less dependent on hand visibility alone
- Consider adding face-distance or owner-distance features more directly into priority selection
- If needed, collect new social-interaction datasets for `OBSERVE`, reuse, and repeated sip-like usage patterns so a later model can learn more of the social timing directly
- If robot-side integration requires stronger formal guarantees, consider adding explicit policy-level cancellation states for abandoned-cup local-check withdrawal instead of only bridge-level trigger inference

## Template

Copy this section for future work days.

## 2026-05-26 - ASK Voice Handoff And Stricter Cup Detection

### Done

- Split ROS2 trigger transport so social ASK events no longer share the same publish topic as robot execution events.
- Updated `configs/config.yaml`:
  - `ros2_trigger.ask_topic: /cup_cleanup/ask_trigger`
  - `ros2_trigger.robot_topic: /cup_cleanup/trigger`
- Updated `integration/ros2_trigger_bridge.py` so:
  - `ASK_TRIGGER`, `CANCEL_ASK_TRIGGER` publish to the ask topic
  - `ROBOT_LIQUID_CHECK_TRIGGER`, `CANCEL_ROBOT_LIQUID_CHECK_TRIGGER` stay on the robot topic
- Verified the intended external handoff architecture:
  - `cup_cleanup` decides ASK
  - voice node receives ASK
  - user `yes` causes downstream `CUP_PICK_TRIGGER`
  - robot then performs detect -> grasp -> place
- Reviewed the current robot-side and policy-side priority split for cleanup:
  - policy selects liquid-check candidates by abandonment-style semantics
  - robot executes the cleanup set by nearest observed distance
- Tightened OpenCV global cup detection in `perception/detect_cups.py`:
  - added stronger contour scoring inputs
  - added `max_area`, `min_box_size`, `min_fill_ratio`, `min_solidity`, `min_circularity`
  - added `min/max_aspect_ratio`, `max_bbox_area_ratio`, `min_score`
  - added per-color `detector_overrides` support
- Tightened `configs/config.yaml` cup HSV and contour thresholds globally, then relaxed green-only thresholds after green under-detection appeared in testing.

### Test

- `python -m py_compile c:\Users\minseok\Desktop\cup_cleanup\perception\detect_cups.py`
- `python -c "from project_utils import load_config; cfg=load_config(r'c:\Users\minseok\Desktop\cup_cleanup\configs\config.yaml'); print(cfg['cup_detection']['colors']['green'])"`
- Static code review of:
  - `integration/ros2_trigger_bridge.py`
  - `policy/state_machine.py`
  - downstream voice / robot coordination paths
- Live runtime confirmation from the operator:
  - ASK changed to voice-side trigger delivery
  - voice response reached the robot-side trigger path

### Result

- ASK is no longer sent directly to the robot-side trigger topic from `cup_cleanup`.
- The live system now supports the intended staged flow:
  - `ASK_TRIGGER` to voice topic
  - voice confirmation
  - robot execution trigger afterward
- Global cup detection is more conservative against false positives such as clothing-colored regions and face-area color blobs.
- Green cup detection remains available through green-specific detector overrides instead of weakening the whole detector globally.

### Issue

- This repository does not contain the external voice-node source of record or the downstream robot repository, so only the `cup_cleanup` side of the transport split is versioned here.
- Current working tree still contains older unrelated modified files, so the branch should not be committed wholesale without selecting task-relevant files.
- The robot-side ASK completion semantics should still be rechecked end-to-end when using `CUP_PICK_TRIGGER`, since success/failure feedback ownership moved away from the old direct ASK path.

### Next

- Commit only the `cup_cleanup` files that belong to this ASK-to-voice split and detector tuning.
- Re-run live policy with `--policy-mode state_machine` when validating ASK transport, since `safety_guard` mode does not export ROS2 ASK triggers.
- If blue clothing or face regions still leak through, add table ROI clipping before contour selection.

## 2026-05-22 - Cleanup Session Alignment

### Done

- Reviewed the current robot-side cleanup flow after the first ROS2 trigger bridge integration.
- Confirmed that the downstream robot stack now interprets `ROBOT_LIQUID_CHECK_TRIGGER` as a cleanup-session start signal rather than a one-cup local-inspection request.
- Updated `integration/ros2_trigger_bridge.py` to align with that robot-side meaning.
- Added bridge-side cleanup session tracking so:
  - one cleanup session starts when the active liquid-check set becomes non-empty
  - brief liquid-check set flicker does not immediately cancel the whole session
  - completed liquid-check results such as `EMPTY` or `NON_EMPTY` do not emit unnecessary cancellation events
- Extended ASK bridge behavior so the currently active ASK cup now emits `CANCEL_ASK_TRIGGER` when it leaves the ASK set entirely, which lets the robot return to observe pose if the policy falls back to `WAIT`.
- Fixed an integration bug where the bridge could cancel immediately after `ASK_TRIGGER`; `ASK_PENDING` and `READY_TO_CLEAR` are now treated as still-active ASK-session states.
- Added integration notes documenting the current handoff boundary between:
  - `cup_cleanup`
  - `pick_and_place_voice_cup`

### Test

- Static syntax validation of `integration/ros2_trigger_bridge.py`
- End-to-end log review against the robot-side cleanup session behavior

### Result

- Bridge semantics are now closer to the real robot integration:
  - policy side still selects liquid-check candidates
  - bridge exports a cleanup-session start or stop signal
  - robot side owns nearest-cup ranking and iterative cleanup execution

### Issue

- Policy-side `selected_for_liquid_check` is still conceptually a per-cup selection even though the bridge now exports a cleanup-session interpretation for robot integration.
- Full end-to-end integration still depends on robot-side motion reliability and conservative cleanup detection quality.

### Next

- Continue annotating core execution files for easier integration debugging.
- Add the next integration stage where the robot requests final user confirmation before descending to grasp.

## 2026-05-22 - ASK Rearm Delay

### Done

- Added `ASK_REARM_DELAY_SEC = 1.5` in `integration/ros2_trigger_bridge.py`.
- Updated pending ASK draining so a newly pending ASK is not published in the same frame immediately after the previous ASK session is cancelled or cleared.
- Added `ASK_STATE_CLEAR_GRACE_SEC` as a long watchdog fallback so an active ASK session is no longer cancelled immediately when policy outputs briefly stop reporting ASK-related state for that cup, and later extended it to `600.0` once robot feedback became the primary ASK session terminator.
- Added robot-feedback-driven ASK session clearing on `/cup_cleanup/robot_feedback`, so the bridge now waits for robot-side `ASK_ACTION_FINISHED` feedback before releasing the active ASK session in the normal path.
- Extended the policy-side ASK clear grace into a long watchdog fallback so short `IDLE` or `WAIT` flicker no longer steals ASK ownership back from the robot mid-grasp.
- Added a bridge-side prediction latch used by `main_demo.py` so the policy overlay/log stream keeps the active robot-owned cup in `ASK` while the robot has not yet reported completion, cancellation, or failure.
- Removed the temporary keyboard `y/n` live-loop hooks from `main_demo.py` and updated overlay text so ASK / ASK_PENDING now describe voice confirmation rather than keypress confirmation.

### Reason

- Runtime logs showed that when one ASK session ended, the next pending ASK could be published again too quickly, which made the robot side feel like ASK events were overlapping even though they were serialized.
- Runtime logs also showed ASK sessions disappearing on their own a few seconds after publication, so the bridge now holds the active ASK session through short state flicker and only silently releases it after a grace period unless explicit reuse cancellation occurs.
- The latest runtime issue suggested that ASK state was still aging out from policy-side transitions before the robot finished its pick path, so ASK ownership is now tied to robot completion/cancel/failure feedback instead of transient policy state alone.

- 

## 2026-05-23 - ROS2 Ask/Cleanup Integration and Coordinate Debugging

### Done

- Added robot-feedback-driven closure for `ASK_PENDING` so the policy-side ask state no longer disappears after a fixed timeout while the robot is still executing.
- Added `apply_robot_ask_feedback(...)` handling in `policy/state_machine.py` so robot-side `completed`, `aborted`, and `cancelled` results can update cup state directly.
- Updated `integration/ros2_trigger_bridge.py` to:
  - consume `ASK_ACTION_FINISHED`
  - expose drained robot feedback to the live app
  - include cleanup target lists in `ROBOT_LIQUID_CHECK_TRIGGER`
  - preserve active ask / liquid-check session tracking for cancel semantics
- Updated `main_demo.py` so drained robot feedback is applied during the live loop.
- Set `configs/config.yaml`:
  - `policy.ask_pending_timeout: 0.0`
  - `ros2_trigger.enabled: true`
- Refined robot-side scenario routing in `robot_control.py`:
  - `ASK_TRIGGER` and `CUP_PICK_TRIGGER` use single-target flow
  - `ROBOT_LIQUID_CHECK_TRIGGER` uses cleanup-session flow
  - cancel / complete / abort return to `JHOME_POS`
  - trigger start uses observe pose
- Added cleanup-session target freezing:
  - the bridge now sends cleanup robot cup ids
  - the robot stores a fixed remaining-target set for the session
  - successfully placed cleanup targets are removed from that set
- Preserved deferred-trigger behavior so busy robot actions queue later work instead of dropping it.
- Reworked home / observe behavior:
  - trigger start goes to observe pose
  - `completed`, `aborted`, and `cancelled` actions return to `JHOME_POS = [0, -30, 90, 0, 90, 0]`
  - cleanup no-candidate and cleanup-failure exits also return home
- Investigated a major observe-pose mismatch:
  - configured `OBSERVE_POSX` remained `[562.779, 63.678, 593.504, 2.918, 152.473, -85.667]`
  - measured pose after `OBSERVE_POSJ` repeatedly landed near `[670.95, 68.33, 387.94, 2.924, 152.483, -85.678]`
  - exact observe-pose verification originally caused false aborts
  - observe entry now proceeds even when the measured TCP does not match the configured reference
- Explored several coordinate-correction strategies and then backed away from them when real motion did not match the corrected numbers:
  - observe translation correction
  - TCP command correction
  - multi-stage refinement with repeated strict single-target detection
- Restored the single-target coordinate pipeline toward the original proven behavior:
  - `ASK_TRIGGER` now uses first single-target detection and direct grasp path
  - multi-stage refinement is bypassed in the main single-target path
  - cleanup still uses snapshot selection, then re-detects only the chosen target once before grasp
- Simplified single-target YOLO selection in `object_detection/yolo.py`:
  - one current frame
  - one target label
  - highest-confidence matching detection
- Restored direct grasp behavior in `robot_control.py`:
  - move directly to detected target pose
  - close gripper
  - lift
  - place
  - return home
- Debugged depth-source handling in `object_detection/realsense.py`:
  - aligned depth only caused the service to stall on systems where aligned depth was not consistently available
  - restored aligned-depth preference with raw-depth fallback
  - kept separate color and depth intrinsics
- Fixed a service crash in `object_detection/detection.py`:
  - `_get_depth()` signature mismatch caused `TypeError: ... takes 3 positional arguments but 4 were given`
  - updated depth helper call path to pass frame coordinates consistently
- Identified the biggest raw-depth coordinate bug of the day:
  - depth was being sampled from raw-depth mapped coordinates
  - but camera 3D conversion still used the original color-frame center pixel
  - this caused large x/y errors even when depth values looked reasonable
- Fixed `_get_depth_at_target(...)` and `_compute_coords_from_detection(...)` so 3D camera coordinates are now computed from the actual pixel where the depth sample was taken.
- Preserved multi-target cleanup snapshot behavior while separating it from single-target `ASK` behavior:
  - `ASK` stays original-style and simple
  - cleanup keeps snapshot candidate discovery and nearest-target selection

### Test

- `python -m py_compile C:\Users\minseok\Desktop\cup_cleanup\main_demo.py`
- `python -m py_compile C:\Users\minseok\Desktop\cup_cleanup\policy\state_machine.py`
- `python -m py_compile C:\Users\minseok\Desktop\cup_cleanup\integration\ros2_trigger_bridge.py`
- `python -m py_compile C:\Users\minseok\Downloads\pick_and_place_voice_cup\pick_and_place_voice\robot_control\robot_control.py`
- `python -m py_compile C:\Users\minseok\Downloads\pick_and_place_voice_cup\pick_and_place_voice\object_detection\detection.py`
- `python -m py_compile C:\Users\minseok\Downloads\pick_and_place_voice_cup\pick_and_place_voice\object_detection\yolo.py`
- `python -m py_compile C:\Users\minseok\Downloads\pick_and_place_voice_cup\pick_and_place_voice\object_detection\realsense.py`
- Live ROS2 runtime checks were performed through repeated `ros2 run pick_and_place_voice robot_control` and `ros2 run pick_and_place_voice object_detection` sessions while watching:
  - `ASK_TRIGGER` dispatch
  - cleanup dispatch
  - observe entry
  - detection service calls
  - base-frame target conversion
  - pose rejection reasons

### Result

- `ASK_PENDING` no longer depends on a hard 20-second timeout and can now stay active until the robot finishes or cancels the ask action.
- ROS2 trigger publishing now works again from the live policy path when `ros2_trigger.enabled` is on.
- `ASK`, cleanup, cancel, and home-return scenario routing are now aligned with the intended behavior:
  - trigger start -> observe
  - finish / abort / cancel -> home
- Cleanup sessions now operate on a fixed target list rather than chasing every framewise change in `NEEDS_LIQUID_CHECK`.
- The robot-side single-target path is simpler and closer to the original version that previously grasped cups successfully.
- The object-detection service no longer crashes on the `_get_depth()` helper mismatch.
- The object-detection service no longer stalls forever only because aligned depth is unavailable; it can now fall back to raw depth.
- The main remaining coordinate pipeline was narrowed down to raw-depth sampling consistency:
  - before the final fix, base-frame targets were often rejected with large x or y offsets
  - the strongest remaining hypothesis was the mismatch between sampled depth pixel and projected camera pixel
- The latest code state includes a direct fix for that sampled-pixel mismatch and is ready for rerun validation.

### Issue

- `OBSERVE_POSJ` and the configured `OBSERVE_POSX` still do not physically correspond on the real robot setup.
- Because the measured observe pose differs strongly from the configured observe reference, there is still unresolved calibration ambiguity between:
  - the pose assumed by hand-eye calibration
  - the pose actually reached by the robot
- Some runtime trials still produced rejected targets such as:
  - green cup x slightly above workspace limit
  - red cup x well outside workspace limit
  - blue cup y far outside workspace limit
- Those bad targets were observed before the final sampled-pixel fix, so a clean rerun is still required to confirm whether the last coordinate bug is fully resolved.
- Cleanup and ask logic are now structurally aligned, but physical pick accuracy still depends on camera-to-base correctness.
- The original refinement-based staged descent remains in the file but is intentionally bypassed for the main single-target path; it may need cleanup later to avoid confusion.

### Next

- Rerun `object_detection` and `robot_control` with the latest sampled-pixel fix and capture fresh logs for:
  - `Received depth position`
  - `Target position in base frame`
  - first grasp command pose
- Verify whether green / red / blue targets now land inside the workspace limits without translation hacks.
- Recheck whether the configured `OBSERVE_POSX` should remain the detection reference or whether the real system needs recalibration instead.
- Compare actual physical cup locations against logged base-frame target coordinates to decide whether the remaining issue is:
  - residual raw-depth projection error
  - hand-eye calibration error
  - observe-pose calibration mismatch
- If coordinate accuracy becomes stable, clean up dead refinement paths and temporary debug scaffolding from `robot_control.py` and `detection.py`.
