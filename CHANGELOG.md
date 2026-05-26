# Changelog

## Unreleased

### Added
- Separate ASK and robot ROS2 trigger topics in runtime config and bridge transport.
- Stricter contour-based cup filtering controls:
  - `max_area`
  - `min_box_size`
  - `min_fill_ratio`
  - `min_solidity`
  - `min_circularity`
  - `min/max_aspect_ratio`
  - `max_bbox_area_ratio`
  - `min_score`
- Per-color detector override support in `perception/detect_cups.py`.
- Robot-feedback-driven ASK completion handling so `ASK_PENDING` can now be closed by downstream `ASK_ACTION_FINISHED` results instead of only by policy-side timeout logic.
- Bridge support for forwarding cleanup target sets (`cleanup_source_cup_ids`, `cleanup_robot_cup_ids`) to the robot-side cleanup session.

### Changed
- Changed ASK publishing so `ASK_TRIGGER` and `CANCEL_ASK_TRIGGER` are routed to the voice-side ask topic instead of the robot execution topic.
- Kept `ROBOT_LIQUID_CHECK_TRIGGER` and robot cleanup cancel on the robot execution topic.
- Tightened global HSV cup detection thresholds and contour acceptance to reduce false positives from clothing and face-colored regions.
- Relaxed green-only detector thresholds through `detector_overrides` after the stricter global detector reduced green-cup recall too much.
- Changed live state-machine timeout behavior so `ASK_PENDING` no longer auto-expires after `20s` when `policy.ask_pending_timeout` is set to `0.0`.
- Changed the ROS2 bridge / live app integration so robot feedback is drained each frame and applied back into the runtime state machine.
- Changed cleanup-session semantics so one `ROBOT_LIQUID_CHECK_TRIGGER` now carries a fixed cleanup target set for the downstream robot session instead of relying only on framewise candidate churn.
- Enabled ROS2 trigger publishing by default in the current runtime config used for live integration testing.

### Changed
- Added a short ASK re-arm delay in `integration/ros2_trigger_bridge.py` so a newly pending ASK is not published in the same frame immediately after the previous ASK session is cancelled or cleared.
- Changed ASK session clearing so leaving the ASK state no longer emits an immediate generic `CANCEL_ASK_TRIGGER`; instead the bridge now holds the active ASK session for a grace window before silently releasing it, while explicit reuse cancel still cancels immediately.
- Changed ASK session ownership again so the bridge now prefers robot feedback on `/cup_cleanup/robot_feedback` to clear an active ASK session; policy-side ASK disappearance only acts as a long watchdog fallback instead of immediately ending the robot-owned ASK run.
- Changed live-policy overlay behavior so an active robot-owned ASK session keeps the corresponding cup in `ASK` state on the policy display while the robot is still executing, instead of letting transient runtime state transitions switch the overlay back to `OBSERVE/IDLE/WAIT`.
- Removed keyboard `y/n` response handling from the live policy loop in preparation for voice-driven confirmation, while keeping the state-machine response APIs available for the upcoming voice input integration.

### Added
- Cleanup-session interpretation for `ROBOT_LIQUID_CHECK_TRIGGER` on the ROS2 bridge side.
- Short cleanup-cancel grace handling so brief liquid-check candidate flicker does not immediately cancel an active robot cleanup session.

### Changed
- Clarified the current integration meaning of ROS2 trigger transport:
  - `ASK_TRIGGER` still starts a social ask flow
  - `ROBOT_LIQUID_CHECK_TRIGGER` is now aligned with downstream robot cleanup-session start behavior
- Updated bridge-side liquid-check publishing so one active cleanup session is started from the current liquid-check candidate set instead of repeatedly treating the event as a per-cup local-inspection request.
- Updated bridge-side ASK handling so the active ASK cup emits `CANCEL_ASK_TRIGGER` not only for reuse cancellation but also when it leaves the ASK set, including transitions back to `WAIT`.
- Treated `ASK_PENDING` and `READY_TO_CLEAR` as ASK-session-active states so the bridge no longer cancels immediately right after `ASK_TRIGGER`.

### Planned
- Add real local liquid verification using the gripper camera
- Connect cleanup actions to Doosan M0609 robot interfaces
- Expand evaluation summaries and ablation records
- Improve person-distance estimation for ASK priority when hands are not visible

## v0.6-social-state-machine

### Added
- Runtime `OBSERVE` state for post-release waiting before asking
- Reuse detection so `OBSERVE`, `ASK_PENDING`, or cooldown can be cancelled when the user grabs the cup again
- Response-aware social states:
  - `ASK_PENDING`
  - `ASK_COOLDOWN`
  - `READY_TO_CLEAR`
  - `NEEDS_LIQUID_CHECK`
- Keyboard `y/n` response handling in live state-machine evaluation
- Keyboard `y/n` mock liquid-check response handling for `NEEDS_LIQUID_CHECK` abandoned-cup testing
- ROS2 trigger bridge support for final live state-machine outputs
- One-shot trigger publishing for:
  - `ASK_TRIGGER`
  - `CANCEL_ASK_TRIGGER`
  - `ROBOT_LIQUID_CHECK_TRIGGER`
- Refined `CANCEL_ASK_TRIGGER` so it is emitted only for cups that actually published a prior `ASK_TRIGGER`
- Added `CANCEL_ROBOT_LIQUID_CHECK_TRIGGER` so robot-side local inspection can be aborted when a previously selected abandoned-cup target leaves the active liquid-check set before completion
- Single-arm ASK priority arbitration so only one cup is asked at a time
- `ask_reason`, ASK rank, and priority logging for explainable social prompting
- Heuristic `drink_count`, `estimated_consumed_ml`, and `estimated_drink_progress`
- ASK milestones at drink counts `5`, `8`, and `10`
- Face-proximity drink gating so sip-like events are counted only when the cup approaches the user's face before release
- Post-accept exclusion so cups confirmed for cleanup are removed from repeated global-policy ASK arbitration

### Changed
- Tightened sip-like event gating so single pick-and-place interactions remain in `OBSERVE` instead of triggering ASK
- Added release debounce, cooldown, and hysteresis to reduce noisy `release_count` spikes
- Separated responsibilities more clearly:
  - global webcam = social timing and cleanup candidacy
  - local or gripper camera = final EMPTY/NON_EMPTY verification before physical cleanup
- Kept `model_only` as pure Behavior Cloning inspection and `safety_guard` as minimal safety correction
- Added live overlay/log feedback for mock local-liquid-check results so abandoned-cup validation can show clear vs restore decisions before robot integration
- Added configurable bridge-side cup-id remapping for ROS2 / robot YOLO integration without rewriting the internal cup IDs used by the policy
- Added bridge-side active-session tracking for ASK and liquid-check triggers so transport events better match one-shot policy intent

## v0.5-trajectory-aware-policy

### Added
- Trajectory-aware active-cup arbitration on top of live model inference
- `IDLE` suppression for untouched cups in multi-cup scenes
- Live overlay for `ACTIVE`, `USED`, confidence, and tracker timing fields
- Trajectory-aware tracker outputs such as:
  - `time_near_cup`
  - `time_since_release`
  - `release_count`
  - `stationary_time`
  - `used_cup_candidate`
- Trajectory-aware policy retraining artifacts:
  - `results/decision_model_trajectory.joblib`
  - `results/classification_report_trajectory.txt`
  - `results/confusion_matrix_trajectory.png`
  - `results/evaluation_summary_trajectory.csv`
- Live evaluation modes:
  - `model_only`
  - `safety_guard`
  - `arbitration`
- Live evaluation CSV logging with `--log-live-eval`

### Changed
- Updated expert-rule labeling to support `IDLE` in trajectory-aware data collection
- Updated real interaction recorder to save active-cup and trajectory features
- Added explicit ASK delay after release and a longer delay for never-active cups
- Tightened cleanup promotion so user presence suppresses stationary-only cleanup

### Notes
- The main improvement in this stage was reducing unnecessary cup-wise `ASK` prompts by requiring trajectory evidence before a cup can become an ASK target
- This stage shifted the project from rule-heavy arbitration toward model-first trajectory-aware behavior cloning with a lightweight safety guard

## v0.4-real-interaction-dataset

### Added
- Real interaction dataset collection workflow with scene-specific CSV files
- Dataset merging and analysis utilities
- Real-data decision policy training pipeline
- Safety-oriented evaluation metrics for real-data validation

### Artifacts
- `data/processed/interaction_green.csv`
- `data/processed/interaction_red.csv`
- `data/processed/interaction_blue.csv`
- `data/processed/interaction_clutter.csv`
- `data/processed/interaction_dataset_all.csv`
- `results/classification_report_real.txt`
- `results/confusion_matrix_real.png`
- `results/evaluation_summary_real.csv`

## v0.3-hand-user-tracking

### Added
- MediaPipe Hands-based hand center detection
- User presence detection
- Hand-cup distance tracking
- Touch count and last-touch timing overlay
- Integrated live perception debug mode in `main_demo.py`

### Notes
- This stage established the real-time hand and user signals required for later interaction tracking and cleanup policy inference

## v0.2-global-cup-detection

### Added
- HSV-based green/red/blue cup detection
- Real USB webcam execution support with `--camera-index` and `--backend`
- Debug visualization for cup bbox, center point, cup ID, color, and area
- Mask cleanup and contour filtering for more stable cup segmentation

### Notes
- This stage established the global webcam perception layer used by all later interaction and policy modules

## v0.1-mock-pipeline

### Added
- Project structure for perception, tracking, policy, and robot modules
- Config-driven expert high-level policy
- Balanced mock dataset generator
- RandomForest and MLP training script
- Uncertainty-aware policy inference
- Mock robot with `WAIT`, `ASK`, `SKIP`, `CLEAR`, and `SPILL_SAFE_CLEAR`
- End-to-end `main_demo.py` mock pipeline
- README, CHANGELOG, DEV_LOG, requirements, and `.gitignore`
- Small evaluation artifacts for portfolio tracking

### Artifacts
- `data/processed/dataset_decision.csv`
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/evaluation_summary.csv`
