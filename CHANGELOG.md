# Changelog

## Unreleased

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
