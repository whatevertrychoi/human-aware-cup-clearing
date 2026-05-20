# Changelog

## Unreleased

### Planned
- Add local liquid verification using the gripper camera
- Connect mock cleanup skills to Doosan M0609 robot interfaces
- Expand evaluation summaries and ablation records

## v0.5-live-policy-inference

### Added
- Trajectory-aware active-cup arbitration on top of live model inference
- `IDLE` suppression for untouched cups in multi-cup scenes
- Live overlay for `ACTIVE`, `USED`, confidence, and tracker timing fields
- Trajectory-aware tracker outputs such as `time_near_cup`, `time_since_release`, `release_count`, and `stationary_time`
- Runtime `OBSERVE` state for post-release waiting before asking
- Reuse detection to cancel `OBSERVE` or `ASK` when the user reuses a cup
- Soft transition state machine for `arbitration` and `state_machine` live modes
- `ASK_PENDING` to avoid repeated prompts every frame
- Single-arm ASK priority arbitration so only one cup is asked at a time
- `ask_reason`, ASK rank, and priority logging for explainable social prompting
- Heuristic `drink_count` and estimated drink progress from hand-cup trajectory
- Stricter sip-like event gating so single pick-and-place interactions stay in `OBSERVE` and do not immediately trigger ASK
- ASK milestones at drink counts `5`, `8`, and `10`, plus release debounce and hysteresis to reduce noisy release spikes
- Face-proximity drink gating so sip-like events are counted when the cup is actually brought near the user
- `NEEDS_LIQUID_CHECK` handoff so cleanup candidates go to local verification instead of direct clear
- `ASK_COOLDOWN` for rejection and timeout handling
- `READY_TO_CLEAR` for accepted cleanup requests
- Keyboard `y/n` response handling in live state-machine evaluation

### Changed
- Updated expert-rule labeling to support `IDLE` in trajectory-aware data collection
- Updated real interaction recorder to save active-cup and trajectory features
- Added explicit `ASK` delay after release and a longer delay for never-active cups
- Tightened cleanup promotion so user presence suppresses stationary-only cleanup

### Notes
- The main fix in this version is reducing unnecessary cup-wise `ASK` prompts by requiring trajectory evidence before a cup can become an `ASK` target
- The longer-term refactor goal is model-first trajectory-aware behavior cloning with only a lightweight safety guard
- Added trajectory-aware retraining artifacts and validation outputs
- Added `model_only`, `safety_guard`, and `arbitration` live evaluation modes for direct comparison

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
