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

### Changed
- Updated expert-rule labeling to support `IDLE` in trajectory-aware data collection
- Updated real interaction recorder to save active-cup and trajectory features

### Notes
- The main fix in this version is reducing unnecessary cup-wise `ASK` prompts by requiring trajectory evidence before a cup can become an `ASK` target
- The longer-term refactor goal is model-first trajectory-aware behavior cloning with only a lightweight safety guard

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
