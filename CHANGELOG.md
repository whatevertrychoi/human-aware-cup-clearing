# Changelog

## Unreleased

### Planned
- Add local liquid verification using the gripper camera
- Connect mock cleanup skills to Doosan M0609 robot interfaces
- Expand evaluation summaries and ablation records

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
