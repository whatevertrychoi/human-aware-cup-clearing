# Changelog

## Unreleased

### Planned
- Add global cup detection with HSV segmentation
- Add MediaPipe-based hand detection and user presence tracking
- Add local liquid verification using the gripper camera
- Connect mock cleanup skills to Doosan M0609 robot interfaces
- Expand evaluation summaries and ablation records

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
