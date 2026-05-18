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
