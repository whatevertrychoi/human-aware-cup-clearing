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
