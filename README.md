# Learning When to Ask: Human-in-the-Loop Cup Clearing for Robotic Bartender

This repository contains a human-in-the-loop cup-clearing module for a fixed Doosan M0609 robotic arm in a bartender environment. The robot does not clear every cup immediately. Instead, it observes user-cup interaction, decides whether to wait or ask, and verifies the cup interior right before pickup.

## Project Overview

The broader bartender system recognizes the user, supports customized drink service, and then manages post-drink cleanup. This repository focuses on the cleanup decision layer and the final execution flow for used cups.

High-level actions:

1. `WAIT`
2. `ASK`
3. `CLEANUP_CANDIDATE`

Final robot skills:

1. `CLEAR`
2. `SPILL_SAFE_CLEAR`
3. `SKIP`

The current implementation first validates the full behavior with a mock robot pipeline and then expands toward real perception and Doosan motion integration.

## Two-Stage Perception

This project uses a two-stage perception design.

1. Global perception with an eye-to-hand camera
   - Observe the full table scene
   - Detect cup position and cup ID
   - Detect hand position
   - Estimate user presence
   - Update interaction history
   - Predict `WAIT`, `ASK`, or `CLEANUP_CANDIDATE`
2. Local perception with an eye-in-hand camera
   - Move close to the target cup
   - Inspect the cup interior
   - Predict `EMPTY` or `NON_EMPTY`
   - Select `CLEAR` or `SPILL_SAFE_CLEAR`

This separation is important because a global camera may not reliably see cup contents, while a gripper camera can perform final verification right before grasping.

## System Flow

1. Observe cups, hand, and user state from the global camera.
2. Update interaction history for each cup.
3. Predict `WAIT`, `ASK`, or `CLEANUP_CANDIDATE`.
4. If the action is `WAIT`, do nothing.
5. If the action is `ASK`, prompt the user.
6. If the user says no, execute `SKIP`.
7. If the user says yes, or if the cup is already a `CLEANUP_CANDIDATE`, move in for local verification.
8. Verify cup interior with the local camera.
9. If `EMPTY`, execute `CLEAR`.
10. If `NON_EMPTY`, execute `SPILL_SAFE_CLEAR`.

## Repository Scope

Git is used as a portfolio record, so only code, configuration, documentation, and small representative artifacts are committed.

Included in Git:

- Source code in `perception/`, `tracking/`, `policy/`, `robot/`
- Entry scripts such as `main_demo.py` and `project_utils.py`
- Config and environment files such as `configs/config.yaml`, `.gitignore`, and `requirements.txt`
- Documentation such as `README.md`, `CHANGELOG.md`, and `DEV_LOG.md`
- Small outputs such as `results/classification_report.txt`, `results/confusion_matrix.png`, `results/evaluation_summary.csv`
- Small mock CSV files in `data/processed/`
- Portfolio images in `docs/sample_images/` and `docs/demo_screenshots/`

Excluded from Git:

- Raw image and video collections under `data/raw/`
- Runtime logs under `data/logs/`
- Large trained model files such as `.joblib`, `.pkl`, `.pt`, `.pth`
- Long demo videos and raw recordings such as `.mp4`, `.avi`, `.mov`
- Virtual environments and Python cache files

## Project Structure

```text
cup_cleanup/
├─ README.md
├─ CHANGELOG.md
├─ DEV_LOG.md
├─ requirements.txt
├─ .gitignore
├─ main_demo.py
├─ project_utils.py
├─ configs/
├─ perception/
├─ policy/
├─ tracking/
├─ robot/
├─ data/
├─ results/
└─ docs/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: `mediapipe` support can vary depending on Python version. The mock pipeline does not require a camera and is the recommended first validation step.

## Execution Order

```bash
python policy/generate_mock_dataset.py --out data/processed/dataset_decision.csv --n 1000
```

```bash
python policy/train_policy.py --data data/processed/dataset_decision.csv --model results/decision_model.joblib --algo rf
```

```bash
python main_demo.py --model results/decision_model.joblib --mock
```

Optional non-interactive demo:

```bash
python main_demo.py --model results/decision_model.joblib --mock --mock-responses y,n
```

## Additional Scripts

## Cup Dataset Capture

Connect the external USB webcam and capture cup images for `v0.2-global-cup-detection`.

```bash
python tools/capture_cup_dataset.py --camera-index 0
```

If the webcam does not open, try a different camera index.

```bash
python tools/capture_cup_dataset.py --camera-index 1
python tools/capture_cup_dataset.py --camera-index 2
```

Optional camera settings:

```bash
python tools/capture_cup_dataset.py --camera-index 1 --width 1280 --height 720 --fps 30
```

Key controls:

- `g`: save a green cup image to `data/raw/green/`
- `r`: save a red cup image to `data/raw/red/`
- `b`: save a blue cup image to `data/raw/blue/`
- `q` or `ESC`: quit

Target collection counts:

- Green: 100 images
- Red: 100 images
- Blue: 100 images

Capture guidance:

- Use the same external USB webcam placement planned for the final demo
- Capture on the real robot work table whenever possible
- Move cups to different positions instead of keeping them centered
- Include scenes where cups are close to one another
- Include lighting and shadow variation

Suggested breakdown per cup:

- 60 images with a single cup at varied positions
- 25 images near other cups
- 15 images with lighting or shadow changes

After collection, the next step is `v0.2-global-cup-detection`, where `perception/detect_cups.py` can be tuned with HSV thresholds using the captured images.

Cup detection:

```bash
python perception/detect_cups.py --config configs/config.yaml
```

Hand detection:

```bash
python perception/detect_hand.py --camera-index 0
```

User presence detection:

```bash
python perception/detect_user_presence.py --camera-index 0
```

Local liquid detection:

```bash
python perception/detect_liquid_local.py --config configs/config.yaml
```

## Data and Results

- Raw data should stay under `data/raw/` and is not committed.
- Runtime logs should stay under `data/logs/` and are not committed.
- Small mock datasets can be kept in `data/processed/`.
- Small evaluation artifacts can be kept in `results/`.
- Sample visuals for the portfolio can be kept in `docs/sample_images/` and `docs/demo_screenshots/`.

Current tracked evaluation artifacts:

- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/evaluation_summary.csv`
- `data/processed/dataset_decision.csv`

## Git Version Plan

- `v0.1-mock-pipeline`: mock dataset generator, expert policy, training, inference, mock robot demo
- `v0.2-global-cup-detection`: HSV-based green/red/blue cup detection and debug visualization
- `v0.3-hand-user-tracking`: MediaPipe Hands, user presence, hand-cup distance, interaction tracking
- `v0.4-decision-policy`: policy training refinements, confusion matrix, classification report, evaluation scripts
- `v0.5-human-in-the-loop`: yes/no ASK interface and `SKIP` flow
- `v0.6-local-liquid-verification`: ROI-based local liquid detection and `CLEAR` vs `SPILL_SAFE_CLEAR`
- `v1.0-final-demo`: integrated pipeline, final documentation, screenshots, evaluation summary

## Development Logging

- `CHANGELOG.md` is for version-based feature history.
- `DEV_LOG.md` is for date-based working notes, tests, issues, and next steps.

## Next Steps

- Connect the global camera to real cup and hand observations
- Improve dataset realism and evaluation reporting
- Replace mock local liquid frames with real gripper-camera input
- Connect Doosan M0609 skills for `CLEAR` and `SPILL_SAFE_CLEAR`
