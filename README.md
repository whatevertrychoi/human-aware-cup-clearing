# Learning When to Ask: Human-in-the-Loop Cup Clearing for Robotic Bartender

This repository contains a human-in-the-loop cup-clearing module for a fixed Doosan M0609 robotic arm in a bartender environment. The robot does not clear every cup immediately. Instead, it observes user-cup interaction, decides whether to wait or ask, and verifies the cup interior right before pickup.

## Project Overview

The broader bartender system recognizes the user, supports customized drink service, and then manages post-drink cleanup. This repository focuses on the cleanup decision layer and the final execution flow for used cups.

High-level actions:

1. `WAIT`
2. `ASK`
3. `CLEANUP_CANDIDATE`

Trajectory-aware live actions:

1. `WAIT`
2. `ASK`
3. `IDLE`
4. `CLEANUP_CANDIDATE`

Final robot skills:

1. `CLEAR`
2. `SPILL_SAFE_CLEAR`
3. `SKIP`

## Conservative Safety-First Policy

The policy is intentionally conservative: uncertain cases are routed to ASK rather than directly to cleanup, because wrong cleanup is more harmful than asking the user for confirmation.

본 시스템은 컵을 빠르게 많이 치우는 것보다, 사용자가 아직 사용하는 컵을 잘못 치우지 않는 것을 우선한다. 따라서 불확실한 상황에서는 로봇이 사용자에게 먼저 확인하도록 보수적으로 설계했다.

Decision philosophy:

1. If the situation is clearly risky, choose `WAIT`.
2. If the situation is ambiguous, choose `ASK`.
3. Only choose `CLEANUP_CANDIDATE` when cleanup is sufficiently safe.
4. Always run local liquid verification before final pickup behavior.

Current uncertainty-aware policy logic:

```python
if raw_action == "WAIT":
    action = "WAIT"
elif confidence < confidence_threshold:
    action = "ASK"
else:
    action = raw_action
```

This means the system is allowed to ask more often than strictly necessary, but it should avoid wrong cleanup whenever possible.

## Trajectory-Aware Live Policy

The older live-policy path treated cups too independently, so unused cups could still surface as `ASK`. The current live-policy layer adds active-cup arbitration and trajectory-aware suppression on top of the trained model.

Core arbitration rules:

1. `WAIT`
   - The cup is the current `active_cup`.
   - A hand is visible.
   - `hand_distance < touch_threshold`.
2. `ASK`
   - `used_cup_candidate = True`.
   - `user_present = 1`.
   - The hand has been released from the cup recently.
3. `IDLE`
   - `user_present = 1`.
   - `used_cup_candidate = False`.
   - The hand is not near the cup.
4. `CLEANUP_CANDIDATE`
   - The user has been absent long enough and the cup has stayed stationary long enough.
   - Or the cup has been untouched for a long time and is stationary.

Current trajectory-aware tracker fields:

- `active_cup_id`
- `is_active_cup`
- `time_near_cup`
- `time_since_release`
- `release_count`
- `cup_motion_distance`
- `stationary_time`
- `was_moved`
- `used_cup_candidate`

This layer ensures that only cups with actual interaction evidence can become `ASK` targets, while untouched cups stay suppressed as `IDLE`.

## Two-Stage Perception

This project uses a two-stage perception design.

1. Global perception with an eye-to-hand camera
   - Observe the full table scene
   - Detect cup position and cup ID
   - Detect hand position
   - Estimate user presence
   - Update interaction history
   - Predict `WAIT`, `ASK`, `IDLE`, or `CLEANUP_CANDIDATE`
2. Local perception with an eye-in-hand camera
   - Move close to the target cup
   - Inspect the cup interior
   - Predict `EMPTY` or `NON_EMPTY`
   - Select `CLEAR` or `SPILL_SAFE_CLEAR`

This separation is important because a global camera may not reliably see cup contents, while a gripper camera can perform final verification right before grasping.

## System Flow

1. Observe cups, hand, and user state from the global camera.
2. Update interaction history for each cup.
3. Predict a model raw action.
4. Apply trajectory-aware arbitration for `WAIT`, `ASK`, `IDLE`, or `CLEANUP_CANDIDATE`.
5. If the action is `WAIT` or `IDLE`, do nothing.
6. If the action is `ASK`, prompt the user.
7. If the user says no, execute `SKIP`.
8. If the user says yes, or if the cup is already a `CLEANUP_CANDIDATE`, move in for local verification.
9. Verify cup interior with the local camera.
10. If `EMPTY`, execute `CLEAR`.
11. If `NON_EMPTY`, execute `SPILL_SAFE_CLEAR`.

## Repository Scope

Git is used as a portfolio record, so only code, configuration, documentation, and small representative artifacts are committed.

Included in Git:

- Source code in `perception/`, `tracking/`, `policy/`, `robot/`, `data_collection/`, and `tools/`
- Entry scripts such as `main_demo.py` and `project_utils.py`
- Config and environment files such as `configs/config.yaml`, `.gitignore`, and `requirements.txt`
- Documentation such as `README.md`, `CHANGELOG.md`, and `DEV_LOG.md`
- Small outputs such as evaluation summaries and representative CSV files in `data/processed/`

Excluded from Git:

- Raw image and video collections under `data/raw/`
- Runtime logs under `data/logs/`
- Large trained model files such as `.joblib`, `.pkl`, `.pt`, `.pth`
- Long demo videos and raw recordings such as `.mp4`, `.avi`, `.mov`
- Virtual environments and Python cache files

## Project Structure

```text
cup_cleanup/
├── README.md
├── CHANGELOG.md
├── DEV_LOG.md
├── requirements.txt
├── .gitignore
├── main_demo.py
├── project_utils.py
├── configs/
├── perception/
├── policy/
├── tracking/
├── robot/
├── data_collection/
├── tools/
├── data/
└── results/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## v0.3 Environment

For `v0.3-hand-user-tracking`, use a Python 3.12 environment with the MediaPipe Solutions API.

```powershell
conda create -n cup_cleanup_mp312 python=3.12 -y
conda activate cup_cleanup_mp312
cd C:\Users\minseok\Desktop\cup_cleanup
pip install -r requirements.txt
pip install mediapipe==0.10.13
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

The second printed line should be `True`.

## Execution Order

Mock pipeline:

```bash
python policy/generate_mock_dataset.py --out data/processed/dataset_decision.csv --n 1000
python policy/train_policy.py --data data/processed/dataset_decision.csv --model results/decision_model.joblib --algo rf
python main_demo.py --model results/decision_model.joblib --mock --mock-responses y,n
```

Global cup detection:

```bash
python perception/detect_cups.py --config configs/config.yaml --camera-index 1 --backend dshow
python perception/detect_cups.py --config configs/config.yaml --camera-index 1 --backend dshow --show-mask-debug
```

Perception debug with cups, hand, and user presence:

```bash
python main_demo.py --camera-index 1 --backend dshow --debug-perception
```

Live policy inference:

```bash
python main_demo.py --camera-index 1 --backend dshow --live-policy --model results/decision_model_real.joblib
```

In this mode the overlay shows:

- predicted action and confidence
- `hand_distance`
- `touch_count`
- `last_touched_time`
- `user_present`
- `user_absent_time`
- `time_near_cup`
- `stationary_time`
- `used_cup_candidate`
- `ACTIVE` and `USED` markers

## Cup Dataset Capture

Connect the external USB webcam and capture cup images for `v0.2-global-cup-detection`.

```bash
python tools/capture_cup_dataset.py --camera-index 1 --backend dshow
```

Key controls:

- `g`: save a green cup image to `data/raw/green/`
- `r`: save a red cup image to `data/raw/red/`
- `b`: save a blue cup image to `data/raw/blue/`
- `q` or `ESC`: quit

## Real Interaction Dataset

`data/processed/dataset_decision.csv` is reserved for the v0.1 mock or synthetic dataset and should never be overwritten by real webcam recordings.

Use scene-specific output files for real interaction capture:

```bash
python data_collection/record_interaction_dataset.py --camera-index 1 --backend dshow --out data/processed/interaction_green.csv --interval 0.5
python data_collection/record_interaction_dataset.py --camera-index 1 --backend dshow --out data/processed/interaction_red.csv --interval 0.5
python data_collection/record_interaction_dataset.py --camera-index 1 --backend dshow --out data/processed/interaction_blue.csv --interval 0.5
python data_collection/record_interaction_dataset.py --camera-index 1 --backend dshow --out data/processed/interaction_clutter.csv --interval 0.5
```

The recorder writes cup-level rows with:

- `scene_id`
- `timestamp`
- `cup_id`
- `x`, `y`
- `hand_distance`
- `last_touched_time`
- `touch_count`
- `moved_recently`
- `distance_to_tray`
- `user_present`
- `user_absent_time`
- `active_cup_id`
- `is_active_cup`
- `time_near_cup`
- `time_since_release`
- `release_count`
- `cup_motion_distance`
- `stationary_time`
- `was_moved`
- `used_cup_candidate`
- expert-rule `label`

Trajectory-aware labels:

- `WAIT`
- `ASK`
- `IDLE`
- `CLEANUP_CANDIDATE`

Recommended real interaction files:

- `data/processed/interaction_green.csv`
- `data/processed/interaction_red.csv`
- `data/processed/interaction_blue.csv`
- `data/processed/interaction_clutter.csv`
- merged dataset: `data/processed/interaction_dataset_all.csv`

Current merged real interaction dataset summary:

- Total rows: `1547`
- Label distribution: `ASK=717`, `WAIT=558`, `CLEANUP_CANDIDATE=272`
- Source file row counts:
  - `interaction_green.csv=305`
  - `interaction_red.csv=266`
  - `interaction_blue.csv=193`
  - `interaction_clutter.csv=783`

Dataset analysis:

```bash
python data_collection/analyze_interaction_dataset.py --data data/processed/interaction_green.csv
```

Dataset merge:

```bash
python data_collection/merge_interaction_datasets.py --out data/processed/interaction_dataset_all.csv
```

Real dataset policy training:

```bash
python policy/train_policy.py --data data/processed/interaction_dataset_all.csv --model results/decision_model_real.joblib --algo rf
```

Mock vs real dataset split:

- Mock dataset: `data/processed/dataset_decision.csv`
- Real scene datasets: `interaction_green.csv`, `interaction_red.csv`, `interaction_blue.csv`, `interaction_clutter.csv`
- Real merged dataset: `data/processed/interaction_dataset_all.csv`

Placeholder clipping for real-data training:

- `hand_distance` is clipped to `2.0`
- `last_touched_time` is clipped to `60.0`
- `user_absent_time` is clipped to `60.0`

This prevents the model from overfitting to sentinel values such as `999` or `1000+` that represent hand-not-visible or never-touched states.

Trajectory-aware dataset note:

- Existing `interaction_green/red/blue/clutter.csv` files are still useful and should not be discarded.
- Additional collection should focus on `IDLE` scenes and stronger cup-use trajectories rather than full recollection.
- Useful supplemental scenes are:
  - user present with no cup touched
  - one active cup while the other cups stay idle
  - cup moved and released
  - user absent with long stationary cups

## Data and Results

- Raw data should stay under `data/raw/` and is not committed.
- Runtime logs should stay under `data/logs/` and are not committed.
- Small processed datasets can be kept in `data/processed/`.
- Small evaluation artifacts can be kept in `results/`.
- Sample visuals for the portfolio can be kept in `docs/sample_images/` and `docs/demo_screenshots/`.

Current tracked evaluation artifacts:

- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/evaluation_summary.csv`
- `data/processed/dataset_decision.csv`
- `results/classification_report_real.txt`
- `results/confusion_matrix_real.png`
- `results/evaluation_summary_real.csv`
- `data/processed/interaction_dataset_all.csv`

Future evaluation should emphasize safety-oriented metrics in addition to plain accuracy:

- `wrong_cleanup_rate`
- `unnecessary_ask_rate`
- `wait_safety_success`
- `ask_override_count`
- `cleanup_candidate_precision`
- `WAIT recall`

The main evaluation principle is that a slightly higher ASK rate is acceptable, while wrong cleanup of a cup still in use should be treated as a major failure.

Current real-data validation metrics:

- `accuracy=1.0000`
- `wrong_cleanup_rate=0.0000`
- `ask_override_count=0`
- `cleanup_candidate_precision=1.0000`
- `WAIT recall=1.0000`

## Git Version Plan

- `v0.1-mock-pipeline`: mock dataset generator, expert policy, training, inference, mock robot demo
- `v0.2-global-cup-detection`: HSV-based green/red/blue cup detection and debug visualization
- `v0.3-hand-user-tracking`: MediaPipe Hands, user presence, hand-cup distance, interaction tracking
- `v0.4-real-interaction-dataset`: real interaction dataset merging, analysis, and policy retraining
- `v0.5-live-policy-inference`: trajectory-aware active-cup arbitration and live real-time policy overlay
- `v0.6-human-in-the-loop`: yes/no ASK interface and `SKIP` flow
- `v0.7-local-liquid-verification`: ROI-based local liquid detection and `CLEAR` vs `SPILL_SAFE_CLEAR`
- `v1.0-final-demo`: integrated pipeline, final documentation, screenshots, evaluation summary

## Development Logging

- `CHANGELOG.md` is for version-based feature history.
- `DEV_LOG.md` is for date-based working notes, tests, issues, and next steps.

## Next Steps

- Collect supplemental `IDLE` and stronger trajectory scenes
- Retrain the policy with trajectory-aware labels once the new scenes are merged
- Connect live `ASK` outputs to the real human-in-the-loop confirmation flow
- Keep local liquid verification as the final gate before any physical cleanup
