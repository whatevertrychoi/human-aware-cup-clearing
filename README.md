# Learning When to Ask: Human-in-the-Loop Cup Clearing for Robotic Bartender

This repository contains a cup-clearing module for a fixed Doosan M0609 robotic arm in a bartender setting. The core idea is to decide when the robot should wait, ask the user, or proceed toward cleanup, then verify cup contents right before grasping.

## Core Idea

The system uses two-stage perception:

1. Global perception with an eye-to-hand camera
   - Detect cups and cup IDs
   - Detect user hand position
   - Estimate user presence
   - Update interaction history
   - Predict `WAIT`, `ASK`, or `CLEANUP_CANDIDATE`
2. Local perception with an eye-in-hand camera
   - Inspect the cup interior right before pickup
   - Predict `EMPTY` or `NON_EMPTY`
   - Choose `CLEAR` or `SPILL_SAFE_CLEAR`

This separates scene understanding from final liquid verification and better matches real robot constraints.

## Current MVP

The first milestone is a mock pipeline:

- Synthetic dataset generation for high-level actions
- Expert policy to label mock data
- RandomForest or MLP classifier training
- Uncertainty-aware `ASK` override
- Mock robot flow for `WAIT`, `ASK`, `CLEAR`, `SPILL_SAFE_CLEAR`, and `SKIP`

## Project Structure

```text
cup_cleanup/
├─ README.md
├─ CHANGELOG.md
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

## Run the Mock Pipeline

1. Generate a balanced mock dataset:

```bash
python policy/generate_mock_dataset.py --out data/processed/dataset_decision.csv --n 1000
```

2. Train the high-level decision policy:

```bash
python policy/train_policy.py --data data/processed/dataset_decision.csv --model results/decision_model.joblib --algo rf
```

3. Run the end-to-end mock demo:

```bash
python main_demo.py --model results/decision_model.joblib --mock
```

If you want non-interactive demo responses:

```bash
python main_demo.py --model results/decision_model.joblib --mock --mock-responses y,n
```

## Perception Utilities

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

## Outputs

- Model: `results/decision_model.joblib`
- Confusion matrix: `results/confusion_matrix.png`
- Classification report: `results/classification_report.txt`
- Mock dataset: `data/processed/dataset_decision.csv`

## Git Version Plan

- `v0.1-mock-pipeline`: mock dataset, expert policy, training, inference, mock robot demo
- `v0.2-global-cup-detection`: HSV-based green/red/blue cup detection
- `v0.3-hand-user-tracking`: hand tracking, user presence, interaction history
- `v0.4-decision-policy`: training and evaluation improvements
- `v0.5-human-in-the-loop`: ASK interface and yes/no flow
- `v0.6-local-liquid-verification`: local cup interior verification
- `v0.7-robot-skill-integration`: Doosan skill integration interface
- `v1.0-final-demo`: integrated demo and portfolio-ready results

## Next TODO

- Connect the global camera stream to real cup and hand tracking
- Replace mock local liquid output with real eye-in-hand camera input
- Add evaluation scripts for ablations and baseline comparisons
- Integrate Doosan M0609 motion skills for `CLEAR` and `SPILL_SAFE_CLEAR`

