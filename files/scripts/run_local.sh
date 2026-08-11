#!/usr/bin/env bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python agent/train_agent.py --config train/config.yaml --epochs 1 --output-dir outputs