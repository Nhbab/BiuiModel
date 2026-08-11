# BiuiModel — Training agent and GitHub Actions workflow

This repo now includes a simple training agent (PyTorch) with a pluggable uploader for Moithub (https://moithub.com).

Files added:
- agent/train_agent.py — CLI training script using a synthetic dataset by default.
- agent/model.py — small MLP model.
- agent/data.py — synthetic data generator and loader (replaceable with real data loader).
- agent/utils.py — checkpoint saving and optional Moithub uploader (uses MOITHUB_API_URL and MOITHUB_TOKEN env vars).
- train/config.yaml — default config
- requirements.txt
- scripts/run_local.sh — example local run script
- .github/workflows/train.yml — GitHub Actions workflow (branch: add-training-agent)

Usage (locally):
1. Create a virtualenv and install dependencies: `pip install -r requirements.txt`
2. Run a quick training: `python agent/train_agent.py --config train/config.yaml --epochs 1 --output-dir outputs`

Moithub upload:
- The uploader will POST to `${MOITHUB_API_URL}/api/v1/upload` with `Authorization: Bearer <token>`.
- Add repository secrets `MOITHUB_API_URL` and `MOITHUB_TOKEN` to enable automatic upload in the CI.