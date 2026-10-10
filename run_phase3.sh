#!/bin/zsh
# Phase-3 pipeline (resumable).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${WORKERS:-9}
echo "== [1/3] DDPG vs TD3  $(date)"
$PY -m experiments.sweep experiments/configs/offpolicy_sweep.yaml --workers $W
echo "== [2/3] twin-Q ablation  $(date)"
$PY -m experiments.sweep experiments/configs/twinq_ablation.yaml --workers $W
echo "== [3/3] analysis  $(date)"
$PY -m analysis.phase3
echo "== done  $(date)"
