#!/bin/zsh
# Phase-4 pipeline (resumable).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${WORKERS:-9}
echo "== [1/3] advantage-scale intervention  $(date)"
$PY -m experiments.sweep experiments/configs/adv_scale.yaml --workers $W
echo "== [2/3] safety-penalty sweep  $(date)"
$PY -m experiments.sweep experiments/configs/safety_penalty.yaml --workers $W
echo "== [3/3] analysis  $(date)"
$PY -m analysis.phase4
echo "== done  $(date)"
