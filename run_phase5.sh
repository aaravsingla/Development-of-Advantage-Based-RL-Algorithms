#!/bin/zsh
# Phase-5 pipeline (resumable). Tuning -> selection -> final 10-seed runs -> evaluation -> figures.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${WORKERS:-9}
echo "== [1/5] tuning  $(date)"
$PY -m experiments.sweep experiments/configs/safenav_tune.yaml --workers $W
$PY -m experiments.sweep experiments/configs/safenav_tune_lag.yaml --workers $W
echo "== [2/5] select coefficients  $(date)"
$PY -m analysis.phase5 --select
echo "== [3/5] final runs  $(date)"
$PY -m experiments.sweep experiments/configs/safenav_final.yaml --workers $W
$PY -m experiments.sweep experiments/configs/safenav_final_lag.yaml --workers $W
echo "== [4/5] evaluation  $(date)"
$PY -m experiments.eval_safenav results/runs/safenav_final --workers $W
echo "== [5/5] analysis  $(date)"
$PY -m analysis.phase5
echo "== done  $(date)"
