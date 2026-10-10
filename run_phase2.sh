#!/bin/zsh
# Full Phase-2 pipeline. Every step is resumable: re-running skips finished work.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${WORKERS:-9}

echo "== [1/5] lambda sweep  $(date)"
$PY -m experiments.sweep experiments/configs/lambda_sweep.yaml --workers $W

echo "== [2/5] gradient probes  $(date)"
for env in Hopper-v5 LunarLanderContinuous-v3; do
  for s in 0 1 2; do
    run=results/runs/lambda_sweep/${env}_lam0.95_s${s}
    out=results/probes/grad_${env}_lam0.95_s${s}
    [ -f $out/grad_stats.csv ] || $PY -m experiments.probe_gradients $run --sigmas 0 1 3 10 --seed $s > $out.log 2>&1 &
  done
done
wait

echo "== [3/5] value probe  $(date)"
[ -f results/probes/values_lambda_sweep.csv ] || $PY -m experiments.probe_values results/runs/lambda_sweep --workers $W

echo "== [4/5] noise sweep  $(date)"
$PY -m experiments.sweep experiments/configs/noise_sweep.yaml --workers $W

echo "== [5/5] analysis  $(date)"
$PY -m analysis.phase2
echo "== done  $(date)"
