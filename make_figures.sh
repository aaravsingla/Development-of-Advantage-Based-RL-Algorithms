#!/bin/zsh
# Regenerate every figure (results/figures) and table (results/tables) from the saved runs and
# probe outputs. No training: takes ~2 min. Run the run_phase*.sh scripts first to produce data.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY -m analysis.phase2
$PY -m analysis.phase3
$PY -m analysis.phase4
$PY -m analysis.phase5
echo "all figures regenerated"
