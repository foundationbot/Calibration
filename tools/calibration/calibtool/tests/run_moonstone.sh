#!/bin/bash
# Run pytest for a calib-tui worktree on Moonstone (the laptop has 14 GB and no test deps; never run suites locally).
# usage: run_moonstone.sh <worktree> <name> [pytest args...]   (default args: tools/calibration -q -rfs -n 8)
#   Syncs <worktree> (code only) to Moonstone ~/calibtool/wt-<name>/ and runs the CI-mirroring venv ~/calibtool/venv.
#   One BLAS/OpenMP thread per xdist worker: without it a 16-worker run drove the shared box to load ~270 (9 Nov 2026).
#   Stored session data for replays: Moonstone ~/calibtool/data/{p10,p21}/ (kit/, etc-okvis/).
set -u
WT=${1:?worktree}; NAME=${2:?name}; shift 2
ARGS="${*:-tools/calibration -q -rfs -n 8}"
rsync -a --delete --exclude .git --exclude 'build*' --exclude models --exclude __pycache__ --exclude .pytest_cache \
  "$WT/" "Moonstone:calibtool/wt-$NAME/" || exit 2
ssh Moonstone "cd ~/calibtool/wt-$NAME && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CALIBTOOL_DATA=\$HOME/calibtool/data && nice -n 10 ~/calibtool/venv/bin/python -m pytest $ARGS \
  -p no:cacheprovider 2>&1 | tail -60"
