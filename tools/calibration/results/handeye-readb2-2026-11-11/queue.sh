#!/bin/bash
# readb2 bench queue (Moonstone): sequential b2.py runs, 8 processes each, niced, one BLAS thread per process.
# usage: queue.sh "TAG|CONFIG|VARIANTS[|EXTRA ARGS, VAR = the variant]" ...
# The runs reported in README.md:
#   queue.sh "fast|fast|base jit f2 f2b start f3"
#   queue.sh "K1|robust|base jit phase f2 f2b start f3" "K2|robust|base jit phase f2 f2b start f3|--pass1 $W/r_K1_VAR.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CALIBTOOL_DATA=~/calibtool/data
PY="nice -n 10 $HOME/calibtool/venv/bin/python"
W=${W:-~/calibtool/readb2-work}
cd "$(dirname "$0")"
for spec in "$@"; do
  IFS="|" read -r TAG CFG VARS EXTRA <<< "$spec"
  for v in $VARS; do
    echo "== $TAG $v $(date +%T)"
    $PY b2.py --config "$CFG" --variant $v --out $W/r_${TAG}_$v.json -j 8 ${EXTRA//VAR/$v} | tail -1
  done
done
