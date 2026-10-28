#!/usr/bin/env bash
# Spurious Shift: train, evaluate, aggregate and plot over a list of seeds.
#
#   bash spurious_shift/run.sh                       # all ten seeds
#   SEEDS="0" SKIP_ABLATIONS=1 SKIP_TRADEOFF=1 bash spurious_shift/run.sh
#
# Env:
#   PYTHON_BIN      interpreter (default: python)
#   SEEDS           space-separated seed list (default: 0-9)
#   SKIP_ABLATIONS  set to 1 to skip the ablation variants
#   SKIP_TRADEOFF   set to 1 to skip the fairness-strength sweep
#
# Outputs go to spurious_shift/results/: per-seed arrays under runs/seed<N>/,
# cross-seed aggregates (results*_agg.npz, results*_summary.csv) and figures/.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SKIP_ABLATIONS="${SKIP_ABLATIONS:-0}"
SKIP_TRADEOFF="${SKIP_TRADEOFF:-0}"
read -r -a PY <<< "${PYTHON_BIN}"
read -r -a RUN_SEEDS <<< "${SEEDS}"

cd "$(dirname "${BASH_SOURCE[0]}")/results"

for s in "${RUN_SEEDS[@]}"; do
  echo "==> [train]    seed ${s}"
  "${PY[@]}" ../train.py --fresh --seed "${s}"
  "${PY[@]}" ../train.py --reeval --seed "${s}"
  if [ "${SKIP_ABLATIONS}" != "1" ]; then
    echo "==> [ablate]   seed ${s}"
    "${PY[@]}" ../run_ablations.py --fresh --seed "${s}"
  fi
  if [ "${SKIP_TRADEOFF}" != "1" ]; then
    echo "==> [tradeoff] seed ${s}"
    "${PY[@]}" ../sweep_tradeoff.py --seed "${s}"
  fi
done

echo "==> [aggregate]"
"${PY[@]}" ../aggregate.py --seeds "${RUN_SEEDS[@]}"
"${PY[@]}" ../aggregate.py --sync-test-view
if [ "${SKIP_ABLATIONS}" != "1" ]; then
  "${PY[@]}" ../aggregate.py --ablations --seeds "${RUN_SEEDS[@]}"
fi
if [ "${SKIP_TRADEOFF}" != "1" ]; then
  "${PY[@]}" ../aggregate.py --tradeoff --seeds "${RUN_SEEDS[@]}"
fi

echo "==> [figures]"
"${PY[@]}" ../plot_motivation.py
"${PY[@]}" ../plot_results.py
if [ "${SKIP_ABLATIONS}" != "1" ]; then
  "${PY[@]}" ../plot_ablations.py
fi
echo "Done. Results and figures are in spurious_shift/results/."
