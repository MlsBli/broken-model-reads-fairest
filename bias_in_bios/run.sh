#!/usr/bin/env bash
# Bias in Bios: train, evaluate every readout, aggregate and plot over a list of
# seeds. Each seed is a full BERT fine-tuning pass over all tasks and methods.
#
#   SEEDS="0 1 2 3 4 5 6 7 8 9" bash bias_in_bios/run.sh
#
# Env:
#   PYTHON_BIN        interpreter (default: python)
#   SEEDS             space-separated seed list (required)
#   SKIP_TRAIN=1      evaluate existing checkpoints without training
#   FORCE_FRESH=1     retrain every seed from scratch
#   FORCE_EVALS=1     recompute evaluation stages that already finished
#   SKIP_AGGREGATE=1  per-seed work only; aggregate once afterwards when several
#                     seed groups run in parallel
#
# Outputs go to bias_in_bios/results/: per-seed arrays under runs/seed<N>/,
# cross-seed aggregates (results*_agg.npz, results*_summary.csv) and figures/.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -z "${SEEDS:-}" ]; then
  echo "Set SEEDS, e.g. SEEDS=\"0 1 2 3 4 5 6 7 8 9\"" >&2
  exit 1
fi
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FORCE_FRESH="${FORCE_FRESH:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
read -r -a PY <<< "${PYTHON_BIN}"
read -r -a RUN_SEEDS <<< "${SEEDS}"

cd "$(dirname "${BASH_SOURCE[0]}")/results"

echo "==> [preflight]"
"${PY[@]}" ../preflight.py --stage final

# Each evaluation stage writes its own artifact, so a finished stage is skipped
# when an interrupted seed is resumed.
stage() {
  local seed="$1" name="$2" marker="$3"; shift 3
  if [ "${FORCE_EVALS:-0}" != "1" ] && [ -f "runs/seed${seed}/${marker}" ]; then
    echo "==> [${name}] seed ${seed}: already done"
    return 0
  fi
  echo "==> [${name}] seed ${seed}"
  "${PY[@]}" "$@" --seed "${seed}"
}

for s in "${RUN_SEEDS[@]}"; do
  if [ "${SKIP_TRAIN}" != "1" ]; then
    if [ "${FORCE_FRESH}" = "1" ]; then
      echo "==> [train] seed ${s}"
      "${PY[@]}" ../train.py --fresh --seed "${s}"
    elif [ -f "runs/seed${s}/results.npz" ]; then
      echo "==> [train] seed ${s}: already done"
    elif [ -f "runs/seed${s}/checkpoint.pt" ]; then
      echo "==> [train] seed ${s}: resuming"
      "${PY[@]}" ../train.py --seed "${s}"
    else
      echo "==> [train] seed ${s}"
      "${PY[@]}" ../train.py --fresh --seed "${s}"
    fi
  fi
  stage "${s}" reeval       results_test.npz              ../train.py --reeval
  stage "${s}" retune       results_retune_fair.npz       ../retune_head.py
  stage "${s}" probe-shared results_probe_shared_fair.npz ../retune_head.py --shared-reference
  stage "${s}" knn          results_knn_shared.npz        ../eval_readouts.py
  stage "${s}" ttbudget     ttbudget.csv                  ../eval_ttbudget.py
  stage "${s}" leakage      results_leakage.npy           ../probe_leakage.py --test
  stage "${s}" ncm          results_ncm_full.npz          ../eval_ncm.py
done

if [ "${SKIP_AGGREGATE}" = "1" ]; then
  echo "Per-seed outputs are in bias_in_bios/results/runs/."
  exit 0
fi

echo "==> [aggregate]"
"${PY[@]}" ../aggregate.py --all --seeds "${RUN_SEEDS[@]}"
"${PY[@]}" ../eval_ttbudget.py --aggregate --seeds "${RUN_SEEDS[@]}"

echo "==> [figures]"
"${PY[@]}" ../plot_results.py --test
"${PY[@]}" ../plot_results.py --retune
"${PY[@]}" ../plot_results.py --retune-fair
"${PY[@]}" ../plot_ncm.py
echo "Done. Results and figures are in bias_in_bios/results/."
