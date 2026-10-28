#!/usr/bin/env bash
# The released ten-seed Bias in Bios campaign, plus the frozen-encoder control.
#
#   BIOS_DATA_PATH=/path/to/bios bash bias_in_bios/run_campaign.sh
#
# Seed 0 runs first, and the campaign stops if it did not produce every expected
# artifact, so a configuration error costs one seed rather than ten. The
# frozen-encoder control runs in a separate copy of the repository with
# BIOS_BERT_FINETUNE = False, and its aggregates are copied back into
# results/frozen_encoder/.
#
# Env: SEEDS (default 0-9), SKIP_FROZEN=1, PYTHON_BIN (default "uv run python"),
#      LOG (default campaign.log at the repository root),
#      FROZEN_COPY (default: a sibling directory of the repository).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
export PYTHON_BIN="${PYTHON_BIN:-uv run python}"
LOG="${LOG:-$ROOT/campaign.log}"
read -r -a PY <<< "${PYTHON_BIN}"
exec > >(tee -a "$LOG") 2>&1

FIRST="${SEEDS%% *}"; REST="${SEEDS#* }"; [ "$REST" = "$SEEDS" ] && REST=""

echo "=== seed ${FIRST} ==="
SEEDS="$FIRST" FORCE_FRESH=1 SKIP_AGGREGATE=1 bash "$HERE/run.sh"
for f in checkpoint.pt results_probe_shared.npz results_knn_shared.npz ttbudget.csv; do
  [ -f "$HERE/results/runs/seed$FIRST/$f" ] || { echo "seed $FIRST is missing $f; stopping."; exit 1; }
done

if [ -n "$REST" ]; then
  echo "=== seeds ${REST} ==="
  SEEDS="$REST" FORCE_FRESH=1 SKIP_AGGREGATE=1 bash "$HERE/run.sh"
fi
SEEDS="$SEEDS" SKIP_TRAIN=1 bash "$HERE/run.sh"

if [ "${SKIP_FROZEN:-0}" != "1" ]; then
  echo "=== frozen-encoder control ==="
  FZ="${FROZEN_COPY:-$ROOT/../$(basename "$ROOT")-frozen}"
  rm -rf "$FZ"
  mkdir -p "$FZ"
  (cd "$ROOT" && tar --exclude=./bias_in_bios/results/runs --exclude=./.git -cf - .) | tar -xf - -C "$FZ"
  sed -i 's/^BIOS_BERT_FINETUNE    = True/BIOS_BERT_FINETUNE    = False/' "$FZ/bias_in_bios/config.py"
  grep -q "^BIOS_BERT_FINETUNE    = False" "$FZ/bias_in_bios/config.py" \
    || { echo "could not set BIOS_BERT_FINETUNE = False"; exit 1; }
  cd "$FZ/bias_in_bios/results"
  for s in $SEEDS; do
    "${PY[@]}" ../train.py --fresh --seed "$s"
    "${PY[@]}" ../train.py --reeval --seed "$s"
    "${PY[@]}" ../retune_head.py --seed "$s"
    "${PY[@]}" ../eval_ncm.py --seed "$s"
  done
  "${PY[@]}" ../aggregate.py --all --seeds $SEEDS
  mkdir -p "$HERE/results/frozen_encoder"
  cp results*_agg.npz results*_summary.csv "$HERE/results/frozen_encoder/"
fi
echo "=== campaign done ==="
