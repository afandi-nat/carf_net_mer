#!/usr/bin/env bash
# End-to-end check on synthetic data: no licensed dataset required.
#
#   bash tools/smoke_test.sh [workdir] [device]
#
# It builds a miniature stand-in dataset in each of the three folder layouts,
# runs prepare -> crop -> build -> train (2 epochs, 1 seed), and reloads one
# checkpoint. Accuracy numbers printed here are meaningless; the point is that
# every stage runs and the checkpoint reproduces its stored probabilities.
#
# The magnify stage is skipped: it clones a third-party repository, and the
# build stage falls back to the apex frame when no amplified frame exists.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
WORK="${1:-/tmp/carf_smoke}"
DEVICE="${2:-cpu}"

echo "== Generating synthetic data in ${WORK}/raw"
python tools/make_synthetic_data.py --out "${WORK}/raw" --subjects 3 --per-class 2

for ds in casme2 samm casme3; do
  label=$(ls "${WORK}/raw/${ds}"/*.xlsx | head -1)
  echo "== ${ds}: prepare"
  python -m carfnet.prepare --dataset "${ds}" --label-file "${label}" \
    --raw-root "${WORK}/raw/${ds}/frames" --labels-dir "${WORK}/labels"
  echo "== ${ds}: crop"
  python -m carfnet.crop --dataset "${ds}" --raw-root "${WORK}/raw/${ds}/frames" \
    --processed-root "${WORK}/processed" --labels-dir "${WORK}/labels"
  echo "== ${ds}: build"
  python -m carfnet.build --dataset "${ds}" --processed-root "${WORK}/processed" \
    --labels-dir "${WORK}/labels" --calib-n 8
done

echo "== train (2 epochs, 1 seed, no ImageNet weights)"
python -m carfnet.train --dataset casme2,samm,casme3 \
  --processed-root "${WORK}/processed" --labels-dir "${WORK}/labels" \
  --epochs 2 --warmup-epochs 1 --seeds 0 --no-pretrained --device "${DEVICE}" \
  --out-dir "${WORK}/run"

echo "== reloading a checkpoint"
python tools/verify_checkpoints.py --run-dir "${WORK}/run" \
  --labels-dir "${WORK}/labels" --limit 2

echo
echo "Smoke test finished. Artefacts are in ${WORK}"
