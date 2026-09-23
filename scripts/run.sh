#!/usr/bin/env bash
# ============================================================
# CARF-Net, one entry point for every dataset. Run from the repository root.
#
#   bash scripts/run.sh <dataset> <stage>
#
# dataset : casme2 | samm | casme3 | all
#           or a comma-separated combination for the train stage
# stage   : prepare   read annotations, map classes, verify frames
#           crop      detect faces and 68 landmarks, crop the needed frames
#           magnify   motion magnification, onset -> apex (optional)
#           build     build carf.npy (flow scale is calibrated automatically)
#           train     LOSO training with SEEDS
#           quick     train with a single seed, to check the setup
#           pre       prepare + crop + magnify + build
#           full      pre + train
#
# Examples:
#   bash scripts/run.sh casme3 full
#   bash scripts/run.sh all pre
#   bash scripts/run.sh casme2,samm,casme3 train    # combined, LOSO over all subjects
#   CLASSES=5 bash scripts/run.sh casme2 full
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/config.sh"
cd "${ROOT}"
mkdir -p logs

TARGET="${1:?Give a dataset: casme2 | samm | casme3 | all}"
STAGE="${2:?Give a stage: prepare | crop | magnify | build | train | quick | pre | full}"

if [ "${TARGET}" = "all" ]; then
  DATASETS=(casme2 samm casme3)
else
  IFS=',' read -r -a DATASETS <<< "${TARGET}"
fi

var () { local name="$1_$2"; echo "${!name:?Set $name in scripts/config.sh}"; }

stage_one () {
  local ds="$1" st="$2" log="logs/${1}_${CLASSES}c_${2}.log"
  case "${st}" in
    prepare)
      python -m carfnet.prepare --dataset "${ds}" --classes "${CLASSES}" \
        --label-file "$(var LABEL "${ds}")" --raw-root "$(var RAW "${ds}")" \
        2>&1 | tee "${log}" ;;
    crop)
      python -m carfnet.crop --dataset "${ds}" --classes "${CLASSES}" \
        --raw-root "$(var RAW "${ds}")" --processed-root "${PROCESSED_ROOT}" \
        2>&1 | tee "${log}" ;;
    magnify)
      python -m carfnet.magnify --dataset "${ds}" --classes "${CLASSES}" \
        --processed-root "${PROCESSED_ROOT}" --device "${DEVICE}" \
        2>&1 | tee "${log}" ;;
    build)
      python -m carfnet.build --dataset "${ds}" --classes "${CLASSES}" \
        --processed-root "${PROCESSED_ROOT}" 2>&1 | tee "${log}" ;;
    *) echo "Unknown stage: ${st}"; exit 1 ;;
  esac
}

train () {
  local tag="$1" seeds="$2"
  python -m carfnet.train --dataset "${tag}" --classes "${CLASSES}" \
    --processed-root "${PROCESSED_ROOT}" --seeds "${seeds}" --epochs "${EPOCHS}" \
    --device "${DEVICE}" --out-dir "runs/carf_${tag//,/+}_${CLASSES}c" \
    --protocols "${PROTOCOLS}" --save-ckpt "${SAVE_CKPT}" --ckpt-seeds "${CKPT_SEEDS}" \
    2>&1 | tee "logs/train_${tag//,/+}_${CLASSES}c.log"
}

case "${STAGE}" in
  prepare|crop|magnify|build)
    for ds in "${DATASETS[@]}"; do stage_one "${ds}" "${STAGE}"; done ;;
  pre|full)
    for ds in "${DATASETS[@]}"; do
      for st in prepare crop magnify build; do stage_one "${ds}" "${st}"; done
    done
    if [ "${STAGE}" = "full" ]; then
      for ds in "${DATASETS[@]}"; do train "${ds}" "${SEEDS}"; done
    fi ;;
  train|quick)
    seeds="${SEEDS}"; [ "${STAGE}" = "quick" ] && seeds=0
    if [ "${TARGET}" = "all" ]; then
      for ds in "${DATASETS[@]}"; do train "${ds}" "${seeds}"; done
    else
      train "${TARGET}" "${seeds}"     # single dataset, or a comma-separated combination
    fi ;;
  *) echo "Unknown stage: ${STAGE}"; exit 1 ;;
esac
