# CARF-Net

**Class-Aware Region Fusion Network for micro-expression recognition**, with one
pipeline that runs on **CASME II**, **SAMM** and **CAS(ME)³**.

Micro-expression datasets differ in frame rate, resolution, folder layout,
annotation format and annotation quirks, so code written for one usually does
not run on another. This repository keeps every dataset-specific detail in a
single registry file and exposes the dataset as a command-line argument:

```bash
bash scripts/run.sh casme2 full
bash scripts/run.sh samm   full
bash scripts/run.sh casme3 full
```

Everything downstream — cropping, alignment, optical flow, training, evaluation
— is shared, so a change is tested on all three datasets at once.

---

## Contents

- [Method in brief](#method-in-brief)
- [Installation](#installation)
- [Try it without any dataset](#try-it-without-any-dataset)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
- [Training and evaluation](#training-and-evaluation)
- [Checkpoints and Grad-CAM](#checkpoints-and-grad-cam)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements and license](#acknowledgements-and-license)

---

## Method in brief

**Input.** Each clip becomes one six-channel 224×224 tensor:

| # | channel | content |
|---|---------|---------|
| 0 | `G_amp` | motion-magnified grayscale apex |
| 1 | `M_on`  | TV-L1 flow magnitude, onset → apex |
| 2 | `M_off` | TV-L1 flow magnitude, offset → apex |
| 3 | `U_on`  | signed horizontal flow component |
| 4 | `V_on`  | signed vertical flow component |
| 5 | `S_on`  | optical strain |

Two more channels (`U_off`, `V_off`) are stored for ablations.

**Model.** A shared ResNet-18 stem splits the aligned face at `h//2` into an
upper-face and a lower-face branch. An Area Weighting Module re-weights the
concatenated branch features, and a class-aware gate fuses the branch logits in
logit space, so different classes may lean on different regions. A joint head
over the concatenated features captures AU combinations that span both regions.

**Two properties worth stating up front:**

*No per-clip statistics anywhere.* Flow is scaled by a fixed pixel constant that
is calibrated once per dataset and stored. Per-sample min-max normalisation, a
common choice in this field, maps a five-pixel motion and a half-pixel motion to
identical images, discarding motion intensity — which is one of the strongest
cues separating surprise from other classes — and it makes normalisation depend
on the test sample itself.

*Head motion is removed twice.* ECC affine alignment before the flow is
computed, and a robust (Cauchy-weighted) global affine fit subtracted from the
flow field afterwards. Rigid head motion on these datasets is often an order of
magnitude larger than the muscle motion of interest, and it correlates with
subject identity rather than class, so it directly fuels cross-subject
overfitting.

---

## Installation

Python 3.9+ and a CUDA GPU are recommended (CPU works, slowly).

```bash
git clone https://github.com/<your-account>/carf-net.git
cd carf-net

# PyTorch, matching your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

> **Exactly one OpenCV package may be installed, and it must be the contrib
> build.** TV-L1 optical flow (`cv2.optflow`) and the LBF face-landmark model
> (`cv2.face`) exist only there, and installing `opencv-python` alongside it
> silently removes them:
>
> ```bash
> pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
> pip install "opencv-contrib-python<5"
> ```

Check the install:

```bash
python -c "import cv2, torch; print(cv2.__version__, hasattr(cv2.face,'createFacemarkLBF'), hasattr(cv2,'optflow'), torch.cuda.is_available())"
# e.g. 4.14.0 True True True
```

Model files (YuNet face detector, LBF landmarks, and the motion-magnification
checkpoint) are downloaded on first use into `weights/` and `third_party/`.

---

## Try it without any dataset

The datasets are licensed and cannot be redistributed, so the repository ships
a generator that fabricates a miniature stand-in — frames **and** annotation
spreadsheets in each dataset's own format — and a script that runs the whole
pipeline on it:

```bash
bash tools/smoke_test.sh /tmp/carf_smoke cpu
```

This exercises prepare → crop → build → train and verifies that a saved
checkpoint reproduces its stored probabilities. It is a plumbing test: the
accuracy it prints is meaningless.

---

## Datasets

Request access from the dataset owners; nothing here is redistributed.

| dataset | fps | access |
|---|---|---|
| CASME II | 200 | [casme.psych.ac.cn/casme/e2](http://casme.psych.ac.cn/casme/e2) |
| SAMM | 200 | [release agreement](https://megc2022.github.io/files/SAMM_ReleaseAgreementV2.pdf) (Manchester Metropolitan University) |
| CAS(ME)³ | 30 | requested from the CASME group, see [MEGC2022](https://megc2022.github.io/challenge.html) |

Expected raw layouts (set the roots in `scripts/config.sh`):

```
CASME2_RAW_selected/sub01/EP03_02/img131.jpg
SAMM/006/006_1_2/006_05562.jpg
CASME3/spNo.1_a_355/355.jpg          <- subject_filename_onset
```

**Three-class mapping** (MEGC 2019 convention). `others` is discarded.

| dataset | positive | negative | surprise | total | subjects |
|---|---|---|---|---|---|
| CASME II | happiness 32 | disgust, repression 90 | 25 | 147 | 24 |
| SAMM | happiness 26 | anger, contempt, disgust, fear, sadness 92 | 15 | 133 | 28 |
| CAS(ME)³ Part A | happy 55 | disgust, fear, anger, sad 457 | 187 | 699 | 88 |

Five-class schemes for CASME II and SAMM are included (`CLASSES=5`). Other
schemes are a few lines in `SCHEMES`, `carfnet/registry.py`.

**Annotation quirks handled automatically**, and reported per clip:

- *Dead apex* (apex ≤ onset): 28 clips in CAS(ME)³, 1 in CASME II, 1 in SAMM.
  The apex is re-selected as the frame with the largest motion relative to the
  onset.
- *Invalid offset* (offset < apex, e.g. offset 0 in CAS(ME)³): repaired and flagged.
- *Case mismatch*: CAS(ME)³ labels say `spNO.1` while folders say `spNo.1`.
- *Frame-rate difference*: the rate-normalisation reference is the median
  onset–apex gap of each dataset (≈33, ≈32 and 6 frames), not a shared constant.

---

## Pipeline

```bash
bash scripts/run.sh <casme2|samm|casme3|all> <prepare|crop|magnify|build|train|quick|pre|full>
```

Stages already completed are skipped, so a run can be resumed.

| stage | what it does | output |
|---|---|---|
| `prepare` | read the annotation file, map emotions to classes, verify frames, repair annotations | `labels/<dataset>_<k>class.csv` |
| `crop` | detect the face and 68 LBF landmarks **once** on the onset frame, apply that box to every frame of the clip | `img<n>.jpg`, `meta.json` |
| `magnify` | learning-based motion magnification, onset → apex (optional) | `amplified.jpg` |
| `build` | alignment, optical flow, masking, channel stacking | `carf.npy` |
| `train` | LOSO training and evaluation | `runs/.../report.json` |

Inside `build`, per clip:

1. IOD-based similarity normalisation onto a canonical 224×224 template, chosen
   so the `h//2` split line lands on the nose bridge for every subject — the
   two-branch design depends on that line being anatomically stable.
2. ECC affine alignment of apex and offset onto onset, restricted to the face mask.
3. Apex re-selection for dead-apex clips.
4. Multi-scale TV-L1 flow with a median filter.
5. Robust global affine motion compensation.
6. Rate normalisation (displacement ÷ gap × reference), so the magnitude channel
   measures intensity rather than annotation duration.
7. Convex-hull face mask with Gaussian feathering; strain is computed *before*
   masking, otherwise the mask edge is painted into the channel as a bright outline.
8. Fixed pixel-unit scaling, calibrated once per dataset and cached in
   `<processed>/<dataset>/carf_calib.json`.

Every stage prints diagnostics (landmark coverage, ECC convergence, flow
saturation, magnitude spread) and writes a per-clip JSON report.

---

## Training and evaluation

Evaluation is leave-one-subject-out. Metrics are accuracy, UF1 (macro F1) and
UAR (macro recall). Defaults:

| setting | value |
|---|---|
| epochs | 60 (5 warm-up, then cosine) |
| batch size | 32 |
| lr, backbone / heads | 2e-4 / 2e-3 |
| loss | logit-adjusted CE + AU-similarity contrastive term |
| regularisation | mixup, random erasing, small affine jitter, frozen BN statistics |
| seeds | 0, 1, 2 |
| protocol | `test_peek` |

Override through the environment: `EPOCHS=100 SEEDS=0 bash scripts/run.sh samm train`,
or call `python -m carfnet.train --help` directly.

---

## Checkpoints and Grad-CAM

Each fold saves the weights it reported, for the first seed by default:

```
runs/carf_<dataset>_<k>c/ckpt/fold-<subject>_seed0_<protocol>.pt
```

About 100 MB per file (≈2.4 GB for CASME II, ≈2.8 GB for SAMM, ≈8.8 GB for
CAS(ME)³). Use `SAVE_CKPT=none` to disable, `CKPT_SEEDS=all` for every seed.

Each file also stores that fold's test clips, labels and probabilities, so an
explanation is guaranteed to describe the model whose numbers were reported.
`tools/verify_checkpoints.py` re-runs a checkpoint and asserts the probabilities
match.

```python
from carfnet.checkpoint import load_checkpoint

model, ckpt = load_checkpoint("runs/carf_casme2_3c/ckpt/fold-01_seed0_test_peek.pt", "cuda:0")
# Grad-CAM targets: model.eyes_branch.layers[2], model.mouth_branch.layers[2]
# score to differentiate: model.predict(eyes, mouth)
# ckpt["test_clips"], ckpt["test_labels"], ckpt["test_probs"], ckpt["classes"]
```

---

## Repository layout

```
carfnet/
  registry.py     dataset specs, class schemes, frame lookup
  prepare.py      stage 0: annotations -> label CSV
  faces.py        face detection (YuNet/Haar) and 68 landmarks (LBF/dlib)
  crop.py         stage 1: crop faces, store landmarks
  magnify.py      stage 2: motion magnification (optional)
  optical.py      alignment, TV-L1 flow, global-motion compensation, strain
  build.py        stage 3: build the input tensors
  data.py         label loading, augmentation, LOSO split
  model.py        CARF-Net
  losses.py       logit-adjusted CE, AU-similarity contrastive loss, mixup
  train.py        LOSO training and reporting
  checkpoint.py   per-fold checkpoints
scripts/          config.sh and the run.sh entry point
tools/            synthetic-data generator, smoke test, checkpoint verifier
```

---

## Troubleshooting

| symptom | check |
|---|---|
| `Missing folders` during prepare | the `RAW_*` paths and the folder layout |
| many `Without landmarks` during crop | detection quality; try `--detector haar` or `--landmark dlib` |
| `flow saturation` above 2% during build | re-run build with `--recalibrate` |
| clips dropped at train time | their `carf.npy` was never built; re-run build |
| `cv2.face` or `createFacemarkLBF` missing | more than one OpenCV package is installed, see [Installation](#installation) |
| `No module named 'numpy.distutils'` | you are installing pinned 2021 packages from some other requirements file; use this repository's `requirements.txt` |

---

## Acknowledgements and license

This repository is the reference implementation of **CARF-Net**, released
alongside our manuscript. The paper is not published yet; once it appears,
please cite it when you use this code in academic work. Author and citation
details are in `CITATION.cff`.

The implementation draws on prior open-source work in micro-expression
recognition, and is built with the following components, each used under its
own license: PyTorch and torchvision, OpenCV (contrib modules, for TV-L1
optical flow and facial landmark fitting), NumPy, pandas, scikit-learn, Pillow
and tqdm. Pretrained face-detection, landmark and motion-magnification models
are fetched from their original distributors on first use and are **not**
redistributed here.

The datasets are third-party and licensed. Obtain them from their owners and
follow the terms of their release agreements. No frames or annotations are
included in this repository, and `.gitignore` excludes `*.xlsx` and `labels/`
so they cannot be committed by accident.

**License:** not yet decided. Add a `LICENSE` file before making the repository
public.
