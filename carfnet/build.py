"""Stage 3 - build the CARF-Net input tensor for every clip (`carf.npy`).

Per-clip pipeline:

  1. 68 landmarks read from `meta.json` (computed once during cropping).
  2. IOD-based similarity normalisation onto a 224x224 canonical template; the
     h//2 split line lands on the nose bridge for every subject.
  3. ECC affine alignment of the apex and offset frames onto the onset frame,
     restricted to the face mask.
  4. Apex re-selection for dead-apex clips: the frame with the largest motion
     relative to the onset.
  5. Multi-scale TV-L1 flow, onset->apex and offset->apex, plus a median filter.
  6. Robust (Cauchy) global affine compensation of head motion.
  7. Rate normalisation: displacement divided by the onset-apex gap and scaled
     by a fixed reference gap, so the channel measures motion intensity rather
     than annotation duration.
  8. Convex-hull face mask with Gaussian feathering.
  9. Fixed pixel-unit scaling (`flow_clip`, `strain_clip`); no per-clip statistic
     is ever used, which keeps magnitudes comparable across clips and folds.
 10. Channel stacking, saved as float16.

Channels (the first six are the CARF-Net input):

    0 G_amp  amplified grayscale       [0, 1]
    1 M_on   magnitude onset->apex     [0, 1]
    2 M_off  magnitude offset->apex    [0, 1]
    3 U_on   u component onset->apex   [-1, 1]
    4 V_on   v component onset->apex   [-1, 1]
    5 S_on   optical strain            [0, 1]
    6 U_off, 7 V_off                   [-1, 1]  (kept for ablations)

`--flow-clip`/`--strain-clip` default to "auto": calibrated once per dataset as
the 90th percentile over per-clip 99th percentiles, cached in
`<processed>/<dataset>/carf_calib.json` and reused. Calibration never looks at
labels.

Usage:
    python -m carfnet.build --dataset casme2 --processed-root /data/carf_processed
"""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from carfnet.optical import (
    SIZE, apply_ecc, ecc_align, fallback_geometry, feather_mask, make_tvl1,
    masks_from_landmarks, multiscale_flow, read_gray_window, remove_global_affine,
    similarity_matrix, strain_from_flow, warp,
)
from carfnet.registry import get_spec


def reselect_apex(base, onset_n, candidates, matrix, hard, args):
    """Return the candidate frame with the largest masked motion w.r.t. onset."""
    onset, _ = read_gray_window(base, onset_n, 0)
    onset = warp(onset.astype(np.uint8), matrix)
    best, best_score, scanned = None, -1.0, 0
    for n in candidates:
        gray, _ = read_gray_window(base, n, 0)
        if gray is None:
            continue
        scanned += 1
        moving = warp(gray.astype(np.uint8), matrix)
        if args.ecc != "off":
            moving, _ = ecc_align(onset, moving, hard, args.ecc)
        flow = make_tvl1().calc(onset, moving, None)
        if args.global_comp:
            flow, _ = remove_global_affine(flow, hard)
        score = float(np.hypot(flow[..., 0], flow[..., 1])[hard].mean())
        if score > best_score:
            best, best_score = n, score
    return best, scanned


def compute_flows(base, row, args, rate_ref):
    """Steps 1-8. Returns flow fields in pixels plus diagnostics."""
    onset_n, apex_n, offset_n = int(row["Onset"]), int(row["Apex"]), int(row["Offset"])
    for n in (onset_n, offset_n):
        if not os.path.isfile(f"{base}/img{n}.jpg"):
            raise FileNotFoundError(f"img{n}.jpg is missing")

    meta_path = f"{base}/meta.json"
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    onset_rgb = np.array(Image.open(f"{base}/img{onset_n}.jpg").convert("RGB"))

    if meta.get("landmarks"):
        pts = np.array(meta["landmarks"], dtype=np.float32)
        matrix = similarity_matrix(pts)
        hard, soft = masks_from_landmarks(pts, matrix, args.feather)
        aligned = True
    else:
        matrix, hard = fallback_geometry(onset_rgb.shape)
        soft = feather_mask(hard, args.feather)
        aligned = False

    annotated_apex = apex_n
    dead = (apex_n <= onset_n or apex_n > offset_n
            or not os.path.isfile(f"{base}/img{apex_n}.jpg"))
    scanned = 0
    if dead or (args.reselect_gap > 0 and apex_n - onset_n <= args.reselect_gap):
        cands = [n for n in range(onset_n + 1, offset_n + 1)
                 if os.path.isfile(f"{base}/img{n}.jpg")]
        new_apex, scanned = reselect_apex(base, onset_n, cands, matrix, hard, args) \
            if cands else (None, 0)
        if new_apex is not None:
            apex_n = new_apex
        elif dead:
            # Nothing to scan: fall back to the frame closest to the midpoint.
            middle = (onset_n + offset_n) // 2
            available = [n for n in (middle + np.arange(-3, 4))
                         if n != onset_n and os.path.isfile(f"{base}/img{n}.jpg")]
            apex_n = min(available, key=lambda n: abs(n - middle), default=offset_n)

    gap = max(1, apex_n - onset_n)
    half = min(args.half_window, max(0, (gap - 1) // 2))
    grays, counts = {}, {}
    for name, n in (("onset", onset_n), ("apex", apex_n), ("offset", offset_n)):
        gray, count = read_gray_window(base, n, half)
        if gray is None:
            raise FileNotFoundError(f"could not read the {name} frame")
        grays[name] = warp(gray.astype(np.uint8), matrix)
        counts[name] = count

    ecc_ok, warp_apex = {}, None
    if args.ecc != "off":
        grays["apex"], warp_apex = ecc_align(grays["onset"], grays["apex"], hard, args.ecc)
        grays["offset"], warp_off = ecc_align(grays["onset"], grays["offset"], hard, args.ecc)
        ecc_ok = {"apex": warp_apex is not None, "offset": warp_off is not None}

    on_flow = multiscale_flow(grays["onset"], grays["apex"], blur=args.median_blur)
    off_flow = multiscale_flow(grays["offset"], grays["apex"], blur=args.median_blur)
    if args.global_comp:
        on_flow, _ = remove_global_affine(on_flow, hard)
        off_flow, _ = remove_global_affine(off_flow, hard)

    if args.rate_normalize:
        on_flow = on_flow * (rate_ref / max(1, apex_n - onset_n))
        off_flow = off_flow * (rate_ref / max(1, offset_n - apex_n))

    # Strain is computed BEFORE masking: masking first makes the mask edge look
    # like a strong gradient and paints a bright outline into the channel.
    strain = strain_from_flow(on_flow) * soft
    soft3 = soft[..., None]
    on_flow, off_flow = on_flow * soft3, off_flow * soft3

    amp_path = f"{base}/amplified.jpg"
    if os.path.isfile(amp_path) and apex_n == annotated_apex:
        amp = np.array(Image.open(amp_path).convert("L"))
        amp = cv2.resize(amp, (onset_rgb.shape[1], onset_rgb.shape[0]))
        ch0 = apply_ecc(warp(amp, matrix), warp_apex)
        used_amp = True
    else:
        ch0 = grays["apex"]
        used_amp = False

    info = {
        "aligned": aligned,
        "apex_annotated": int(annotated_apex),
        "apex_used": int(apex_n),
        "reselected": bool(apex_n != annotated_apex),
        "frames_scanned": int(scanned),
        "ecc": ecc_ok,
        "used_amplified": used_amp,
        "frames_averaged": counts,
        "mask_ratio": float(hard.mean()),
    }
    return on_flow, off_flow, strain, ch0, hard, info


def to_channels(on_flow, off_flow, strain, ch0, flow_clip, strain_clip):
    def signed(component):
        return np.clip(component / flow_clip, -1.0, 1.0)

    def magnitude(flow):
        return np.clip(np.hypot(flow[..., 0], flow[..., 1]) / flow_clip, 0.0, 1.0)

    return np.stack([
        ch0.astype(np.float32) / 255.0,
        magnitude(on_flow),
        magnitude(off_flow),
        signed(on_flow[..., 0]),
        signed(on_flow[..., 1]),
        np.clip(strain / strain_clip, 0.0, 1.0),
        signed(off_flow[..., 0]),
        signed(off_flow[..., 1]),
    ], axis=-1).astype(np.float16)


def calibrate(data, args, rate_ref):
    """Measure the flow and strain scale on a spread-out sample of clips."""
    idxs = np.unique(np.linspace(0, len(data) - 1,
                                 min(args.calib_n, len(data))).astype(int))
    flows, strains = [], []
    for i in tqdm(idxs, desc="Calibrating scale"):
        row = data.loc[i]
        try:
            on_flow, _, strain, _, hard, _ = compute_flows(
                os.path.join(args.processed_root, row["clip_dir"]), row, args, rate_ref)
        except Exception:  # noqa: BLE001
            continue
        flows.append(float(np.percentile(np.abs(on_flow[hard]), 99)))
        strains.append(float(np.percentile(strain[hard], 99)))
    if not flows:
        raise SystemExit("Calibration failed: no clip could be read.")
    return (round(float(np.percentile(flows, 90)), 2),
            round(float(np.percentile(strains, 90)), 3), len(flows))


def resolve_scales(data, args, rate_ref, calib_path):
    if args.flow_clip != "auto" and args.strain_clip != "auto":
        return float(args.flow_clip), float(args.strain_clip)
    key = {k: getattr(args, k) for k in
           ("rate_normalize", "global_comp", "ecc", "half_window", "feather",
            "reselect_gap")}
    calib = json.load(open(calib_path)) if os.path.isfile(calib_path) else None
    if calib is None or calib.get("config") != key or args.recalibrate:
        flow_clip, strain_clip, n = calibrate(data, args, rate_ref)
        calib = {"flow_clip": flow_clip, "strain_clip": strain_clip,
                 "n_clips": n, "rate_ref": rate_ref, "config": key}
        os.makedirs(os.path.dirname(calib_path), exist_ok=True)
        json.dump(calib, open(calib_path, "w"), indent=2)
        print(f">> Calibrated on {n} clips: flow_clip {flow_clip}, "
              f"strain_clip {strain_clip} -> {calib_path}")
    else:
        print(f">> Reusing the cached calibration in {calib_path}")
    return (calib["flow_clip"] if args.flow_clip == "auto" else float(args.flow_clip),
            calib["strain_clip"] if args.strain_clip == "auto" else float(args.strain_clip))


def main(args):
    spec = get_spec(args.dataset)
    data = pd.read_csv(args.csv_file, dtype={"Subject": str, "Filename": str})
    dead = data["flag"].fillna("").str.contains("dead_apex")
    rate_ref = args.rate_ref or float((data["Apex"] - data["Onset"])[~dead].median())

    calib_path = os.path.join(args.processed_root, spec.name,
                              f"{args.out_name[:-4]}_calib.json")
    flow_clip, strain_clip = resolve_scales(data, args, rate_ref, calib_path)
    print(f">> {spec.title}: flow_clip {flow_clip}, strain_clip {strain_clip}, "
          f"rate_ref {rate_ref if args.rate_normalize else 'off'}, "
          f"ECC {args.ecc}, feather {args.feather}")

    reports, failures, skipped = {}, [], 0
    for i in tqdm(range(len(data)), desc=f"Building {args.out_name}"):
        row = data.loc[i]
        base = os.path.join(args.processed_root, row["clip_dir"])
        out_path = os.path.join(base, args.out_name)
        if not args.overwrite and os.path.isfile(out_path):
            skipped += 1
            continue
        try:
            on_flow, off_flow, strain, ch0, _, info = compute_flows(base, row, args, rate_ref)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{row['clip_dir']}: {exc}")
            continue
        stack = to_channels(on_flow, off_flow, strain, ch0, flow_clip, strain_clip)
        np.save(out_path, stack)
        info["mean_magnitude"] = float(stack[..., 1].astype(np.float32).mean())
        info["saturated_flow"] = float((np.abs(stack[..., 3:5]) >= 0.999).mean())
        reports[row["clip_dir"]] = info

    print(f"\nSkipped (existing) : {skipped}")
    print(f"Built              : {len(reports)}")
    print(f"Failures           : {len(failures)}")
    for item in failures[:20]:
        print(f"    {item}")

    if reports:
        vals = list(reports.values())
        print("\nDiagnostics:")
        print(f"    landmark-aligned   : {sum(v['aligned'] for v in vals)} of {len(vals)}")
        print(f"    apex re-selected   : {sum(v['reselected'] for v in vals)}")
        print(f"    used amplified     : {sum(v['used_amplified'] for v in vals)}")
        if args.ecc != "off":
            print(f"    ECC apex converged : {sum(v['ecc'].get('apex', False) for v in vals)}")
        mag = np.array([v["mean_magnitude"] for v in vals])
        sat = float(np.mean([v["saturated_flow"] for v in vals]))
        print(f"    magnitude spread   : min {mag.min():.4f} median {np.median(mag):.4f} "
              f"max {mag.max():.4f}")
        print(f"    flow saturation    : {sat:.4f}")
        if sat > 0.02:
            print("    WARNING: saturation above 2%; raise --flow-clip or --recalibrate.")
        report_path = os.path.join(args.processed_root, spec.name,
                                   args.out_name.replace(".npy", "_report.json"))
        json.dump(reports, open(report_path, "w"), indent=1)
        print(f"\nPer-clip report: {report_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--processed-root", required=True)
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--labels-dir", default="labels")
    p.add_argument("--csv-file", default="")
    p.add_argument("--out-name", default="carf.npy")
    p.add_argument("--flow-clip", default="auto")
    p.add_argument("--strain-clip", default="auto")
    p.add_argument("--calib-n", type=int, default=60)
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--rate-normalize", type=int, default=1)
    p.add_argument("--rate-ref", type=float, default=0.0,
                   help="0 = median onset-apex gap of this dataset.")
    p.add_argument("--ecc", default="affine", choices=["affine", "euclidean", "off"])
    p.add_argument("--feather", type=float, default=4.0,
                   help="Mask feathering sigma in pixels. 0 = hard mask.")
    p.add_argument("--global-comp", type=int, default=1)
    p.add_argument("--half-window", type=int, default=0,
                   help="Temporal averaging half-window; needs crop --half-window.")
    p.add_argument("--median-blur", type=int, default=5)
    p.add_argument("--reselect-gap", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if not args.csv_file:
        args.csv_file = os.path.join(
            args.labels_dir, f"{get_spec(args.dataset).name}_{args.classes}class.csv")
    return args


if __name__ == "__main__":
    main(parse_args())
