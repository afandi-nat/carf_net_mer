"""Stage 1 - crop faces from the raw frames into a uniform processed layout.

The face box and the 68 landmarks are computed ONCE, on the onset frame, and
the same box is applied to every frame of that clip so no drift is introduced
between frames. The landmarks are stored in `meta.json` in crop coordinates, so
the build stage never has to re-detect landmarks on an already cropped face.

Frames exported per clip: onset, apex and offset, plus a +-half-window around
each when requested, plus the range onset+1 .. onset+scan_cap for dead-apex
clips (and for clips with a short onset-apex gap when `--reselect-gap` is set),
which is the material the build stage uses to re-select an apex.

Usage:
    python -m carfnet.crop --dataset casme3 --raw-root /data/CASME3 \
        --processed-root /data/carf_processed
"""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from carfnet.faces import FaceDetector, Landmarker, plausible
from carfnet.registry import get_spec, index_frames


def frames_needed(row, half_window, reselect_gap, scan_cap):
    onset, apex, offset = int(row["Onset"]), int(row["Apex"]), int(row["Offset"])
    wanted = set()
    for n in (onset, apex, offset):
        wanted.update(range(n - half_window, n + half_window + 1))
    dead = "dead_apex" in str(row.get("flag", ""))
    if dead or (reselect_gap > 0 and apex - onset <= reselect_gap):
        wanted.update(range(onset + 1, min(offset, onset + scan_cap) + 1))
    return sorted(wanted)


def expand_box(x0, y0, x1, y1, margin, width, height):
    w, h = x1 - x0, y1 - y0
    return (
        int(max(0, np.floor(x0 - margin * w))),
        int(max(0, np.floor(y0 - margin * h * 1.6))),   # a little more forehead
        int(min(width, np.ceil(x1 + margin * w))),
        int(min(height, np.ceil(y1 + margin * h * 0.6))),
    )


def main(args):
    spec = get_spec(args.dataset)
    data = pd.read_csv(args.csv_file, dtype={"Subject": str, "Filename": str})
    detector = FaceDetector(args.detector)
    landmarker = Landmarker(args.landmark)
    scan_cap = args.scan_cap or spec.scan_cap
    print(f">> {spec.title}: detector {detector.kind}, landmarks {args.landmark}, "
          f"scan cap {scan_cap}")

    subject_box = {}
    stats = {"ok": 0, "skipped": 0, "no_landmark": 0, "cached_box": 0}
    failures = []

    for idx in tqdm(range(len(data)), desc=f"Cropping {spec.name}"):
        row = data.loc[idx]
        src_dir = os.path.join(args.raw_root, row["raw_dir"])
        out_dir = os.path.join(args.processed_root, row["clip_dir"])
        meta_path = os.path.join(out_dir, "meta.json")
        wanted = frames_needed(row, args.half_window, args.reselect_gap, scan_cap)

        if not args.overwrite and os.path.isfile(meta_path) and all(
                os.path.isfile(os.path.join(out_dir, f"img{n}.jpg"))
                for n in (row["Onset"], row["Apex"], row["Offset"])):
            stats["skipped"] += 1
            continue

        frames = index_frames(src_dir)
        onset_path = frames.get(int(row["Onset"]))
        if onset_path is None:
            failures.append(f"{row['raw_dir']}: onset frame missing")
            continue
        onset = cv2.imread(onset_path, cv2.IMREAD_COLOR)
        if onset is None:
            failures.append(f"{row['raw_dir']}: onset frame unreadable")
            continue
        height, width = onset.shape[:2]

        box = detector(onset)
        pts = None
        if box is not None:
            pts = landmarker(onset, box)
            if not plausible(pts, box):
                pts = None

        if box is None:
            crop = subject_box.get(row["Subject"])
            if crop is None:
                failures.append(f"{row['raw_dir']}: no face detected")
                continue
            stats["cached_box"] += 1
        elif pts is not None:
            (lx0, ly0), (lx1, ly1) = pts.min(0), pts.max(0)
            crop = expand_box(lx0, ly0, lx1, ly1, args.margin, width, height)
        else:
            x, y, w, h = box
            crop = expand_box(x, y, x + w, y + h, args.margin * 0.5, width, height)
        if pts is None:
            stats["no_landmark"] += 1
        subject_box[row["Subject"]] = crop

        x0, y0, x1, y1 = crop
        os.makedirs(out_dir, exist_ok=True)
        written = []
        for n in wanted:
            path = frames.get(n)
            if path is None:
                continue
            img = onset if n == int(row["Onset"]) else cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (height, width):
                continue
            cv2.imwrite(os.path.join(out_dir, f"img{n}.jpg"), img[y0:y1, x0:x1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            written.append(n)

        with open(meta_path, "w") as handle:
            json.dump({
                "raw_dir": row["raw_dir"],
                "crop_box": [int(v) for v in crop],
                "raw_size": [int(width), int(height)],
                "detector": detector.kind if box is not None else "subject_cache",
                "landmark": args.landmark if pts is not None else None,
                "landmarks": None if pts is None else (pts - [x0, y0]).round(2).tolist(),
                "frames": written,
            }, handle)
        stats["ok"] += 1

    print(f"\nCropped            : {stats['ok']}")
    print(f"Skipped (existing) : {stats['skipped']}")
    print(f"Without landmarks  : {stats['no_landmark']}  (build falls back to an ellipse mask)")
    print(f"Reused subject box : {stats['cached_box']}")
    print(f"Failures           : {len(failures)}")
    for item in failures[:20]:
        print(f"    {item}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--processed-root", required=True)
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--labels-dir", default="labels")
    p.add_argument("--csv-file", default="")
    p.add_argument("--detector", default="auto", choices=["auto", "yunet", "haar"])
    p.add_argument("--landmark", default="lbf", choices=["lbf", "dlib"])
    p.add_argument("--margin", type=float, default=0.12)
    p.add_argument("--half-window", type=int, default=0,
                   help="Also export +-N neighbouring frames for temporal averaging.")
    p.add_argument("--reselect-gap", type=int, default=0,
                   help="Also export the scan range for clips whose onset-apex gap <= N.")
    p.add_argument("--scan-cap", type=int, default=0,
                   help="Frames after onset scanned for apex re-selection. 0 = dataset default.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if not args.csv_file:
        args.csv_file = os.path.join(
            args.labels_dir, f"{get_spec(args.dataset).name}_{args.classes}class.csv")
    return args


if __name__ == "__main__":
    main(parse_args())
