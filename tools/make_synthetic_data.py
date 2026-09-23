"""Create a small synthetic dataset for smoke-testing the pipeline.

The micro-expression datasets are licensed and cannot be redistributed, so this
script fabricates a miniature stand-in: it warps one face photo with local,
class-dependent deformations and writes both the frames and annotation
spreadsheets in each dataset's official format and folder layout.

It is a plumbing test, not a benchmark. Accuracy numbers obtained on it mean
nothing.

Usage:
    python tools/make_synthetic_data.py --out /tmp/carf_demo
    python tools/make_synthetic_data.py --out /tmp/carf_demo --face my_face.jpg
"""

import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carfnet.faces import FaceDetector, Landmarker  # noqa: E402

FACE_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
            "lena.jpg")

# subjects x clips per class; canvas size and colour mode per dataset
LAYOUT = {
    "casme2": {"canvas": (640, 480), "gray": False},
    "samm": {"canvas": (2040, 1088), "gray": True},
    "casme3": {"canvas": (1280, 720), "gray": False},
}
EMOTION = {
    "casme2": {"positive": "happiness", "negative": "disgust", "surprise": "surprise"},
    "samm": {"positive": "Happiness", "negative": "Anger", "surprise": "Surprise"},
    "casme3": {"positive": "happy", "negative": "disgust", "surprise": "surprise"},
}
AUS = {"positive": "12", "negative": "4+7", "surprise": "1+2+5"}


def load_face(path, url):
    if path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read {path}")
        return img
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sample_face.jpg")
    if not os.path.isfile(cache):
        print(f">> Downloading a sample face from {url}")
        urllib.request.urlretrieve(url, cache)
    return cv2.imread(cache, cv2.IMREAD_COLOR)


def deformation_fields(face):
    """Per-class displacement fields anchored on detected landmarks."""
    detector, landmarker = FaceDetector("auto"), Landmarker("lbf")
    box = detector(face)
    if box is None:
        raise SystemExit("No face detected in the source image.")
    pts = landmarker(face, box)
    if pts is None:
        raise SystemExit("Landmark fitting failed on the source image.")

    height, width = face.shape[:2]
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)

    def bump(centre, sigma=18.0):
        return np.exp(-((grid_x - centre[0]) ** 2 + (grid_y - centre[1]) ** 2)
                      / (2 * sigma ** 2))

    brow = pts[17:27].mean(0)
    mouth = pts[48:68].mean(0)
    left_corner, right_corner = pts[48], pts[54]
    zero = np.zeros_like(grid_x)

    fields = {
        # mouth corners pulled up and outwards
        "positive": (bump(right_corner) - bump(left_corner),
                     -(bump(left_corner) + bump(right_corner))),
        # brows pulled down
        "negative": (zero, 1.2 * bump(brow, 30)),
        # brows raised, mouth opened
        "surprise": (zero, -1.2 * bump(brow, 30) + 1.0 * bump(mouth + [0, 15], 20)),
    }
    return fields, grid_x, grid_y


def render(face, fields, grid_x, grid_y, cls, amplitude, shift, canvas, gray):
    dx, dy = fields[cls]
    amp = 3.0 * amplitude
    warped = cv2.remap(face,
                       (grid_x - amp * dx - shift[0]).astype(np.float32),
                       (grid_y - amp * dy - shift[1]).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    scale = canvas[1] * 0.6 / face.shape[0]
    warped = cv2.resize(warped, None, fx=scale, fy=scale)
    out = np.full((canvas[1], canvas[0], 3), 90, np.uint8)
    y0 = (canvas[1] - warped.shape[0]) // 2
    x0 = (canvas[0] - warped.shape[1]) // 2
    out[y0:y0 + warped.shape[0], x0:x0 + warped.shape[1]] = warped
    if gray:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return out


def clip_plan(rng, n_subjects, n_per_class, fps):
    """(subject index, clip index, class, onset, apex, offset) tuples."""
    classes = ["positive", "negative", "surprise"]
    step = max(2, fps // 8)
    plan = []
    for s in range(n_subjects):
        counter = 0
        for cls in classes:
            for _ in range(n_per_class):
                onset = int(rng.integers(20, 200))
                apex = onset + int(rng.integers(step, step * 2))
                offset = apex + int(rng.integers(step, step * 2))
                plan.append((s, counter, cls, onset, apex, offset))
                counter += 1
    return plan


def write_casme2(root, plan, make_frames):
    rows = []
    for s, c, cls, on, ap, off in plan:
        subject, clip = f"{s + 1:02d}", f"EP{c + 1:02d}_01"
        make_frames(f"{root}/frames/sub{subject}/{clip}", cls, on, ap, off,
                    lambda n: f"img{n}.jpg", "casme2")
        rows.append({"Subject": subject, "Filename": clip, "Unnamed: 2": None,
                     "OnsetFrame": on, "ApexFrame": ap, "OffsetFrame": off,
                     "Unnamed: 6": None, "Action Units": AUS[cls],
                     "Estimated Emotion": EMOTION["casme2"][cls]})
    path = f"{root}/CASME2-coding-synthetic.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def write_samm(root, plan, make_frames):
    rows = []
    for s, c, cls, on, ap, off in plan:
        subject, clip = f"{s + 6:03d}", f"{s + 6:03d}_{c + 1}_1"
        make_frames(f"{root}/frames/{subject}/{clip}", cls, on, ap, off,
                    lambda n, sub=subject: f"{sub}_{n:05d}.jpg", "samm")
        rows.append([subject, clip, 1, on, ap, off, off - on, "Micro - 1/2",
                     AUS[cls], EMOTION["samm"][cls], 3, ""])
    # SAMM keeps a notes block above the header row; reproduce that shape.
    preamble = [["SAMM Dataset - Synthetic FACS Codes"] + [""] * 11 for _ in range(13)]
    header = ["Subject", "Filename", "Inducement Code", "Onset Frame",
              "Apex Frame", "Offset Frame", "Duration", "Micro", "Action Units",
              "Estimated Emotion", "Objective Classes", "Notes"]
    table = pd.DataFrame(preamble + [header] + rows)
    path = f"{root}/SAMM_Micro_FACS_Codes_synthetic.xlsx"
    table.to_excel(path, index=False, header=False, sheet_name="MICRO_ONLY")
    return path


def write_casme3(root, plan, make_frames):
    rows = []
    for s, c, cls, on, ap, off in plan:
        # CAS(ME)3 writes spNO in the labels but spNo in the folder names
        subject, name = f"spNO.{s + 1}", chr(ord("a") + c)
        make_frames(f"{root}/frames/spNo.{s + 1}_{name}_{on}", cls, on, ap, off,
                    lambda n: f"{n}.jpg", "casme3")
        rows.append({"Subject": subject, "Filename": name, "Onset": on, "Apex": ap,
                     "Offset": off, "AU": AUS[cls], "Objective class": "II",
                     "emotion": EMOTION["casme3"][cls]})
    path = f"{root}/casme3_part_A_ME_label_synthetic.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="label", index=False)
    return path


def main(args):
    face = load_face(args.face, args.face_url)
    fields, grid_x, grid_y = deformation_fields(face)
    rng = np.random.default_rng(args.seed)

    def make_frames(clip_dir, cls, onset, apex, offset, name_fn, dataset):
        os.makedirs(clip_dir, exist_ok=True)
        canvas = LAYOUT[dataset]["canvas"]
        gray = LAYOUT[dataset]["gray"]
        base_shift = rng.uniform(-3, 3, 2)
        for n in range(max(0, onset - 2), offset + 3):
            if n <= apex:
                amplitude = np.clip((n - onset) / max(1, apex - onset), 0, 1)
            else:
                amplitude = np.clip(1 - (n - apex) / max(1, offset - apex), 0, 1)
            # slow head drift, so global-motion compensation has work to do
            shift = base_shift + 0.02 * (n - onset) * np.array([1.0, 0.5])
            img = render(face, fields, grid_x, grid_y, cls, amplitude, shift,
                         canvas, gray)
            cv2.imwrite(os.path.join(clip_dir, name_fn(n)), img)

    writers = {"casme2": write_casme2, "samm": write_samm, "casme3": write_casme3}
    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        root = os.path.join(args.out, dataset)
        os.makedirs(root, exist_ok=True)
        plan = clip_plan(rng, args.subjects, args.per_class,
                         30 if dataset == "casme3" else 200)
        label_path = writers[dataset](root, plan, make_frames)
        print(f"{dataset:8s} {len(plan)} clips  frames: {root}/frames  labels: {label_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--datasets", default="casme2,samm,casme3")
    p.add_argument("--subjects", type=int, default=3)
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--face", default="", help="source face image; downloaded if empty")
    p.add_argument("--face-url", default=FACE_URL)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
