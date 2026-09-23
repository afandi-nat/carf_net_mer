"""Stage 0 - read the official annotations, map emotions to classes, check frames.

Output: `labels/<dataset>_<k>class.csv`, one row per clip, consumed by every
later stage. The annotation files themselves are never redistributed with this
repository; obtain them from the dataset owners.

Annotation fixes applied here are recorded in the `flag` column:

    dead_apex     apex <= onset, or apex beyond offset. The build stage will
                  re-select the apex as the frame with the largest motion.
                  CAS(ME)3 has 28 such clips in the 3-class subset, CASME II 1,
                  SAMM 1.
    offset_fixed  offset < apex (e.g. offset 0 in CAS(ME)3 spNO.40/e). The
                  offset becomes apex + (apex - onset), clipped to the last
                  available frame.

Usage:
    python -m carfnet.prepare --dataset casme2 \
        --label-file /data/CASME2-coding-20140508.xlsx \
        --raw-root /data/CASME2_RAW_selected --classes 3
"""

import argparse
import os

import numpy as np
import pandas as pd

from carfnet.registry import DirResolver, get_scheme, get_spec, index_frames

COLUMNS = ["dataset", "Subject", "Filename", "clip", "Onset", "Apex", "Offset",
           "Action Units", "Label", "str_label", "emo_label", "raw_dir",
           "clip_dir", "flag"]


def build_table(args):
    spec = get_spec(args.dataset)
    scheme = get_scheme(spec.name, args.classes)
    df = spec.reader(args.label_file)

    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df["str_label"] = df["Label"].map(scheme["map"])
    dropped = df[df["str_label"].isna()]["Label"].value_counts()
    df = df[df["str_label"].notna()].copy()
    df["emo_label"] = df["str_label"].map({n: i for i, n in enumerate(scheme["names"])})

    for col in ("Onset", "Apex", "Offset"):
        df[col] = df[col].astype(int)
    df["Action Units"] = df["Action Units"].map(
        lambda v: "" if pd.isna(v) else str(v).strip())
    df["dataset"] = spec.name
    df["clip"] = df.apply(spec.clip_name, axis=1)
    df["clip_dir"] = df.apply(lambda r: f"{spec.name}/{r['Subject']}/{r['clip']}", axis=1)
    df["raw_dir"] = df.apply(spec.raw_dir, axis=1)
    df["flag"] = ""

    check = bool(args.raw_root) and os.path.isdir(args.raw_root)
    if not check:
        print(f">> WARNING: raw root not found ({args.raw_root}); frames are not verified.")
    resolver = DirResolver(args.raw_root or "", spec.case_insensitive_dirs)

    missing_dir, frame_issues, keep = [], [], []
    for idx, row in df.iterrows():
        onset, apex, offset = row["Onset"], row["Apex"], row["Offset"]
        flags, last, frames = [], None, {}

        if check:
            rel = resolver(row["raw_dir"])
            if rel is None:
                missing_dir.append(row["raw_dir"])
                keep.append(False)
                continue
            df.at[idx, "raw_dir"] = rel
            frames = index_frames(os.path.join(args.raw_root, rel))
            last = max(frames) if frames else None

        if offset < apex:
            gap = max(1, apex - onset)
            offset = apex + gap if last is None else min(last, apex + gap)
            flags.append("offset_fixed")
        if apex <= onset or apex > offset:
            flags.append("dead_apex")

        if check:
            if onset not in frames:
                frame_issues.append(f"{rel}: onset frame {onset} missing")
                keep.append(False)
                continue
            if apex not in frames and "dead_apex" not in flags:
                frame_issues.append(f"{rel}: apex frame {apex} missing")
                flags.append("dead_apex")
            if offset not in frames:
                frame_issues.append(f"{rel}: offset frame {offset} missing")
                below = [n for n in frames if n <= offset]
                if below:
                    offset = max(below)
                    flags.append("offset_fixed")

        df.at[idx, "Offset"] = offset
        df.at[idx, "flag"] = "+".join(dict.fromkeys(flags))
        keep.append(True)

    df = df[np.array(keep, dtype=bool)].reset_index(drop=True)
    return spec, scheme, df[COLUMNS], dropped, missing_dir, frame_issues


def main(args):
    spec, scheme, df, dropped, missing_dir, frame_issues = build_table(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n=== {spec.title}, {args.classes} classes ===")
    print(f"Clips kept        : {len(df)}")
    print(f"Subjects          : {df['Subject'].nunique()}")
    print("Per class         :")
    for i, name in enumerate(scheme["names"]):
        print(f"    {i} {name:10s} {int((df['emo_label'] == i).sum())}")
    if len(dropped):
        print("Dropped (outside the scheme): "
              + ", ".join(f"{k} {v}" for k, v in dropped.items()))
    print(f"Dead apex         : {df['flag'].str.contains('dead_apex').sum()} "
          f"(apex re-selected during build)")
    print(f"Offsets repaired  : {df['flag'].str.contains('offset_fixed').sum()}")
    gaps = (df["Apex"] - df["Onset"])[~df["flag"].str.contains("dead_apex")]
    if len(gaps):
        print(f"Onset-apex gap    : median {gaps.median():.0f} frames "
              f"(used as the default rate reference)")
    if missing_dir:
        print(f"Missing folders   : {len(missing_dir)}")
        for item in missing_dir[:10]:
            print(f"    {item}")
    if frame_issues:
        print(f"Frame issues      : {len(frame_issues)}")
        for item in frame_issues[:10]:
            print(f"    {item}")
    print(f"\nWritten to {args.out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, help="casme2, samm or casme3")
    p.add_argument("--label-file", required=True, help="official annotation .xlsx")
    p.add_argument("--raw-root", default="", help="root of the raw frame folders")
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--labels-dir", default="labels")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    if not args.out:
        args.out = os.path.join(
            args.labels_dir, f"{get_spec(args.dataset).name}_{args.classes}class.csv")
    return args


if __name__ == "__main__":
    main(parse_args())
