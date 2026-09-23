"""Dataset registry.

Every dataset-specific detail lives here: how to read the official annotation
file, how emotions map to classes, how raw frame folders are laid out, and how
frame files are named. Adding a new dataset means adding one entry to `SPECS`
and one class mapping to `SCHEMES`.

Supported raw layouts:

    casme2   <raw>/sub01/EP03_02/img131.jpg
    samm     <raw>/006/006_1_2/006_05562.jpg
    casme3   <raw>/spNo.1_a_355/355.jpg        (subject_filename_onset)

Unified processed layout produced by the pipeline:

    <processed>/<dataset>/<subject>/<clip>/img<n>.jpg
                                          /meta.json
                                          /amplified.jpg
                                          /carf.npy
"""

import os
import re
from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------------------- #
# Class schemes
# --------------------------------------------------------------------------- #
# The order of `names` defines the integer label. Emotions absent from `map`
# are dropped from that scheme.
#
# The three-class schemes follow the MEGC 2019 composite protocol:
#   CASME II : sadness and fear are NOT folded into negative, which reproduces
#              the 147-sample split used by Ruan et al. (2022).
#   SAMM     : anger, contempt, disgust, fear, sadness -> negative (133 samples).
#   CAS(ME)3 : happy -> positive; disgust, fear, anger, sad -> negative.
SCHEMES = {
    "casme2": {
        3: {
            "names": ["positive", "negative", "surprise"],
            "map": {
                "happiness": "positive",
                "disgust": "negative",
                "repression": "negative",
                "surprise": "surprise",
            },
        },
        5: {
            "names": ["disgust", "happiness", "others", "repression", "surprise"],
            "map": {k: k for k in
                    ["disgust", "happiness", "others", "repression", "surprise"]},
        },
    },
    "samm": {
        3: {
            "names": ["positive", "negative", "surprise"],
            "map": {
                "happiness": "positive",
                "anger": "negative",
                "contempt": "negative",
                "disgust": "negative",
                "fear": "negative",
                "sadness": "negative",
                "surprise": "surprise",
            },
        },
        5: {
            "names": ["anger", "happiness", "contempt", "other", "surprise"],
            "map": {k: k for k in
                    ["anger", "happiness", "contempt", "other", "surprise"]},
        },
    },
    "casme3": {
        3: {
            "names": ["positive", "negative", "surprise"],
            "map": {
                "happy": "positive",
                "disgust": "negative",
                "fear": "negative",
                "anger": "negative",
                "sad": "negative",
                "surprise": "surprise",
            },
        },
    },
}


# --------------------------------------------------------------------------- #
# Annotation readers
# --------------------------------------------------------------------------- #
def _read_casme2(path):
    """CASME2-coding-*.xlsx"""
    df = pd.read_excel(path, sheet_name=0, dtype={"Subject": str})
    df = df.rename(columns={"OnsetFrame": "Onset", "ApexFrame": "Apex",
                            "OffsetFrame": "Offset", "Estimated Emotion": "Label"})
    df["Subject"] = df["Subject"].str.strip().str.zfill(2)
    return df


def _read_samm(path):
    """SAMM_Micro_FACS_Codes_*.xlsx. The table header sits below a notes block,
    so it is located rather than hard-coded."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_row = next(i for i in range(len(raw))
                      if str(raw.iloc[i, 0]).strip() == "Subject")
    df = pd.read_excel(path, sheet_name=0, header=header_row, dtype={"Subject": str})
    df = df.rename(columns={"Onset Frame": "Onset", "Apex Frame": "Apex",
                            "Offset Frame": "Offset", "Estimated Emotion": "Label"})
    df["Subject"] = df["Subject"].str.strip().str.zfill(3)
    return df


def _read_casme3(path):
    """cas(me)3_part_A_ME_label_JpgIndex_*.xlsx, sheet `label`."""
    df = pd.read_excel(path, sheet_name="label")
    df = df.rename(columns={"AU": "Action Units", "emotion": "Label"})
    df["Subject"] = df["Subject"].astype(str).str.strip()
    df["Filename"] = df["Filename"].astype(str).str.strip()
    return df


# --------------------------------------------------------------------------- #
# Raw folders
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    name: str
    title: str
    fps: int
    reader: callable
    raw_dir: callable
    clip_name: callable
    #: CAS(ME)3 writes `spNO` in the labels but `spNo` in folder names
    case_insensitive_dirs: bool = False
    #: cap on how many frames after onset are scanned when re-selecting an apex
    scan_cap: int = 40

    def schemes(self):
        return SCHEMES[self.name]


SPECS = {
    "casme2": DatasetSpec(
        name="casme2", title="CASME II", fps=200, reader=_read_casme2,
        raw_dir=lambda r: f"sub{r['Subject']}/{r['Filename']}",
        clip_name=lambda r: str(r["Filename"]),
        scan_cap=60,
    ),
    "samm": DatasetSpec(
        name="samm", title="SAMM", fps=200, reader=_read_samm,
        raw_dir=lambda r: f"{r['Subject']}/{r['Filename']}",
        clip_name=lambda r: str(r["Filename"]),
        scan_cap=60,
    ),
    "casme3": DatasetSpec(
        name="casme3", title="CAS(ME)3 Part A", fps=30, reader=_read_casme3,
        raw_dir=lambda r: f"{r['Subject']}_{r['Filename']}_{int(r['Onset'])}",
        # one CAS(ME)3 filename can hold several MEs, so the onset is part of
        # the clip identifier
        clip_name=lambda r: f"{r['Filename']}_{int(r['Onset'])}",
        case_insensitive_dirs=True,
        scan_cap=20,
    ),
}


def get_spec(name):
    key = name.lower().replace("(", "").replace(")", "").replace("_", "")
    key = {"casmeii": "casme2", "casme3parta": "casme3"}.get(key, key)
    if key not in SPECS:
        raise SystemExit(f"Unknown dataset: {name}. Choose from: {', '.join(SPECS)}")
    return SPECS[key]


def get_scheme(dataset, num_classes):
    schemes = get_spec(dataset).schemes()
    if num_classes not in schemes:
        raise SystemExit(
            f"No {num_classes}-class scheme defined for {dataset}. "
            f"Available: {sorted(schemes)}. Add one in carfnet/registry.py."
        )
    return schemes[num_classes]


# --------------------------------------------------------------------------- #
# Frame lookup
# --------------------------------------------------------------------------- #
FRAME_RE = re.compile(r"(\d+)\.(jpe?g|png|bmp)$", re.IGNORECASE)


def index_frames(clip_dir):
    """Map frame number -> path, using the last number in the file name.

    Works for `img131.jpg`, `006_05562.jpg` and `355.jpg` alike, so no
    per-dataset naming pattern is needed.
    """
    table = {}
    try:
        names = sorted(os.listdir(clip_dir), key=len)
    except FileNotFoundError:
        return table
    for name in names:
        match = FRAME_RE.search(name)
        if match:
            table.setdefault(int(match.group(1)), os.path.join(clip_dir, name))
    return table


class DirResolver:
    """Resolve a clip folder, optionally ignoring letter case."""

    def __init__(self, root, case_insensitive):
        self.root = root
        self.case_insensitive = case_insensitive
        self._index = None

    def __call__(self, rel):
        if os.path.isdir(os.path.join(self.root, rel)):
            return rel
        if not self.case_insensitive:
            return None
        if self._index is None:
            self._index = {}
            for dirpath, dirnames, _ in os.walk(self.root):
                depth = os.path.relpath(dirpath, self.root).count(os.sep)
                for d in dirnames:
                    rel_d = os.path.relpath(os.path.join(dirpath, d), self.root)
                    self._index.setdefault(rel_d.lower(), rel_d)
                if depth >= 1:
                    dirnames[:] = []
        return self._index.get(rel.lower())


def extract_aus(text):
    """AU numbers from strings such as 'R14A or 17A', '4+L10', 'A1B+A2C'."""
    return sorted({int(x) for x in re.findall(r"\d+", str(text)) if int(x) < 100})
