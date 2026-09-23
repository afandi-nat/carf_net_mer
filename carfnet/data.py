"""Label loading, augmentation and the LOSO split.

Normalisation uses fixed constants rather than dataset or per-fold statistics,
which closes a subtle leakage path (statistics computed over data that includes
the test fold).

Horizontal flipping negates the signed u channels: mirroring an image reverses
horizontal displacement, so without the sign flip the augmentation would teach
the model that leftward and rightward motion are the same thing, erasing
exactly the direction information those channels exist to carry.
"""

import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from carfnet.registry import get_scheme, get_spec

# Channel 0 lives in [0, 1]; magnitude and strain in [0, 1]; signed channels in
# [-1, 1]. These constants centre everything near zero with O(1) spread.
MEAN = [0.45, 0.06, 0.06, 0.00, 0.00, 0.10, 0.00, 0.00]
STD = [0.25, 0.12, 0.12, 0.15, 0.15, 0.15, 0.15, 0.15]

#: channels whose sign must flip on a horizontal mirror
SIGNED_U = (3, 6)

MAX_AU = 100


def au_multi_hot(own_aus, exchanged_aus):
    """Multi-hot AU vector over the clip's own AUs plus those of the exchanged
    lower half, which is what the AU-similarity contrastive term compares."""
    vector = np.zeros((MAX_AU,), dtype=np.float32)
    for text in (own_aus, exchanged_aus):
        for au in re.findall(r"\d+", str(text)):
            if int(au) < MAX_AU:
                vector[int(au)] = 1.0
    return torch.from_numpy(vector)


def loso_folds(data, column="Subject"):
    """Leave-one-subject-out: (train frames, test frames, subject ids)."""
    train_list, test_list, subjects = [], [], sorted(data[column].unique())
    for subject in subjects:
        mask = data[column] == subject
        train_list.append(data[~mask].reset_index(drop=True))
        test_list.append(data[mask].reset_index(drop=True))
    return train_list, test_list, subjects


def load_labels(datasets, num_classes, labels_dir="labels"):
    """Concatenate the label CSVs of the selected datasets.

    When several datasets are combined, subject ids are prefixed with the
    dataset name so LOSO never mixes, say, CASME II subject 01 with another
    dataset's subject 01.
    """
    names, frames = None, []
    for ds in datasets:
        spec = get_spec(ds)
        scheme = get_scheme(spec.name, num_classes)
        if names is None:
            names = scheme["names"]
        elif names != scheme["names"]:
            raise SystemExit(
                f"The {num_classes}-class scheme of {spec.name} ({scheme['names']}) "
                f"differs from the other datasets ({names}); they cannot be combined."
            )
        path = os.path.join(labels_dir, f"{spec.name}_{num_classes}class.csv")
        if not os.path.isfile(path):
            raise SystemExit(f"{path} does not exist. Run the prepare stage for {spec.name}.")
        df = pd.read_csv(path, dtype={"Subject": str, "Filename": str})
        df["Action Units"] = df["Action Units"].fillna("").astype(str)
        if len(datasets) > 1:
            df["Subject"] = spec.name + "_" + df["Subject"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True), names


def drop_unbuilt(df, processed_root, npy_name):
    """Drop clips whose tensor has not been built yet, and say which."""
    exists = df["clip_dir"].map(
        lambda c: os.path.isfile(os.path.join(processed_root, c, npy_name)))
    missing = df[~exists]
    if len(missing):
        print(f">> WARNING: {len(missing)} clips have no {npy_name} and were dropped:")
        for clip in missing["clip_dir"].head(10):
            print(f"       {clip}")
    return df[exists].reset_index(drop=True)


class CARFDataset(Dataset):
    """One sample = (upper half, lower half) of a clip's tensor.

    During training the lower half is swapped, with probability 0.5, for the
    lower half of another clip carrying the same label, which is the
    annotation-mimicking augmentation of the original method.
    """

    def __init__(self, processed_root, frame, mode, num_channels=6,
                 npy_name="carf.npy", aug_strength=1.0, mag_jitter=0.0):
        self.data = frame.reset_index(drop=True)
        self.root = processed_root
        self.mode = mode
        self.npy_name = npy_name
        self.num_channels = num_channels
        self.is_train = "training" in mode
        self.aug_strength = aug_strength
        self.mag_jitter = mag_jitter
        self.mean = torch.tensor(MEAN[:num_channels]).view(-1, 1, 1)
        self.std = torch.tensor(STD[:num_channels]).view(-1, 1, 1)
        self.signed_idx = [c for c in SIGNED_U if c < num_channels]
        self._cache = {}

    def __len__(self):
        return len(self.data)

    def class_counts(self, num_classes, column="emo_label"):
        counts = torch.zeros(num_classes)
        for label, n in self.data.groupby(column).size().items():
            counts[int(label)] = float(n)
        return counts

    # ------------------------------------------------------------------ #
    def _raw(self, clip_dir):
        if clip_dir not in self._cache:
            arr = np.load(os.path.join(self.root, clip_dir, self.npy_name))
            arr = arr.astype(np.float32)[..., : self.num_channels]
            self._cache[clip_dir] = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        return self._cache[clip_dir].clone()

    def _flip(self, tensor):
        tensor = torch.flip(tensor, dims=[-1])
        for c in self.signed_idx:
            tensor[c] = -tensor[c]
        return tensor

    def _geometric(self, tensor):
        """Small affine jitter. Rotation technically also rotates the flow
        vectors; the angle is capped at five degrees so that error stays well
        below apex-annotation noise."""
        angle = random.uniform(-5.0, 5.0) * self.aug_strength
        tx = random.uniform(-0.04, 0.04) * self.aug_strength
        ty = random.uniform(-0.04, 0.04) * self.aug_strength
        scale = 1.0 + random.uniform(-0.05, 0.05) * self.aug_strength
        theta = np.deg2rad(angle)
        cos, sin = np.cos(theta) / scale, np.sin(theta) / scale
        matrix = torch.tensor([[cos, -sin, tx], [sin, cos, ty]],
                              dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(matrix, (1, *tensor.shape), align_corners=False)
        out = F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False,
                            padding_mode="zeros")
        return out.squeeze(0)

    def _erase(self, tensor):
        _, height, width = tensor.shape
        area = height * width * random.uniform(0.02, 0.15)
        ratio = random.uniform(0.5, 2.0)
        eh, ew = int(round(np.sqrt(area * ratio))), int(round(np.sqrt(area / ratio)))
        if eh >= height or ew >= width or eh < 1 or ew < 1:
            return tensor
        top = random.randint(0, height - eh)
        left = random.randint(0, width - ew)
        tensor[:, top: top + eh, left: left + ew] = 0.0
        return tensor

    def _jitter(self, tensor):
        """Scale every flow channel by one random factor.

        Median motion magnitude differs by several times between subjects, so
        under LOSO the model is trained on one motion scale and tested on
        another. Channel 0 is appearance, not flow, so it is left alone.
        """
        factor = random.uniform(1.0 - self.mag_jitter, 1.0 + self.mag_jitter)
        tensor[1:] = tensor[1:] * factor
        return tensor

    def _load(self, clip_dir, flip=None):
        tensor = self._raw(clip_dir)
        if self.is_train and self.mag_jitter > 0:
            tensor = self._jitter(tensor)
        if self.is_train:
            if flip is None:
                flip = random.random() < 0.5
            if flip:
                tensor = self._flip(tensor)
            if self.aug_strength > 0 and random.random() < 0.7:
                tensor = self._geometric(tensor)
        tensor = (tensor - self.mean) / self.std
        if self.is_train and random.random() < 0.5:
            tensor = self._erase(tensor)
        return tensor

    def _exchange_lower(self, emo_label, flip):
        pool = self.data[self.data["emo_label"] == emo_label]
        pick = pool.sample().reset_index(drop=True)
        frame = self._load(pick.loc[0, "clip_dir"], flip=flip)
        return frame[:, frame.shape[1] // 2:, :], pick.loc[0, "Action Units"]

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        aus = row["Action Units"]
        label = int(row["emo_label"])

        # The same flip is applied to both halves so the composed face stays
        # consistent in direction.
        flip = random.random() < 0.5 if self.is_train else False
        frame = self._load(row["clip_dir"], flip=flip)
        height = frame.shape[1]
        eyes = frame[:, : height // 2, :]

        if self.is_train and random.random() > 0.5:
            mouth, new_aus = self._exchange_lower(label, flip)
        else:
            mouth, new_aus = frame[:, height // 2:, :], aus

        if self.is_train:
            return (eyes, mouth), (label, au_multi_hot(aus, new_aus))
        return (eyes, mouth), label


def flip_batch(eyes, mouth, num_channels):
    """Horizontal flip for test-time augmentation, with the sign flip."""
    signed = [c for c in SIGNED_U if c < num_channels]
    eyes, mouth = torch.flip(eyes, dims=[-1]), torch.flip(mouth, dims=[-1])
    for c in signed:
        eyes[:, c] = -eyes[:, c]
        mouth[:, c] = -mouth[:, c]
    return eyes, mouth


def make_loader(processed_root, frame, batch_size, mode, num_channels=6,
                npy_name="carf.npy", num_workers=0, aug_strength=1.0,
                mag_jitter=0.0, generator=None, drop_last=False):
    dataset = CARFDataset(processed_root, frame, mode, num_channels=num_channels,
                          npy_name=npy_name, aug_strength=aug_strength,
                          mag_jitter=mag_jitter)
    is_train = "training" in mode
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                        num_workers=num_workers,
                        generator=generator if is_train else None,
                        drop_last=drop_last and is_train)
    return loader, dataset
