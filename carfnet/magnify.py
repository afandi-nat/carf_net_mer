"""Stage 2 (optional) - learning-based motion magnification, onset -> apex.

Writes `amplified.jpg` per clip, which becomes channel 0 of the input tensor.
If this stage is skipped, the build stage uses the plain apex frame instead and
says so in its diagnostics.

The magnifier is the PyTorch re-implementation of Oh et al. (2018),
`kaist-ami/Deep-Motion-Mag-Pytorch`, which ships its own pretrained checkpoint.
It is cloned into `third_party/` on first use; nothing else is downloaded. The
call mirrors that repository's static-mode inference with the onset frame as
reference and amplification factor 6, as used by Ruan et al. (2022).

Usage:
    python -m carfnet.magnify --dataset samm --processed-root /data/carf_processed
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from carfnet.registry import get_spec

REPO_URL = "https://github.com/kaist-ami/Deep-Motion-Mag-Pytorch.git"


def ensure_repo(third_party_dir):
    repo_dir = os.path.join(third_party_dir, "Deep-Motion-Mag-Pytorch")
    if not os.path.isdir(repo_dir):
        print(">> Cloning Deep-Motion-Mag-Pytorch (one time only)")
        os.makedirs(third_party_dir, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, repo_dir])
    ckpt = os.path.join(repo_dir, "model", "epoch50.tar")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"Checkpoint not found at {ckpt}")
    return repo_dir, ckpt


def load_model(repo_dir, ckpt_path, device):
    sys.path.insert(0, repo_dir)
    from module import magnet  # noqa: E402  (provided by the cloned repository)

    model = magnet().to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob.get("model_state_dict", blob)
    # The checkpoint was saved through DataParallel, hence the `module.` prefix.
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f">> Tensors left uninitialised: {len(missing)}")
    if unexpected:
        print(f">> Unused tensors in the checkpoint: {len(unexpected)}")
    model.eval()
    return model


def _to_input(path, device):
    """Normalise to [-1, 1] and crop to a multiple of 4, as the original does."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype="float32") / 127.5 - 1.0
    height, width = arr.shape[:2]
    arr = arr[: height // 4 * 4, : width // 4 * 4, :]
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


@torch.no_grad()
def magnify_pair(model, onset_path, apex_path, amp_factor, device):
    frame_a = _to_input(onset_path, device)
    frame_b = _to_input(apex_path, device)
    mag = torch.tensor(np.array(amp_factor, dtype="float32"),
                       device=device).reshape(1, 1, 1, 1)
    _, shape_a = model.encoder(frame_a)
    texture_b, shape_b = model.encoder(frame_b)
    out = model.decoder(texture_b, model.res_manipulator(shape_a, shape_b, mag))
    out = out.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
    return ((out + 1.0) / 2.0 * 255.0).astype("uint8")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo_dir, ckpt = ensure_repo(os.path.abspath(args.third_party))
    model = load_model(repo_dir, ckpt, device)
    data = pd.read_csv(args.csv_file, dtype={"Subject": str, "Filename": str})

    failures, skipped = [], 0
    for idx in tqdm(range(len(data)), desc=f"Magnifying {args.dataset}"):
        row = data.loc[idx]
        base = os.path.join(args.processed_root, row["clip_dir"])
        out_path = os.path.join(base, args.out_name)
        if not args.overwrite and os.path.isfile(out_path):
            skipped += 1
            continue
        onset = os.path.join(base, f"img{row['Onset']}.jpg")
        apex = os.path.join(base, f"img{row['Apex']}.jpg")
        if not (os.path.isfile(onset) and os.path.isfile(apex)):
            failures.append(f"{row['clip_dir']}: missing frames")
            continue
        try:
            out = magnify_pair(model, onset, apex, args.amp_factor, device)
            Image.fromarray(out).save(out_path, quality=100)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{row['clip_dir']}: {exc}")

    print(f"\nSkipped (existing) : {skipped}")
    print(f"Failures           : {len(failures)}")
    for item in failures[:20]:
        print(f"    {item}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--processed-root", required=True)
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--labels-dir", default="labels")
    p.add_argument("--csv-file", default="")
    p.add_argument("--out-name", default="amplified.jpg")
    p.add_argument("--amp-factor", type=float, default=6.0)
    p.add_argument("--third-party", default="third_party")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if not args.csv_file:
        args.csv_file = os.path.join(
            args.labels_dir, f"{get_spec(args.dataset).name}_{args.classes}class.csv")
    return args


if __name__ == "__main__":
    main(parse_args())
