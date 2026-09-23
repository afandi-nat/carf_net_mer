"""Check that saved checkpoints reproduce the probabilities stored with them.

Each checkpoint carries the test clips of its fold and the probabilities that
went into `report.json`. Re-running the model over those clips must return the
same numbers; if it does not, the saved weights are not the ones that produced
the reported result.

Usage:
    python tools/verify_checkpoints.py --run-dir runs/carf_casme2_3c
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carfnet.checkpoint import load_checkpoint  # noqa: E402
from carfnet.data import load_labels, make_loader  # noqa: E402
from carfnet.train import predict_probs  # noqa: E402


def main(args):
    files = sorted(glob.glob(os.path.join(args.run_dir, "ckpt", "*.pt")))
    if not files:
        raise SystemExit(f"No checkpoint found in {args.run_dir}/ckpt")
    if args.limit:
        files = files[: args.limit]

    worst = 0.0
    for path in files:
        model, ckpt = load_checkpoint(path, args.device)
        datasets = [d.strip() for d in ckpt["dataset"].split(",")]
        data, _ = load_labels(datasets, len(ckpt["classes"]), args.labels_dir)
        test = data.set_index("clip_dir").loc[ckpt["test_clips"]].reset_index()
        loader, _ = make_loader(ckpt["processed_root"], test, len(test), "testing",
                                num_channels=ckpt["num_channels"],
                                npy_name=ckpt["npy_name"])
        probs, trues = predict_probs(loader, model, args.device,
                                     ckpt["num_channels"], tta=True)
        assert list(trues) == ckpt["test_labels"], f"{path}: label mismatch"
        delta = float(np.abs(probs - np.array(ckpt["test_probs"])).max())
        worst = max(worst, delta)
        print(f"{os.path.basename(path):48s} subject {ckpt['subject']:>8s}  "
              f"fold acc {ckpt['fold_accuracy']:.3f}  max delta {delta:.2e}")

    print(f"\n{len(files)} checkpoints verified, largest probability difference {worst:.2e}")
    if worst > 1e-5:
        raise SystemExit("Checkpoints do not reproduce their stored probabilities.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--labels-dir", default="labels")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0, help="check only the first N files")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
