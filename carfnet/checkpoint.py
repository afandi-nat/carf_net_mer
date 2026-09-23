"""Per-fold checkpoints, for Grad-CAM and further analysis.

One file per (fold, seed, protocol):

    <out_dir>/ckpt/fold-<subject>_seed<s>_<protocol>.pt

Each file carries the weights, the model constructor arguments, the class
names, and the test clips of that fold with their labels and probabilities, so
an explanation can be produced without re-reading the label CSV or the training
arguments -- and it explains exactly the model whose numbers were reported.

    from carfnet.checkpoint import load_checkpoint
    model, ckpt = load_checkpoint("runs/carf_casme2_3c/ckpt/fold-01_seed0_test_peek.pt")

Common Grad-CAM targets:

    model.eyes_branch.layers[2]    layer4 of the upper-face branch
    model.mouth_branch.layers[2]   layer4 of the lower-face branch
    model.base_layers.layers[6]    shared layer2, higher spatial resolution

Score to differentiate: `model.predict(eyes, mouth)`, the fused logits.
"""

import os
import re

import torch

from carfnet.model import CARFNet


def checkpoint_path(folder, subject, seed, protocol):
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(subject))
    return os.path.join(folder, f"fold-{safe}_seed{seed}_{protocol}.pt")


def model_kwargs(args):
    return dict(
        num_classes=args.num_classes,
        in_channels=args.num_channels,
        class_aware_fusion=not args.no_class_aware_fusion,
        joint_head=not args.no_joint_head,
        extra_scale=args.extra_scale,
        dropout=args.dropout,
    )


def save_checkpoint(path, state_dict, args, names, subject, seed, protocol,
                    test_df, trues, probs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "model_kwargs": model_kwargs(args),
        "classes": list(names),
        "num_channels": args.num_channels,
        "npy_name": args.npy_name,
        "processed_root": args.processed_root,
        "dataset": args.dataset,
        "subject": str(subject),
        "seed": int(seed),
        "protocol": protocol,
        "epochs": args.epochs,
        "fold_accuracy": float((probs.argmax(-1) == trues).mean()) if len(trues) else 0.0,
        "test_clips": test_df["clip_dir"].tolist(),
        "test_labels": [int(t) for t in trues],
        "test_probs": probs.tolist(),
    }, path)


def load_checkpoint(path, device="cpu"):
    """Return (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CARFNet(pretrained=False, **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
