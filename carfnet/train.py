"""Leave-one-subject-out training and evaluation.

    python -m carfnet.train --dataset casme2 --processed-root /data/carf_processed
    python -m carfnet.train --dataset casme2,samm,casme3 ...   # combined, LOSO over all subjects

Checkpoint-selection protocols. Pick them with `--protocols`:

    test_peek  per fold, the epoch with the highest accuracy on the held-out
               subject. This is the protocol of Ruan et al. (2022) and the
               default here, so numbers are comparable with that line of work.
               The test fold participates in selecting the reported weights;
               state this explicitly when reporting results.
    ema        exponential moving average of the weights at the last epoch. No
               checkpoint selection at all, so the test fold is never inspected.
    last       plain last-epoch weights.
    inner      best epoch on an inner validation split of training subjects
               (enable with --inner-holdout).

Metrics follow MEGC: accuracy, UF1 (macro F1) and UAR (macro recall).
"""

import argparse
import copy
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from carfnet.checkpoint import checkpoint_path, save_checkpoint
from carfnet.data import drop_unbuilt, flip_batch, load_labels, loso_folds, make_loader
from carfnet.losses import WeightedSupCoL, au_similarity, build_criterion, mixup
from carfnet.model import CARFNet

PROTOCOLS = ("test_peek", "ema", "last", "inner")
NOTES = {
    "test_peek": "epoch selected on the test fold",
    "ema": "no checkpoint selection",
    "last": "no checkpoint selection",
    "inner": "selected on inner validation",
}


# --------------------------------------------------------------------------- #
def set_seed(seed, deterministic=False):
    """Seed every source of randomness.

    torch.manual_seed alone is not enough on GPU: cuDNN picks convolution
    algorithms with an autotuner whose outcome depends on the current load, and
    some kernels use non-deterministic atomics. Over dozens of folds those
    rounding differences accumulate into whole samples.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


class EMA:
    """Exponential moving average of weights and buffers, with a warm-up decay."""

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.step_count = 0
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        decay = min(self.decay, (1.0 + self.step_count) / (10.0 + self.step_count))
        for shadow, live in zip(self.shadow.parameters(), model.parameters()):
            shadow.mul_(decay).add_(live.detach(), alpha=1.0 - decay)
        for shadow, live in zip(self.shadow.buffers(), model.buffers()):
            shadow.copy_(live)


def warmup_cosine(optimizer, warmup_steps, total_steps, min_ratio=0.02):
    def curve(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, curve)


@torch.no_grad()
def predict_probs(loader, model, device, num_channels, tta=True):
    """Per-sample probabilities. TTA averages the prediction with a mirrored
    copy, using the sign-aware flip so it stays physically consistent."""
    model.eval()
    probs, trues = [], []
    for (eyes, mouth), labels in loader:
        eyes, mouth = eyes.to(device), mouth.to(device)
        prob = F.softmax(model.predict(eyes, mouth), dim=-1)
        if tta:
            f_eyes, f_mouth = flip_batch(eyes.clone(), mouth.clone(), num_channels)
            prob = 0.5 * (prob + F.softmax(model.predict(f_eyes, f_mouth), dim=-1))
        probs.append(prob.cpu())
        trues.extend(labels.view(-1).tolist())
    return torch.cat(probs).numpy(), np.array(trues, dtype=np.int64)


def accuracy_of(probs, trues):
    return float((probs.argmax(-1) == trues).mean()) if len(trues) else 0.0


def split_inner(train_df, ratio, rng):
    """Hold out whole subjects, not samples, so validation stays cross-subject."""
    subjects = sorted(train_df["Subject"].unique())
    n_hold = max(1, int(round(len(subjects) * ratio)))
    held = set(rng.choice(subjects, size=n_hold, replace=False).tolist())
    fit = train_df[~train_df["Subject"].isin(held)].reset_index(drop=True)
    val = train_df[train_df["Subject"].isin(held)].reset_index(drop=True)
    return fit, val, sorted(held)


def summarize(probs, trues, num_classes):
    preds = probs.argmax(-1)
    labels = list(range(num_classes))
    return {
        "accuracy": float((preds == trues).mean()),
        "uf1": float(f1_score(trues, preds, average="macro")),
        "uar": float(recall_score(trues, preds, average="macro")),
        "per_class_f1": [float(v) for v in
                         f1_score(trues, preds, average=None, labels=labels)],
        "per_class_recall": [float(v) for v in
                             recall_score(trues, preds, average=None, labels=labels)],
        "confusion_matrix": confusion_matrix(trues, preds, labels=labels).tolist(),
    }


# --------------------------------------------------------------------------- #
def train_one_fold(args, train_df, test_df, device, rng, names):
    if args.inner_holdout > 0:
        fit_df, val_df, held = split_inner(train_df, args.inner_holdout, rng)
    else:
        fit_df, val_df, held = train_df, None, []

    common = dict(num_channels=args.num_channels, npy_name=args.npy_name)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader, train_set = make_loader(
        args.processed_root, fit_df, args.batch_size, "training",
        num_workers=args.num_workers, aug_strength=args.aug_strength,
        mag_jitter=args.mag_jitter, generator=generator,
        drop_last=len(fit_df) > 2 * args.batch_size, **common)
    test_loader, _ = make_loader(args.processed_root, test_df,
                                 max(1, len(test_df)), "testing", **common)
    val_loader = None
    if val_df is not None and len(val_df):
        val_loader, _ = make_loader(args.processed_root, val_df,
                                    min(64, len(val_df)), "testing", **common)

    model = CARFNet(
        num_classes=args.num_classes, in_channels=args.num_channels,
        pretrained=not args.no_pretrained,
        class_aware_fusion=not args.no_class_aware_fusion,
        joint_head=not args.no_joint_head,
        extra_scale=args.extra_scale, dropout=args.dropout).to(device)
    if args.freeze_bn:
        model.set_bn_frozen(True)

    criterion = build_criterion(args.loss, train_set.class_counts(args.num_classes),
                                device, tau=args.tau,
                                label_smoothing=args.label_smoothing,
                                gamma=args.focal_gamma)
    au_criterion = WeightedSupCoL()
    optimizer = torch.optim.AdamW(model.param_groups(
        args.lr_backbone, args.lr_head, args.gate_lr_mult, args.weight_decay))
    steps = max(1, len(train_loader))
    scheduler = warmup_cosine(optimizer, steps * args.warmup_epochs, steps * args.epochs)
    ema = EMA(model, decay=args.ema_decay) if "ema" in args.selected else None

    history, trues = [], None
    best_peek_acc, best_peek_state = -1.0, None

    for _ in range(args.epochs):
        model.train()
        for (eyes, mouth), (labels, aus) in train_loader:
            eyes, mouth = eyes.to(device), mouth.to(device)
            labels, aus = labels.to(device), aus.to(device)

            use_mixup = args.mixup_alpha > 0 and random.random() < args.mixup_prob
            if use_mixup:
                eyes, mouth, target = mixup(eyes, mouth, labels, args.num_classes,
                                            args.mixup_alpha, args.label_smoothing)
            else:
                target = labels

            eyes_logits, mouth_logits, joint_logits, contrast = model(eyes, mouth)
            cls_loss = criterion(eyes_logits, target) + criterion(mouth_logits, target)
            if joint_logits is not None:
                cls_loss = cls_loss + criterion(joint_logits, target)
            fused = model.fuse_logits(eyes_logits, mouth_logits, joint_logits)
            cls_loss = cls_loss + args.fusion_loss_weight * criterion(fused, target)

            # The AU term is skipped on mixed batches: after mixup the AU
            # targets no longer correspond to the sample pairs.
            total = cls_loss
            if not use_mixup:
                au_loss = au_criterion(contrast, au_similarity(aus))
                if au_loss is not None:
                    total = (1 - args.lambda_) * cls_loss + args.lambda_ * au_loss

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

        test_acc, test_probs = -1.0, None
        if "test_peek" in args.selected or val_loader is not None:
            test_probs, trues = predict_probs(test_loader, model, device,
                                              args.num_channels, args.tta)
            test_acc = accuracy_of(test_probs, trues)
        val_acc = -1.0
        if val_loader is not None:
            val_probs, val_trues = predict_probs(val_loader, model, device,
                                                 args.num_channels, args.tta)
            val_acc = accuracy_of(val_probs, val_trues)
        history.append((test_probs, test_acc, val_acc))

        # Strict '>' matches np.argmax: the FIRST epoch reaching the maximum.
        if "test_peek" in args.selected and test_acc > best_peek_acc:
            best_peek_acc = test_acc
            best_peek_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}

    result, states = {}, {}
    if ema is not None:
        result["ema"], trues = predict_probs(test_loader, ema.shadow, device,
                                             args.num_channels, args.tta)
        states["ema"] = ema.shadow.state_dict()
    if "last" in args.selected:
        result["last"], trues = predict_probs(test_loader, model, device,
                                              args.num_channels, args.tta)
        states["last"] = model.state_dict()
    if "test_peek" in args.selected:
        result["test_peek"] = history[int(np.argmax([h[1] for h in history]))][0]
        states["test_peek"] = best_peek_state
    if val_loader is not None and "inner" in args.selected:
        result["inner"] = history[int(np.argmax([h[2] for h in history]))][0]
    if trues is None:
        _, trues = predict_probs(test_loader, model, device, args.num_channels, args.tta)
    return result, trues, held, model.gate_report(names), states


# --------------------------------------------------------------------------- #
def main(args):
    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    data, names = load_labels(datasets, args.classes, args.labels_dir)
    data = drop_unbuilt(data, args.processed_root, args.npy_name)
    args.num_classes = len(names)
    args.selected = [p for p in PROTOCOLS
                     if p in {t.strip() for t in args.protocols.split(",")}]
    if not args.selected:
        raise SystemExit(f"--protocols matched nothing. Choose from: {PROTOCOLS}")
    if not args.out_dir:
        args.out_dir = os.path.join("runs", f"carf_{'+'.join(datasets)}_{args.classes}c")
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    save_protocols = [x.strip() for x in args.save_ckpt.split(",")
                      if x.strip() not in ("", "none")]
    bad = [x for x in save_protocols if x not in args.selected or x == "inner"]
    if bad:
        raise SystemExit(f"--save-ckpt {bad} is not being computed. Active: {args.selected}")
    if args.ckpt_seeds == "all":
        ckpt_seeds = set(seeds)
    elif args.ckpt_seeds == "first":
        ckpt_seeds = {seeds[0]}
    else:
        ckpt_seeds = {int(x) for x in args.ckpt_seeds.split(",")}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f">> Dataset {'+'.join(datasets)}: {len(data)} clips, "
          f"{data['Subject'].nunique()} subjects, classes {names}, device {device}")
    print("   Per class: " + ", ".join(
        f"{n} {int((data['emo_label'] == i).sum())}" for i, n in enumerate(names)))

    train_list, test_list, subjects = loso_folds(data)
    ckpt_dir = os.path.join(args.out_dir, "ckpt")
    if save_protocols:
        n_files = len(train_list) * len(ckpt_seeds & set(seeds)) * len(save_protocols)
        print(f">> Saving {'+'.join(save_protocols)} checkpoints for seeds "
              f"{sorted(ckpt_seeds & set(seeds))}: {n_files} files, "
              f"about {n_files * 0.1:.1f} GB, in {ckpt_dir}")

    pooled = {p: {s: [] for s in seeds} for p in args.selected}
    all_trues, fold_frames, gate = [], [], None

    for k in range(len(train_list)):
        fold_trues = None
        for seed in seeds:
            set_seed(seed, args.deterministic)
            args.seed = seed
            rng = np.random.default_rng(seed)
            result, trues, _, gate, states = train_one_fold(
                args, train_list[k], test_list[k], device, rng, names)
            fold_trues = trues
            if seed in ckpt_seeds:
                subject = str(test_list[k]["Subject"].iloc[0])
                for protocol in save_protocols:
                    save_checkpoint(checkpoint_path(ckpt_dir, subject, seed, protocol),
                                    states[protocol], args, names, subject, seed,
                                    protocol, test_list[k], trues, result[protocol])
            for protocol in args.selected:
                if protocol in result:
                    pooled[protocol][seed].append(result[protocol])
        all_trues.append(fold_trues)
        fold_frames.append(test_list[k])
        line = " ".join(f"{p}={accuracy_of(pooled[p][seeds[0]][-1], fold_trues):.3f}"
                        for p in args.selected if pooled[p][seeds[0]])
        print(f"LOSO {subjects[k]}  n={len(fold_trues)}  {line}", flush=True)

    trues = np.concatenate(all_trues)
    origin = pd.concat(fold_frames, ignore_index=True)["dataset"].to_numpy()
    report = {"config": vars(args), "classes": names, "fusion_gate": gate,
              "protocols": {}}

    print("\n" + "=" * 72)
    print(f"{'protocol':<12}{'seed':<8}{'ACC':>8}{'UF1':>8}{'UAR':>8}   note")
    print("=" * 72)
    for protocol in args.selected:
        if not pooled[protocol][seeds[0]]:
            continue
        entry, per_seed = {}, []
        for seed in seeds:
            probs = np.concatenate(pooled[protocol][seed])
            per_seed.append(probs)
            stats = summarize(probs, trues, args.num_classes)
            entry[f"seed_{seed}"] = stats
            print(f"{protocol:<12}{seed:<8}{stats['accuracy']:>8.4f}"
                  f"{stats['uf1']:>8.4f}{stats['uar']:>8.4f}   {NOTES[protocol]}")
        mean_probs = np.mean(per_seed, axis=0)
        if len(seeds) > 1:
            ens = summarize(mean_probs, trues, args.num_classes)
            entry["ensemble"] = ens
            accs = [entry[f"seed_{s}"]["accuracy"] for s in seeds]
            f1s = [entry[f"seed_{s}"]["uf1"] for s in seeds]
            print(f"{protocol:<12}{'ens':<8}{ens['accuracy']:>8.4f}{ens['uf1']:>8.4f}"
                  f"{ens['uar']:>8.4f}   softmax average of {len(seeds)} seeds")
            print(f"{protocol:<12}{'mean':<8}{np.mean(accs):>8.4f}{np.mean(f1s):>8.4f}"
                  f"{'':>8}   +-{np.std(accs):.4f} / +-{np.std(f1s):.4f}")
        if len(datasets) > 1:
            entry["per_dataset"] = {}
            for ds in datasets:
                mask = origin == ds
                stats = summarize(mean_probs[mask], trues[mask], args.num_classes)
                entry["per_dataset"][ds] = stats
                print(f"{'':<12}{ds:<8}{stats['accuracy']:>8.4f}{stats['uf1']:>8.4f}"
                      f"{stats['uar']:>8.4f}")
        report["protocols"][protocol] = entry

    head = report["protocols"].get(args.selected[0], {})
    best = head.get("ensemble") or head.get(f"seed_{seeds[0]}")
    if best:
        print(f"\nPer class, protocol {args.selected[0]}:")
        for i, name in enumerate(names):
            print(f"   {name:10s} recall {best['per_class_recall'][i]:.3f}"
                  f"   F1 {best['per_class_f1'][i]:.3f}")
        print("\nConfusion matrix (rows: true, columns: predicted):")
        print(f"   {'':10s}" + "".join(f"{n[:8]:>9s}" for n in names))
        for i, row in enumerate(best["confusion_matrix"]):
            print(f"   {names[i]:10s}" + "".join(f"{v:>9d}" for v in row))
    if gate:
        print(f"\nFusion gate (1 = leaning on the upper-face branch): {gate}")

    out = os.path.join(args.out_dir, "report.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nFull report: {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True,
                   help="casme2 | samm | casme3, or a comma-separated combination")
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--processed-root", required=True)
    p.add_argument("--labels-dir", default="labels")
    p.add_argument("--npy-name", default="carf.npy")
    p.add_argument("--out-dir", default="")
    p.add_argument("--num-channels", type=int, default=6)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr-backbone", type=float, default=2e-4)
    p.add_argument("--lr-head", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=5.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--freeze-bn", type=int, default=1)

    p.add_argument("--loss", default="logit_adjusted",
                   choices=["logit_adjusted", "weighted_ce", "plain_ce"])
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--lambda-", dest="lambda_", type=float, default=0.3,
                   help="weight of the AU contrastive term")
    p.add_argument("--fusion-loss-weight", type=float, default=1.0)

    p.add_argument("--aug-strength", type=float, default=1.0)
    p.add_argument("--mag-jitter", type=float, default=0.0,
                   help="random flow scaling 1+-j during training; 0 disables it")
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--tta", type=int, default=1)

    p.add_argument("--no-class-aware-fusion", action="store_true")
    p.add_argument("--no-joint-head", action="store_true")
    p.add_argument("--extra-scale", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--gate-lr-mult", type=float, default=20.0)
    p.add_argument("--no-pretrained", action="store_true",
                   help="skip ImageNet weights (smoke tests only)")

    p.add_argument("--protocols", default="test_peek",
                   help="test_peek (default), ema, last, inner, or a comma list")
    p.add_argument("--save-ckpt", default="test_peek",
                   help="checkpoints to save per fold; must be among --protocols, or none")
    p.add_argument("--ckpt-seeds", default="first",
                   help="first, all, or a list such as 0,2 (about 100 MB per file)")
    p.add_argument("--inner-holdout", type=float, default=0.0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
