import argparse
import csv
import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Overwriting .* in registry.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from lib.bdfrnet import BDFRNet
from utils.dataset_fhps import FHPSDataset


ROOT = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def dice_ce_loss(logits, target, class_weights=None):
    ce_map = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
    pt = torch.exp(-ce_map)
    ce = (((1.0 - pt) ** 1.0) * ce_map).mean()
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = torch.sum(probs * one_hot, dims)
    denom = torch.sum(probs + one_hot, dims)
    dice = 1.0 - torch.mean((2.0 * inter + 1e-6) / (denom + 1e-6))
    fp = torch.sum(probs * (1.0 - one_hot), dims)
    fn = torch.sum((1.0 - probs) * one_hot, dims)
    tversky = 1.0 - torch.mean((inter + 1e-6) / (inter + 0.3 * fp + 0.7 * fn + 1e-6))
    return 0.65 * dice + 0.35 * ce + 0.2 * tversky


def per_class_metrics(pred, target):
    metrics = {}
    for cls in (1, 2):
        p = pred == cls
        t = target == cls
        inter = torch.logical_and(p, t).sum().item()
        p_sum = p.sum().item()
        t_sum = t.sum().item()
        union = torch.logical_or(p, t).sum().item()
        metrics[f"dice_{cls}"] = (2 * inter + 1e-6) / (p_sum + t_sum + 1e-6)
        metrics[f"iou_{cls}"] = (inter + 1e-6) / (union + 1e-6)
    metrics["mean_dice"] = (metrics["dice_1"] + metrics["dice_2"]) / 2.0
    metrics["mean_iou"] = (metrics["iou_1"] + metrics["iou_2"]) / 2.0
    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {"dice_1": 0.0, "dice_2": 0.0, "iou_1": 0.0, "iou_2": 0.0, "mean_dice": 0.0, "mean_iou": 0.0}
    count = 0
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        outputs = model(image)
        logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        pred = torch.argmax(logits, dim=1)
        for i in range(pred.shape[0]):
            metrics = per_class_metrics(pred[i], mask[i])
            for key, value in metrics.items():
                totals[key] += value
            count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(ROOT / "data" / "fhps_aop"))
    parser.add_argument("--pretrained_dir", default=str(ROOT / "pretrained_pth" / "pvt"))
    parser.add_argument("--output_dir", default=str(ROOT / "runs" / "fhps_bdfrnet_b2"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = FHPSDataset(args.data_root, "train", img_size=args.img_size, augment=True)
    val_set = FHPSDataset(args.data_root, "val", img_size=args.img_size, augment=False)
    if args.limit_train > 0:
        train_set = Subset(train_set, range(min(args.limit_train, len(train_set))))
    if args.limit_val > 0:
        val_set = Subset(val_set, range(min(args.limit_val, len(val_set))))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    model = BDFRNet(num_classes=3, encoder="pvt_v2_b2", pretrained_dir=args.pretrained_dir, pretrain=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    class_weights = torch.tensor([0.3, 2.0, 1.0], dtype=torch.float32, device=device)

    best_dice = -1.0
    log_path = output_dir / "metrics.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_dice_ps", "val_dice_fh", "val_mean_dice", "val_mean_iou", "lr"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    outputs = model(image)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    loss = sum(dice_ce_loss(out, mask, class_weights=class_weights) for out in outputs) / len(outputs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())

            val_metrics = evaluate(model, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_dice_ps": val_metrics["dice_1"],
                "val_dice_fh": val_metrics["dice_2"],
                "val_mean_dice": val_metrics["mean_dice"],
                "val_mean_iou": val_metrics["mean_iou"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()
            print(
                f"Epoch {epoch:03d}/{args.epochs} loss={row['train_loss']:.4f} "
                f"val_ps={row['val_dice_ps']:.4f} val_fh={row['val_dice_fh']:.4f} "
                f"val_mean={row['val_mean_dice']:.4f}",
                flush=True,
            )
            if val_metrics["mean_dice"] > best_dice:
                best_dice = val_metrics["mean_dice"]
                torch.save(model.state_dict(), output_dir / "best.pth")


if __name__ == "__main__":
    main()
