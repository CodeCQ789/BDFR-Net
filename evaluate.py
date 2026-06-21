from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="Overwriting .* in registry.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import DataLoader
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt

from lib.bdfrnet import BDFRNet
from utils.dataset_fhps import FHPSDataset


ROOT = Path(__file__).resolve().parent


def _to_image(mask):
    return sitk.GetImageFromArray(mask.astype(np.uint8))


def _surface(mask):
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    return mask ^ binary_erosion(mask)


def _average_surface_distance(pred, target):
    pred_surface = _surface(pred)
    target_surface = _surface(target)
    if not pred_surface.any() or not target_surface.any():
        return 100.0
    pred_to_target = distance_transform_edt(~target_surface)[pred_surface].sum()
    target_to_pred = distance_transform_edt(~pred_surface)[target_surface].sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 0.0
    return float((pred_to_target + target_to_pred) / denom)


def _hausdorff_distance(pred, target):
    pred_image = sitk.Cast(sitk.RescaleIntensity(_to_image(pred)), sitk.sitkUInt8)
    target_image = sitk.Cast(sitk.RescaleIntensity(_to_image(target)), sitk.sitkUInt8)
    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(pred_image, target_image)
    return float(hd_filter.GetHausdorffDistance())


def per_class_metrics(pred, target):
    metrics = {}
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    for cls in (1, 2):
        p = pred == cls
        t = target == cls
        if not p.any() or not t.any():
            metrics[f"dice_{cls}"] = 0.0
            metrics[f"asd_{cls}"] = 100.0
            metrics[f"hd_{cls}"] = 100.0
            continue
        inter = np.logical_and(p, t).sum()
        p_sum = p.sum()
        t_sum = t.sum()
        metrics[f"dice_{cls}"] = (2.0 * inter + 1e-6) / (p_sum + t_sum + 1e-6)
        metrics[f"asd_{cls}"] = _average_surface_distance(p, t)
        metrics[f"hd_{cls}"] = _hausdorff_distance(p, t)
    metrics["mean_dice"] = (metrics["dice_1"] + metrics["dice_2"]) / 2.0
    metrics["mean_asd"] = (metrics["asd_1"] + metrics["asd_2"]) / 2.0
    metrics["mean_hd"] = (metrics["hd_1"] + metrics["hd_2"]) / 2.0
    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {
        "dice_1": 0.0,
        "asd_1": 0.0,
        "hd_1": 0.0,
        "dice_2": 0.0,
        "asd_2": 0.0,
        "hd_2": 0.0,
        "mean_dice": 0.0,
        "mean_asd": 0.0,
        "mean_hd": 0.0,
    }
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
    data_root = ROOT / "data" / "fhps_aop"
    weight_path = ROOT / "weights" / "best.pth"
    pretrained_dir = ROOT / "pretrained_pth" / "pvt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BDFRNet(num_classes=3, encoder="pvt_v2_b2", pretrained_dir=str(pretrained_dir), pretrain=False).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    loader = DataLoader(FHPSDataset(data_root, "val", img_size=256, augment=False), batch_size=16, shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, device)
    print(f"PS Dice: {metrics['dice_1']:.6f}")
    print(f"PS ASD: {metrics['asd_1']:.6f}")
    print(f"PS HD: {metrics['hd_1']:.6f}")
    print(f"FH Dice: {metrics['dice_2']:.6f}")
    print(f"FH ASD: {metrics['asd_2']:.6f}")
    print(f"FH HD: {metrics['hd_2']:.6f}")
    print(f"Mean Dice: {metrics['mean_dice']:.6f}")
    print(f"Mean ASD: {metrics['mean_asd']:.6f}")
    print(f"Mean HD: {metrics['mean_hd']:.6f}")


if __name__ == "__main__":
    main()
