import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


MARKER = b"ElementDataFile = LOCAL"


def read_mha(path):
    raw = Path(path).read_bytes()
    marker_pos = raw.find(MARKER)
    if marker_pos < 0:
        raise ValueError(f"Unsupported MHA without LOCAL data: {path}")
    header_end = raw.find(b"\n", marker_pos)
    if header_end < 0:
        header_end = marker_pos + len(MARKER)
    else:
        header_end += 1
    header = raw[:header_end].decode("ascii", errors="ignore")
    fields = {}
    for line in header.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    dim_size = tuple(int(v) for v in fields["DimSize"].split())
    element_type = fields["ElementType"]
    dtype_map = {
        "MET_UCHAR": np.uint8,
        "MET_CHAR": np.int8,
        "MET_USHORT": np.uint16,
        "MET_SHORT": np.int16,
        "MET_FLOAT": np.float32,
    }
    if element_type not in dtype_map:
        raise ValueError(f"Unsupported ElementType {element_type} in {path}")
    arr = np.frombuffer(raw[header_end:], dtype=dtype_map[element_type])
    expected = int(np.prod(dim_size))
    if arr.size != expected:
        raise ValueError(f"Unexpected payload size in {path}: got {arr.size}, expected {expected}")
    if len(dim_size) == 2:
        width, height = dim_size
        arr = arr.reshape(height, width)
    elif len(dim_size) == 3 and dim_size[2] in (1, 3, 4):
        width, height, channels = dim_size
        arr = arr.reshape(channels, height, width).transpose(1, 2, 0)
        if channels == 1:
            arr = arr[:, :, 0]
    else:
        arr = arr.reshape(tuple(reversed(dim_size)))
    return arr.copy(), fields


def collect_pairs(source_root):
    source_root = Path(source_root)
    pairs = {}
    for image_path in source_root.rglob("images/*.mha"):
        mask_path = image_path.parent.parent / "masks" / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}")
        if image_path.name in pairs:
            raise ValueError(f"Duplicate case id {image_path.name}; refusing ambiguous split")
        pairs[image_path.name] = (image_path, mask_path)
    return [pairs[name] for name in sorted(pairs)]


def write_png_pair(image_path, mask_path, out_image, out_mask):
    image, _ = read_mha(image_path)
    mask, _ = read_mha(mask_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape {image.shape}: {image_path}")
    mask = np.where(mask == 1, 1, np.where(mask == 2, 2, 0)).astype(np.uint8)

    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_mask), mask)


def main():
    parser = argparse.ArgumentParser(description="Prepare FH-PS-AoP data for segmentation experiments.")
    parser.add_argument("--source_root", default=r"E:\miccai2026rebuttle\Pubic Symphysis-Fetal Head_dataValDataset")
    parser.add_argument("--out_root", default="./data/fhps_aop")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    pairs = collect_pairs(args.source_root)
    if len(pairs) != 4000:
        raise ValueError(f"Expected 4000 image/mask pairs for FH-PS-AoP training set, found {len(pairs)}")

    rng = random.Random(args.seed)
    pairs = pairs[:]
    rng.shuffle(pairs)
    train_count = int(round(len(pairs) * args.train_ratio))
    split_pairs = {"train": pairs[:train_count], "val": pairs[train_count:]}

    manifest_rows = []
    for split, items in split_pairs.items():
        for image_path, mask_path in items:
            out_name = image_path.with_suffix(".png").name
            write_png_pair(
                image_path,
                mask_path,
                out_root / split / "images" / out_name,
                out_root / split / "masks" / out_name,
            )
            manifest_rows.append({
                "split": split,
                "case": image_path.stem,
                "image": str(image_path),
                "mask": str(mask_path),
            })

    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "case", "image", "mask"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source_root": str(Path(args.source_root)),
        "out_root": str(out_root),
        "seed": args.seed,
        "split_protocol": "FH-PS-AoP 4000-image training set split 8:2 into training and validation sets",
        "train": len(split_pairs["train"]),
        "val": len(split_pairs["val"]),
        "classes": {"0": "background", "1": "pubic symphysis", "2": "fetal head"},
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
