from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.transforms.v2 import functional as F

from .loading import load_model
from .utils import calculate_metrics, get_image_filenames, threshold


def _get_image_tensor(filename: str, size: int | None = None) -> torch.Tensor:
    image = cv2.imread(filename)
    if image is None:
        raise ValueError(f"Failed to read image: {filename}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = F.to_image(image)
    image = F.crop(image, 0, 0, image.shape[1] - 2, image.shape[2])

    if size is not None:
        image = F.resize(image, size, antialias=None)

    image = F.to_dtype(image, torch.float32, scale=True)
    return image


def _predict_mask_unet(model: torch.nn.Module, image_filename: str, *, device: torch.device, size: int | None) -> np.ndarray:
    original_image = _get_image_tensor(image_filename, size)
    with torch.no_grad():
        image = torch.clone(original_image).to(device).unsqueeze(0)
        output = model(image).squeeze(0, 1)
        output = (output > 0.5).float()

    return (output.type(torch.uint8) * 255).cpu().numpy()


def _predict_mask_yolo(model: object, image_filename: str, *, conf: float) -> np.ndarray:
    results = model.predict(source=image_filename, conf=conf, save=False, verbose=False)
    if not results:
        img = cv2.imread(image_filename)
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    result = results[0]
    if not hasattr(result, "masks") or result.masks is None:
        img = cv2.imread(image_filename)
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    mask_data = result.masks.data.cpu().numpy()
    if mask_data.size == 0:
        img = cv2.imread(image_filename)
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if len(mask_data.shape) == 3:
        mask = mask_data.max(axis=0)
    else:
        mask = mask_data[0] if len(mask_data.shape) > 2 else mask_data

    return (mask > 0.5).astype(np.uint8) * 255


def _save_mask(output_dir: Path, input_root: Path, image_filename: str, mask: np.ndarray) -> None:
    rel_dir = Path(os.path.relpath(os.path.dirname(image_filename), input_root))
    out_dir = output_dir / "mask" / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / Path(image_filename).name), mask)


def _save_comparison(output_dir: Path, input_root: Path, image_filename: str, mask: np.ndarray) -> None:
    rel_dir = Path(os.path.relpath(os.path.dirname(image_filename), input_root))
    out_dir = output_dir / "compare" / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    original_img = cv2.imread(image_filename)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(2, 1, 1)
    plt.title("Image")
    plt.imshow(original_img)
    fig.add_subplot(2, 1, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.savefig(str(out_dir / f"{Path(image_filename).stem}.png"))
    plt.close(fig)


def cmd_predict(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, args.checkpoint, device=("cuda" if args.cuda else "cpu"))

    input_root = Path(args.input)
    image_filenames = get_image_filenames(args.input, args.recursive)
    if not image_filenames:
        raise SystemExit(f"No images found in {args.input}")

    rows = []
    for image_filename in image_filenames:
        if model.name == "yolo":
            mask = _predict_mask_yolo(model.model, image_filename, conf=args.confidence)
        else:
            mask = _predict_mask_unet(model.model, image_filename, device=model.device, size=args.size)

        if args.threshold_area > 0:
            mask = threshold(mask, args.threshold_area)

        if args.save_mask:
            _save_mask(output_dir, input_root, image_filename, mask)
        if args.save_comparison:
            _save_comparison(output_dir, input_root, image_filename, mask)

        metrics = calculate_metrics(mask, args.scaling_factor)
        rows.append({"image": image_filename, **metrics})

    pd.DataFrame(rows).round(4).to_csv(output_dir / "measurements.csv", index=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="subterra_model")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="Run inference over a directory of images")
    p.add_argument("--model", choices=["unet", "yolo"], required=True)
    p.add_argument("--checkpoint", required=True, help="Path to .pth (UNet) or .pt (YOLO) checkpoint")
    p.add_argument("--input", required=True, help="Input directory containing images")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--recursive", action="store_true", help="Recursively search for images")
    p.add_argument("--save-mask", dest="save_mask", action="store_true", default=True, help="Save predicted masks")
    p.add_argument("--no-save-mask", dest="save_mask", action="store_false", help="Do not save predicted masks")
    p.add_argument("--save-comparison", action="store_true", help="Save side-by-side image+mask PNGs")
    p.add_argument("--size", type=int, default=None, help="Resize images to this size (UNet only)")
    p.add_argument("--scaling-factor", type=float, default=0.2581)
    p.add_argument("--threshold-area", type=int, default=15)
    p.add_argument("--confidence", type=float, default=0.3, help="YOLO confidence threshold")
    p.add_argument("--cuda", action="store_true", help="Use CUDA if available (UNet only)")
    p.set_defaults(func=cmd_predict)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
