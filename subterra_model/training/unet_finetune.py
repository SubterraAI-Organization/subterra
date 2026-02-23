from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as F

from subterra_model.loading import _load_unet


def _is_image_file(path: Path) -> bool:
    # Keep in sync with formats we accept from the UI/API. CI-600 exports are
    # commonly TIFF, and older annotations may rely on this fallback scanner.
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _extract_camera_model(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""
    if isinstance(meta.get("camera_model"), str):
        return meta.get("camera_model") or ""
    camera = meta.get("camera") or {}
    if isinstance(camera, dict) and isinstance(camera.get("camera_model"), str):
        return camera.get("camera_model") or ""
    nested_meta = meta.get("meta") or {}
    if isinstance(nested_meta, dict) and isinstance(nested_meta.get("camera_model"), str):
        return nested_meta.get("camera_model") or ""
    mini = meta.get("minirhizotron") or {}
    if isinstance(mini, dict) and isinstance(mini.get("camera_model"), str):
        return mini.get("camera_model") or ""
    return ""


def _is_corrected(meta: dict) -> bool:
    if not isinstance(meta, dict):
        return False
    corrected = meta.get("corrected")
    if isinstance(corrected, bool):
        return corrected
    nested = meta.get("meta") or {}
    if isinstance(nested, dict) and isinstance(nested.get("corrected"), bool):
        return bool(nested.get("corrected"))
    return False


def _extract_run_id(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""
    if isinstance(meta.get("run_id"), str):
        return meta.get("run_id") or ""
    nested = meta.get("meta") or {}
    if isinstance(nested, dict) and isinstance(nested.get("run_id"), str):
        return nested.get("run_id") or ""
    return ""


def _pick_pair(annotation_dir: Path) -> Optional[tuple[Path, Path, dict]]:
    meta_path = annotation_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            img_name = meta.get("image_filename")
            mask_name = meta.get("mask_filename")
            if img_name and mask_name:
                img_path = annotation_dir / str(img_name)
                mask_path = annotation_dir / str(mask_name)
                if img_path.exists() and mask_path.exists():
                    return img_path, mask_path, meta
        except Exception:
            pass

    files = [p for p in annotation_dir.iterdir() if p.is_file() and _is_image_file(p)]
    if not files:
        return None

    mask_candidates = [p for p in files if "mask" in p.stem.lower()]
    img_candidates = [p for p in files if "mask" not in p.stem.lower()]

    if mask_candidates and img_candidates:
        return img_candidates[0], mask_candidates[0], {}

    if len(files) == 2:
        a, b = files
        if "mask" in a.stem.lower() and "mask" not in b.stem.lower():
            return b, a, {}
        if "mask" in b.stem.lower() and "mask" not in a.stem.lower():
            return a, b, {}
        # fallback: treat second as mask
        return a, b, {}

    return None


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    image_size: int


class AnnotationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        annotations_dir: Path,
        *,
        image_size: int,
        seed: int = 0,
        camera_model: Optional[str] = None,
        only_corrected: bool = False,
        preserve_aspect_ratio: bool = True,
        run_id: Optional[str] = None,
    ):
        self.annotations_dir = annotations_dir
        self.image_size = image_size
        self.seed = seed
        self.camera_model = (camera_model or "").strip()
        self.only_corrected = bool(only_corrected)
        self.preserve_aspect_ratio = bool(preserve_aspect_ratio)
        self.run_id = (run_id or "").strip()

        pairs: list[tuple[Path, Path]] = []
        for child in sorted(annotations_dir.iterdir()):
            if not child.is_dir():
                continue
            pair = _pick_pair(child)
            if pair:
                img_path, mask_path, meta = pair
                if self.run_id:
                    rid = _extract_run_id(meta).strip()
                    if rid != self.run_id:
                        continue
                if self.camera_model:
                    cam = _extract_camera_model(meta).strip()
                    if cam.lower() != self.camera_model.lower():
                        continue
                if self.only_corrected and not _is_corrected(meta):
                    continue
                pairs.append((img_path, mask_path))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _resize_and_pad(self, image: Image.Image, *, resample: Image.Resampling) -> Image.Image:
        target = int(self.image_size)
        if target <= 0:
            return image
        w, h = image.size
        if w <= 0 or h <= 0:
            return image
        if w == target and h == target:
            return image

        scale = min(target / w, target / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = image.resize((new_w, new_h), resample=resample)

        background = (0, 0, 0) if resized.mode == "RGB" else 0
        canvas = Image.new(resized.mode, (target, target), color=background)
        x0 = (target - new_w) // 2
        y0 = (target - new_h) // 2
        canvas.paste(resized, (x0, y0))
        return canvas

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_size > 0:
            if self.preserve_aspect_ratio:
                image = self._resize_and_pad(image, resample=Image.Resampling.BILINEAR)
                mask = self._resize_and_pad(mask, resample=Image.Resampling.NEAREST)
            else:
                image = image.resize((self.image_size, self.image_size), resample=Image.Resampling.BILINEAR)
                mask = mask.resize((self.image_size, self.image_size), resample=Image.Resampling.NEAREST)

        image_t = F.to_image(image)
        mask_t = F.to_image(mask)

        image_t = F.to_dtype(image_t, torch.float32, scale=True)
        mask_t = F.to_dtype(mask_t, torch.float32, scale=True)

        # random flips
        rng = random.Random(self.seed + idx)
        if rng.random() < 0.5:
            image_t = F.horizontal_flip(image_t)
            mask_t = F.horizontal_flip(mask_t)
        if rng.random() < 0.25:
            image_t = F.vertical_flip(image_t)
            mask_t = F.vertical_flip(mask_t)

        # binarize mask to {0,1}
        mask_t = (mask_t > 0.5).float()
        return image_t, mask_t


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # pred/target: [B,1,H,W]
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return (1 - dice).mean()


def finetune_unet_from_annotations(
    *,
    annotations_dir: Path,
    base_checkpoint_path: str,
    output_checkpoint_path: Path,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-4,
    image_size: int = 512,
    camera_model: Optional[str] = None,
    only_corrected: bool = False,
    preserve_aspect_ratio: bool = True,
    run_id: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    cfg = TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr, image_size=image_size)
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Build an unfiltered view first so we can give a helpful error if filters
    # exclude everything (common when camera metadata is missing).
    all_pairs = AnnotationDataset(
        Path(annotations_dir),
        image_size=cfg.image_size,
        seed=0,
        camera_model=None,
        only_corrected=False,
        preserve_aspect_ratio=preserve_aspect_ratio,
        run_id=(run_id or "").strip() or None,
    )

    dataset = AnnotationDataset(
        Path(annotations_dir),
        image_size=cfg.image_size,
        seed=0,
        camera_model=camera_model,
        only_corrected=only_corrected,
        preserve_aspect_ratio=preserve_aspect_ratio,
        run_id=(run_id or "").strip() or None,
    )
    if len(dataset) == 0:
        if len(all_pairs) > 0:
            raise RuntimeError(
                "No training pairs match the requested filters "
                f"(camera_model={camera_model!r}, only_corrected={bool(only_corrected)}, run_id={(run_id or '').strip() or None!r}). "
                f"Found {len(all_pairs)} total training pairs under {annotations_dir}."
            )
        raise RuntimeError(
            f"No training pairs found under {annotations_dir}. "
            "Expected one subfolder per annotation containing an image+mask pair (e.g. *_mask.png) "
            "or a meta.json with image_filename and mask_filename."
        )

    flags = []
    if run_id and str(run_id).strip():
        flags.append(f"run_id={str(run_id).strip()}")
    if camera_model:
        flags.append(f"camera={camera_model}")
    if only_corrected:
        flags.append("only_corrected=true")
    flags.append(f"preserve_aspect_ratio={bool(preserve_aspect_ratio)}")
    suffix = f" ({', '.join(flags)})" if flags else ""
    log(f"Found {len(dataset)} annotation pairs{suffix}")

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    loaded = _load_unet(base_checkpoint_path, device)
    model: torch.nn.Module = loaded.model
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = torch.nn.BCELoss()

    last_loss = None
    for epoch in range(cfg.epochs):
        running = 0.0
        n = 0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            if out.ndim == 3:
                out = out.unsqueeze(1)

            loss = bce(out, masks) + _dice_loss(out, masks)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            n += 1

        last_loss = running / max(1, n)
        log(f"epoch {epoch + 1}/{cfg.epochs} loss={last_loss:.6f}")

    checkpoint = {"state_dict": model.state_dict()}
    torch.save(checkpoint, str(output_checkpoint_path))

    return {
        "num_samples": len(dataset),
        "final_loss": float(last_loss) if last_loss is not None else None,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "image_size": cfg.image_size,
    }
