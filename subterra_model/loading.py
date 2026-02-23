from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from .models.unet import UNet


ModelName = Literal["unet", "yolo"]


@dataclass(frozen=True)
class LoadedModel:
    name: ModelName
    model: object
    device: torch.device


def _load_unet(checkpoint_path: str, device: torch.device) -> LoadedModel:
    model = UNet(in_channels=3, out_channels=1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return LoadedModel(name="unet", model=model, device=device)


def _load_yolo(checkpoint_path: str, device: torch.device) -> LoadedModel:
    from ultralytics import YOLO

    model = YOLO(checkpoint_path, task="segment")
    return LoadedModel(name="yolo", model=model, device=device)


def load_model(name: ModelName, checkpoint_path: str, *, device: Optional[str] = None) -> LoadedModel:
    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "unet":
        return _load_unet(checkpoint_path, resolved_device)
    if name == "yolo":
        return _load_yolo(checkpoint_path, resolved_device)
    raise ValueError(f"Unsupported model name: {name}")

