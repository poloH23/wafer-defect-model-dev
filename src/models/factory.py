# Build classification backbones adapted to the task's input size, channels and number of labels.

from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn

SUPPROTED_MODELS = ("resnet18", "resnet34", "resnet50")


def build_model(name: str, num_classes: int, in_channels: int, pretrained: bool = False, small_input_stem: bool = True) -> nn.Module:
    """
    ResNet with its input stem and classifier head adapted to the task.

    small_input_stem:
        The ImageNet stem (7x7 conv stride 2 + 3x3 max pool) shrinks the input 4x before the first residual block.
        That suits 244x244 photos, turns a 52x52 map into 13x13 immediately.
        The CIFAR-style replacement (3x3 conv stride 1, no max pool) keeps the special detail:
        52 -> 52 -> 26 -> 13 -> 7 through the 4 stages.

    pretrained:
        Loads ImageNet weight. If the stem or the number of input channels changes, conv1 is re-initialized and only
        the remaining layers keep their pretrained weights.

    The head outputs raw logits (no sigmoid): BCEWithLogitsLoss applies the sigmoid itself, in a numerically stable way.
    """
    if name not in SUPPROTED_MODELS:
        raise ValueError(f"Unsupported model '{name}', choose from {SUPPROTED_MODELS}")

    model = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)

    if small_input_stem or in_channels != 3:
        kernel_size, stride, padding = (3, 1, 1) if small_input_stem else (7, 2, 3)

        model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=model.conv1.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if small_input_stem:
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """
    Rebuild a model from a checkpoint saved by scripts/train.py, in evaluation mode.

    The checkpoint stores everything needed to rebuild the architecture, so evaluation never depends
    on config files that may have changed since training. pretrained=False because every weight is
    restored from the checkpoint anyway, so nothing needs to be downloaded.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_cfg = checkpoint["model_config"]
    model = build_model(model_cfg["name"], checkpoint["num_classes"], checkpoint["in_channels"], pretrained=False, small_input_stem=model_cfg["small_input_stem"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
