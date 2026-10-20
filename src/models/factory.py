"""
Build classification backbones adapted to this task's input size, channels and number of labels.
"""

from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn
from torchvision.models.resnet import BasicBlock

from src.common.config_schema import ModelConfig

SUPPORTED_MODELS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "slim_resnet",
)

# Every supported architecture downsamples its input 32x in total, in five stride-2 steps, a design
# made for 224x224 photos. Removing the first two steps keeps a 52x52 map at 7x7 before the
# classifier instead of 2x2, the same as the small-input ResNet stem.
SMALL_INPUT_REMOVED_DOWNSAMPLING = 2


class SlimResNet(nn.Module):
    """
    A ResNet whose width and depth are configurable, built from torchvision's BasicBlock.

    width: channels of the first stage; each following stage doubles them (width, 2w, 4w, 8w).
    depths: number of residual blocks in each of the four stages.

    With width=64, depths=(2, 2, 2, 2) it is exactly ResNet-18. Narrower and shallower versions
    keep ResNet's structure, a few dense convolutions that GPUs run efficiently, while cutting FLOPs
    toward the level of mobile architectures.
    """

    def __init__(self, num_classes: int, in_channels: int, width: int = 64, depths: tuple[int, ...] = (2, 2, 2, 2), small_input_stem: bool = True):
        super().__init__()
        if len(depths) != 4 or min(depths) < 1:
            raise ValueError(f"depths must give at least one block for each of the 4 stages, got {depths}")
        if small_input_stem:
            # Same as the small-input stem of build_model: 3x3 stride-1 conv, no max pool
            self.stem = nn.Sequential(nn.Conv2d(in_channels, width, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, width, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        stages, channels = [], width
        for index, depth in enumerate(depths):
            out_channels = width * 2**index
            stride = 1 if index == 0 else 2
            downsample = None
            if stride != 1 or channels != out_channels:
                downsample = nn.Sequential(nn.Conv2d(channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
            blocks = [BasicBlock(channels, out_channels, stride=stride, downsample=downsample)]
            blocks += [BasicBlock(out_channels, out_channels) for _ in range(depth - 1)]
            stages.append(nn.Sequential(*blocks))
            channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

        # Same initialization as torchvision's ResNet
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.pool(self.stages(self.stem(x))), 1))


def build_model(
    name: str,
    num_classes: int,
    in_channels: int,
    pretrained: bool = False,
    small_input_stem: bool = True,
    width: int = 64,
    depths: tuple[int, ...] = (2, 2, 2, 2),
) -> nn.Module:
    """
    A torchvision architecture with its input layer, early downsampling and classifier head
    adapted to the task.

    small_input_stem:
        ResNet: the ImageNet stem (7x7 conv stride 2 + max pool) is replaced by a 3x3 stride-1 conv
        without max pool (the CIFAR-style stem). Other architectures: their first two stride-2
        operations become stride 1. Either way a 52x52 input reaches the classifier as 7x7.

    pretrained:
        Loads ImageNet weights. Layers that are replaced (the first conv when the number of input
        channels changes, the ResNet stem, the classifier head) are re-initialized.

    width, depths:
        Only used by slim_resnet (see SlimResNet), which is always trained from scratch.

    The head outputs raw logits (no sigmoid): BCEWithLogitsLoss applies the sigmoid itself,
    in a numerically stable way.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{name}', choose from {SUPPORTED_MODELS}")
    if name == "slim_resnet":
        if pretrained:
            raise ValueError("slim_resnet has no pretrained weights")
        return SlimResNet(num_classes, in_channels, width=width, depths=tuple(depths), small_input_stem=small_input_stem)

    model = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
    if name.startswith("resnet"):
        # Kept exactly as before, so checkpoints of earlier runs still load into the same architecture
        _adapt_resnet_stem(model, name, in_channels, small_input_stem)
    else:
        if small_input_stem:
            _remove_first_downsampling(model, SMALL_INPUT_REMOVED_DOWNSAMPLING)
        _set_input_channels(model, in_channels)
    _replace_last_linear(model, num_classes)
    return model


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the submodule at a dotted path such as "features.0.0"."""
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def _adapt_resnet_stem(model: nn.Module, name: str, in_channels: int, small_input_stem: bool) -> None:
    # get_model is typed to return a plain nn.Module, so type checkers cannot know conv1 exists.
    # The isinstance check tells them, and fails with a clear message otherwise.
    conv1 = model.conv1
    if not isinstance(conv1, nn.Conv2d):
        raise TypeError(f"'{name}' does not have the expected ResNet stem (conv1)")
    if small_input_stem or in_channels != 3:
        kernel_size, stride, padding = (3, 1, 1) if small_input_stem else (7, 2, 3)
        model.conv1 = nn.Conv2d(in_channels, conv1.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if small_input_stem:
        model.maxpool = nn.Identity()


def _remove_first_downsampling(model: nn.Module, count: int) -> None:
    """
    Make the first `count` downsampling operations stride 1, in the order the modules are defined,
    which for the supported architectures is the forward order. A stride-2 convolution keeps its
    padding, so it now preserves the spatial size; a stride-2 max pool is removed.

    Only safe for the stem and first stage: deeper stages of some architectures (e.g. the two
    parallel branches of a ShuffleNet block) must be changed together to keep shapes consistent.
    """
    targets = []
    for name, module in model.named_modules():
        if len(targets) == count:
            break
        if _is_downsampling(module):
            targets.append((name, module))
    if len(targets) < count:
        raise ValueError(f"Found only {len(targets)} downsampling operations, expected at least {count}")

    for name, module in targets:
        if isinstance(module, nn.Conv2d):
            module.stride = (1, 1)
        else:
            _set_module(model, name, nn.Identity())


def _is_downsampling(module: nn.Module) -> bool:
    """A stride-2 convolution or max pool, i.e. a step that halves the spatial size."""
    if isinstance(module, nn.Conv2d):
        return module.stride == (2, 2)
    if isinstance(module, nn.MaxPool2d):
        return module.stride in (2, (2, 2))
    return False


def _set_input_channels(model: nn.Module, in_channels: int) -> None:
    """Replace the first convolution when the number of input channels differs; everything else is kept."""
    name, conv = next((n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d))
    if conv.in_channels == in_channels:
        return
    _set_module(
        model,
        name,
        nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        ),
    )


def _replace_last_linear(model: nn.Module, num_classes: int) -> None:
    """Replace the final fully connected layer (the classifier head) with one producing `num_classes` logits."""
    name, linear = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    _set_module(model, name, nn.Linear(linear.in_features, num_classes))


def build_model_from_config(model_cfg: ModelConfig, num_classes: int, in_channels: int) -> nn.Module:
    """
    The single place where a model config becomes a model. Training, benchmarking and checkpoint
    loading all go through it, so a new architecture option cannot be forgotten in one of them.
    """
    return build_model(
        model_cfg.name,
        num_classes,
        in_channels,
        pretrained=model_cfg.pretrained,
        small_input_stem=model_cfg.small_input_stem,
        width=model_cfg.width,
        depths=tuple(model_cfg.depths),
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """
    Rebuild a model from a checkpoint saved by scripts/train.py, in evaluation mode.

    The checkpoint stores everything needed to rebuild the architecture, so evaluation never depends
    on config files that may have changed since training. pretrained=False because every weight is
    restored from the checkpoint anyway, so nothing needs to be downloaded.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    # Checkpoints saved before slim_resnet existed have no width/depths: the config defaults apply.
    # pretrained=False: every weight comes from the checkpoint, nothing needs to be downloaded.
    model_cfg = ModelConfig.model_validate({**checkpoint["model_config"], "pretrained": False})
    model = build_model_from_config(model_cfg, checkpoint["num_classes"], checkpoint["in_channels"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
