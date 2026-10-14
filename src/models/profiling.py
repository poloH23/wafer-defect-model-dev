"""
Hardware-independent size and compute measures of a model. Measured latency depends on the
hardware and is noisy; these numbers are exact and comparable across machines, so they serve as
the efficiency objective during the search. Latency is measured separately, on the final candidates.
"""

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """
    Floating-point operations of one forward pass on an input of `input_shape` (batch included),
    counted by torch's FlopCounterMode: a multiply-accumulate counts as 2 FLOPs. Only matrix
    multiplications and convolutions are counted, the usual convention; normalization layers and
    activations, cheap by comparison, are not.
    """
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(torch.zeros(input_shape, device=device))
    finally:
        model.train(was_training)
    return int(counter.get_total_flops())
