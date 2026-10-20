# Loss construction. The class-balance logic itself lives in the NumPy-only class_balance module.
# This function only wraps its result into a torch loss.

import numpy as np
import torch
from torch import nn


def build_loss(pos_weight: np.ndarray | None = None) -> nn.Module:
    """
    BCEWithLogitsLoss for multi-label classification: one independent sigmoid per class.
    With pos_weight, the loss of a positive label of class c is multiplied by pos_weight[c].
    """
    weight = torch.as_tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None
    return nn.BCEWithLogitsLoss(pos_weight=weight)
