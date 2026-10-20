import numpy as np
import torch

from src.training.losses import build_loss


def test_plain_bce_without_pos_weight():
    loss = build_loss(None)
    assert loss.pos_weight is None


def test_pos_weight_scales_positive_labels_only():
    logits = torch.zeros(1, 2)
    positive = torch.ones(1, 2)
    negative = torch.zeros(1, 2)
    weighted = build_loss(np.array([1.0, 3.0], dtype=np.float32))
    plain = build_loss(None)
    # Positive targets: the class-1 term is tripled, so the mean grows from L to (L + 3L) / 2 = 2L
    assert torch.isclose(weighted(logits, positive), 2 * plain(logits, positive))
    # Negative targets are unaffected by pos_weight
    assert torch.isclose(weighted(logits, negative), plain(logits, negative))
