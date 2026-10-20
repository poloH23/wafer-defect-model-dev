# A checkpoint saved in the same format as scripts/train.py must rebuild an identical model.

import torch

from src.models.factory import build_model, load_checkpoint


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = build_model("resnet18", num_classes=8, in_channels=3, small_input_stem=True).eval()
    path = tmp_path / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {"name": "resnet18", "pretrained": False, "small_input_stem": True},
            "num_classes": 8,
            "in_channels": 3,
            "encoding": "onehot",
            "class_names": [f"c{i}" for i in range(8)],
            "best_epoch": 1,
        },
        path,
    )

    restored, checkpoint = load_checkpoint(path, torch.device("cpu"))
    x = torch.randn(2, 3, 52, 52)
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x))
    assert not restored.training
    assert checkpoint["encoding"] == "onehot"
