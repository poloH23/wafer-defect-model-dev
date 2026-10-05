import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import fit, overfit_single_batch, predict, set_seed

# ================= synthesised settings =================

NUM_FEATURES, NUM_CLASSES = 12, 3
CLASS_NAMES = ["a", "b", "c"]


# ================= Test Cases =================


def make_loader(num_samples: int, seed: int, shuffle: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, NUM_FEATURES, generator=generator)
    true_weights = torch.randn(NUM_FEATURES, NUM_CLASSES, generator=torch.Generator().manual_seed(0))
    y = (x @ true_weights > 0).float()
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=shuffle)


def tiny_model() -> nn.Module:
    return nn.Linear(NUM_FEATURES, NUM_CLASSES)


def test_predict_shapes_and_probability_range():
    loader = make_loader(50, seed=1)
    y_true, y_prob, loss = predict(tiny_model(), loader, nn.BCEWithLogitsLoss(), torch.device("cpu"))
    assert y_true.shape == y_prob.shape == (50, NUM_CLASSES)
    assert ((y_prob >= 0) & (y_prob <= 1)).all()
    assert loss > 0


def test_fit_learns_and_restores_best_weights():
    set_seed(0)
    model = tiny_model()
    loss_fn = nn.BCEWithLogitsLoss()
    val_loader = make_loader(200, seed=2)
    logged = []

    result = fit(
        model,
        make_loader(512, seed=1, shuffle=True),
        val_loader,
        loss_fn,
        torch.optim.Adam(model.parameters(), lr=0.05),
        epochs=15,
        device=torch.device("cpu"),
        class_names=CLASS_NAMES,
        on_epoch_end=lambda record, epoch: logged.append((epoch, record)),
    )

    assert result.best_score > 0.95
    assert [epoch for epoch, _ in logged] == list(range(1, result.epochs_run + 1))
    assert {"train_loss", "val_loss", "lr", "val_macro_ap", "val_ap_a"} <= set(logged[0][1])
    # The model returned by fit holds the best epoch's weights, not the last epoch's
    best_record = dict(logged)[result.best_epoch]
    assert best_record["val_macro_ap"] == pytest.approx(result.best_score)


def test_early_stopping_when_nothing_improves():
    model = tiny_model()
    # lr=0: the weights never change, so no epoch after the first can improve
    result = fit(
        model,
        make_loader(64, seed=1),
        make_loader(64, seed=2),
        nn.BCEWithLogitsLoss(),
        torch.optim.SGD(model.parameters(), lr=0.0),
        epochs=20,
        device=torch.device("cpu"),
        class_names=CLASS_NAMES,
        early_stopping_patience=3,
    )
    assert result.stopped_early
    assert result.best_epoch == 1
    assert result.epochs_run == 4


def test_unknown_monitor_raises():
    model = tiny_model()
    with pytest.raises(KeyError, match="monitor"):
        fit(
            model,
            make_loader(32, seed=1),
            make_loader(32, seed=2),
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            epochs=1,
            device=torch.device("cpu"),
            class_names=CLASS_NAMES,
            monitor="val_accuracy",
        )


def test_overfit_single_batch_fits_every_label():
    set_seed(0)
    x, y = next(iter(make_loader(32, seed=3)))
    model = tiny_model()
    result = overfit_single_batch(model, x, y, nn.BCEWithLogitsLoss(), torch.optim.Adam(model.parameters(), lr=0.1), torch.device("cpu"), steps=500)
    assert result["exact_match"] == 1.0
    assert result["final_loss"] < 0.1 * result["initial_loss"]
