from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from src.eda.img_packer import ImagePacker


@pytest.fixture
def mock_raw_dataset(tmp_path: Path):
    """Fixture: Create a temporary dynamic dataset."""
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Randomly generate 2 52x52 dummy images and save them in .npy format.
    img1 = np.random.randint(0, 255, size=(52, 52), dtype=np.uint8)
    img2 = np.random.randint(0, 255, size=(52, 52), dtype=np.uint8)

    np.save(image_dir / "image_001.npy", img1)
    np.save(image_dir / "image_002.npy", img2)

    csv_path = tmp_path / "labels.csv"
    df = pd.DataFrame(
        {
            "images": ["image_001.npy", "image_002.npy"],
            "defect_0": [1, 0],
            "defect_1": [1, 0],
        }
    )
    df.to_csv(csv_path, index=False)
    return image_dir, csv_path


# ================= Test Cases =================


def test_pack_to_npz(mock_raw_dataset, tmp_path: Path):
    image_dir, csv_path = mock_raw_dataset
    output_npz = tmp_path / "output_dataset.npz"

    packer = ImagePacker()
    packer.pack(
        image_dir=image_dir,
        csv_path=csv_path,
        output_path=output_npz,
        image_col="images",
        label_key="labels",
        output_format="npz",
    )

    assert output_npz.exists(), "Cannot generate NPZ file."

    # Check the dimesion and content of the output file
    data = np.load(output_npz)
    assert "images" in data.files, "images array is missing in NPZ file."
    assert "labels" in data.files, "labels array is missing in NPZ file."

    # 2 52x52 images and 2 labels for each image
    assert data["images"].shape == (2, 52, 52)
    assert data["labels"].shape == (2, 2)


def test_pack_to_h5(mock_raw_dataset, tmp_path: Path):
    image_dir, csv_path = mock_raw_dataset
    output_h5 = tmp_path / "output_dataset.h5"

    packer = ImagePacker()
    packer.pack(
        image_dir=image_dir,
        csv_path=csv_path,
        output_path=output_h5,
        image_col="images",
        label_key="labels",
        output_format="h5",
    )

    assert output_h5.exists(), "Cannot generate HDF5 file."

    with h5py.File(output_h5, "r") as f:
        assert "images" in f, "images Dataset is missing in HDF5 file."
        assert "labels" in f, "labels Dataset is missing in HDF5 file."

        assert f["images"].shape == (2, 52, 52)  # type: ignore
        assert f["labels"].shape == (2, 2)  # type: ignore


def test_pack_missing_csv(tmp_path: Path):
    packer = ImagePacker()

    with pytest.raises(FileNotFoundError):
        packer.pack(
            image_dir=tmp_path,
            csv_path=tmp_path / "not_exist.csv",
            output_path=tmp_path / "out.npz",
        )
