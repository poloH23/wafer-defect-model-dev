import pytest
import numpy as np
from pathlib import Path
from src.eda.img_reader import ImageReader

@pytest.fixture
def mock_hdf5_file(tmp_path: Path):
    """Fixture: Create a temporary HDF5 file."""
    import h5py
    file_path = tmp_path / "mock.h5"
    fake_img_data = np.random.randint(0, 255, size=(3, 244, 244), dtype=np.uint8)
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("images", data=fake_img_data)
    return file_path

# ================= Test Cases =================

def test_read_hdf5_success(mock_hdf5_file: Path):
    reader = ImageReader(output_type="numpy")
    img, metadata = reader.read(file_path=mock_hdf5_file, dataset_name="images")
    assert isinstance(img, np.ndarray), "the images datatype should be NumPy array."
    assert img.shape == (3, 244, 244), "the shape of the image should be (3, 244, 244)"
    assert metadata["format"] == "hdf5"
    assert metadata["original_shape"] == (3, 244, 244)

def test_read_hdf5_missing_dataset(mock_hdf5_file: Path):
    reader = ImageReader()
    with pytest.raises(ValueError) as excinfo:
        reader.read(file_path=mock_hdf5_file, dataset_name="wrong_dataset_name")

def test_read_hdf5_output_torch(mock_hdf5_file: Path):
    import torch
    reader = ImageReader(output_type="torch")
    img, metadata = reader.read(file_path=mock_hdf5_file, dataset_name="images")
    assert isinstance(img, torch.Tensor)
    assert img.shape == torch.Size([3, 244, 244])
