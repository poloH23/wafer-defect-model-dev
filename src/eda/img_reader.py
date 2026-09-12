from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch


class ImageReader:
    """
    Universal image reader and standardization adapter.
    Suports various scientific and industrial image formats, and outputs image arrays and metadata.
    """

    SUPPORTED_FORMATS = {
        "cv2": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
        "numpy": [".npz", ".npy"],
        "hdf5": [".h5", ".hdf5"],
        "raw": [".raw", ".cr2", ".nef"],
    }

    def __init__(self, output_type: Literal["numpy", "torch"] = "numpy"):
        """
        Initialize the ImageReader.
        :param output_type: Specify the output type, either "numpy" for NumPy array or "torch" for PyTorch tensor.
        """
        self.output_type = output_type
        self.metadata = {
            "original_shape": None,
            "format": None,
            "dtype": None,
            "labels": None,
            "extra_info": dict(),
        }

    def _convert_type(self, data: np.ndarray) -> np.ndarray | torch.Tensor:
        """
        Internal method: Convert the data to the specified output type according to the setting.
        :param data: Input image data as a NumPy array.
        """
        if self.output_type == "torch":
            return torch.from_numpy(data)
        return data

    def read(self, file_path: str | Path, **kwargs) -> tuple[np.ndarray | torch.Tensor, dict[str, Any]]:
        """
        A single entry point for image retrieval.
        :return: A tuple containing the image data and metadata (image data, metadata).
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix in self.SUPPORTED_FORMATS["cv2"]:
            img_array, metadata = self._read_cv2(path)
        elif suffix in self.SUPPORTED_FORMATS["numpy"]:
            img_array, metadata = self._read_numpy(path, **kwargs)
        elif suffix in self.SUPPORTED_FORMATS["hdf5"]:
            img_array, metadata = self._read_hdf5(path, **kwargs)
        elif suffix in self.SUPPORTED_FORMATS["raw"]:
            img_array, metadata = self._read_raw(path)
        else:
            raise ValueError(f"Unsupported image format: {suffix}")

        final_img = self._convert_type(img_array)
        if metadata.get("labels") is not None and isinstance(metadata["labels"], np.ndarray):
            metadata["labels"] = self._convert_type(metadata["labels"])
        return final_img, metadata

    def supported_formats(self) -> dict[str, list[str]]:
        return self.SUPPORTED_FORMATS

    # ========== Proprietary format reading logic ==========

    def _read_cv2(self, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        """CV2 defaults to reading in BGR; convert to RGB."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"CV2 Unable to read the file: {path}")
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.metadata.update({"original_shape": img.shape, "format": path.suffix.lower(), "dtype": str(img.dtype)})
        return img, self.metadata

    def _read_numpy(self, path: Path, image_key: str = "arr_0", label_key: str = "arr_1") -> tuple[np.ndarray, dict[str, Any]]:
        if path.suffix == ".npy":
            img = np.load(path)
            labels = None
        else:
            # .npz file
            data = np.load(path)
            img = data[image_key]
            labels = data[label_key] if label_key in data.files else None
        self.metadata.update(
            {
                "original_shape": img.shape,
                "format": path.suffix.lower(),
                "dtype": str(img.dtype),
                "labels": labels,
            }
        )
        return img, self.metadata

    def _read_hdf5(self, path: Path, dataset_name: str = "images") -> tuple[np.ndarray | Any, dict[str, Any]]:
        import h5py

        with h5py.File(path, "r") as f:
            if dataset_name not in f:
                raise ValueError(f"HDF5 files not found: {dataset_name}")
            img = f[dataset_name][:]  # type: ignore
            if isinstance(img, np.ndarray):
                self.metadata.update(
                    {
                        "original_shape": img.shape,
                        "format": path.suffix.lower(),
                        "dtype": str(img.dtype),
                        "extra_info": {"group_keys": list(f.keys())},
                    }
                )
        return img, self.metadata

    def _read_raw(self, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            img = raw.postprocess()
            self.metadata.update(
                {
                    "original_shape": img.shape,
                    "format": path.suffix.lower(),
                    "dtype": str(img.dtype),
                }
            )
        return img, self.metadata
