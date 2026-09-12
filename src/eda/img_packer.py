from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.eda.img_reader import ImageReader


class ImagePacker:
    """
    Dataset packer.
    Package the images and CSV label mapping tables into efficient npz or hdf5 formats.
    """

    def __init__(self, reader: ImageReader | None = None):
        """
        Allows passing an external reader; if none is provided, a reader that outputs a NumPy array is instantiated by default.
        """
        self.reader = reader or ImageReader(output_type="numpy")

    def pack(
        self,
        image_dir: str | Path,
        csv_path: str | Path,
        output_path: str | Path,
        image_col: str = "images",
        output_format: Literal["npz", "h5"] | None = None,
    ) -> None:
        """
        Encapsulates the core execution functions of the dataset.
        """
        image_dir = Path(image_dir)
        csv_path = Path(csv_path)
        output_path = Path(output_path).resolve()

        # Determine the output format or auto-determine depending on the extension
        if output_format is None:
            ext = output_path.suffix.lower()
            if ext in [".h5", ".hdf5"]:
                output_format = "h5"
            elif ext == ".npz":
                output_format = "npz"
            else:
                raise ValueError("Cannot determine the output format by the extension, please specify the argument 'output_format' ('npz' 或 'h5')")

        # Read the CSV file (image names and labels)
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find the CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        if image_col not in df.columns:
            raise ValueError(f"Cannot find the image column name in the CSV file: {image_col}")

        # Load the images and labels
        image_list, label_list, error_count = self._load_data(df, image_dir, image_col)

        if len(image_list) == 0:
            raise RuntimeError("Cannot successfully read any images, failed to package the image dataset.")

        # Stack into an array
        print("Stacking the images into an array")
        final_images = np.stack(image_list)
        final_labels = np.stack(label_list)
        print(f"   - image dimension: {final_images.shape}")
        print(f"   - label dimension: {final_labels.shape}")

        # Check the directory path and ouput results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "npz":
            self._save_as_npz(final_images, final_labels, output_path)
        elif output_format == "h5":
            self._save_as_h5(final_images, final_labels, output_path)
        print(f"Packaging porcess is successfully done: {output_path}")

    def _load_data(self, df: pd.DataFrame, image_dir: Path, image_col: str) -> tuple[list[str], list[str], int]:
        """Perform image and label reading operations."""
        image_list = list()
        label_list = list()
        error_count = 0

        print(f"Start converting data: {len(df)}")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_name = row[image_col]
            img_path = image_dir / str(img_name)
            labels = row.drop(labels=[image_col]).values.astype(np.int32)
            try:
                img, _ = self.reader.read(img_path)
                image_list.append(img)
                label_list.append(labels)
            except Exception:
                error_count += 1
        return image_list, label_list, error_count

    def _save_as_npz(self, images: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
        """ "Saved dataset into the NPZ format file."""
        print("Writing NPZ archive...")
        np.savez_compressed(output_path, images=images, labels=labels)
        return None

    def _save_as_h5(self, images: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
        """ "Stored in HDF5 format (using GZIP compression)."""
        print("Writing HDF5 archive...")
        with h5py.File(output_path, "w") as f:
            f.create_dataset("images", data=images, compression="gzip", compression_opts=4)
            f.create_dataset("labels", data=labels, compression="gzip", compression_opts=4)
