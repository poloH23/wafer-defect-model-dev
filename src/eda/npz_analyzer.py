import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Union, Optional
from pathlib import Path

class NPZAnalyzer:
    """
    Data exploration and analysis module designed for the NPZ files.
    Supports CLI archiving and interactive display in Jupyter.
    """
    def __init__(self, npz_path: Path, image_key: str, label_key: str):
        self.npz_path = Path(npz_path).resolve()

        if not self.npz_path.exists() or self.npz_path.suffix != ".npz":
            raise FileNotFoundError("Cannot find the NPZ data file.")

        try:
            # Read the data
            data = np.load(self.npz_path)
            if image_key not in data.keys() or label_key not in data.keys():
                raise ValueError(f"Wrong keys. Available keys are: {data.files}")
            self.images = data[image_key]
            self.labels = data[label_key]

            # Determine the image number
            # Determine the image chanel and dimenssions (2D grayscale or 3D RGB)
            self.num_samples = self.images.shape[0]
            self.image_shape = self.images.shape[1:]

            # Determine the label type (1D singel-label or 2D multi-labels One-Hot)
            self._analyze_label_structure()
        except Exception as e:
            raise RuntimeError(f"An error occurred when rading the NPZ data: {e}")

    def _analyze_label_structure(self):
        """Determine whether the labels and class indices with shape (N,) or multi-labels with shape (N, C)."""
        if self.labels.ndim == 1:
            self.label_mode = "single-label"
            self.num_classes = len(np.unique(self.labels))
        elif self.labels.ndim == 2:
            # Determine whether the annotation is multi-labels or not
            max_labels_per_sample = np.max(np.sum(self.labels > 0, axis=1))
            self.label_mode = f"multi-label (max: {max_labels_per_sample})" if max_labels_per_sample > 1 else "one-hot-single-label"
            self.num_classes = self.labels.shape[1]
        else:
            raise ValueError(f"Unsupported label dimension: {self.labels.shape}")

    def get_distribution(self) -> Dict[str, int]:
        """Calculate the label distribution."""
        if self.labels.ndim == 1:
            unique, counts = np.unique(self.labels, return_counts=True)
            return {f"class_{u}": int(c) for u, c in zip(unique, counts)}
        else:
            counts = np.sum(self.labels, axis=0)
            return {f"class_{i}": c for i, c in enumerate(counts)}
    
    def get_label_type(self) -> str:
        """Returns the analyzed tag type to avoid redundant calculations and dimensionality errors."""
        return self.label_mode

    def create_sample_plot(
            self,
            index: int,
            cmap: Optional[str] = "gray",
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            custom_patches: Optional[List[mpatches.Patch]] = None
    ) -> plt.Figure: # type: ignore
        """Core plotting logic: Parameterize color and legend settings."""
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Index out of range. Valid range: 0 to {self.num_samples - 1}")

        img = self.images[index]
        label = self.labels[index]
        fig, ax = plt.subplots(figsize=(9, 8))

        # If the image is 3D and the channel dimension is at the beginning (C, H, W), 
        # convert it to the (H, W, C) format supported by imshow.
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1: # single-channel flattening
                img = img.squeeze(-1)

        # Plotting parameters
        im_kwargs = {}
        if img.ndim == 2: # grayscale or single-channel
            im_kwargs['cmap'] = cmap
        if vmin is not None: im_kwargs['vmin'] = vmin
        if vmax is not None: im_kwargs['vmax'] = vmax

        im = ax.imshow(img, **im_kwargs)
        ax.set_title(f"Sample #{index} | Label: {label}")
        ax.axis('off')

        if custom_patches:
            ax.legend(handles=custom_patches, bbox_to_anchor=(1.05, 1), loc="upper left")
            
        fig.tight_layout()
        return fig

    def show_sample(self, index: int, **kwargs):
        """Display the figure for Jupyter Notebook."""
        fig = self.create_sample_plot(index, **kwargs)
        plt.show()
        plt.close(fig)

    def save_sample(self, index: int, save_path: Union[str, Path]):
        """Save the figures when analyzing with CLI or Jupyter Notebook."""
        save_path = Path(save_path)
        if save_path is None:
            raise ValueError("The argument 'save_path' cannot be empty.")

        save_dir = save_path.parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        fig = self.create_sample_plot(index)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path
