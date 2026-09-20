from pathlib import Path
from typing import Any, Literal

from src.eda.img_packer import ImagePacker
from src.eda.npz_analyzer import NPZAnalyzer


class DatasetAnalysisPipeline:
    """
    A general dataset analysis pipeline.
    It automatically identifies data sources, determines whether data packaging is required, and generates standardized analysis reports and sample image files.
    """

    def __init__(self, report_dir: str | Path, data_dir: str | Path | None = None):
        """
        Initialize the pipeline and automatically create the folder structure for storing reports.
        """
        self.report_dir = Path(report_dir)
        self.sample_dir = self.report_dir / "samples"
        self.data_dir = Path(data_dir) if data_dir is not None else self.report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        data_source: str | Path,
        csv_path: str | Path | None = None,
        image_key: str = "images",
        label_key: str = "labels",
        pack_format: Literal["npz", "h5"] | None = "npz",
        pure_class_examples: bool = True,
    ):
        """
        The main entry point for pipeline execution.
        """
        if csv_path is not None:
            csv_path = Path(csv_path)
        data_source = Path(data_source)

        print("=" * 40)
        print("Starting Data Analysis Execution...")
        print("=" * 40)

        # 1. Prepare the image dataset
        target_file = self._prepare_data(data_source, csv_path, image_key, label_key, pack_format)

        # 2. execute data analysis and output reports
        print(f"\nStarting dataset analysis: {target_file.name}")
        self._analysis_and_report(target_file, image_key, label_key, pure_class_examples)
        print(f"\nDataset analysis finished, reports saved: {self.report_dir}")

    def _prepare_data(
        self, data_source: Path, csv_path: Path | None, image_key: str = "images", label_key: str = "labels", pack_format: Literal["npz", "h5"] | None = None
    ) -> Path:
        """
        Determine whether packaging is required and return the path to the final file ready for analysis.
        """
        if data_source.is_dir():
            print("'data_source' is a directory, starting dataset packaging with ImagePacker...")
            if not csv_path:
                raise ValueError("'csv_path' is required when 'data_source' is a directory")
            packed_file_path = self.data_dir / f"packed_dataset.{pack_format}"

            # Start packaging
            packer = ImagePacker()
            packer.pack(data_source, csv_path, Path(packed_file_path), image_key, label_key, pack_format)
            return packed_file_path
        elif data_source.is_file() and data_source.suffix in [".npz", ".h5", ".hdf5"]:
            print(f"This data is a packed filed already, skipping the packaging process: {data_source.suffix}")
            return data_source
        else:
            raise ValueError(f"Unsupported data format: {data_source}")

    def _analysis_and_report(self, target_file: Path, image_key: str, label_key: str, pure_class_examples: bool = True, plot_kwargs: dict[str, Any] | None = None) -> None:
        """Execute the analysis using NPZAnalyzer and output the Markdown report."""
        # Protection Mechanism: NPZ file supported only
        if target_file.suffix != ".npz":
            print("Warning: The current analyzer module only supports .npz files.")
            print(f"The file {target_file.name} will not generate an analysis report, but it is ready for subsequent training.")
            return

        analyzer = NPZAnalyzer(npz_path=target_file, image_key=image_key, label_key=label_key)

        # Get the statistics info
        num_samples = analyzer.num_samples
        img_shape = analyzer.image_shape
        label_type = analyzer.get_label_type()
        distribution = analyzer.get_distribution()

        # Sample images to save (3 images represent the initial, intermediate, and final stages, respectively)
        print("Capturing and saving sampled image archive...")

        class_representative = analyzer.get_class_representative_indices(prefer_pure=pure_class_examples)
        saved_samples: dict[str, str] = dict()

        plot_kwargs = plot_kwargs or {}
        for class_name, idx in class_representative.items():
            save_path = self.sample_dir / f"sample_{class_name}.png"
            analyzer.save_sample(index=idx, save_path=save_path, **plot_kwargs)
            saved_samples[class_name] = save_path.name

        # Generate the Markdown report
        report_content = self._generate_markdown(target_file.name, num_samples, img_shape, label_type, distribution, saved_samples)

        # Save into the report file
        report_path = self.report_dir / "dataset_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

    def _generate_markdown(self, filename: str, num_samples: int, img_shape: Any, label_type: str, distribution: dict[str, int], saved_samples: dict[str, str]) -> str:
        """Pure string manipulation: assembling Markdown content"""
        md = "# Automated Dataset Analysis Report (EDA Report)\n\n"
        md += "## Basic Information\n"
        md += f"- **File Source**: `{filename}`\n"
        md += f"- **Total Samples**: `{num_samples}`\n"
        md += f"- **Image Dimension**: `{img_shape}`\n"
        md += f"- **Label Type**: `{label_type}`\n\n"

        md += "## Class Distribution\n"
        md += "| Class | Number |\n|---|---|\n"
        for k, v in distribution.items():
            md += f"| {k} | {v} |\n"

        md += "\n## Sampling Inspection\n"
        for class_name, img_name in saved_samples.items():
            md += f"### {class_name}\n"
            md += f"![Sample Image](samples/{img_name})\n\n"
        return md
