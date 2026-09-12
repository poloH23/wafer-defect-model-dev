from pathlib import Path
from typing import Literal


class DatasetAnalysisPipeline:
    """
    A general dataset analysis pipeline.
    It automatically identifies data sources, determines whether data packaging is required, and generates standardized analysis reports and sample image files.
    """

    def __init__(self, output_dir: str | Path):
        """
        Initialize the pipeline and automatically create the folder structure for storing reports.
        """
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / "eda" / "reports"
        self.sample_dir = self.report_dir / "samples"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        data_source: str | Path,
        csv_path: str | Path | None = None,
        image_col: str = "images",
        pack_format: Literal["npz", "h5"] = "npz",
    ):
        """
        The main entry point for pipeline execution.
        """
        data_source = Path(data_source)

        print("=" * 40)
        print("Starting Data Analysis Execution...")
        print("=" * 40)

        # 1. Prepare the image dataset
