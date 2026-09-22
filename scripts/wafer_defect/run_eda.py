from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from src.common.path import get_project_root
from src.pipelines.eda_pipeline import DatasetAnalysisPipeline  # noqa: E402

# ================= Project-specific Settings =================


# Array key names from official source data (docs/wafer_defect_desc/Description.pdf)
IMAGE_KEY = "arr_0"
LABEL_KEY = "arr_1"

# 8 basic defects
CLASS_NAMES = {
    "class_0": "Center",
    "class_1": "Donut",
    "class_2": "Edge_Loc",
    "class_3": "Edge_Ring",
    "class_4": "Loc",
    "class_5": "Near_Full",
    "class_6": "Scratch",
    "class_7": "Random",
}

# The pixel values ​​of the wafer map have only 3 discrete states
# 0: empty; 1: pass; 2: defect
WAFER_CMAP = ListedColormap(["#f5f5f5", "#8ecae6", "#023047"])
WAFER_NORM = BoundaryNorm([0, 1, 2, 3], WAFER_CMAP.N)

PROJECT_ROOT = get_project_root()
DATA_SOURCE = PROJECT_ROOT / "data" / "raw" / "Wafer_Map_Datasets.npz"
REPORT_DIR = PROJECT_ROOT / "reports" / "eda" / "wafer_defect"


def main() -> None:
    pipeline = DatasetAnalysisPipeline(report_dir=REPORT_DIR)
    pipeline.run(data_source=DATA_SOURCE, image_key=IMAGE_KEY, label_key=LABEL_KEY, plot_kwargs={"cmap": WAFER_CMAP, "norm": WAFER_NORM}, class_names=CLASS_NAMES)


if __name__ == "__main__":
    main()
