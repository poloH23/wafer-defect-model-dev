import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

os.environ["DATA_DIR"] = "tests/mock_data"
os.environ["MODEL_SAVE_PATH"] = "tests/mock_outputs"

print(f"\n[Pytest Setup] Project directory has been set: {project_root}")
