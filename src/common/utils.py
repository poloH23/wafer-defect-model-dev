import time
import logging
from pathlib import Path
from functools import wraps
from datetime import datetime
from typing import Any, Callable, Union

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

FMT_FILNAME = "%Y%m%d_%H%M%S" # format for file name, e.g., 20260123_150800.
FMT_DATE_ONLY = "%Y%m%d" # format for date, e.g., 20260123.
FMT_LOG = "%Y-%m-%d %H:%M:%S" # format for log (easily read), e.g., 2026-01-23 15:08:00.

# --- Time Format and Timestamp Process ---
def get_timestamp_str(fmt: str = FMT_FILNAME) -> str:
    """
    Get the current timestamp, e.g., 20260123_150800.

    Args:
        fmt: defaults to FMT_FILNAME
    Return:
        The timestamp string.
    """
    return datetime.now().strftime(fmt)

def add_timestamp_to_filename(
        filename: Union[str, Path],
        fmt: str = FMT_FILNAME,
        sep: str = '_'
) -> str:
    """
    Add the timestamp at the filename.
    
    Example:
        input: "report.png"
        output: "report_20260123_150800.png"
        
        input: "data/raw.csv"
        output: "data/raw_20260123_150800.csv"
    Args:
        filename: target to add the timestamp
        fmt: defaults to FMT_FILNAME
        sep: defaults to '_'
    Returns:
        The filename with a timestamp.
    """
    path = Path(filename)
    timestamp = get_timestamp_str(fmt)
    new_stem = f"{path.stem}{sep}{timestamp}"
    return str(path.with_name(f"{new_stem}{path.suffix}"))

def timer_duration_str(start: float, end: float) -> str:
    return f"{end - start:.2f}"

# --- Performance Timer ---
def timer_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorators can be used to enable automatic timing for any function, thus increasing its versatility.

    Usage:
        @timer_decorator
        def my_process(data):
            ...
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            duration = timer_duration_str(end_time, start_time)
            logger.info(f"[Timer] Function '{func.__name__}' finished in {duration} seconds.")
    return wrapper

class Timer:
    """
    Calculate the time duration of a specific block.

    Usage:
        with Timer("Data Loading"):
            df = pd.read_csv(...)
    """
    def __init__(self, name: str = "Block"):
        self.name = name
        self.start_time: float = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.info(f"[Timer] Start: {self.name}...")
        return self
    
    def __exit__(self):
        end_time = time.perf_counter()
        duration = timer_duration_str(self.start_time, end_time)
        logger.info(f"[Timer] Done: {self.name}. Cost: {duration}")
