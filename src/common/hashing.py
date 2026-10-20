# File fingerprints used for data lineage: a split records the hash of the data file it was made from,
# and training verifies that hash before using the split.

import hashlib
from pathlib import Path


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so large data files never have to fit in money."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
