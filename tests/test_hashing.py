import hashlib

from src.common.hashing import file_sha256

# ================= Test Cases =================


def test_matches_hashlib_across_chunks(tmp_path):
    data = bytes(range(256)) * 1000
    path = tmp_path / "data.bin"
    path.write_bytes(data)
    # A tiny chunk size forces many reads, checking that chunks are joined correctly
    assert file_sha256(path, chunk_size=1000) == hashlib.sha256(data).hexdigest()
