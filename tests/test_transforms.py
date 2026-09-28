import numpy as np
import pytest

from src.data.transforms import RandomDihedral, apply_dihedral, encode_discrete

# ================= fixture: synthesised data =================


@pytest.fixture
def asymmetric_map() -> np.ndarray:
    """A 4x4 map with no rotational or mirror symmetry, so all 8 D4 variants ate distinct."""
    return np.array(
        [
            [0, 1, 1, 2],
            [0, 0, 1, 2],
            [2, 0, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )


# ================= encode_discrete =================


class TestEncodeDiscrete:
    def test_onehot_shape_and_exclusivity(self, asymmetric_map):
        x = encode_discrete(asymmetric_map, num_values=3, mode="onehot")
        assert x.shape == (3, 4, 4)
        assert x.dtype == np.float32
        assert np.array_equal(x.sum(axis=0), np.ones((4, 4)))  # exactly one channel active per pixel
        assert np.array_equal(x.argmax(axis=0), asymmetric_map)  # lossless: original map recoverable

    def test_scaled_shape_and_range(self, asymmetric_map):
        x = encode_discrete(asymmetric_map, num_values=3, mode="scaled")
        assert x.shape == (1, 4, 4)
        assert x.dtype == np.float32
        assert set(np.unique(x).tolist()) == {0.0, 0.5, 1.0}

    def test_value_out_of_range_raises(self, asymmetric_map):
        with pytest.raises(ValueError, match="Pixel values"):
            encode_discrete(asymmetric_map, num_values=2)  # the map contains the value 2

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError, match="2D"):
            encode_discrete(np.zeros((1, 4, 4), dtype=np.uint8), num_values=3)

    def test_unknown_mode_raises(self, asymmetric_map):
        with pytest.raises(ValueError, match="Unsupported"):
            encode_discrete(asymmetric_map, num_values=3, mode="bogus")  # type: ignore[arg-type]


# ========== apply_dihedral ==========


class TestApplyDihedral:
    def test_eight_distinct_variants(self, asymmetric_map):
        variants = {apply_dihedral(asymmetric_map, k, flip).tobytes() for k in range(4) for flip in (False, True)}
        assert len(variants) == 8

    def test_values_are_preserved(self, asymmetric_map):
        for k in range(4):
            for flip in (False, True):
                out = apply_dihedral(asymmetric_map, k, flip)
                assert out.dtype == asymmetric_map.dtype
                assert np.array_equal(np.sort(out, axis=None), np.sort(asymmetric_map, axis=None))

    def test_returns_contiguous_array(self, asymmetric_map):
        assert apply_dihedral(asymmetric_map, 1, True).flags["C_CONTIGUOUS"]

    def test_rotation_is_invertible(self, asymmetric_map):
        rotated = apply_dihedral(asymmetric_map, 1, False)
        assert np.array_equal(apply_dihedral(rotated, 3, False), asymmetric_map)

    def test_same_operation_on_every_channel(self, asymmetric_map):
        encoded = encode_discrete(asymmetric_map, num_values=3)
        out = apply_dihedral(encoded, 1, True)
        for c in range(3):
            assert np.array_equal(out[c], apply_dihedral(encoded[c], 1, True))

    def test_commutes_with_encoding(self, asymmetric_map):
        # Augmenting before or after encoding gives the same result
        a = apply_dihedral(encode_discrete(asymmetric_map, 3), 3, True)
        b = encode_discrete(apply_dihedral(asymmetric_map, 3, True), 3)
        assert np.array_equal(a, b)


# ========== RandomDihedral ==========


class TestRandomDihedral:
    def test_same_seed_same_sequence(self, asymmetric_map):
        a, b = RandomDihedral(seed=0), RandomDihedral(seed=0)
        for _ in range(20):
            assert np.array_equal(a(asymmetric_map), b(asymmetric_map))

    def test_reseed_restarts_the_stream(self, asymmetric_map):
        t = RandomDihedral(seed=0)
        first = [t(asymmetric_map).tobytes() for _ in range(10)]
        t.reseed(0)
        assert [t(asymmetric_map).tobytes() for _ in range(10)] == first

    def test_covers_all_eight_variants(self, asymmetric_map):
        t = RandomDihedral(seed=123)
        seen = {t(asymmetric_map).tobytes() for _ in range(400)}
        assert len(seen) == 8
