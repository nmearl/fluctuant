import numpy as np
import pytest

from fluctuant.generators import SyntheticTransientGenerator


@pytest.fixture(scope='module')
def gen():
    return SyntheticTransientGenerator(seed=0)


# ---------------------------------------------------------------------------
# Irregular time sampling
# ---------------------------------------------------------------------------

class TestIrregularTimes:
    def test_returns_numpy_array(self, gen):
        times = gen.generate_irregular_times(duration=200, mean_cadence=3)
        assert isinstance(times, np.ndarray)

    def test_monotonically_increasing(self, gen):
        times = gen.generate_irregular_times(duration=200, mean_cadence=3)
        assert np.all(np.diff(times) > 0)

    def test_all_within_duration(self, gen):
        times = gen.generate_irregular_times(duration=200, mean_cadence=3)
        assert times[-1] < 200

    def test_non_empty(self, gen):
        times = gen.generate_irregular_times(duration=100, mean_cadence=3)
        assert len(times) > 0


# ---------------------------------------------------------------------------
# AGN light curves
# ---------------------------------------------------------------------------

class TestGenerateAGN:
    def test_correct_count(self, gen):
        lcs = gen.generate_agn(n_objects=12, cadence_days=5, duration_days=300)
        assert len(lcs) == 12

    def test_three_column_arrays(self, gen):
        lcs = gen.generate_agn(n_objects=5, cadence_days=5, duration_days=300)
        for lc in lcs:
            assert lc.ndim == 2 and lc.shape[1] == 3

    def test_times_increasing(self, gen):
        lcs = gen.generate_agn(n_objects=5, cadence_days=5, duration_days=300)
        for lc in lcs:
            assert np.all(np.diff(lc[:, 0]) > 0)

    def test_errors_positive(self, gen):
        lcs = gen.generate_agn(n_objects=5, cadence_days=5, duration_days=300)
        for lc in lcs:
            assert np.all(lc[:, 2] > 0)

    def test_all_finite(self, gen):
        lcs = gen.generate_agn(n_objects=5, cadence_days=5, duration_days=300)
        for lc in lcs:
            assert np.all(np.isfinite(lc))


# ---------------------------------------------------------------------------
# TDE light curves
# ---------------------------------------------------------------------------

class TestGenerateTDE:
    def test_correct_count(self, gen):
        lcs = gen.generate_tde(n_objects=12, cadence_days=5, duration_days=300)
        assert len(lcs) == 12

    def test_three_column_arrays(self, gen):
        lcs = gen.generate_tde(n_objects=5, cadence_days=5, duration_days=300)
        for lc in lcs:
            assert lc.ndim == 2 and lc.shape[1] == 3

    def test_tde_brightens_relative_to_baseline(self, gen):
        # In magnitude space, brightening = smaller value than baseline (19.0).
        lcs = gen.generate_tde(n_objects=20, cadence_days=5, duration_days=500)
        brightened = [lc[:, 1].min() < 19.0 for lc in lcs]
        assert sum(brightened) > len(lcs) * 0.8

    def test_errors_positive(self, gen):
        lcs = gen.generate_tde(n_objects=5, cadence_days=5, duration_days=300)
        for lc in lcs:
            assert np.all(lc[:, 2] > 0)


# ---------------------------------------------------------------------------
# Component-level generation
# ---------------------------------------------------------------------------

class TestWithComponents:
    def test_agn_component_keys(self, gen):
        result = gen.generate_agn_with_components(cadence_days=5, duration_days=300)
        for key in ('times', 'mags', 'errors', 'drw_model', 'flare_model', 'full_model'):
            assert key in result

    def test_tde_component_keys(self, gen):
        result = gen.generate_tde_with_components(cadence_days=5, duration_days=300)
        for key in ('times', 'mags', 'errors', 'model'):
            assert key in result

    def test_agn_component_lengths_consistent(self, gen):
        r = gen.generate_agn_with_components(cadence_days=5, duration_days=300)
        n = len(r['times'])
        for key in ('mags', 'errors', 'drw_model', 'flare_model', 'full_model'):
            assert len(r[key]) == n, f"Length mismatch for key '{key}'"

    def test_tde_component_lengths_consistent(self, gen):
        r = gen.generate_tde_with_components(cadence_days=5, duration_days=300)
        n = len(r['times'])
        for key in ('mags', 'errors', 'model'):
            assert len(r[key]) == n, f"Length mismatch for key '{key}'"
