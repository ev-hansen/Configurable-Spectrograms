import numpy as np
import pandas as pd
from pathlib import Path
import types

# Import target module
import runpy
import sys

MODULE_PATH = Path(__file__).resolve().parent.parent / 'batch_multi_plot_spectrogram.py'
ns = runpy.run_path(str(MODULE_PATH))

# Patch: Use a simple namespace for mod to avoid attribute errors
import types
mod = types.SimpleNamespace(**ns)


def test_get_cdf_file_type():
    """
    Test extraction of CDF file type from filename.

    Ensures correct type is returned for known patterns and None for unknown.
    """
    assert mod.get_cdf_file_type('foo_ees_bar.cdf') == 'ees', "Failed to detect 'ees' type"
    assert mod.get_cdf_file_type('something_orb_data.cdf') == 'orb', "Failed to detect 'orb' type"
    assert mod.get_cdf_file_type('no_match.cdf') is None, "Failed to return None for no match"


def test_get_variable_shape(tmp_path):
    """
    Test retrieval of variable shape from a CDF file.

    Uses a fake CDF object to simulate cdflib.CDF behavior.
    """
    class FakeCDF:
        def __init__(self, file_path):
            self.file_path = file_path  # Used for completeness, not referenced further
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_value, traceback):
            return False
        def varget(self, variable_name):
            if variable_name == 'ok':
                return np.zeros((2, 3))
            raise ValueError('Variable not found')
    original_cdf_class = mod.cdflib.CDF
    try:
        mod.cdflib.CDF = FakeCDF  # type: ignore
        assert mod.get_variable_shape('dummy.cdf', 'ok') == (2, 3), "Expected shape (2, 3)"
        assert mod.get_variable_shape('dummy.cdf', 'missing') is None, "Expected None for missing variable"
    finally:
        mod.cdflib.CDF = original_cdf_class  # type: ignore


def test_get_timestamps_for_orbit():
    """
    Test extraction of timestamps for a given orbit.

    Checks correct slicing of the time array for each orbit.
    """
    orbit_df = pd.DataFrame({
        'Orbit': [10, 11],
        'EES Min Index': [0, 2],
        'EES Max Index': [1, 4],
    })
    time_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    # Test for first orbit
    assert mod.get_timestamps_for_orbit(orbit_df, 10, 'ees', time_array) == [100.0, 200.0], "Expected [100.0, 200.0]"
    # Test for single-point orbit
    orbit_df.loc[1, 'EES Max Index'] = 2
    assert mod.get_timestamps_for_orbit(orbit_df, 11, 'ees', time_array) == [300.0], "Expected [300.0]"


def test_get_cdf_var_shapes(tmp_path):
    """
    Test retrieval of variable shapes for all CDF files in a folder.

    Patches get_variable_shape to return a fixed shape for testing.
    """
    def patched_get_variable_shape(file_path, variable_name):
        return (1,)
    original_get_variable_shape = mod.get_variable_shape
    mod.get_variable_shape = patched_get_variable_shape
    data_folder = tmp_path / 'data'
    data_folder.mkdir()
    (data_folder / 'a.cdf').write_text('x')
    (data_folder / 'b.CDF').write_text('x')
    result = mod.get_cdf_var_shapes(str(data_folder), ['ok'])
    assert 'ok' in result and len(result['ok']) == 2, f"Expected 2 shapes, got {result}"
    assert all(shape == (1,) for shape in result['ok']), f"Expected all shapes (1,), got {result['ok']}"
    mod.get_variable_shape = original_get_variable_shape
