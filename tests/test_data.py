import numpy as np
import pytest

from multispec.data import DataSet


@pytest.fixture
def vtvh_file(tmp_path):
    p = tmp_path / "vtvh_spectra.txt"
    content = (
        "Energy 5.0K_7.0T 1.7K_1.0T 2.1K_1.0T 2.1K_7.0T 5.0K_1.0T\n"
        "15000.0 4.0 99.0 1.0 3.0 0.5\n"
        "15500.0 8.0 99.0 2.0 6.0 1.0\n"
    )
    p.write_text(content)
    return p


def test_vtvh_loading_reorders_columns_and_picks_closest_temp(vtvh_file):
    cfg = {
        "filename": str(vtvh_file),
        "temperatures": [2.0, 5.0],
        "fields": [1.0, 7.0],
        "temp_tolerance": 0.5,
    }
    ds = DataSet(cfg)

    assert ds.is_vtvh is True
    assert ds.conditions == ["2.0K_1.0T", "2.0K_7.0T", "5.0K_1.0T", "5.0K_7.0T"]
    assert ds.matched_headers == ["2.1K_1.0T", "2.1K_7.0T", "5.0K_1.0T", "5.0K_7.0T"]
    assert ds.real_temperatures == [2.1, 2.1, 5.0, 5.0]
    assert ds.real_fields == [1.0, 7.0, 1.0, 7.0]
    np.testing.assert_allclose(ds.get_x(), [15000.0, 15500.0])
    np.testing.assert_allclose(ds.get_y_flat(), [1.0, 2.0, 3.0, 6.0, 0.5, 1.0, 4.0, 8.0])


def test_vtvh_subset_selection(vtvh_file):
    cfg = {
        "filename": str(vtvh_file),
        "temperatures": [2.0, 5.0],
        "fields": [1.0, 7.0],
    }
    ds = DataSet(cfg, select_fields=[7.0], select_temps=[5.0])

    assert ds.active_field_indices == [1]
    assert ds.active_temp_indices == [1]
    assert ds.toml_fields == [7.0]
    assert ds.toml_temperatures == [5.0]
    assert ds.conditions == ["5.0K_7.0T"]
    np.testing.assert_allclose(ds.get_y_flat(), [4.0, 8.0])


def test_missing_data_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        DataSet({"filename": str(tmp_path / "nonexistent.txt")})


@pytest.mark.parametrize(
    "kwargs, tol, match_msg",
    [
        ({"select_fields": [14.0]}, 0.5, "None of the requested fields"),
        ({"select_temps": [25.0]}, 0.5, "None of the requested temperatures"),
        ({}, 0.05, "Could not find a column for 2.0K_1.0T"),
    ],
)
def test_vtvh_invalid_filters_or_tolerance_raises(vtvh_file, kwargs, tol, match_msg):
    cfg = {
        "filename": str(vtvh_file),
        "temperatures": [2.0, 5.0],
        "fields": [1.0, 7.0],
        "temp_tolerance": tol,
    }
    with pytest.raises(ValueError, match=match_msg):
        DataSet(cfg, **kwargs)


@pytest.fixture
def standard_file(tmp_path):
    p = tmp_path / "abs_spectra.txt"
    content = "Energy Abs_RT MCD_RT\n20000.0 0.45 1.20\n20500.0 0.80 -0.90\n"
    p.write_text(content)
    return p


def test_standard_explicit_column_loading(standard_file):
    ds = DataSet({"filename": str(standard_file), "columns": ["MCD_RT"]})

    assert ds.is_vtvh is False
    assert ds.active_field_indices is None
    assert ds.conditions == ["MCD_RT"]
    assert ds.real_temperatures == [0.0]
    assert ds.real_fields == [0.0]
    np.testing.assert_allclose(ds.get_y_flat(), [1.20, -0.90])


def test_standard_default_all_columns(standard_file):
    ds = DataSet({"filename": str(standard_file)})

    assert ds.is_vtvh is False
    assert ds.conditions == ["Abs_RT", "MCD_RT"]
    assert ds.real_temperatures == [0.0, 0.0]
    np.testing.assert_allclose(ds.get_y_flat(), [0.45, 0.80, 1.20, -0.90])


def test_standard_missing_column_raises(standard_file):
    with pytest.raises(ValueError, match="Column 'CD_77K' not found"):
        DataSet({"filename": str(standard_file), "columns": ["CD_77K"]})
