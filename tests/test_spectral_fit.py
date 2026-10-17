import pickle

import numpy as np
import pytest

from multispec.bands import GaussianBand
from multispec.data import DataSet
from multispec.spectral_fit import GlobalFitter


@pytest.fixture
def synthetic_dataset(tmp_path):
    x = np.linspace(14000.0, 16000.0, 41)
    c_true, w_true = 15000.0, 300.0
    g_shape = np.exp(-4 * np.log(2) * ((x - c_true) / w_true) ** 2)

    y_col1 = 2.0 * g_shape
    y_col2 = 5.0 * g_shape

    data_file = tmp_path / "synth_vtvh.txt"
    lines = ["Energy 2.0K_1.0T 2.0K_7.0T"]
    for xv, y1, y2 in zip(x, y_col1, y_col2):
        lines.append(f"{xv:.1f} {y1:.6f} {y2:.6f}")
    data_file.write_text("\n".join(lines))

    return DataSet(
        {
            "filename": str(data_file),
            "temperatures": [2.0],
            "fields": [1.0, 7.0],
        }
    )


def test_global_fitter_gathers_floating_params_and_bounds(synthetic_dataset):
    band = GaussianBand(
        "Band1",
        {
            "type": "Gaussian",
            "center": {"value": 15000.0, "vary": True},
            "width": {"value": 300.0, "min": 50.0, "vary": True},
            "temp_broadening": {"value": 10.0, "vary": False},
            "amplitudes": {"2.0": [10.0, 20.0]},
        },
        expected_keys=["2.0"],
    )
    fitter = GlobalFitter(synthetic_dataset, [band])

    assert [p.name for p in fitter.floating_params] == [
        "Band1_center",
        "Band1_width",
        "Band1_amplitudes",
    ]
    np.testing.assert_allclose(fitter._get_initial_guesses(), [15000.0, 300.0, 10.0, 20.0])

    default_bounds = fitter._get_bounds(finite_only=False)
    assert default_bounds == [
        (13800.0, 16200.0),
        (50.0, float("inf")),
        (-float("inf"), float("inf")),
        (-float("inf"), float("inf")),
    ]

    finite_bounds = fitter._get_bounds(finite_only=True)
    assert finite_bounds == [
        (13800.0, 16200.0),
        (50.0, 1800.0),
        (-40.0, 60.0),
        (-40.0, 80.0),
    ]


def test_global_fitter_pickle_reinitializes_asteval(synthetic_dataset):
    band = GaussianBand(
        "Band1",
        {
            "type": "Gaussian",
            "center": {"value": 15000.0, "vary": True},
            "width": {"value": 300.0, "vary": False},
            "amplitudes": {"2.0": [2.0, 5.0]},
        },
        expected_keys=["2.0"],
    )
    fitter = GlobalFitter(synthetic_dataset, [band])
    restored = pickle.loads(pickle.dumps(fitter))

    assert restored.aeval is not None
    assert np.isclose(restored.cost_function(np.array([15000.0, 2.0, 5.0])), 0.0, atol=1e-9)


def test_global_fitter_least_squares_recovers_parameters(synthetic_dataset):
    band = GaussianBand(
        "Band1",
        {
            "type": "Gaussian",
            "center": {"value": 14920.0, "vary": True},
            "width": {"value": 340.0, "vary": True},
            "amplitudes": {"2.0": [1.6, 4.4]},
        },
        expected_keys=["2.0"],
    )
    fitter = GlobalFitter(synthetic_dataset, [band], method="least_squares")

    res = fitter.run()
    assert res.success
    assert fitter.cost_function(res.x) < 1e-7
    assert np.isclose(band.center.value, 15000.0, atol=1e-2)
    assert np.isclose(band.width.value, 300.0, atol=1e-2)
    np.testing.assert_allclose(band.amplitudes.value, [2.0, 5.0], atol=1e-2)


def test_global_fitter_coupled_band_expressions(synthetic_dataset):
    b1 = GaussianBand(
        "Band1",
        {
            "type": "Gaussian",
            "center": {"value": 14950.0, "vary": True},
            "width": {"value": 300.0, "vary": False},
            "amplitudes": {"2.0": [0.8, 2.0]},
        },
        expected_keys=["2.0"],
    )
    b2 = GaussianBand(
        "Band2",
        {
            "type": "Gaussian",
            "center": {"expr": "Band1.center"},
            "width": {"expr": "Band1.width"},
            "amplitudes": {"expr": "Band1.amplitudes * 1.5"},
        },
        expected_keys=["2.0"],
    )

    fitter = GlobalFitter(synthetic_dataset, [b1, b2], method="least_squares")
    res = fitter.run()

    assert res.success
    assert np.isclose(b1.center.value, 15000.0, atol=1e-2)
    np.testing.assert_allclose(b1.amplitudes.value, [0.8, 2.0], atol=1e-2)

    ns = fitter._build_namespace()
    np.testing.assert_allclose(b2.amplitudes.get_value(ns), [1.2, 3.0], atol=1e-2)


def test_global_fitter_lbfgsb_and_invalid_method(synthetic_dataset):
    band = GaussianBand(
        "Band1",
        {
            "type": "Gaussian",
            "center": {"value": 15000.0, "vary": False},
            "width": {"value": 300.0, "vary": False},
            "amplitudes": {"2.0": [1.8, 4.7]},
        },
        expected_keys=["2.0"],
    )
    fitter = GlobalFitter(synthetic_dataset, [band], method="L-BFGS-B")
    res = fitter.run()

    assert res.success
    np.testing.assert_allclose(band.amplitudes.value, [2.0, 5.0], atol=1e-2)

    bad_fitter = GlobalFitter(synthetic_dataset, [band], method="unsupported_solver")
    with pytest.raises(ValueError, match="Invalid optimization method"):
        bad_fitter.run()
