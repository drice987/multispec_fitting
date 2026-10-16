import numpy as np
import pytest

from multispec.bands import GaussianBand, PseudoVoigtBand, SpectralBand
from multispec.constants import k_B


@pytest.fixture
def base_band_cfg():
    return {
        "type": "Gaussian",
        "center": {"value": 20000.0, "vary": True},
        "width": {"value": 400.0, "vary": True},
        "amplitudes": {"2.0": [2.0, 4.0]},
    }


def test_spectral_band_parameter_registration(base_band_cfg):
    band = SpectralBand("base", base_band_cfg)
    assert [p.name for p in band.get_parameters()] == [
        "base_center",
        "base_width",
        "base_amplitudes",
    ]
    with pytest.raises(NotImplementedError):
        band.evaluate(np.linspace(19000.0, 21000.0, 5))

    therm_cfg = {
        **base_band_cfg,
        "temp_broadening": {"value": 50.0, "vary": True},
        "vib_energy": {"value": 150.0, "vary": False},
    }
    therm_band = SpectralBand("therm", therm_cfg)
    assert len(therm_band.get_parameters()) == 5


def test_gaussian_band_fwhm_and_amplitudes(base_band_cfg):
    band = GaussianBand("g1", base_band_cfg)
    x = np.array([19800.0, 20000.0, 20200.0])
    y = band.evaluate(x, namespace={"__real_temperatures__": [2.0, 2.0]})

    assert y.shape == (2, 3)
    np.testing.assert_allclose(y[0], [1.0, 2.0, 1.0])
    np.testing.assert_allclose(y[1], [2.0, 4.0, 2.0])


def test_gaussian_band_thermal_broadening(base_band_cfg):
    cfg = {
        **base_band_cfg,
        "temp_broadening": {"value": 50.0, "vary": True},
        "vib_energy": {"value": 100.0, "vary": False},
    }
    band = GaussianBand("g_therm", cfg)

    temps = np.array([2.0, 150.0])
    expected_widths = 400.0 + 50.0 / np.tanh(100.0 / (2 * k_B * (temps + 1e-5)))

    x = 20000.0 + expected_widths / 2.0
    y = band.evaluate(x, namespace={"__real_temperatures__": temps})

    assert np.isclose(y[0, 0], 1.0)
    assert np.isclose(y[1, 1], 2.0)

def test_pseudovoigt_default_lorentz_fraction(base_band_cfg):
    band = PseudoVoigtBand("pv", base_band_cfg)
    params = band.get_parameters()

    assert len(params) == 4
    assert band.lorentz_frac.value == 0.5
    assert (band.lorentz_frac.min_val, band.lorentz_frac.max_val) == (0.0, 1.0)


@pytest.mark.parametrize(
    "eta, expected_wing",
    [
        (0.0, 0.125),
        (0.5, 0.2625),
        (1.0, 0.4),
    ],
)
def test_pseudovoigt_lineshape_mixing(base_band_cfg, eta, expected_wing):
    cfg = {**base_band_cfg, "lorentz_frac": {"value": eta, "vary": False}}
    band = PseudoVoigtBand("pv", cfg)

    x = np.array([20000.0, 20200.0, 20400.0])
    y = band.evaluate(x, namespace={"__real_temperatures__": [2.0, 2.0]})

    np.testing.assert_allclose(y[0], [2.0, 1.0, expected_wing])