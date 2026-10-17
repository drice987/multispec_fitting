import csv
import tomllib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from multispec.bands import GaussianBand, VibronicBand
from multispec.data import DataSet
from multispec.outputs import (
    plot_results,
    save_results_to_csv,
    save_results_to_toml,
    save_spectra_to_csv,
)
from multispec.spectral_fit import GlobalFitter


@pytest.fixture
def vtvh_2x2_setup(tmp_path):
    x = np.linspace(19500.0, 20500.0, 11)
    g = np.exp(-4 * np.log(2) * ((x - 20000.0) / 250.0) ** 2)

    data_file = tmp_path / "vtvh_2x2.txt"
    lines = ["Energy 2.0K_1.0T 2.0K_7.0T 5.0K_1.0T 5.0K_7.0T"]
    for xv, gv in zip(x, g):
        lines.append(f"{xv:.1f} {-1.0 * gv:.5f} {-3.0 * gv:.5f} {-0.5 * gv:.5f} {-1.5 * gv:.5f}")
    data_file.write_text("\n".join(lines))

    ds = DataSet(
        {
            "filename": str(data_file),
            "temperatures": [2.0, 5.0],
            "fields": [1.0, 7.0],
        }
    )
    b1 = GaussianBand(
        "Band1",
        {
            "type": "Gaussian",
            "center": {"value": 20000.0, "vary": True},
            "width": {"value": 250.0, "vary": False},
            "amplitudes": {"2.0": [-1.0, -3.0], "5.0": [-0.5, -1.5]},
        },
        expected_keys=["2.0", "5.0"],
    )
    b2 = VibronicBand(
        "Band2",
        {
            "type": "Vibronic",
            "n_levels": 3,
            "center": {"expr": "Band1.center + 400"},
            "width": {"value": 250.0, "vary": False},
            "vib_spacing": {"value": 500.0, "vary": False},
            "huang_rhys": {"value": 0.5, "vary": False},
            "amplitudes": {"expr": "Band1.amplitudes * 0.5"},
        },
        expected_keys=["2.0", "5.0"],
    )
    fitter = GlobalFitter(ds, [b1, b2])
    return ds, [b1, b2], fitter._build_namespace()


def test_save_results_to_toml_and_csvs(tmp_path, vtvh_2x2_setup):
    ds, bands, ns = vtvh_2x2_setup

    toml_out = tmp_path / "fitted_results.toml"
    param_csv = tmp_path / "output_parameters.csv"
    amp_csv = tmp_path / "output_amplitudes.csv"
    spec_csv = tmp_path / "output_spectra.csv"

    save_results_to_toml(str(toml_out), ds, bands)
    save_results_to_csv(ds, bands, ns, str(param_csv), str(amp_csv))
    save_spectra_to_csv(ds, bands, ns, str(spec_csv))

    with open(toml_out, "rb") as f:
        parsed_toml = tomllib.load(f)
    assert parsed_toml["dataset"]["fields"] == [1.0, 7.0]
    assert parsed_toml["bands"]["Band1"]["center"]["value"] == 20000.0
    assert parsed_toml["bands"]["Band2"]["n_levels"] == 3
    assert parsed_toml["bands"]["Band2"]["center"]["expr"] == "Band1.center + 400"

    with open(param_csv, newline="") as f:
        param_rows = list(csv.reader(f))
    assert param_rows[0] == ["Band_Name", "Type", "Center", "Width", "Vib_Spacing", "Huang_Rhys"]
    assert param_rows[1] == ["Band1", "Gaussian", "20000.0000", "250.0000", "", ""]
    assert param_rows[2][2] == "Band1.center + 400"

    with open(amp_csv, newline="") as f:
        amp_rows = list(csv.reader(f))
    assert amp_rows[0] == ["Band_Name", "2.0K_1.0T", "2.0K_7.0T", "5.0K_1.0T", "5.0K_7.0T"]
    assert [float(v) for v in amp_rows[2][1:]] == [-0.5, -1.5, -0.25, -0.75]

    with open(spec_csv, newline="") as f:
        spec_rows = list(csv.reader(f))
    assert spec_rows[0][:5] == [
        "X_Value",
        "Exp_2.0K_1.0T",
        "Fit_2.0K_1.0T",
        "Band1_2.0K_1.0T",
        "Band2_2.0K_1.0T",
    ]
    mid_row = [float(v) for v in spec_rows[6]]
    assert np.isclose(mid_row[2], mid_row[3] + mid_row[4], atol=1e-3)


def test_plot_results_2d_and_fallback_layouts(tmp_path, vtvh_2x2_setup):
    ds, bands, ns = vtvh_2x2_setup
    grid_plot = tmp_path / "grid_fit.png"
    plot_results(ds, bands, ns, filename=str(grid_plot))
    assert grid_plot.exists() and grid_plot.stat().st_size > 0

    std_file = tmp_path / "std_4col.txt"
    std_file.write_text("Energy C1 C2 C3 C4\n19500.0 1.0 2.0 3.0 4.0\n20000.0 2.0 4.0 6.0 8.0\n")
    std_ds = DataSet({"filename": str(std_file)})
    std_band = GaussianBand(
        "B1",
        {
            "type": "Gaussian",
            "center": {"value": 20000.0},
            "width": {"value": 300.0},
            "amplitudes": {"C1": 2.0, "C2": 4.0, "C3": 6.0, "C4": 8.0},
        },
    )
    std_ns = GlobalFitter(std_ds, [std_band])._build_namespace()

    fallback_plot = tmp_path / "fallback_fit.png"
    save_results_to_toml(str(tmp_path / "std_out.toml"), std_ds, [std_band])
    plot_results(std_ds, [std_band], std_ns, filename=str(fallback_plot))
    assert fallback_plot.exists() and fallback_plot.stat().st_size > 0
