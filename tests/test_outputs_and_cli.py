import csv
import sys
import tomllib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from multispec.bands import GaussianBand, VibronicBand
from multispec.data import DataSet
from multispec.main import (
    extract_fitted_g_tensor,
    load_config,
    main,
    parse_args,
    run_standalone_sh,
    unpack_band_dipoles,
)
from multispec.outputs import (
    plot_results,
    save_results_to_csv,
    save_results_to_toml,
    save_spectra_to_csv,
)
from multispec.parameters import FitParameter
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


@pytest.mark.parametrize(
    "symmetry, expected_g, expected_dipoles",
    [
        ("isotropic", [2.1, 2.1, 2.1], (0.2, 0.2, 0.2)),
        ("axial", [2.0, 2.0, 2.3], (0.3, 0.4, 0.4)),
        ("rhombic", [1.9, 2.1, 2.4], (0.4, 0.5, 0.6)),
    ],
)
def test_g_tensor_and_dipole_unpacking_all_symmetries(symmetry, expected_g, expected_dipoles):
    sh_params = {
        "g": FitParameter("g", {"value": 2.1}),
        "gx": FitParameter("gx", {"value": expected_g[0]}),
        "gy": FitParameter("gy", {"value": expected_g[1]}),
        "gz": FitParameter("gz", {"value": expected_g[2]}),
    }
    assert extract_fitted_g_tensor(sh_params, symmetry) == expected_g

    raw_dipoles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert unpack_band_dipoles(raw_dipoles, index=1, symmetry_mode=symmetry) == expected_dipoles


def test_parse_args_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["multispec", "input.toml", "--sh-only", "--fields", "1.0", "7.0", "--temps", "2.0", "5.0"],
    )
    args = parse_args()
    assert args.config_file == "input.toml"
    assert args.sh_only is True
    assert args.fields == [1.0, 7.0]
    assert args.temps == [2.0, 5.0]


def test_config_and_standalone_sh_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")

    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("[dataset]\nfilename = 'data.txt'\n")
    with pytest.raises(ValueError, match="missing 'dataset' or 'bands'"):
        load_config(bad_toml)

    with pytest.raises(ValueError, match="Missing \\[sh\\] block"):
        run_standalone_sh({"dataset": {}, "bands": {}})

    std_file = tmp_path / "std.txt"
    std_file.write_text("Energy Abs\n20000.0 1.0\n20500.0 1.5\n")
    with pytest.raises(ValueError, match="not configured as VTVH"):
        run_standalone_sh({"dataset": {"filename": str(std_file)}, "sh": {}, "bands": {}})


def test_end_to_end_cli_full_and_sh_only_pipelines(tmp_path, monkeypatch, vtvh_2x2_setup):
    monkeypatch.chdir(tmp_path)

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[dataset]
filename = "vtvh_2x2.txt"
temperatures = [2.0, 5.0]
fields = [1.0, 7.0]

[fit_settings]
method = "least_squares"

[sh]
spin = 0.5
symmetry = "isotropic"
plot_reduced_mag = false
g = { value = 2.0, vary = false }
D = { value = 0.0, vary = false }

[bands.Band1]
type = "Gaussian"
center = { value = 19985.0, vary = true }
width = { value = 255.0, vary = true }

[bands.Band1.amplitudes]
"2.0" = [-0.9, -2.9]
"5.0" = [-0.45, -1.45]
"""
    )

    monkeypatch.setattr(sys, "argv", ["multispec", str(cfg_path)])
    main()

    assert (tmp_path / "fitted_results.toml").exists()
    assert (tmp_path / "fit_results.png").exists()
    assert (tmp_path / "sh_outputs" / "magnetization_fits.png").exists()
    assert (tmp_path / "sh_outputs" / "isofield_7.0T.png").exists()
    assert (tmp_path / "sh_outputs" / "sh_simulated_curves.csv").exists()

    with open(tmp_path / "sh_outputs" / "sh_fit_parameters.csv", newline="") as f:
        sh_rows = list(csv.reader(f))
    band_row = sh_rows[-1]
    assert band_row[0] == "Band1"
    assert float(band_row[1]) > 0.0

    monkeypatch.setattr(sys, "argv", ["multispec", str(cfg_path), "--sh-only", "--fields", "7.0"])
    main()
    assert (tmp_path / "sh_outputs" / "isofield_7.0T.png").exists()


def test_cli_unknown_band_type_raises(tmp_path, monkeypatch, vtvh_2x2_setup):
    monkeypatch.chdir(tmp_path)
    bad_cfg = tmp_path / "bad_band.toml"
    bad_cfg.write_text(
        """
[dataset]
filename = "vtvh_2x2.txt"
temperatures = [2.0, 5.0]
fields = [1.0, 7.0]

[bands.Band1]
type = "Lorentzian"
center = { value = 20000.0 }
width = { value = 250.0 }

[bands.Band1.amplitudes]
"2.0" = [-1.0, -3.0]
"5.0" = [-0.5, -1.5]
"""
    )
    monkeypatch.setattr(sys, "argv", ["multispec", str(bad_cfg)])
    with pytest.raises(ValueError, match="Unknown band type 'Lorentzian'"):
        main()
