import argparse
import tomllib
from pathlib import Path

import numpy as np

from . import __version__
from .bands import GaussianBand, PseudoVoigtBand, SpectralBand, VibronicBand
from .data import DataSet
from .magnetization import (
    MagnetizationFitter,
    SpinHamiltonian,
    build_sh_parameters,
    calculate_polarizations,
    generate_simulation_grids,
)
from .outputs import (
    plot_isofield_summary,
    plot_magnetization_fit_grid,
    plot_results,
    save_results_to_csv,
    save_results_to_toml,
    save_sh_results_to_csv,
    save_simulated_curves_to_csv,
    save_spectra_to_csv,
)
from .parameters import ArrayParameter
from .spectral_fit import GlobalFitter


def load_config(filepath: str | Path) -> dict:

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file missing: {path}")

    with open(path, "rb") as file:
        config = tomllib.load(file)

    if "dataset" not in config or "bands" not in config:
        raise ValueError("Config File is missing 'dataset' or 'bands' blocks.")

    return config


def parse_args() -> argparse.Namespace:
    """Parses command line arguments to find input"""
    parser = argparse.ArgumentParser(
        description="Run the global fitting routine with parameters from TOML input file."
    )

    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument("config_file", type=str, help="Path to TOML configuration file")
    parser.add_argument(
        "--sh-only",
        action="store_true",
        help="Skip spectral fitting and only run the Spin-Hamiltonian solver using amplitudes in the TOML.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        type=float,
        default=None,
        help="Subset of magnetic fields to include in the fit (e.g. --fields 10.0)",
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=None,
        help="Subset of temperatures to include in the fit (e.g. --temps 2.0 5.0)",
    )
    return parser.parse_args()


def extract_fitted_g_tensor(sh_params: dict, symmetry_mode: str) -> list[float]:
    """Reconstructs the 3-element [gx, gy, gz] list from the parameter dictionary."""
    if symmetry_mode == "isotropic":
        g_val = sh_params["g"].value
        return [g_val, g_val, g_val]
    if symmetry_mode == "axial":
        gx = sh_params["gx"].value
        return [gx, gx, sh_params["gz"].value]
    return [sh_params["gx"].value, sh_params["gy"].value, sh_params["gz"].value]


def unpack_band_dipoles(
    band_params: np.ndarray, index: int, symmetry_mode: str
) -> tuple[float, float, float]:
    """Slices raw optimizer parameter array into Mxy, Myz, Mxz based on symmetry."""
    if symmetry_mode == "isotropic":
        val = band_params[index]
        return val, val, val
    elif symmetry_mode == "axial":
        idx = index * 2
        mxy, mxz = band_params[idx : idx + 2]
        return mxy, mxz, mxz
    else:  # rhombic
        idx = index * 3
        return tuple(band_params[idx : idx + 3])


def run_global_sh_fit(
    bands_dict: dict,
    flat_real_temps: list[float],
    flat_fields: list[float],
    flat_nominal_temps: list[float],
    mol_config: dict,
) -> None:
    S = mol_config.get("spin", 1.0)
    method = mol_config.get("method", "least_squares")
    plot_reduced_mag = mol_config.get("plot_reduced_mag", True)
    symmetry_mode = mol_config.get("symmetry", "isotropic").lower()

    if symmetry_mode not in ["isotropic", "axial", "rhombic"]:
        raise ValueError("Invalid symmetry in input file. Please use isotropic, axial, or rhombic.")

    # 1. Configure & Run Fit
    sh_params = build_sh_parameters(mol_config, symmetry_mode)
    fitter = MagnetizationFitter(
        S, flat_real_temps, flat_fields, bands_dict, sh_params, symmetry_mode=symmetry_mode
    )
    result = fitter.run_fit(method=method)

    if not result.success:
        print(f"Global fit failed: {result.message}")
        return

    # 2. Update Parameters
    for i, param in enumerate(fitter.floating_sh_params):
        param.set_value(result.x[i])

    g_tensor = extract_fitted_g_tensor(sh_params, symmetry_mode)
    D_fit = sh_params["D"].value
    E_fit = sh_params["E"].value

    # 3. Simulate High-Resolution Curves
    engine = SpinHamiltonian(S, D_fit, E_fit, g_tensor)
    unique_nom_temps = np.unique(flat_nominal_temps)
    target_field = float(np.max(flat_fields))
    smooth_fields, mag_basis, smooth_temps, iso_basis = generate_simulation_grids(
        engine, unique_nom_temps, target_field
    )

    # 4. Process Dipoles & Polarizations per Band
    band_params = result.x[fitter.num_floating_sh :]
    all_plot_params = {}
    sh_csv_data = []

    for i, name in enumerate(fitter.band_names):
        Mxy, Myz, Mxz = unpack_band_dipoles(band_params, i, symmetry_mode)
        pol = calculate_polarizations(Mxy, Myz, Mxz)
        sh_csv_data.append([name, pol.Mxy, pol.Myz, pol.Mxz, pol.perc_x, pol.perc_y, pol.perc_z])

        sf = fitter.scale_factors[name]
        all_plot_params[name] = [D_fit, E_fit, pol.Mxy * sf, pol.Myz * sf, pol.Mxz * sf]

    # 5. Output Results
    out_dir = Path("sh_outputs")
    out_dir.mkdir(exist_ok=True)

    curve_data_rows = plot_magnetization_fit_grid(
        bands_dict,
        flat_real_temps,
        flat_fields,
        flat_nominal_temps,
        all_plot_params,
        mag_basis,
        smooth_fields,
        out_dir,
        plot_reduced_mag=plot_reduced_mag,
    )
    plot_isofield_summary(
        bands_dict,
        flat_real_temps,
        flat_fields,
        all_plot_params,
        iso_basis,
        smooth_temps,
        target_field,
        out_dir,
    )
    save_sh_results_to_csv(sh_csv_data, D_fit, E_fit, g_tensor, out_dir)
    save_simulated_curves_to_csv(curve_data_rows, out_dir)


def run_standalone_sh(
    config: dict,
    selected_fields: list[float] | None = None,
    selected_temps: list[float] | None = None,
) -> None:
    if "sh" not in config:
        raise ValueError("Cannot run SH solver: Missing [sh] block in TOML.")

    dataset = DataSet(config["dataset"], select_fields=selected_fields, select_temps=selected_temps)
    if not dataset.is_vtvh:
        raise ValueError("Cannot run SH solver: Dataset is not configured as VTVH.")

    mol_config = config["sh"]
    flat_nominal_temps = [float(t) for t in dataset.toml_temperatures for _ in dataset.toml_fields]
    flat_real_temps = dataset.real_temperatures
    flat_fields = dataset.real_fields
    expected_keys = [str(t) for t in dataset.toml_temperatures]

    bands_dict = {}
    for band_name, band_setup in config["bands"].items():
        if "amplitudes" not in band_setup or "expr" in band_setup["amplitudes"]:
            continue

        param = ArrayParameter(
            f"{band_name}_amplitudes",
            band_setup["amplitudes"],
            expected_keys=expected_keys,
            field_indices=dataset.active_field_indices,
        )
        if param.value is not None and len(param.value) > 0:
            bands_dict[band_name] = param.value

    if bands_dict:
        run_global_sh_fit(bands_dict, flat_real_temps, flat_fields, flat_nominal_temps, mol_config)


def run_magnetization_pipeline(
    dataset: DataSet, bands: list[SpectralBand], namespace: dict, mol_config: dict
) -> None:
    flat_real_temps = dataset.real_temperatures
    flat_fields = dataset.real_fields
    flat_nominal_temps = [float(t) for t in dataset.toml_temperatures for f in dataset.toml_fields]

    x_axis = dataset.get_x()
    bands_dict = {}

    for band in bands:
        has_amps = any(isinstance(p, ArrayParameter) for p in band.get_parameters())

        if has_amps:
            band_matrix = band.evaluate(x_axis, namespace)

            true_peak_amps = []
            for row in band_matrix:
                max_idx = np.argmax(np.abs(row))
                true_peak_amps.append(row[max_idx])

            bands_dict[band.name] = np.array(true_peak_amps)

    if bands_dict:
        run_global_sh_fit(bands_dict, flat_real_temps, flat_fields, flat_nominal_temps, mol_config)


def main() -> None:
    args = parse_args()
    config = load_config(args.config_file)

    dataset_cfg = config.get("dataset", {})
    selected_fields = args.fields or dataset_cfg.get("select_fields", None)
    selected_temps = args.temps or dataset_cfg.get("select_temperatures", None)

    if args.sh_only:
        run_standalone_sh(config, selected_fields=selected_fields, selected_temps=selected_temps)
        return

    dataset = DataSet(dataset_cfg, select_fields=selected_fields, select_temps=selected_temps)

    expected_keys = [str(t) for t in dataset.toml_temperatures] if dataset.is_vtvh else None

    bands = []

    for band_name, band_setup in config["bands"].items():
        band_type = band_setup.get("type")
        band_kwargs = {
            "expected_keys": expected_keys,
            "field_indices": dataset.active_field_indices if dataset.is_vtvh else None,
        }

        if band_type == "Gaussian":
            bands.append(GaussianBand(band_name, band_setup, **band_kwargs))
        elif band_type == "Vibronic":
            bands.append(VibronicBand(band_name, band_setup, **band_kwargs))
        elif band_type == "PseudoVoigt":
            bands.append(PseudoVoigtBand(band_name, band_setup, **band_kwargs))
        else:
            raise ValueError(f"Unknown band type '{band_type}' found in [{band_name}].")

    fit_settings = config.get("fit_settings", {})
    method = fit_settings.get("method", "least_squares")
    fitter = GlobalFitter(dataset, bands, method=method)

    result = fitter.run()

    if result.success:
        final_namespace = fitter._build_namespace()
        save_results_to_toml("fitted_results.toml", dataset, bands)
        save_results_to_csv(dataset, bands, final_namespace)
        save_spectra_to_csv(dataset, bands, final_namespace)
        plot_results(dataset, bands, final_namespace)

        if "sh" in config:
            if dataset.is_vtvh:
                run_magnetization_pipeline(dataset, bands, final_namespace, config["sh"])


if __name__ == "__main__":
    main()
