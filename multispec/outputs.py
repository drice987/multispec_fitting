import csv
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .constants import mu_b, k_B
from .data import DataSet
from .bands import SpectralBand
from .parameters import ArrayParameter

def plot_results(dataset: DataSet, bands: list[SpectralBand], namespace: dict, filename: str = "fit_results.png") -> None:
    x_axis = dataset.get_x()
    conditions = dataset.conditions
    n_plots = len(conditions)

    is_full_2d_grid = (
        dataset.is_vtvh 
        and len(dataset.toml_temperatures) > 1 
        and len(dataset.toml_fields) > 1
        and n_plots == len(dataset.toml_temperatures) * len(dataset.toml_fields)
    )

    if is_full_2d_grid:
        n_rows = len(dataset.toml_temperatures)
        n_cols = len(dataset.toml_fields)
    else:
        n_cols = min(3, n_plots)
        n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(4.5 * n_cols, 3.2 * n_rows), 
        squeeze=False
    )
    axes_flat = axes.flatten()

    handles, labels = None, None
    band_matrices = [band.evaluate(x_axis, namespace) for band in bands]

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]

        y_raw = dataset.y_matrix.T[idx]
        ax.plot(x_axis, y_raw, color='grey', label='Experimental', alpha=0.7)

        total_sim = np.zeros_like(x_axis)
        for b_idx, band in enumerate(bands):
            band_y = band_matrices[b_idx][idx]
            ax.plot(x_axis, band_y, linestyle='--', linewidth=1, label=band.name)
            total_sim += band_y

        ax.plot(x_axis, total_sim, color='black', linewidth=1.2, label='Total Fit')

        ax.set_title(cond, fontsize=10, fontweight='bold')
        ax.invert_xaxis()
        ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9)
        ax.set_ylabel("MCD Intensity", fontsize=9)
        ax.tick_params(direction='in', labelsize=8)

        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()

    for j in range(len(conditions), len(axes_flat)):
        axes_flat[j].set_visible(False)

    if handles and labels:
        fig.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(labels), 6),
            fontsize='small',
            frameon=True
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_results_to_toml(filename: str, dataset: DataSet, bands: list[SpectralBand]) -> None:
    """Writes the fit results into a TOML output file."""
    with open(filename, 'w', encoding='utf-8') as f:
        
        f.write("[dataset]\n")
        f.write(f'filename = "{dataset.filename.name}"\n')
        if dataset.is_vtvh:
            f.write(f"fields = {dataset.toml_fields}\n")
            temps_formatted = ", ".join([f'"{t}"' if isinstance(t, str) else str(t) for t in dataset.toml_temperatures])
            f.write(f"temperatures = [{temps_formatted}]\n\n")
        else:
            cols_formatted = ", ".join([f'"{c}"' for c in dataset.toml_columns])
            f.write(f"columns = [{cols_formatted}]\n\n")
            
        for band in bands:
            f.write(f"[bands.{band.name}]\n")
            f.write(f'type = "{band.type}"\n')

            if hasattr(band, 'n_levels'):
                f.write(f'n_levels = {band.n_levels}\n')

            for param in band.get_parameters():
                name = param.name.removeprefix(f"{band.name}_")
                if not isinstance(param, ArrayParameter):
                    if param.expr:
                        f.write(f'{name} = {{ expr = "{param.expr}" }}\n')
                    else:
                        vary_str = "true" if param.vary else "false"
                        f.write(f'{name} = {{ value = {param.value:.4f}, vary = {vary_str} }}\n')
            f.write("\n")
            
            for param in band.get_parameters():
                name = param.name.removeprefix(f"{band.name}_")
                if isinstance(param, ArrayParameter) and not param.expr:
                    f.write(f"[bands.{band.name}.{name}]\n")
                    current_idx = 0
                    for key, size in zip(param.row_keys, param.row_sizes):
                        chunk = param.value[current_idx : current_idx + size]
                        chunk_str = ", ".join([f"{val:.3f}" for val in chunk])
                        f.write(f'"{key}" = [{chunk_str}]\n')
                        current_idx += size
                    f.write("\n")

def save_results_to_csv(dataset: DataSet, bands: list[SpectralBand], namespace: dict, param_filename: str = "output_parameters.csv", amp_filename: str = "output_amplitudes.csv") -> None:
    
    # --- Table 1: Scalar Parameters ---
    with open(param_filename, mode='w', newline='') as f_param:
        writer = csv.writer(f_param)
        
        unique_param_names = []
        for band in bands:
            for param in band.get_parameters():
                if not isinstance(param, ArrayParameter):
                    base_name = param.name.removeprefix(f"{band.name}_")
                    if base_name not in unique_param_names:
                        unique_param_names.append(base_name)
        headers = ["Band_Name", "Type"] + [name.title() for name in unique_param_names]
        writer.writerow(headers)

        for band in bands:
            row_dict = {name: "" for name in unique_param_names}
            for param in band.get_parameters():
                if not isinstance(param, ArrayParameter):
                    base_name = param.name.removeprefix(f"{band.name}_")
                    row_dict[base_name] = str(param.expr) if param.expr else f"{param.value:.4f}"
            
            row = [band.name, band.type]
            for name in unique_param_names:
                row.append(row_dict[name])
            writer.writerow(row)  
            
    # --- Table 2: Amplitudes ---
    with open(amp_filename, mode='w', newline='') as f_amp:
        writer = csv.writer(f_amp)
        
        headers = ["Band_Name"] + dataset.conditions
        writer.writerow(headers)
        
        for band in bands:
            row = [band.name]
            for param in band.get_parameters():
                if isinstance(param, ArrayParameter):
                    amps = param.get_value(namespace)
                    row.extend([f"{val:.4f}" for val in amps])
            writer.writerow(row)

def save_sh_results_to_csv(sh_data: list, D: float, E: float, g: float | list[float], out_dir: Path, filename: str = "sh_fit_parameters.csv") -> None:
    filepath = out_dir / filename
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["--- Global Spin-Hamiltonian Parameters ---"])
        writer.writerow(["D (cm-1)", f"{D:.4f}"])
        writer.writerow(["E (cm-1)", f"{E:.4f}"])
        writer.writerow(["gx", f"{g[0]:.4f}"])
        writer.writerow(["gy", f"{g[1]:.4f}"])
        writer.writerow(["gz", f"{g[2]:.4f}"])
        writer.writerow([])
        writer.writerow(["Band_Name", "Mxy", "Myz", "Mxz", "%x", "%y", "%z"])
        for row in sh_data:
            name, Mxy, Myz, Mxz, px, py, pz = row
            writer.writerow([name, f"{Mxy:.2f}", f"{Myz:.2f}", f"{Mxz:.2f}", 
                             f"{px:.2f}", f"{py:.2f}", f"{pz:.2f}"])

def save_spectra_to_csv(dataset: DataSet, bands: list[SpectralBand], namespace: dict, filename: str = 'output_spectra.csv') -> None:
    x_axis = dataset.get_x()
    conditions = dataset.conditions
    band_matrices = [band.evaluate(x_axis, namespace) for band in bands]

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)

        headers = ["X_Value"]
        for cond in conditions:
            headers.extend([f"Exp_{cond}", f"Fit_{cond}"])
            for band in bands:
                headers.append(f"{band.name}_{cond}")
        writer.writerow(headers)

        for row_idx, x_val in enumerate(x_axis):
            row_data = [f"{x_val:.4f}"]
            
            for idx, cond in enumerate(conditions):
                exp_val = dataset.y_matrix[row_idx, idx]
                row_data.append(f"{exp_val:.4f}")
                
                total_fit = 0.0
                band_vals = []
                for b_idx, band in enumerate(bands):
                    val = band_matrices[b_idx][idx][row_idx]
                    total_fit += val
                    band_vals.append(f"{val:.4f}")
                    
                row_data.append(f"{total_fit:.4f}")
                row_data.extend(band_vals)
                
            writer.writerow(row_data)

def plot_sh_curves(ax: plt.Axes, band_name: str, flat_real_temps: list[float], flat_fields: list[float], exp_amps: np.ndarray, fit_params: list[float], mag_basis: dict, smooth_fields: np.ndarray, curve_data_rows: list, flat_nominal_temps: list[float], plot_reduced_mag: bool = True) -> None:
    D, E, Mxy, Myz, Mxz = fit_params
    
    real_temps_arr = np.array(flat_real_temps)
    nom_temps_arr = np.array(flat_nominal_temps)
    fields_arr = np.array(flat_fields)
    amps_arr = np.array(exp_amps)
    
    unique_nom_temps = np.unique(nom_temps_arr)
    
    for nom_temp in unique_nom_temps:
        idx = np.where(nom_temps_arr == nom_temp)[0]
        
        t_fields = fields_arr[idx]
        t_amps = amps_arr[idx]
        t_real_temps = real_temps_arr[idx]
        
        if plot_reduced_mag:
            x_exp = (mu_b * t_fields) / (2 * k_B * t_real_temps)
        else:
            x_exp = t_fields
            
        p = ax.plot(x_exp, t_amps, marker='o', linestyle='none', label=f'{nom_temp} K')
        line_color = p[0].get_color()
        
        for x, y, r_t in zip(x_exp, t_amps, t_real_temps):
            curve_data_rows.append([band_name, r_t, "Experimental", x, y])
        
        t_smooth_amps = []
        x_smooth = []
        for i, b_mag in enumerate(smooth_fields):
            ave_xy, ave_yz, ave_zx = mag_basis[nom_temp][i]
            sim_val = Mxy * ave_xy + Myz * ave_yz + Mxz * ave_zx
            t_smooth_amps.append(sim_val)
            
            if plot_reduced_mag:
                x_smooth.append((mu_b * b_mag) / (2 * k_B * nom_temp))
            else:
                x_smooth.append(b_mag)
                
        for x, y in zip(x_smooth, t_smooth_amps):
            curve_data_rows.append([band_name, nom_temp, "Fit", x, y])
                
        ax.plot(x_smooth, t_smooth_amps, linestyle='-', color=line_color)

    if plot_reduced_mag:
        ax.set_xlabel("$\\mu_B B / 2kT$")
    else:
        ax.set_xlabel("Magnetic Field (T)")
    ax.set_ylabel("MCD Amplitude")
    ax.set_title(band_name)
    ax.legend(fontsize='small', loc='best')

def plot_isofield_summary(bands_dict: dict, flat_temps: list[float], flat_fields: list[float], all_plot_params: dict, iso_basis: list, smooth_temps: np.ndarray, target_field: float, out_dir: Path) -> None:
    temps_arr = np.array(flat_temps)
    fields_arr = np.array(flat_fields)
        
    plt.figure(figsize=(6, 9))
    
    for band_name, amps in bands_dict.items():
        amps_arr = np.array(amps)
        
        idx = np.where(np.isclose(fields_arr, target_field, atol=1e-3))[0]
        if len(idx) == 0: 
            continue
            
        target_temps = temps_arr[idx]
        target_amps = amps_arr[idx]
        target_x = (mu_b * target_field) / (2 * k_B * target_temps)
        
        p = plt.plot(target_x, target_amps, marker='+', linestyle='none', markersize=6)
        line_color = p[0].get_color()
        
        D, E, Mxy, Myz, Mxz = all_plot_params[band_name]
        
        t_smooth_amps = []
        x_smooth = []
        for i, temp in enumerate(smooth_temps):
            ave_xy, ave_yz, ave_zx = iso_basis[i]
            sim_val = Mxy * ave_xy + Myz * ave_yz + Mxz * ave_zx
            t_smooth_amps.append(sim_val)
            x_smooth.append((mu_b * target_field) / (2 * k_B * temp))
            
        plt.plot(x_smooth, t_smooth_amps, linestyle='-', color=line_color)
        
        max_x_idx = np.argmax(x_smooth)
        band_num = band_name.replace('Band', '')
        plt.text(x_smooth[max_x_idx], t_smooth_amps[max_x_idx], f" {band_num}", 
                 fontsize=12, verticalalignment='bottom')

    plt.xlabel("$\\mu_B B / 2kT$")
    plt.ylabel("MCD Intensity (mdeg)")
    plt.gca().tick_params(direction='in')
    plt.tight_layout()
    filename = out_dir / f"isofield_{target_field}T.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_magnetization_fit_grid(bands_dict: dict,flat_real_temps: list[float],flat_fields: list[float],flat_nominal_temps: list[float],all_plot_params: dict,mag_basis: dict,
    smooth_fields: np.ndarray,out_dir: Path,plot_reduced_mag: bool = True) -> list[list]:
    """Generates the multi-panel grid for VTVH magnetization curves and returns curve rows."""
    num_bands = len(bands_dict)
    n_cols = min(3, num_bands)
    n_rows = math.ceil(num_bands / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    curve_data_rows = [["Band_Name", "Temperature_K", "Data_Type", "X_Value", "MCD_Intensity"]]

    for i, (name, exp_amps) in enumerate(bands_dict.items()):
        ax = axes_flat[i]
        plot_params = all_plot_params[name]
        plot_sh_curves(
            ax, name, flat_real_temps, flat_fields, exp_amps,
            plot_params, mag_basis, smooth_fields, curve_data_rows,
            flat_nominal_temps, plot_reduced_mag=plot_reduced_mag
        )

    for j in range(num_bands, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / "magnetization_fits.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return curve_data_rows

def save_simulated_curves_to_csv(curve_data_rows: list, out_dir: Path, filename: str = "sh_simulated_curves.csv") -> None:
    with open(out_dir / filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(curve_data_rows)