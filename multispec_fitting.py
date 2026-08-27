import tomllib, math
import csv
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt
import re
from typing import Any
from asteval import Interpreter

mu_b = 0.46686  # Bohr magneton in cm-1/T
k_B = 0.695     # Boltzmann constant in cm-1/K

def load_config(filepath: str | Path) -> dict:

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'Config file missing: {path}')

    with open(path, "rb") as file:
        config = tomllib.load(file)

    if 'dataset' not in config or 'bands' not in config:
        raise ValueError("Config File is missing 'dataset' or 'bands' blocks.")

    return config

class FitParameter:
    """
    Reads in fit parameters from input file
    """
    def __init__(self, name: str, setup_dict: dict) -> None:
        self.name = name

        self.value = setup_dict.get('value', 0.0)
        self.min_val = setup_dict.get('min', -float('inf'))
        self.max_val = setup_dict.get('max', float('inf'))
        self.vary = setup_dict.get('vary', True)
        self.expr = setup_dict.get('expr', None)
        if self.expr is not None:
            self.vary = False

    def get_value(self, namespace: dict | None = None, evaluator: Any = None) -> float | np.ndarray:
        if self.expr:
            try:
                if evaluator is None:
                    evaluator = Interpreter()                
                safe_expr = self.expr.replace('.', '_')
                for key, val in namespace.items():
                    evaluator.symtable[key] = val
                return evaluator(safe_expr)
            except Exception as e:
                name = getattr(self, 'name', 'Array') 
                raise RuntimeError(f"Failed to evaluate expression '{self.expr}' for {name}: {e}")
                
        return self.value

    def set_value(self, new_value: float) -> None:
        self.value = max(self.min_val, min(self.max_val, new_value))

    def __repr__(self) -> str:
        state = f"expr='{self.expr}'" if self.expr else f"vary={self.vary}"
        return f"<FitParameter '{self.name}': value={self.value:.4f}, {state}>"

class ArrayParameter:
    "Reads in array of amplitudes from input file, separate class than scalars"
    def __init__(self, name: str, array_dict: dict, expected_keys: list[str] | None = None) -> None:
        self.name = name
        self.expr = array_dict.get('expr', None)

        if self.expr is not None:
            self.vary = False
            self.value = None
            self.min_val = None
            self.max_val = None
        else:
            flat_values = []
            self.row_keys = []
            self.row_sizes = []
            keys_to_load = expected_keys if expected_keys else [k for k in array_dict]

            for key in keys_to_load:
                if key in array_dict:
                    values = array_dict[key]
                    self.row_keys.append(key)
                    self.row_sizes.append(len(values))
                    flat_values.extend(values)

            if len(flat_values) == 0:
                raise ValueError(f"No amplitude data was loaded for {self.name}"
                                 f"\nExpected keys : {keys_to_load}"
                                 f"\nFound keys : {list(array_dict.keys())}")
               
            self.value = np.array(flat_values, dtype=float)
            self.vary = True
            self.min_val = array_dict.get('min', -np.inf)
            self.max_val = array_dict.get('max', np.inf)

    def get_value(self, namespace: dict | None = None, evaluator: Any = None) -> np.ndarray | float:
        if self.expr:
            try:
                if evaluator is None:
                    from asteval import Interpreter
                    evaluator = Interpreter()                
                safe_expr = self.expr.replace('.', '_')
                for key, val in namespace.items():
                    evaluator.symtable[key] = val
                return evaluator(safe_expr)
            except Exception as e:
                name = getattr(self, 'name', 'Array') 
                raise RuntimeError(f"Failed to evaluate expression '{self.expr}' for {name}: {e}")
                        
        return self.value

    def set_value(self, new_array: np.ndarray) -> None:
        if self.expr is not None:
            raise ValueError(f"Cannot manually set '{self.name}'; constrained by {self.expr}")
            
        self.value = np.clip(new_array, self.min_val, self.max_val)

class SpectralBand:
    """
    Default band type used to read in parameters present for all band types (center, width, amplitudes)
    """
    def __init__(self, name: str, config_dict: dict, expected_keys: list[str] | None = None) -> None:
        self.name = name
        self.type = config_dict.get('type')
        
        self.center = FitParameter(f"{name}_center", config_dict['center'])
        self.width = FitParameter(f"{name}_width", config_dict['width'])
        self.amplitudes = ArrayParameter(f"{name}_amplitudes", config_dict['amplitudes'], expected_keys)
        self.has_temp = 'temp_broadening' in config_dict
        self.has_vib = 'vib_energy' in config_dict
        self.temp_broadening = FitParameter(
            f"{name}_temp_broadening", 
            config_dict.get('temp_broadening', {'value': 0.0, 'vary': False})
        )
        self.vib_energy = FitParameter(
            f"{name}_vib_energy", 
            config_dict.get('vib_energy', {'value': 200.0, 'vary': False})
        )
    def get_parameters(self) -> list:
        params = [self.center, self.width, self.amplitudes]
        if self.has_temp: params.append(self.temp_broadening)
        if self.has_vib: params.append(self.vib_energy)
        return params
        
    def evaluate(self, x: np.ndarray, namespace: dict | None = None, evaluator: Any = None) -> np.ndarray:
        raise NotImplementedError("Must be implemented by a subclass.")

class GaussianBand(SpectralBand):
    """Class used for specific Gaussian band, used to calculate the lineshape.
    Gets center, width, and amplitudes from SpectralBand.
    """
    def __init__(self, name: str, config_dict: dict, expected_keys: list[str] | None = None) -> None:
        super().__init__(name, config_dict, expected_keys)

    def get_parameters(self) -> list:
        return super().get_parameters()

    def evaluate(self, x: np.ndarray, namespace: dict | None = None, evaluator: Any = None) -> np.ndarray:
        center = self.center.get_value(namespace,evaluator)
        w0 = self.width.get_value(namespace,evaluator)
        A = self.temp_broadening.get_value(namespace,evaluator)
        E_vib = self.vib_energy.get_value(namespace,evaluator)
        amps = self.amplitudes.get_value(namespace,evaluator)
        temps = np.array(namespace['__real_temperatures__'])

        safe_E_vib = max(abs(E_vib), 1.0) 
        thermal_term = A / np.tanh(safe_E_vib / (2 * k_B * (temps + 1e-5)))
        widths = np.maximum(w0 + thermal_term, 1e-5)
        
        exponent = -4 * np.log(2) * ((x[np.newaxis, :] - center) / widths[:, np.newaxis])**2
        base_shape = np.exp(exponent)
        
        return amps[:, np.newaxis] * base_shape

class PseudoVoigtBand(SpectralBand):
    """
    Calculated Pseudo-Voigt lineshape, a linear combination of a Gaussian and a Lorentzian profile.
    Gets center, width, and amplitude from SpectralBand.
    """
    def __init__(self, name: str, config_dict: dict, expected_keys: list[str] | None = None) -> None:
        super().__init__(name, config_dict, expected_keys)
        self.lorentz_frac = FitParameter(
                f"{name}_lorentz_frac", 
                config_dict.get('lorentz_frac', {'value': 0.5, 'min': 0.0, 'max': 1.0})
            )

    def get_parameters(self) -> list:
        base_params = super().get_parameters()
        return base_params + [self.lorentz_frac]

    def evaluate(self, x: np.ndarray, namespace: dict | None = None, evaluator: Any = None) -> np.ndarray:
        center = self.center.get_value(namespace,evaluator)
        w0 = self.width.get_value(namespace,evaluator)
        eta = self.lorentz_frac.get_value(namespace,evaluator)
        A = self.temp_broadening.get_value(namespace,evaluator)
        E_vib = self.vib_energy.get_value(namespace,evaluator)
        amps = self.amplitudes.get_value(namespace,evaluator)
        temps = np.array(namespace['__real_temperatures__'])
        safe_E_vib = max(abs(E_vib), 1.0) 
        thermal_term = A / np.tanh(safe_E_vib / (2 * k_B * (temps + 1e-5)))
        widths = np.maximum(w0 + thermal_term, 1e-5)[:, np.newaxis]

        dx = x[np.newaxis, :] - center  

        gaussian = np.exp(-4 * np.log(2) * (dx / widths)**2)
        lorentzian = 1.0 / (1.0 + 4 * (dx / widths)**2)

        base_shape = eta * lorentzian + (1 - eta) * gaussian
        
        return amps[:, np.newaxis] * base_shape

class VibronicBand(SpectralBand):
    """
    Generates vibronic progression using the Huang-Rhys factor and Poisson distribution
    Uses standard Gaussian shape for each band.
    Gets center, width, and amplitudes from SpectralBand.
    """
    def __init__(self, name: str, config_dict: dict, expected_keys: list[str] | None = None) -> None:
        super().__init__(name, config_dict, expected_keys)
        
        # Add the vibronic-specific parameters
        self.vib_spacing = FitParameter(f"{name}_vib_spacing", config_dict['vib_spacing'])
        self.huang_rhys = FitParameter(f"{name}_huang_rhys", config_dict['huang_rhys'])
        self.n_levels = config_dict.get('n_levels', 10)

    def get_parameters(self) -> list:
        """Override to include the new parameters alongside the base ones."""
        base_params = super().get_parameters()
        return base_params + [self.vib_spacing, self.huang_rhys]
        
    def evaluate(self, x: np.ndarray, namespace: dict | None = None, evaluator: Any = None) -> np.ndarray:
        """Calculates the full sum of the vibronic progression."""
        c = self.center.get_value(namespace,evaluator)
        w = self.width.get_value(namespace,evaluator)
        amps = self.amplitudes.get_value(namespace,evaluator)
        spacing = self.vib_spacing.get_value(namespace,evaluator)
        s = self.huang_rhys.get_value(namespace,evaluator)
        
        A = self.temp_broadening.get_value(namespace,evaluator)
        E_vib = self.vib_energy.get_value(namespace,evaluator)

        temps = np.array(namespace['__real_temperatures__'])
        safe_E_vib = max(abs(E_vib), 1.0)
        thermal_term = A / np.tanh(safe_E_vib / (2 * k_B * (temps + 1e-5)))
        widths = np.maximum(w + thermal_term, 1e-5)
        sigmas = widths / (2 * np.sqrt(2 * np.log(2)))       

        total_base_shape = np.zeros((len(amps), len(x)))
        
        for n in range(self.n_levels):
            c_n = c + (n * spacing)
            
            # Calculate the Franck-Condon relative intensity
            fc_factor = math.exp(-s) * (s**n / math.factorial(n))
            
            exponent = -((x[np.newaxis, :] - c_n) ** 2) / (2 * sigmas[:, np.newaxis] ** 2)
            total_base_shape += fc_factor * np.exp(exponent)
            
        return amps[:, np.newaxis] * total_base_shape

class DataSet:
    """
    Loads experimental VTVH data, aligns it with the requested TOML conditions,
    and flattens it for the least-squares minimizer.
    """
    def __init__(self, filename: str | Path, toml_fields: list[float], toml_temperatures: list[float], temp_tolerance: float = 0.5) -> None:
        self.filename = Path(filename)
        self.toml_fields = toml_fields            
        self.toml_temperatures = toml_temperatures 
        
        self.temp_tolerance = temp_tolerance 
        
        self.x = None
        self.y_matrix = None     
        self.y_flat = None       
        
        self.real_temperatures = []
        self.real_fields = []
        self.matched_headers = []
        
        self._load_and_flatten()

    def _load_and_flatten(self) -> None:
        """Reads the text file and formats the data for the fitter."""
        if not self.filename.exists():
            raise FileNotFoundError(f"Data file not found: {self.filename}")
            
        with open(self.filename, 'r') as f:
            header_line = f.readline().strip()
        headers = header_line.split()
        raw_data = np.loadtxt(self.filename, skiprows=1)
        self.x = raw_data[:, 0]

        # Parse all available headers to find T and B
        header_pattern = re.compile(r"^([0-9]*\.?[0-9]+)K_([0-9]*\.?[0-9]+)T$")
        parsed_headers = []
        for idx, h in enumerate(headers):
            m = header_pattern.match(h)
            if m:
                parsed_headers.append({
                    'T': float(m.group(1)),
                    'B': float(m.group(2)),
                    'idx': idx,
                    'header': h
                })

        selected_columns = []
        
        for set_t in self.toml_temperatures:
            for set_b in self.toml_fields:
                matches = [p for p in parsed_headers 
                           if np.isclose(p['B'], float(set_b)) and abs(p['T'] - float(set_t)) <= self.temp_tolerance]
                if not matches:
                    raise ValueError(f"Could not find a column for {set_t}K_{set_b}T within +/- {self.temp_tolerance}K window.")
                best_match = min(matches, key=lambda x: abs(x['T'] - float(set_t)))
                
                selected_columns.append(best_match['idx'])
                self.real_temperatures.append(best_match['T'])
                self.real_fields.append(best_match['B'])
                self.matched_headers.append(best_match['header'])
             
        self.y_matrix = raw_data[:, selected_columns]        
        self.y_flat = self.y_matrix.T.flatten()

    def get_x(self) -> np.ndarray:
        return self.x
        
    def get_y_flat(self) -> np.ndarray:
        return self.y_flat

class GlobalFitter:
    def __init__(self, dataset: DataSet, bands: list[SpectralBand], method: str = 'least_squares') -> None:
        self.dataset = dataset
        self.bands = bands  
        self.method = method
        self.aeval = Interpreter()
        self.floating_params = []
        self._gather_floating_parameters()

    def _gather_floating_parameters(self) -> None:
        """Finds every parameterwhere vary == True."""
        for band in self.bands:
            for param in band.get_parameters():
                if param.vary:
                    self.floating_params.append(param)

    def _build_namespace(self) -> dict:
        """
        Builds a dictionary of all current parameter values so that constrained parameters can evaluate.
        """
        namespace = {}
        namespace['__temperatures__'] = self.dataset.toml_temperatures
        namespace['__real_temperatures__'] = self.dataset.real_temperatures
        for band in self.bands:
            for param in band.get_parameters():
                namespace[param.name] = param.value
            
        return namespace

    def _get_initial_guesses(self) -> np.ndarray:
        """Grabs initial values for fitting"""
        x0 = []
        for param in self.floating_params:
            val = param.value
            if isinstance(val, np.ndarray):
                x0.extend(val.flatten())
            else:
                x0.append(val)
        return np.array(x0)

    def residual(self, scipy_x: np.ndarray) -> np.ndarray:
        """
        Core function for minimization, calculate residualss
        """
        idx = 0
        for param in self.floating_params:
            if isinstance(param.value, np.ndarray):
                size = param.value.size
                param.set_value(scipy_x[idx : idx+size])
                idx += size
            else:
                param.set_value(scipy_x[idx])
                idx += 1
                
        namespace = self._build_namespace()
        
        # Calculate the total simulated spectrum
        x_axis = self.dataset.get_x()
        total_simulation = np.zeros_like(self.dataset.get_y_flat())

        for band in self.bands:
            simulated_matrix = band.evaluate(x_axis, namespace, evaluator=self.aeval)
            total_simulation += simulated_matrix.flatten()
            
        # Return the 1D residual
        return total_simulation - self.dataset.get_y_flat()

    def cost_function(self, scipy_x: np.ndarray) -> float:
        """Converts residual array into a single scalar value for DE."""
        res_array = self.residual(scipy_x)
        return np.sum(res_array **2)

    def _get_bounds(self) -> list[tuple[float, float]]:
        bounds = []

        # Calculates spectral window with buffer for default min/max
        x_data = self.dataset.get_x()
        x_min, x_max = np.min(x_data), np.max(x_data)
        buffer = (x_max - x_min) * 0.1
        
        safe_center_min = x_min - buffer
        safe_center_max = x_max + buffer
        
        for param in self.floating_params:
            p_min = param.min_val
            p_max = param.max_val
            
            if isinstance(param.value, np.ndarray):
                # Amplitudes
                for val in param.value.flatten():
                    c_min = val - abs(val) if np.isinf(p_min) else p_min
                    c_max = val + abs(val) if np.isinf(p_max) else p_max
                    bounds.append((c_min, c_max))
            else:
                # Scalars
                val = param.value

                if 'center' in param.name.lower():
                    # Centers default to the data window + 10% if left blank
                    c_min = safe_center_min if np.isinf(p_min) else p_min
                    c_max = safe_center_max if np.isinf(p_max) else p_max
                else:
                    # Other parameters float +/- 1000%
                    margin = abs(val) if val != 0 else 1000.0
                    c_min = val - margin if np.isinf(p_min) else p_min
                    c_max = val + margin if np.isinf(p_max) else p_max
                    
                bounds.append((c_min, c_max))
                
        return bounds

    def run(self) -> Any:
        """Executes the fit using least squares or differential evolution."""
        x0 = self._get_initial_guesses()

        if self.method == 'differential_evolution':    
            bounds = self._get_bounds()

            result = differential_evolution(
                self.cost_function,
                bounds = bounds,
                x0 = x0,
                polish = True,
                workers = -1,
                updating = 'deferred',
                disp = True
            )
        else:
            result = least_squares(
                self.residual, 
                x0, 
                method='trf', 
                ftol=1e-6, 
                xtol=1e-6
            )
        
        self.residual(result.x)
        return result

def plot_results(dataset: DataSet, bands: list[SpectralBand], namespace: dict) -> None:
    x_axis = dataset.get_x()
    temps = dataset.toml_temperatures
    fields = dataset.toml_fields
    
    n_rows = len(temps)
    n_cols = len(fields)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    
    for i, temp in enumerate(temps):
        for j, field in enumerate(fields):
            ax = axes[i, j]
            matrix_idx = (i * n_cols) + j
            
            y_raw = dataset.y_matrix.T[matrix_idx]
            ax.plot(x_axis, y_raw, color='grey', label='Experimental', alpha=0.7)
            
            total_sim = np.zeros_like(x_axis)
            
            for band in bands:
                band_matrix = band.evaluate(x_axis, namespace)
                band_y = band_matrix[matrix_idx]
                ax.plot(x_axis, band_y, linestyle='--', linewidth=1, label=band.name)
                total_sim += band_y
                
            ax.plot(x_axis, total_sim, color='black', linewidth=1.2, label='Total Fit')
            
            ax.set_title(f"{temp}K {field}T")
            ax.invert_xaxis()
            
            if i == 0 and j == n_cols - 1:
                ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=2, fontsize='small')

    plt.tight_layout()
    plt.savefig("fit_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    


def save_results_to_toml(filename: str, dataset: DataSet, bands: list[SpectralBand]) -> None:
    """Writes the fit results into a TOML output file."""
    with open(filename, 'w', encoding='utf-8') as f:
        
        # Write the Dataset block
        f.write("[dataset]\n")
        f.write(f'filename = "{dataset.filename.name}"\n')
        f.write(f"fields = {dataset.toml_fields}\n")
        temps_formatted = ", ".join([f'"{t}"' if isinstance(t, str) else str(t) for t in dataset.toml_temperatures])
        f.write(f"temperatures = [{temps_formatted}]\n\n")
        
        # Write the Bands
        for band in bands:
            f.write(f"[bands.{band.name}]\n")
            f.write(f'type = "{band.type}"\n')

            #Write n-levels if exists
            if hasattr(band, 'n_levels'):
                f.write(f'n_levels = {band.n_levels}\n')
            # Write all scalars
            for param in band.get_parameters():
                name = param.name.split('_', 1)[1] 
                
                # Expressions
                if param.expr:
                    f.write(f'{name} = {{ expr = "{param.expr}" }}\n')
                elif not (hasattr(param.value, "size") and param.value.size > 1):
                    vary_str = "true" if param.vary else "false"
                    f.write(f'{name} = {{ value = {param.value:.4f}, vary = {vary_str} }}\n')
            
            f.write("\n")
            
            # Write all amplitudes
            for param in band.get_parameters():
                name = param.name.split('_', 1)[1]
                
                if not param.expr and hasattr(param.value, "size") and param.value.size > 1:
                    f.write(f"[bands.{band.name}.{name}]\n")
                    
                    current_idx = 0
                    for key, size in zip(param.row_keys, param.row_sizes):
                        chunk = param.value[current_idx : current_idx + size]
                        chunk_str = ", ".join([f"{val:.3f}" for val in chunk])
                        f.write(f'"{key}" = [{chunk_str}]\n')
                        current_idx += size
                    f.write("\n")

def save_results_to_csv(dataset: DataSet, bands: list[SpectralBand], namespace: dict, param_filename: str = "output_parameters.csv", amp_filename: str = "output_amplitudes.csv") -> None:
    """
    Exports the optimized fit results into two separate CSV files.
    """
    
    # --- Table 1: Scalar Parameters ---
    with open(param_filename, mode='w', newline='') as f_param:
        writer = csv.writer(f_param)
        
        # Find all unique parameter names across every loaded band
        unique_param_names = []
        for band in bands:
            for param in band.get_parameters():
                if not isinstance(param, ArrayParameter):
                    base_name = param.name.split('_', 1)[1]
                    if base_name not in unique_param_names:
                        unique_param_names.append(base_name)
        headers = ["Band_Name", "Type"] + [name.title() for name in unique_param_names]
        writer.writerow(headers)

        for band in bands:
            # Initialize a dictionary for this row with empty strings
            row_dict = {name: "" for name in unique_param_names}
            
            # Fill in the values for the band
            for param in band.get_parameters():
                if not isinstance(param, ArrayParameter):
                    base_name = param.name.split('_', 1)[1]
                    row_dict[base_name] = str(param.expr) if param.expr else f"{param.value:.4f}"
            
            # Place row in the proper order
            row = [band.name, band.type]
            for name in unique_param_names:
                row.append(row_dict[name])
                
            writer.writerow(row)    
            
    # --- Table 2: VTVH Amplitudes ---
    with open(amp_filename, mode='w', newline='') as f_amp:
        writer = csv.writer(f_amp)
        
        headers = ["Band_Name"]
        for temp in dataset.toml_temperatures:
            for field in dataset.toml_fields:
                headers.append(f"{temp}K_{field}T")
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

def parse_args() -> argparse.Namespace:
    """Parses command line arguments to find input"""
    parser = argparse.ArgumentParser(description="Run the global fitting routine with parameters from TOML input file.")

    parser.add_argument(
        "config_file",
        type = str,
        help = "Path to TOML configuration file"
    )
    parser.add_argument(
        "--sh-only",
        action = "store_true",
        help = "Skip spectral fitting and only run the Spin-Hamiltonian solver using amplitudes in the TOML."
    )
    return parser.parse_args()

def save_spectra_to_csv(dataset: DataSet, bands: list[SpectralBand], namespace: dict, filename: str = 'output_spectra.csv') -> None:
    """Exports raw x-axis, expeirmenta y-values, total fit, and individual band fits"""
    x_axis = dataset.get_x()
    temps = dataset.toml_temperatures
    fields = dataset.toml_fields
    n_cols = len(fields)
    band_matrices = [band.evaluate(x_axis, namespace) for band in bands]

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Build Headers
        headers = ["X_Value"]
        for i, temp in enumerate(temps):
            for j, field in enumerate(fields):
                cond = f"{temp}K_{field}T"
                headers.extend([f"Exp_{cond}", f"Fit_{cond}"])
                for band in bands:
                    headers.append(f"{band.name}_{cond}")
        writer.writerow(headers)

        # Write Data Row by Row
        for row_idx, x_val in enumerate(x_axis):
            row_data = [f"{x_val:.4f}"]
            
            for i, temp in enumerate(temps):
                for j, field in enumerate(fields):
                    matrix_idx = (i * n_cols) + j
                    
                    # Experimental Y
                    exp_val = dataset.y_matrix[row_idx, matrix_idx]
                    row_data.append(f"{exp_val:.4f}")
                    
                    # Total Fit and Individual Bands
                    total_fit = 0.0
                    band_vals = []
                    for b_idx, band in enumerate(bands):
                        val = band_matrices[b_idx][matrix_idx][row_idx]
                        total_fit += val
                        band_vals.append(f"{val:.4f}")
                        
                    row_data.append(f"{total_fit:.4f}")
                    row_data.extend(band_vals)
                    
            writer.writerow(row_data)

def get_spin_matrices(S: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates the Sx, Sy, and Sz spin operator matrices for a given spin S.
    Returns complex numpy arrays of shape (2S+1, 2S+1).
    """
    dim = int(2 * S + 1)
    Sz = np.zeros((dim, dim), dtype=complex)
    Sp = np.zeros((dim, dim), dtype=complex) # S_+ 
    Sm = np.zeros((dim, dim), dtype=complex) # S_- 
    
    for i in range(dim):
        m = S - i
        Sz[i, i] = m
        
        if i > 0:
            Sp[i-1, i] = np.sqrt(S * (S + 1) - m * (m + 1))
            
        if i < dim - 1:
            Sm[i+1, i] = np.sqrt(S * (S + 1) - m * (m - 1))
            
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)
    
    return Sx, Sy, Sz

class SpinHamiltonian:
    """
    Calculates the energy levels and wavefunctions for a given spin system
    subject to Zero-Field Splitting and an external magnetic field.

    The VTVH MCD magnetization and effective transition dipole extraction 
    formalism is based on the theoretical framework developed in:
    Ref: Frank Neese, Edward I. Solomon; MCD C-Term Signs, Saturation Behavior, and Determination 
    of Band Polarizations in Randomly Oriented Systems with Spin S ≥ 1/2. Applications to S = 1/2 and S = 5/2. 
    Inorg. Chem. 19 April 1999; 38 (8): 1847–1865.
    """
    def __init__(self, S: float, D: float, E: float, g: float | list[float]) -> None:
        self.S = S
        self.Sx, self.Sy, self.Sz = get_spin_matrices(S)
        
        # Handle isotropic g-value or anisotropic g-tensor
        if isinstance(g, (int, float)):
            self.gx = self.gy = self.gz = float(g)
        else:
            self.gx, self.gy, self.gz = g
            
        # Bohr magneton
        self.mu_b = mu_b
        
        # Build the Zero-Field Splitting Matrix 
        S_sq = S * (S + 1)
        identity = np.eye(int(2 * S + 1), dtype=complex)
        
        self.H_zfs = D * (self.Sz @ self.Sz - (S_sq / 3.0) * identity) + \
                     E * (self.Sx @ self.Sx - self.Sy @ self.Sy)

    def solve(self, B_vector: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies the magnetic field (Zeeman effect), diagonalizes the matrix,
        and returns the energies and spin expectation values.
        """
        Bx, By, Bz = B_vector
        
        # Build the Zeeman Matrix 
        H_zeeman = self.mu_b * (self.gx * Bx * self.Sx + 
                                self.gy * By * self.Sy + 
                                self.gz * Bz * self.Sz)
                                
        H_total = self.H_zfs + H_zeeman
        
        # Diagonalize the Hermitian matrix
        energies, wavefunctions = np.linalg.eigh(H_total)
        
        # Calculate Spin Expectation Values for each state
        exp_Sx = np.diag(wavefunctions.conj().T @ self.Sx @ wavefunctions).real
        exp_Sy = np.diag(wavefunctions.conj().T @ self.Sy @ wavefunctions).real
        exp_Sz = np.diag(wavefunctions.conj().T @ self.Sz @ wavefunctions).real
        exp_S = np.vstack((exp_Sx, exp_Sy, exp_Sz)).T
        
        return energies, exp_S

    def get_mcd_components(self, B_mag: float, temp: float, n_theta: int = 30, n_phi: int = 30) -> tuple[float, float, float]:
        """
        Calculates the orientation-averaged MCD basis components (xy, yz, zx)
        decoupled from the transition dipoles.
        """
        ave_xy, ave_yz, ave_zx = 0.0, 0.0, 0.0
        weight_sum = 0.0
        
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        kT = k_B * temp  
        
        for theta in thetas:
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            weight = sin_t 
            
            for phi in phis:
                sin_p = np.sin(phi)
                cos_p = np.cos(phi)
                
                ux = sin_t * cos_p
                uy = sin_t * sin_p
                uz = cos_t
                
                B_vector = [B_mag * ux, B_mag * uy, B_mag * uz]
                energies, exp_S = self.solve(B_vector)
                
                exp_terms = np.exp(-(energies - energies[0]) / kT)
                populations = exp_terms / np.sum(exp_terms)
                
                comp_xy, comp_yz, comp_zx = 0.0, 0.0, 0.0
                for i in range(len(energies)):
                    Sx, Sy, Sz = exp_S[i]
                    comp_xy += populations[i] * (uz * Sz)
                    comp_yz += populations[i] * (ux * Sx)
                    comp_zx += populations[i] * (uy * Sy)
                                                          
                ave_xy += comp_xy * weight
                ave_yz += comp_yz * weight
                ave_zx += comp_zx * weight
                weight_sum += weight
                
        return ave_xy / weight_sum, ave_yz / weight_sum, ave_zx / weight_sum

class MagnetizationFitter:
    """
    Fits experimental VTVH amplitudes to extract Zero-Field Splitting (D, E),
    an isotropic g-value, and effective transition dipole moments.
    """
    def __init__(self, S: float, temps: list[float], fields: list[float], exp_norm_dict: dict, sh_params: dict, symmetry_mode: str = "isotropic") -> None:
        self.S = S
        self.temps = temps
        self.fields = fields
        self.exp_norm_dict = {}  
        self.band_names = list(exp_norm_dict.keys())
        self.symmetry_mode = symmetry_mode.lower()
        
        self.sh_params = sh_params
        self.floating_sh_params = [p for p in self.sh_params.values() if p.vary]
        self.num_floating_sh = len(self.floating_sh_params)
        
        self.scale_factors = {}
        for name, amps in exp_norm_dict.items():
            sf = np.max(np.abs(amps))
            if sf == 0: sf = 1.0
            self.scale_factors[name] = sf
            self.exp_norm_dict[name] = np.array(amps) / sf

    def residual(self, params: list[float]) -> np.ndarray:
        for i, param in enumerate(self.floating_sh_params):
            param.set_value(params[i])
            
        if self.symmetry_mode == "isotropic":
            gx = gy = gz = self.sh_params['g'].value
        elif self.symmetry_mode == "axial":
            gx = gy = self.sh_params['gx'].value
            gz = self.sh_params['gz'].value
        elif self.symmetry_mode == "rhombic":
            gx = self.sh_params['gx'].value
            gy = self.sh_params['gy'].value
            gz = self.sh_params['gz'].value
            
        D = self.sh_params['D'].value
        E = self.sh_params['E'].value
            
        if D != 0 and abs(E / D) > 1/3:
            return np.ones(len(self.temps) * len(self.band_names)) * 1e6
            
        engine = SpinHamiltonian(self.S, D, E, [gx, gy, gz])
        
        basis_matrix = {}
        for i, t in enumerate(self.temps):
            b = self.fields[i]
            if (t, b) not in basis_matrix:
                basis_matrix[(t, b)] = engine.get_mcd_components(b, t, n_theta=15, n_phi=15)
        
        all_residuals = []
        
        band_params = params[self.num_floating_sh:]
        
        for i, name in enumerate(self.band_names):
            if self.symmetry_mode == "isotropic":
                idx = i
                Mxy = Myz = Mxz = band_params[idx]
            elif self.symmetry_mode == "axial":
                idx = i * 2
                Mxy, Mxz = band_params[idx : idx+2]
                Myz = Mxz
            elif self.symmetry_mode == "rhombic":
                idx = i * 3
                Mxy, Myz, Mxz = band_params[idx : idx+3]
                
            simulated_norm = np.zeros(len(self.temps))
            for j in range(len(self.temps)):
                t = self.temps[j]
                b = self.fields[j]
                ave_xy, ave_yz, ave_zx = basis_matrix[(t, b)]
                simulated_norm[j] = Mxy * ave_xy + Myz * ave_yz + Mxz * ave_zx
                
            band_residual = simulated_norm - self.exp_norm_dict[name]
            all_residuals.append(band_residual)
            
        return np.concatenate(all_residuals)

    def cost_function(self, params: list[float]) -> float:
        res_array = self.residual(params)
        return np.sum(res_array ** 2)

    def run_fit(self, method: str = 'least_squares') -> Any:
        guess, lb, ub = [], [], []
        
        # Load bounds for floating Spin-Hamiltonian parameters
        for p in self.floating_sh_params:
            guess.append(p.value)
            lb.append(p.min_val)
            ub.append(p.max_val)
            
        # Append bounds for the Transition Dipoles
        num_bands = len(self.band_names)
        if self.symmetry_mode == "isotropic":
            guess.extend([0.1] * num_bands)
            lb.extend([-50.0] * num_bands)
            ub.extend([ 50.0] * num_bands)
        elif self.symmetry_mode == "axial":
            guess.extend([0.1, 0.1] * num_bands)
            lb.extend([-50.0, -50.0] * num_bands)
            ub.extend([ 50.0,  50.0] * num_bands)
        elif self.symmetry_mode == "rhombic":
            guess.extend([0.1, 0.1, 0.1] * num_bands)
            lb.extend([-50.0, -50.0, -50.0] * num_bands)
            ub.extend([ 50.0,  50.0,  50.0] * num_bands)

        if method == 'differential_evolution':
            de_bounds = list(zip(lb, ub))
            result = differential_evolution(
                self.cost_function, bounds=de_bounds, x0=guess, polish=True, workers=-1, disp=True
            )
        else:
            result = least_squares(
                self.residual, x0=guess, bounds=(lb, ub), method='trf', xtol=1e-4, ftol=1e-4
            )
            
        return result


def run_global_sh_fit(bands_dict: dict, flat_real_temps: list[float], flat_fields: list[float], flat_nominal_temps: list[float], mol_config: dict) -> None:
    S = mol_config.get('spin', 1.0)
    method = mol_config.get('method', 'least_squares')
    plot_reduced_mag = mol_config.get('plot_reduced_mag', True)
    
    symmetry_mode = mol_config.get('symmetry', 'isotropic').lower()
    if symmetry_mode not in ["isotropic", "axial", "rhombic"]:
        raise ValueError("Invalid symmetry in input file. Please use isotropic, axial, or rhombic.")

    def_g = {'value': 2.00, 'vary': False}
    def_D = {'value': 5.0, 'vary': False}
    def_E = {'value': 0.0, 'vary': False}
    
    sh_params = {}
    if symmetry_mode == "isotropic":
        sh_params['g'] = FitParameter("g", mol_config.get('g', def_g))
    elif symmetry_mode == "axial":
        sh_params['gx'] = FitParameter("gx", mol_config.get('gx', def_g))
        sh_params['gz'] = FitParameter("gz", mol_config.get('gz', def_g))
    elif symmetry_mode == "rhombic":
        sh_params['gx'] = FitParameter("gx", mol_config.get('gx', def_g))
        sh_params['gy'] = FitParameter("gy", mol_config.get('gy', def_g))
        sh_params['gz'] = FitParameter("gz", mol_config.get('gz', def_g))
        
    sh_params['D'] = FitParameter("D", mol_config.get('D', def_D))
    sh_params['E'] = FitParameter("E", mol_config.get('E', def_E))

    fitter = MagnetizationFitter(S, flat_real_temps, flat_fields, bands_dict, sh_params, symmetry_mode=symmetry_mode)
    result = fitter.run_fit(method=method)

    if result.success:        
        for i, param in enumerate(fitter.floating_sh_params):
            param.set_value(result.x[i])
            
        if symmetry_mode == "isotropic":
            gx_fit = gy_fit = gz_fit = sh_params['g'].value
        elif symmetry_mode == "axial":
            gx_fit = gy_fit = sh_params['gx'].value
            gz_fit = sh_params['gz'].value
        elif symmetry_mode == "rhombic":
            gx_fit = sh_params['gx'].value
            gy_fit = sh_params['gy'].value
            gz_fit = sh_params['gz'].value
            
        D_fit = sh_params['D'].value
        E_fit = sh_params['E'].value
        g_tensor = [gx_fit, gy_fit, gz_fit]
                
        engine = SpinHamiltonian(S, D_fit, E_fit, g_tensor)       
        unique_nom_temps = np.unique(flat_nominal_temps)
        max_field = np.max(flat_fields)
        
        smooth_fields = np.linspace(0.1, max_field * 1.05, 50)
        mag_basis = {}
        for t in unique_nom_temps:
            mag_basis[t] = []
            for b in smooth_fields:
                mag_basis[t].append(engine.get_mcd_components(b, t, n_theta=30, n_phi=30))
                
        target_field = max_field
        min_t, max_t = np.min(unique_nom_temps), np.max(unique_nom_temps)
        smooth_temps = np.logspace(np.log10(min_t * 0.8), np.log10(max_t * 5), 100)
        
        iso_basis = []
        for t in smooth_temps:
            iso_basis.append(engine.get_mcd_components(target_field, t, n_theta=30, n_phi=30))

        out_dir = Path("sh_outputs")
        out_dir.mkdir(exist_ok=True)
        
        all_plot_params = {}
        sh_csv_data = []
        curve_data_rows = [["Band_Name", "Temperature_K", "Data_Type", "X_Value", "MCD_Intensity"]]
        
        band_params = result.x[fitter.num_floating_sh:]
        
        num_bands = len(fitter.band_names)
        n_cols = min(3, num_bands) 
        n_rows = math.ceil(num_bands / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        if num_bands > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        for i, name in enumerate(fitter.band_names):
            if symmetry_mode == "isotropic":
                idx = i
                Mxy = Myz = Mxz = band_params[idx]
            elif symmetry_mode == "axial":
                idx = i * 2
                Mxy, Mxz = band_params[idx : idx+2]
                Myz = Mxz
            elif symmetry_mode == "rhombic":
                idx = i * 3
                Mxy, Myz, Mxz = band_params[idx : idx+3]
            
            eps = 1e-12 
            Px = abs((Mxy * Mxz) / (Myz + eps))
            Py = abs((Mxy * Myz) / (Mxz + eps))
            Pz = abs((Myz * Mxz) / (Mxy + eps))
            
            total_P = Px + Py + Pz
            if total_P > 0:
                perc_x = (Px / total_P) * 100
                perc_y = (Py / total_P) * 100
                perc_z = (Pz / total_P) * 100
            else:
                perc_x = perc_y = perc_z = 0.0
                
            sh_csv_data.append([name, Mxy, Myz, Mxz, perc_x, perc_y, perc_z])
            sf = fitter.scale_factors[name]
            plot_params = [D_fit, E_fit, Mxy * sf, Myz * sf, Mxz * sf]
            all_plot_params[name] = plot_params
            
            ax = axes_flat[i]
            plot_sh_curves(ax, name, flat_real_temps, flat_fields, bands_dict[name], plot_params, mag_basis, smooth_fields, curve_data_rows, flat_nominal_temps, plot_reduced_mag=plot_reduced_mag)
                
        for j in range(num_bands, len(axes_flat)):
            axes_flat[j].set_visible(False)
            
        plt.tight_layout()
        fig.savefig(out_dir / "magnetization_fits.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        plot_isofield_summary(bands_dict, flat_real_temps, flat_fields, all_plot_params, iso_basis, smooth_temps, target_field, out_dir)
        save_sh_results_to_csv(sh_csv_data, D_fit, E_fit, g_tensor, out_dir)
        
        with open(out_dir / "sh_simulated_curves.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(curve_data_rows)
            
    else:
        print(f"Global fit failed: {result.message}")

def run_standalone_sh(config: dict) -> None:
    if 'sh' not in config:
        raise ValueError("Cannot run SH solver: Missing [sh] block in TOML.")
        
    mol_config = config['sh']
    temps = config['dataset']['temperatures']
    fields = config['dataset']['fields']
    flat_nominal_temps = [float(t) for t in temps for f in fields]
    flat_real_temps = flat_nominal_temps 
    flat_fields = [float(f) for t in temps for f in fields]
    
    bands_dict = {}
    for band_name, band_setup in config['bands'].items():
        if 'amplitudes' not in band_setup or 'expr' in band_setup['amplitudes']:
            continue
            
        amps_dict = band_setup['amplitudes']
        amps = []
        for t in temps:
            if str(t) in amps_dict:
                amps.extend(amps_dict[str(t)])
                
        if amps:
            bands_dict[band_name] = amps

    if bands_dict:
        run_global_sh_fit(bands_dict, flat_real_temps, flat_fields, flat_nominal_temps, mol_config)

def run_magnetization_pipeline(dataset: DataSet, bands: list[SpectralBand], namespace: dict, mol_config: dict) -> None:
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
        
        # Convert experimental X-axis to reduced magnetization
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

def main() -> None:
    args = parse_args()
    config = load_config(args.config_file)

    if args.sh_only:
        run_standalone_sh(config)
        return
    
    dataset = DataSet(
        filename=config['dataset']['filename'],
        toml_fields=config['dataset']['fields'],
        toml_temperatures=config['dataset']['temperatures'],
        temp_tolerance=config['dataset'].get('temp_tolerance', 0.5)
    )

    expected_keys = [str(t) for t in dataset.toml_temperatures]

    bands = []
    
    for band_name, band_setup in config['bands'].items():
        band_type = band_setup.get('type')
        
        if band_type == 'Gaussian':
            bands.append(GaussianBand(band_name, band_setup, expected_keys))
        elif band_type == 'Vibronic':
            bands.append(VibronicBand(band_name, band_setup, expected_keys))
        elif band_type == 'PseudoVoigt':
            bands.append(PseudoVoigtBand(band_name, band_setup, expected_keys))
        else:
            raise ValueError(f"Unknown band type '{band_type}' found in [{band_name}].")

    fit_settings = config.get('fit_settings', {})
    method = fit_settings.get('method','least_squares')
    fitter = GlobalFitter(dataset, bands, method=method)
    
    # Run the Optimization
    result = fitter.run()
    
    if result.success:
        final_namespace = fitter._build_namespace()
        save_results_to_toml('fitted_results.toml', dataset, bands)
        save_results_to_csv(dataset, bands, final_namespace)
        save_spectra_to_csv(dataset, bands, final_namespace)
        plot_results(dataset, bands, final_namespace)
        
        if 'sh' in config:
            run_magnetization_pipeline(dataset, bands, final_namespace, config['sh'])

if __name__ == "__main__":
    main()