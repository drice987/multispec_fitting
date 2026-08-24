import tomllib, math
import csv
import argparse
from asteval import Interpreter
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt

aeval = Interpreter()

def load_config(filepath):

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
    def __init__(self, name, setup_dict):
        self.name = name

        self.value = setup_dict.get('value', 0.0)
        self.min_val = setup_dict.get('min', -float('inf'))
        self.max_val = setup_dict.get('max', float('inf'))
        self.vary = setup_dict.get('vary', True)
        self.expr = setup_dict.get('expr', None)
        if self.expr is not None:
            self.vary = False

    def get_value(self, namespace=None):
        if self.expr:
            try:
                safe_expr = self.expr.replace('.', '_')
                for key, val in namespace.items():
                    aeval.symtable[key] = val
                return aeval(safe_expr)
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate expression '{self.expr}' for {self.name}: {e}")

        return self.value

    def set_value(self, new_value):
        self.value = max(self.min_val, min(self.max_val, new_value))

    def __repr__(self):
        state = f"expr='{self.expr}'" if self.expr else f"vary={self.vary}"
        return f"<FitParameter '{self.name}': value={self.value:.4f}, {state}>"

class ArrayParameter:
    "Reads in array of amplitudes from input file, separate class than scalars"
    def __init__(self, name, array_dict, expected_keys = None):
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

    def get_value(self, namespace=None):
        if self.expr:
            try:
                safe_expr = self.expr.replace('.', '_')
                for key, val in namespace.items():
                    aeval.symtable[key] = val
                return aeval(safe_expr)
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate expression '{self.expr}': {e}")
                
        return self.value 

    def set_value(self, new_array):
        if self.expr is not None:
            raise ValueError(f"Cannot manually set '{self.name}'; constrained by {self.expr}")
            
        self.value = np.clip(new_array, self.min_val, self.max_val)

class SpectralBand:
    """
    Default band type used to read in parameters present for all band types (center, width, amplitudes)
    """
    def __init__(self, name, config_dict,expected_keys=None):
        self.name = name
        self.type = config_dict.get('type')
        
        self.center = FitParameter(f"{name}_center", config_dict['center'])
        self.width = FitParameter(f"{name}_width", config_dict['width'])
        self.amplitudes = ArrayParameter(f"{name}_amplitudes", config_dict['amplitudes'], expected_keys)
        
    def get_parameters(self):
        return [self.center, self.width, self.amplitudes]
        
    def evaluate(self, x, namespace=None):
        raise NotImplementedError("Must be implemented by a subclass.")

class GaussianBand(SpectralBand):
    """Class used for specific Gaussian band, used to calculate the lineshape.
    Gets center, width, and amplitudes from SpectralBand.
    """
    def __init__(self, name, config_dict, expected_keys = None):
        super().__init__(name, config_dict, expected_keys)

        #Optional thermal broadening
        self.temp_broadening = FitParameter(f"{name}_temp_broadening", config_dict.get('temp_broadening', {'value': 0.0, 'vary': False}))
        self.vib_energy = FitParameter(f"{name}_vib_energy", config_dict.get('vib_energy', {'value': 200.0, 'vary': False}))

    def get_parameters(self):
        return super().get_parameters() + [self.temp_broadening, self.vib_energy]

    def evaluate(self, x, namespace=None):
        center = self.center.get_value(namespace)
        w0 = self.width.get_value(namespace)
        A = self.temp_broadening.get_value(namespace)
        E_vib = self.vib_energy.get_value(namespace)
        amps = self.amplitudes.get_value(namespace)

        k_B = 0.695
        temps = np.array(namespace['__temperatures__'])
        
        safe_E_vib = max(abs(E_vib), 1.0) 
        thermal_term = A / np.tanh(safe_E_vib / (2 * k_B * (temps + 1e-5)))
        widths = np.maximum(w0 + thermal_term, 1e-5)
        
        num_fields = len(amps) // len(temps)
        full_array = np.repeat(widths, num_fields)
        
        exponent = -4 * np.log(2) * ((x[np.newaxis, :] - center) / full_array[:, np.newaxis])**2
        base_shape = np.exp(exponent)
        
        return amps[:, np.newaxis] * base_shape

class VibronicBand(SpectralBand):
    """
    Generates vibronic progression using the Huang-Rhys factor and Poisson distribution
    Uses standard Gaussian shape for each band.
    Gets center, width, and amplitudes from SpectralBand.
    """
    def __init__(self, name, config_dict, expected_keys=None):
        super().__init__(name, config_dict, expected_keys)
        
        # Add the vibronic-specific parameters
        self.vib_spacing = FitParameter(f"{name}_vib_spacing", config_dict['vib_spacing'])
        self.huang_rhys = FitParameter(f"{name}_huang_rhys", config_dict['huang_rhys'])
        self.n_levels = config_dict.get('n_levels', 10)
        
        # Optional thermal broadening
        self.temp_broadening = FitParameter(f"{name}_temp_broadening", config_dict.get('temp_broadening', {'value': 0.0, 'vary': False}))
        self.vib_energy = FitParameter(f"{name}_vib_energy", config_dict.get('vib_energy', {'value': 200.0, 'vary': False}))

    def get_parameters(self):
        """Override to include the new parameters alongside the base ones."""
        base_params = super().get_parameters()
        return base_params + [self.vib_spacing, self.huang_rhys, self.temp_broadening, self.vib_energy]
        
    def evaluate(self, x, namespace=None):
        """Calculates the full sum of the vibronic progression."""
        c = self.center.get_value(namespace)
        w = self.width.get_value(namespace)
        amps = self.amplitudes.get_value(namespace)
        spacing = self.vib_spacing.get_value(namespace)
        s = self.huang_rhys.get_value(namespace)
        
        A = self.temp_broadening.get_value(namespace)
        E_vib = self.vib_energy.get_value(namespace)

        # Thermal Broadening
        k_B = 0.695
        temps = np.array(namespace['__temperatures__'])
        safe_E_vib = max(abs(E_vib), 1.0)
        thermal_term = A / np.tanh(safe_E_vib / (2 * k_B * (temps + 1e-5)))
        widths = np.maximum(w + thermal_term, 1e-5)

        #Convert to sigmas
        sigmas = widths / (2 * np.sqrt(2 * np.log(2)))
        
        # Temps x Fields
        num_fields = len(amps) // len(temps)
        full_sigma_array = np.repeat(sigmas, num_fields)
        
        total_base_shape = np.zeros((len(amps), len(x)))
        
        for n in range(self.n_levels):
            c_n = c + (n * spacing)
            
            # Calculate the Franck-Condon relative intensity
            fc_factor = math.exp(-s) * (s**n / math.factorial(n))
            
            exponent = -((x[np.newaxis, :] - c_n) ** 2) / (2 * full_sigma_array[:, np.newaxis] ** 2)
            total_base_shape += fc_factor * np.exp(exponent)
            
        return amps[:, np.newaxis] * total_base_shape

class DataSet:
    """
    Loads experimental VTVH data, aligns it with the requested TOML conditions,
    and flattens it for the least-squares minimizer.
    """
    def __init__(self, filename, toml_fields, toml_temperatures):
        self.filename = Path(filename)
        self.toml_fields = toml_fields            
        self.toml_temperatures = toml_temperatures 
        
        self.x = None
        self.y_matrix = None     
        self.y_flat = None       
        
        self._load_and_flatten()

    def _load_and_flatten(self):
        """Reads the text file and formats the data for the fitter."""
        if not self.filename.exists():
            raise FileNotFoundError(f"Data file not found: {self.filename}")
            
        with open(self.filename, 'r') as f:
            header_line = f.readline().strip()
        headers = header_line.split()
        raw_data = np.loadtxt(self.filename, skiprows=1)
        self.x = raw_data[:, 0]

        selected_columns = []
        for temp in self.toml_temperatures:
            for field in self.toml_fields:
                expected_header = f"{temp}K_{field}T"

                try:
                    col_idx = headers.index(expected_header)
                    selected_columns.append(col_idx)
                except ValueError:
                    raise ValueError(f"Could not find '{expected_header}' in data file.")
             
        self.y_matrix = raw_data[:, selected_columns]
        
        self.y_flat = self.y_matrix.T.flatten()

    def get_x(self):
        return self.x
        
    def get_y_flat(self):
        return self.y_flat

class GlobalFitter:
    def __init__(self, dataset, bands, method = 'least_squares'):
        self.dataset = dataset
        self.bands = bands  
        self.method = method

        self.floating_params = []
        self._gather_floating_parameters()

    def _gather_floating_parameters(self):
        """Finds every parameterwhere vary == True."""
        for band in self.bands:
            for param in band.get_parameters():
                if param.vary:
                    self.floating_params.append(param)

    def _build_namespace(self):
        """
        Builds a dictionary of all current parameter values so that constrained parameters can evaluate.
        """
        namespace = {}
        namespace['__temperatures__'] = self.dataset.toml_temperatures

        for band in self.bands:
            for param in band.get_parameters():
                namespace[param.name] = param.value
            
        return namespace

    def _get_initial_guesses(self):
        """Grabs initial values for fitting"""
        x0 = []
        for param in self.floating_params:
            val = param.value
            if isinstance(val, np.ndarray):
                x0.extend(val.flatten())
            else:
                x0.append(val)
        return np.array(x0)

    def residual(self, scipy_x):
        """
        Core function for minimization, calculate residualss
        """
        # Map the flat array back into parameters
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
            simulated_matrix = band.evaluate(x_axis, namespace)
            total_simulation += simulated_matrix.flatten()
            
        # 4. Return the 1D residual
        return total_simulation - self.dataset.get_y_flat()

    def cost_function(self, scipy_x):
        """Converts residual array into a single scalar value for DE."""
        res_array = self.residual(scipy_x)
        return np.sum(res_array **2)

    def _get_bounds(self):
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

    def run(self):
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

def plot_results(dataset, bands, namespace):
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
    plt.show()

def save_results_to_toml(filename, dataset, bands):
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

def save_results_to_csv(dataset, bands, namespace, param_filename="output_parameters.csv", amp_filename="output_amplitudes.csv"):
    """
    Exports the optimized fit results into two separate CSV files.
    """
    
    # --- Table 1: Scalar Parameters ---
    with open(param_filename, mode='w', newline='') as f_param:
        writer = csv.writer(f_param)
        
        writer.writerow(["Band_Name", "Type", "Center", "Width", "Vib_Spacing", "Huang_Rhys"])
        
        for band in bands:            
            for param in band.get_parameters():
                if isinstance(param, ArrayParameter):
                    continue
                
                name = param.name.split('_', 1)[1]
                val_str = str(param.expr) if param.expr else f"{param.value:.4f}"
                
                if name == "center": center = val_str
                elif name == "width": width = val_str
                elif name == "vib_spacing": vib_spacing = val_str
                elif name == "huang_rhys": huang_rhys = val_str
                    
            writer.writerow([band.name, band.type, center, width, vib_spacing, huang_rhys])
            
            
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

def parse_args():
    """Parses command line arguments to find input"""
    parser = argparse.ArgumentParser(description="Run the global fitting routine with parameters from TOML input file.")

    parser.add_argument(
        "config_file",
        type = str,
        help = "Path to TOML configuration file"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config_file)
    # Data Pipeline: Load, align, and flatten the .txt file
    dataset = DataSet(
        filename=config['dataset']['filename'],
        toml_fields=config['dataset']['fields'],
        toml_temperatures=config['dataset']['temperatures']
    )

    expected_keys = [str(t) for t in dataset.toml_temperatures]

    
    
    
    bands = []
    
    for band_name, band_setup in config['bands'].items():
        band_type = band_setup.get('type')
        
        if band_type == 'Gaussian':
            bands.append(GaussianBand(band_name, band_setup, expected_keys))
        elif band_type == 'Vibronic':
            bands.append(VibronicBand(band_name, band_setup, expected_keys))
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
        plot_results(dataset, bands, final_namespace)

if __name__ == "__main__":
    main()