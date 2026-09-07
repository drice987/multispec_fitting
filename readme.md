# Multispec Fitting Software

## Overview
Analyzing spectroscopic series across changing physical conditions (temperature, magnetic field, pH, time) often suffers from over-parameterization when fitting spectra independently. Multispec performs simultaneous global fitting across arbitrary 1D or 2D condition series. Peak positions, widths, and vibronic progressions can be coupled or constrained across conditions while allowing intensities and thermal parameters to vary according to physical models.

For VTVH MCD data, the software includes a coupled Spin Hamiltonian solver based on the orientation-averaging formalism to extract zero-field splitting ($D, E$), $g$-tensors, and effective transition dipole products ($M_{xy}, M_{yz}, M_{xz}$) and percentage polarizations ($`\%\,x, \%\,y, \%\,z`$).

## Key Features

* **Global Spectral Deconvolution:** Simultaneously fits multiple spectra, enforcing identical band positions, shared widths, or user-defined mathematical relationships across all conditions.
* **Flexible Lineshapes:**
  * **Gaussian:** Standard symmetric absorption/MCD lineshape.
  * **Pseudo-Voigt:** Linear combination of Gaussian and Lorentzian profiles with mixing parameter ($\eta$).
  * **Vibronic Progression:** Franck–Condon progressions modeled with Huang–Rhys factors ($S$) and Poisson distributions.
  * **Thermal Broadening:** Temperature-dependent linewidth broadening modeled via $w(T) = w_0 + A / \tanh\left(\frac{\hbar\omega}{2 k_B T}\right)$.
* **Mathematical Parameter Tying:** Define relational constraints directly in the input TOML file using mathematical expressions via `asteval` (e.g., opposite-signed pseudo-A pairs like `amplitudes = { expr = "-Band1.amplitudes" }`).
* **Condition Filtering:** Rapidly fit a subset of magnetic fields or temperatures on the fly (`--fields`, `--temps`) without having to modify initial guesses in your configuration file.
* **Integrated Spin Hamiltonian Solver:**
  * Supports isotropic ($g$), axial ($g_x = g_y, g_z$), and rhombic ($g_x, g_y, g_z$) spin symmetries.
  * Simulates orientation-averaged VTVH magnetization saturation curves.
  * Decomposes band polarizations into effective transition dipole products ($M_{xy}, M_{yz}, M_{xz}$) and percentage polarizations ($`\%\,x, \%\,y, \%\,z`$).
* **Multiple Numerical Optimizers:** Gradient-based Levenberg–Marquardt / TRF (`least_squares`) with physical boundary enforcement, alongside global stochastic methods (`differential_evolution`, `dual_annealing`, `Nelder-Mead`, `L-BFGS-B`).
* **Non-VTVH Compatibility:** Supports arbitrary multi-column datasets (e.g., pH titrations, kinetic runs, electrochemical series).

## Example Single Spectrum from VTVH fit
<p align="center">
  <img src="./example/example_image.png" alt="Fitted 2.5K 10T data from the VTVH fit">
</p>
*Fitted example using VTVH MCD data [1]*

## Requirements & Installation

Multispec requires **Python 3.11+** (for native `tomllib` support).

```bash
# Clone repository
git clone git@github.com:drice987/multispec_fitting.git
cd multispec_fitting

# Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### Dependencies
* `numpy`
* `scipy`
* `matplotlib`
* `asteval`

## Quick Start
Execution is driven from the command line using a single `.toml` file that specifies data paths, fitting algorithms, initial guesses, and band models.

### 1. Run Full Global Fitting & Spin Hamiltonian Analysis

```bash
python -m multispec.main example/N2Q_SH_with_global-fit.toml
```

### 2. Fit a Specific Subset of Conditions

Quickly test initial guesses or refine band positions against high-signal spectra without altering the TOML file:

```bash
# Fit only the 10.0 T and 7.0 T spectra across all temperatures
python -m multispec.main example/N2Q_SH_with_global-fit.toml --fields 10.0 7.0

# Fit only the 2.5 K isotherm across all fields
python -m multispec.main example/N2Q_SH_with_global-fit.toml --temps 2.5

# Fit a single condition (e.g., 2.5 K at 10.0 T)
python -m multispec.main example/N2Q_SH_with_global-fit.toml --temps 2.5 --fields 10.0
```

### 3. Run Standalone Spin Hamiltonian Solver

Skip the spectral deconvolution step and fit the Spin Hamiltonian directly using the amplitude arrays already saved in the TOML configuration:

```bash
python -m multispec.main example/N2Q_SH_with_global-fit.toml --sh-only
```

## Configuration Overview (`.toml`)

Configurations are divided into four main blocks: `[dataset]`, `[fit_settings]`, `[sh]`, and `[bands.<Name>]`.

```toml
[dataset]
filename = "./example/N2Q_MCD_VTVH.txt"
temperatures = [2.5, 5, 10, 20, 40]
fields = [10, 7, 5, 3]
temp_tolerance = 1.0  # Search tolerance matching headers like '2.5K_10T'

[fit_settings]
method = "least_squares"  # 'least_squares', 'differential_evolution', etc.

[sh]
symmetry = "axial"       # 'isotropic', 'axial', or 'rhombic'
spin = 1.0
method = "least_squares"
plot_reduced_mag = true   # Plots against mu_B * B / (2 * k_B * T)
gx = { value = 2.00, vary = false }
gz = { value = 2.08, vary = false }
D  = { value = 24.3, vary = false }
E  = { value = 0.0,  vary = false }

# Floating Gaussian Band
[bands.Band4]
type = "Gaussian"
center = { value = 11184.8, vary = true }
width  = { value = 1509.5, vary = true }

[bands.Band4.amplitudes]
min = 0.0
"2.5" = [619.95, 498.17, 381.82, 241.19]
"5"   = [591.94, 478.47, 368.13, 232.54]
"10"  = [453.93, 367.27, 281.78, 177.09]
"20"  = [151.64, 118.82, 88.76, 54.80]
"40"  = [10.0, 7.0, 5.0, 3.0]

# Vibronic Band with Constraint Tying
[bands.Band0]
type = "Vibronic"
center = { value = 10098.1, vary = true }
width  = { value = 355.6, vary = true }
n_levels = 7
vib_spacing = { value = 545.5, vary = true }
huang_rhys = { expr = "Band1.huang_rhys" }       # Coupled to Band1
amplitudes = { expr = "-Band1.amplitudes" }      # Opposite-sign pseudo-A partner
```

## Output Files

Each successful run produces organized parameter tables, simulated coordinates, and plots:

### Spectral Deconvolution Outputs (Root Directory)
| File | Description |
| :--- | :--- |
| `fit_results.png` | Multi-panel grid visual showing raw data, individual band deconvolutions, and total fits. Adapts layout dynamically based on active conditions. |
| `fitted_results.toml` | Updated configuration file populated with optimized parameters, ready to be reused for subsequent fits. |
| `output_parameters.csv` | Table of optimized scalar parameters (centers, widths, vibronic spacings, Huang–Rhys factors). |
| `output_amplitudes.csv` | Optimized band amplitudes across all experimental conditions. |
| `output_spectra.csv` | Full spectral profile decomposition (Energy, Experimental, Total Fit, and individual Band columns). |

### Spin Hamiltonian Outputs (`sh_outputs/`)
| File | Description |
| :--- | :--- |
| `magnetization_fits.png` | Multi-panel VTVH saturation curves per band with experimental data vs. Spin Hamiltonian simulations. |
| `isofield_{field}T.png` | Temperature-dependent intensity summary across all bands at a selected magnetic field. |
| `sh_fit_parameters.csv` | Fitted spin parameters ($D, E$, $g$-tensor), effective transition dipole products ($M_{xy}, M_{yz}, M_{xz}$), and percentage polarizations ($`\%\,x, \%\,y, \%\,z`$). |
| `sh_simulated_curves.csv` | Long-format coordinates of all experimental and simulated VTVH curves. |

## Interactive / Jupyter Notebook Usage

`multispec` can also be imported directly for programmatic use in Jupyter Notebooks or custom Python scripts:

```python
from multispec import DataSet, GlobalFitter, SpinHamiltonian, GaussianBand

# Load dataset with condition filter
dataset = DataSet(dataset_cfg, select_fields=[7.0, 10.0])

# Initialize Spin Hamiltonian engine directly
engine = SpinHamiltonian(S=1.0, D=24.3, E=0.0, g=[2.0, 2.0, 2.08])
energies, exp_S = engine.solve(B_vector=[0.0, 0.0, 7.0])
```

## References
1. Derek B. Rice et al., *Sci. Adv.* **10**, eado1603(2024). DOI:10.1126/sciadv.ado1603