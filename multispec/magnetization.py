import numpy as np
from typing import Any
from scipy.optimize import differential_evolution, least_squares, minimize, dual_annealing
from dataclasses import dataclass
from typing import NamedTuple
from .constants import mu_b, k_B
from .parameters import FitParameter

class PolarizationResult(NamedTuple):
    Mxy: float
    Myz: float
    Mxz: float
    perc_x: float
    perc_y: float
    perc_z: float

def build_sh_parameters(mol_config: dict, symmetry_mode: str) -> dict[str, FitParameter]:
    """Constructs FitParameter dictionary based on point-group symmetry."""
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
    return sh_params

def calculate_polarizations(Mxy: float, Myz: float, Mxz: float, eps: float = 1e-12) -> PolarizationResult:
    """Calculates directional transition dipole projections and polarization percentages."""
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

    return PolarizationResult(Mxy, Myz, Mxz, perc_x, perc_y, perc_z)

def generate_simulation_grids(engine: SpinHamiltonian, unique_temps: np.ndarray, max_field: float) -> tuple[np.ndarray, dict, np.ndarray, list]:
    """Computes smooth field and temperature basis sets for curve plotting."""
    smooth_fields = np.linspace(0.1, max_field * 1.05, 50)
    mag_basis = {
        t: [engine.get_mcd_components(b, t, n_theta=30, n_phi=30) for b in smooth_fields]
        for t in unique_temps
    }

    min_t, max_t = np.min(unique_temps), np.max(unique_temps)
    smooth_temps = np.logspace(np.log10(min_t * 0.8), np.log10(max_t * 5), 100)
    iso_basis = [engine.get_mcd_components(max_field, t, n_theta=30, n_phi=30) for t in smooth_temps]

    return smooth_fields, mag_basis, smooth_temps, iso_basis

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
        
        # ZFS
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
        
        # Zeeman
        H_zeeman = self.mu_b * (self.gx * Bx * self.Sx + 
                                self.gy * By * self.Sy + 
                                self.gz * Bz * self.Sz)
                                
        H_total = self.H_zfs + H_zeeman
        
        # Diagonalize
        energies, wavefunctions = np.linalg.eigh(H_total)
        
        # Spin Expectation Values for each state
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
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
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
        
        for p in self.floating_sh_params:
            guess.append(p.value)
            lb.append(p.min_val)
            ub.append(p.max_val)
            
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
        
        x0=guess
        bounds = list(zip(lb, ub))
                
        if not method:
            method = 'least_squares'

        supported_methods = ['least_squares', 'differential_evolution', 'dual_annealing', 'L-BFGS-B', 'Nelder-Mead']

        if method not in supported_methods:
            raise ValueError(f"Invalid optimization method: '{method}'. Supported methods are: {', '.join(supported_methods)} ")

        elif method == 'dual_annealing':
            result = dual_annealing(self.cost_function, bounds=bounds, x0=x0)
            
        elif method in ['L-BFGS-B', 'Nelder-Mead']:
            result = minimize(self.cost_function, x0, method=method, bounds=bounds)

        elif method == 'differential_evolution':
            result = differential_evolution(
                self.cost_function, bounds=bounds, x0=x0, polish=True, workers=-1, disp=True
            )
        else:
            result = least_squares(
                self.residual, x0=guess, bounds=(lb, ub), method='trf', xtol=1e-4, ftol=1e-4
            )
            
        return result

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