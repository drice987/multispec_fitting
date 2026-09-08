import numpy as np
from typing import Any, NamedTuple
from scipy.optimize import differential_evolution, least_squares, minimize, dual_annealing
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

    # Configure D
    d_cfg = mol_config.get('D', def_D)
    d_dict = d_cfg.copy() if isinstance(d_cfg, dict) else {'value': float(d_cfg)}
    
    d_min = d_dict.get('min', -float('inf'))
    d_max = d_dict.get('max', float('inf'))
    d_val = d_dict.get('value', 5.0)

    if d_dict.get('vary', True) and abs(d_val) < 1e-3:
        if d_min >= 0:
            val = max(0.1, d_min)
        elif d_max <= 0:
            val = min(-0.1, d_max)
        else:
            val = -0.1 if (d_val < 0 or np.signbit(d_val)) else 0.1
        d_dict['value'] = val

    sh_params['D'] = FitParameter("D", d_dict)
    d_val = sh_params['D'].value

    # Configure E and eta
    e_cfg = mol_config.get('E', def_E)
    e_dict = e_cfg.copy() if isinstance(e_cfg, dict) else {'value': float(e_cfg)}
    
    if symmetry_mode != "rhombic":
        e_dict['value'] = 0.0
        e_dict['vary'] = False

    e_vary = e_dict.get('vary', False) and (symmetry_mode == "rhombic")

    if e_vary:
        eta_init = abs(e_dict.get('value', 0.0) / d_val) if d_val != 0 else 0.0
        eta_init = min(max(eta_init, 0.0), 1/3.0)
        sh_params['eta'] = FitParameter("eta", {'value': eta_init, 'min': 0.0, 'max': 1/3.0, 'vary': True})
        sh_params['E'] = FitParameter("E", {'value': eta_init * abs(d_val), 'vary': False})
    else:
        sh_params['E'] = FitParameter("E", e_dict)

    return sh_params

def calculate_polarizations(Mxy: float, Myz: float, Mxz: float, eps: float = 1e-12) -> PolarizationResult:
    """Calculates directional transition dipole projections and polarization percentages."""
    Px = abs(Mxy * Mxz) / (abs(Myz) + eps)
    Py = abs(Mxy * Myz) / (abs(Mxz) + eps)
    Pz = abs(Myz * Mxz) / (abs(Mxy) + eps)

    total_P = Px + Py + Pz
    if total_P > 0:
        perc_x = (Px / total_P) * 100
        perc_y = (Py / total_P) * 100
        perc_z = (Pz / total_P) * 100
    else:
        perc_x = perc_y = perc_z = 0.0

    return PolarizationResult(Mxy, Myz, Mxz, perc_x, perc_y, perc_z)

def generate_simulation_grids(engine: "SpinHamiltonian", unique_temps: np.ndarray, max_field: float) -> tuple[np.ndarray, dict, np.ndarray, list]:
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
    """
    def __init__(self, S: float, D: float, E: float, g: float | list[float]) -> None:
        self.S = S
        self.Sx, self.Sy, self.Sz = get_spin_matrices(S)
        
        if isinstance(g, (int, float)):
            self.gx = self.gy = self.gz = float(g)
        else:
            self.gx, self.gy, self.gz = g
            
        self.mu_b = mu_b
        
        S_sq = S * (S + 1)
        identity = np.eye(int(2 * S + 1), dtype=complex)
        
        self.H_zfs = D * (self.Sz @ self.Sz - (S_sq / 3.0) * identity) + \
                     E * (self.Sx @ self.Sx - self.Sy @ self.Sy)

    def solve(self, B_vector: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Applies magnetic field, diagonalizes the matrix, and returns energies and spin expectation values."""
        Bx, By, Bz = B_vector
        H_zeeman = self.mu_b * (self.gx * Bx * self.Sx + 
                                self.gy * By * self.Sy + 
                                self.gz * Bz * self.Sz)
                                
        H_total = self.H_zfs + H_zeeman
        energies, wavefunctions = np.linalg.eigh(H_total)
        
        exp_Sx = np.diag(wavefunctions.conj().T @ self.Sx @ wavefunctions).real
        exp_Sy = np.diag(wavefunctions.conj().T @ self.Sy @ wavefunctions).real
        exp_Sz = np.diag(wavefunctions.conj().T @ self.Sz @ wavefunctions).real
        exp_S = np.vstack((exp_Sx, exp_Sy, exp_Sz)).T
        
        return energies, exp_S

    def get_mcd_components(self, B_mag: float, temp: float, n_theta: int = 30, n_phi: int = 30) -> tuple[float, float, float]:
        """Calculates orientation-averaged MCD basis components (xy, yz, zx) vectorized across spherical orientations."""
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        kT = max(k_B * temp, 1e-9)

        theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
        sin_t = np.sin(theta_grid).ravel()
        cos_t = np.cos(theta_grid).ravel()
        sin_p = np.sin(phi_grid).ravel()
        cos_p = np.cos(phi_grid).ravel()

        ux = sin_t * cos_p
        uy = sin_t * sin_p
        uz = cos_t
        weights = sin_t  

        H_zeeman = (self.mu_b * B_mag) * (
            self.gx * ux[:, None, None] * self.Sx
            + self.gy * uy[:, None, None] * self.Sy
            + self.gz * uz[:, None, None] * self.Sz
        )
        H_total = self.H_zfs[None, :, :] + H_zeeman

        energies, wavefunctions = np.linalg.eigh(H_total)

        exp_Sx = np.diagonal(wavefunctions.conj().swapaxes(-1, -2) @ self.Sx @ wavefunctions, axis1=-2, axis2=-1).real
        exp_Sy = np.diagonal(wavefunctions.conj().swapaxes(-1, -2) @ self.Sy @ wavefunctions, axis1=-2, axis2=-1).real
        exp_Sz = np.diagonal(wavefunctions.conj().swapaxes(-1, -2) @ self.Sz @ wavefunctions, axis1=-2, axis2=-1).real

        delta_E = energies - energies[:, [0]]
        exp_terms = np.exp(-delta_E / kT)
        populations = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

        sum_Sx = np.sum(populations * exp_Sx, axis=1)  
        sum_Sy = np.sum(populations * exp_Sy, axis=1)
        sum_Sz = np.sum(populations * exp_Sz, axis=1)

        comp_xy = uz * sum_Sz
        comp_yz = ux * sum_Sx
        comp_zx = uy * sum_Sy

        total_weight = np.sum(weights)
        ave_xy = float(np.sum(comp_xy * weights) / total_weight)
        ave_yz = float(np.sum(comp_yz * weights) / total_weight)
        ave_zx = float(np.sum(comp_zx * weights) / total_weight)

        return ave_xy, ave_yz, ave_zx

class MagnetizationFitter:
    """Fits experimental VTVH amplitudes to extract Zero-Field Splitting, g-values, and transition dipoles."""
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
        
        if 'eta' in self.sh_params and self.sh_params['eta'].vary:
            eta = np.clip(self.sh_params['eta'].value, 0.0, 1/3.0)
            E = eta * abs(D)
            self.sh_params['E'].value = E
        else:
            E = self.sh_params['E'].value
            
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
        finite_only = method in ('differential_evolution', 'dual_annealing')
        
        for p in self.floating_sh_params:
            if p.name == 'D' and abs(p.value) < 1e-4:
                if p.min_val >= 0:
                    val = max(0.1, p.min_val)
                elif p.max_val <= 0:
                    val = min(-0.1, p.max_val)
                else:
                    val = -0.1 if (p.value < 0 or np.signbit(p.value)) else 0.1
                p.set_value(val)

            guess.append(p.value)
            p_min, p_max = p.min_val, p.max_val
            
            if finite_only:
                if 'g' in p.name:
                    c_min = 1.0 if np.isinf(p_min) else p_min
                    c_max = 3.0 if np.isinf(p_max) else p_max
                elif p.name == 'D':
                    margin = max(abs(p.value) * 5.0, 50.0)
                    c_min = p.value - margin if np.isinf(p_min) else p_min
                    c_max = p.value + margin if np.isinf(p_max) else p_max
                else:
                    c_min = p_min
                    c_max = p_max
                lb.append(c_min)
                ub.append(c_max)
            else:
                lb.append(p_min)
                ub.append(p_max)
            
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
        
        x0 = guess
        bounds = list(zip(lb, ub))
                
        if not method:
            method = 'least_squares'

        supported_methods = ['least_squares', 'differential_evolution', 'dual_annealing', 'L-BFGS-B', 'Nelder-Mead']

        if method not in supported_methods:
            raise ValueError(f"Invalid optimization method: '{method}'. Supported methods are: {', '.join(supported_methods)}")

        if method == 'dual_annealing':
            result = dual_annealing(self.cost_function, bounds=bounds, x0=x0)
        elif method in ['L-BFGS-B', 'Nelder-Mead']:
            result = minimize(self.cost_function, x0, method=method, bounds=bounds)
        elif method == 'differential_evolution':
            result = differential_evolution(
                self.cost_function, bounds=bounds, x0=x0, polish=True, workers=-1, updating='deferred', disp=True
            )
        else:
            result = least_squares(
                self.residual, x0=guess, bounds=(lb, ub), method='trf', xtol=1e-4, ftol=1e-4
            )
            
        self.residual(result.x)
        return result

def get_spin_matrices(S: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates the Sx, Sy, and Sz spin operator matrices for a given spin S."""
    m = np.arange(S, -S - 1, -1, dtype=float)
    m_raising = m[1:]
    s_plus_diag = np.sqrt(S * (S + 1) - m_raising * (m_raising + 1))
    
    Sp = np.diag(s_plus_diag, k=1).astype(complex)
    Sm = Sp.conj().T
    
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)
    Sz = np.diag(m).astype(complex)
    
    return Sx, Sy, Sz