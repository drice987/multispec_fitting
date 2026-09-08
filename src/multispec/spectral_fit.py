from typing import Any

import numpy as np
from asteval import Interpreter
from scipy.optimize import differential_evolution, dual_annealing, least_squares, minimize

from .bands import SpectralBand
from .data import DataSet


class GlobalFitter:
    def __init__(
        self, dataset: DataSet, bands: list[SpectralBand], method: str = "least_squares"
    ) -> None:
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
        namespace["__temperatures__"] = self.dataset.toml_temperatures
        namespace["__real_temperatures__"] = self.dataset.real_temperatures
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
                param.set_value(scipy_x[idx : idx + size])
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
        return np.sum(res_array**2)

    def _get_bounds(self, finite_only: bool = False) -> list[tuple[float, float]]:
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
                    if finite_only:
                        margin = max(abs(val) * 3.0, 50.0)
                        c_min = val - margin if np.isinf(p_min) else p_min
                        c_max = val + margin if np.isinf(p_max) else p_max
                    else:
                        c_min = p_min
                        c_max = p_max
                    bounds.append((c_min, c_max))
            else:
                # Scalars
                val = param.value

                if "center" in param.name.lower():
                    c_min = safe_center_min if np.isinf(p_min) else p_min
                    c_max = safe_center_max if np.isinf(p_max) else p_max
                elif finite_only:
                    margin = max(abs(val) * 5.0, 100.0)
                    c_min = val - margin if np.isinf(p_min) else p_min
                    c_max = val + margin if np.isinf(p_max) else p_max
                else:
                    c_min = p_min
                    c_max = p_max

                bounds.append((c_min, c_max))

        return bounds

    def run(self) -> Any:
        """Executes the fit using least squares or differential evolution."""
        x0 = self._get_initial_guesses()
        if self.method in ("differential_evolution", "dual_annealing"):
            bounds = self._get_bounds(finite_only=True)
        else:
            bounds = self._get_bounds(finite_only=False)

        supported_methods = [
            "least_squares",
            "differential_evolution",
            "dual_annealing",
            "L-BFGS-B",
            "Nelder-Mead",
        ]

        if self.method not in supported_methods:
            raise ValueError(
                f"Invalid optimization method: '{self.method}'. Supported methods are: {', '.join(supported_methods)} "
            )

        if self.method == "differential_evolution":
            result = differential_evolution(
                self.cost_function,
                bounds=bounds,
                x0=x0,
                polish=True,
                workers=-1,
                updating="deferred",
                disp=True,
            )
        elif self.method == "dual_annealing":
            result = dual_annealing(self.cost_function, bounds=bounds, x0=x0)

        elif self.method in ["L-BFGS-B", "Nelder-Mead"]:
            result = minimize(self.cost_function, x0, method=self.method, bounds=bounds)

        else:  # least_squares
            lb = [b[0] for b in bounds]
            ub = [b[1] for b in bounds]
            result = least_squares(
                self.residual, x0=x0, bounds=(lb, ub), method="trf", ftol=1e-6, xtol=1e-6
            )

        self.residual(result.x)
        return result

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("aeval", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        from asteval import Interpreter

        self.aeval = Interpreter()
