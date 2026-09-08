# multispec/bands.py
import math
from typing import Any

import numpy as np

from .constants import k_B
from .parameters import ArrayParameter, FitParameter


class SpectralBand:
    """
    Default band type used to read in parameters present for all band types (center, width, amplitudes)
    """

    def __init__(
        self,
        name: str,
        config_dict: dict,
        expected_keys: list[str] | None = None,
        field_indices: list[int] | None = None,
    ) -> None:
        self.name = name
        self.type = config_dict.get("type")

        self.center = FitParameter(f"{name}_center", config_dict["center"])
        self.width = FitParameter(f"{name}_width", config_dict["width"])
        self.amplitudes = ArrayParameter(
            f"{name}_amplitudes",
            config_dict["amplitudes"],
            expected_keys=expected_keys,
            field_indices=field_indices,
        )
        self.has_temp = "temp_broadening" in config_dict
        self.has_vib = "vib_energy" in config_dict
        self.temp_broadening = FitParameter(
            f"{name}_temp_broadening",
            config_dict.get("temp_broadening", {"value": 0.0, "vary": False}),
        )
        self.vib_energy = FitParameter(
            f"{name}_vib_energy", config_dict.get("vib_energy", {"value": 200.0, "vary": False})
        )

    def get_parameters(self) -> list:
        params = [self.center, self.width, self.amplitudes]
        if self.has_temp:
            params.append(self.temp_broadening)
        if self.has_vib:
            params.append(self.vib_energy)
        return params

    def evaluate(
        self, x: np.ndarray, namespace: dict | None = None, evaluator: Any = None
    ) -> np.ndarray:
        raise NotImplementedError("Must be implemented by a subclass.")


class GaussianBand(SpectralBand):
    """Class used for specific Gaussian band, used to calculate the lineshape.
    Gets center, width, and amplitudes from SpectralBand.
    """

    def __init__(
        self,
        name: str,
        config_dict: dict,
        expected_keys: list[str] | None = None,
        field_indices: list[int] | None = None,
    ) -> None:
        super().__init__(name, config_dict, expected_keys, field_indices)

    def get_parameters(self) -> list:
        return super().get_parameters()

    def evaluate(
        self, x: np.ndarray, namespace: dict | None = None, evaluator: Any = None
    ) -> np.ndarray:
        center = self.center.get_value(namespace, evaluator)
        w0 = self.width.get_value(namespace, evaluator)
        A = self.temp_broadening.get_value(namespace, evaluator)
        E_vib = self.vib_energy.get_value(namespace, evaluator)
        amps = self.amplitudes.get_value(namespace, evaluator)
        temps = np.array(namespace["__real_temperatures__"])

        safe_E_vib = max(abs(E_vib), 1.0)
        thermal_term = A / np.tanh(safe_E_vib / (2 * k_B * (temps + 1e-5)))
        widths = np.maximum(w0 + thermal_term, 1e-5)

        exponent = -4 * np.log(2) * ((x[np.newaxis, :] - center) / widths[:, np.newaxis]) ** 2
        base_shape = np.exp(exponent)

        return amps[:, np.newaxis] * base_shape


class PseudoVoigtBand(SpectralBand):
    """
    Calculated Pseudo-Voigt lineshape, a linear combination of a Gaussian and a Lorentzian profile.
    Gets center, width, and amplitude from SpectralBand.
    """

    def __init__(
        self,
        name: str,
        config_dict: dict,
        expected_keys: list[str] | None = None,
        field_indices: list[int] | None = None,
    ) -> None:
        super().__init__(name, config_dict, expected_keys, field_indices)
        self.lorentz_frac = FitParameter(
            f"{name}_lorentz_frac",
            config_dict.get("lorentz_frac", {"value": 0.5, "min": 0.0, "max": 1.0}),
        )

    def get_parameters(self) -> list:
        base_params = super().get_parameters()
        return base_params + [self.lorentz_frac]

    def evaluate(
        self, x: np.ndarray, namespace: dict | None = None, evaluator: Any = None
    ) -> np.ndarray:
        center = self.center.get_value(namespace, evaluator)
        w0 = self.width.get_value(namespace, evaluator)
        eta = self.lorentz_frac.get_value(namespace, evaluator)
        A = self.temp_broadening.get_value(namespace, evaluator)
        E_vib = self.vib_energy.get_value(namespace, evaluator)
        amps = self.amplitudes.get_value(namespace, evaluator)
        temps = np.array(namespace["__real_temperatures__"])
        safe_E_vib = max(abs(E_vib), 1.0)
        thermal_term = A / np.tanh(safe_E_vib / (2 * k_B * (temps + 1e-5)))
        widths = np.maximum(w0 + thermal_term, 1e-5)[:, np.newaxis]

        dx = x[np.newaxis, :] - center

        gaussian = np.exp(-4 * np.log(2) * (dx / widths) ** 2)
        lorentzian = 1.0 / (1.0 + 4 * (dx / widths) ** 2)

        base_shape = eta * lorentzian + (1 - eta) * gaussian

        return amps[:, np.newaxis] * base_shape


class VibronicBand(SpectralBand):
    """
    Generates vibronic progression using the Huang-Rhys factor and Poisson distribution
    Uses standard Gaussian shape for each band.
    Gets center, width, and amplitudes from SpectralBand.
    """

    def __init__(
        self,
        name: str,
        config_dict: dict,
        expected_keys: list[str] | None = None,
        field_indices: list[int] | None = None,
    ) -> None:
        super().__init__(name, config_dict, expected_keys, field_indices)

        # Add the vibronic-specific parameters
        self.vib_spacing = FitParameter(f"{name}_vib_spacing", config_dict["vib_spacing"])
        self.huang_rhys = FitParameter(f"{name}_huang_rhys", config_dict["huang_rhys"])
        self.n_levels = config_dict.get("n_levels", 10)

    def get_parameters(self) -> list:
        """Override to include the new parameters alongside the base ones."""
        base_params = super().get_parameters()
        return base_params + [self.vib_spacing, self.huang_rhys]

    def evaluate(
        self, x: np.ndarray, namespace: dict | None = None, evaluator: Any = None
    ) -> np.ndarray:
        """Calculates the full sum of the vibronic progression."""
        c = self.center.get_value(namespace, evaluator)
        w = self.width.get_value(namespace, evaluator)
        amps = self.amplitudes.get_value(namespace, evaluator)
        spacing = self.vib_spacing.get_value(namespace, evaluator)
        s = self.huang_rhys.get_value(namespace, evaluator)

        A = self.temp_broadening.get_value(namespace, evaluator)
        E_vib = self.vib_energy.get_value(namespace, evaluator)

        temps = np.array(namespace["__real_temperatures__"])
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
