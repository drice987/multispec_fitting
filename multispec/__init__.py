__version__ = "2.0.0"

from .data import DataSet
from .spectral_fit import GlobalFitter
from .magnetization import SpinHamiltonian, MagnetizationFitter
from .bands import GaussianBand, VibronicBand, PseudoVoigtBand

__all__ = [
    "DataSet",
    "GlobalFitter",
    "SpinHamiltonian",
    "MagnetizationFitter",
    "GaussianBand",
    "VibronicBand",
    "PseudoVoigtBand",
]
