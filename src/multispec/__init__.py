from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multispec")
except PackageNotFoundError:
    __version__ = "unknown"

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
