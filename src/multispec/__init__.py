from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multispec")
except PackageNotFoundError:
    __version__ = "unknown"

from .bands import GaussianBand, PseudoVoigtBand, VibronicBand
from .data import DataSet
from .magnetization import MagnetizationFitter, SpinHamiltonian
from .spectral_fit import GlobalFitter

__all__ = [
    "DataSet",
    "GlobalFitter",
    "SpinHamiltonian",
    "MagnetizationFitter",
    "GaussianBand",
    "VibronicBand",
    "PseudoVoigtBand",
]
