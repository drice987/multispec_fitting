# Multispec Fitting Software

## Overview
Python designed for simultaneous fitting of multiple spectra, particularly useful for analyzing datasets with shared spectral features across changing conditions, such as Variable Temperature, Variable Field (VTVH) Magnetic Circular Dichroism (MCD) spectroscopy.

## Key Features & Physics Implemented
* **Simultaneous Global Fitting:** Fit multiple spectra simultaneously, tying parameters (peak centers) across datasets while allowing amplitudes to float independently.
* **Gaussian Deconvolution:** Robust fitting of Gaussian lineshapes.
* **Amplitude and Sign Constraints:** Set specific boundaries on parameters, including opposite-signed amplitudes (useful for fitting MCD pseudo A-terms).
* **Vibronic Progressions:** Built-in support for modeling vibronic coupling using the Huang-Rhys factor and Poisson distributions, automatically calculating vibrational spacing.
* **Custom Parameter Constraints:** Define explicit mathematical relationships between different spectral bands directly in input file.

## Example Single Spectra from VTVH fit
<p align="center">
  <img src="example_image.png" alt="Fitted 2.5K 10T data from the VTVH fit">
</p>
*Fitted example using VTVH MCD data [1]*

## Installation

Clone the repository and install the required dependencies:

```bash
git clone git@github.com:drice987/multispec_fitting.git
cd multispec_fitting
pip install -r requirements.txt
```

## Quick Start
The script reads initial guesses, bounds, and constraints from structured text files. Please see the Jupyter Notebook located at **'examples/example_jupyter.ipynb'** for a walkthrough. The notebook demonstrates setting parameters, executing the global fit, and plotting the results using VTVH MCD data [1].

## References
Derek B. Rice et al., *Sci. Adv.* **10**, eado1603(2024). DOI:10.1126/sciadv.ado1603