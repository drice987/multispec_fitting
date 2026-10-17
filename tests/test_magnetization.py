import numpy as np
import pytest

from multispec.constants import mu_b
from multispec.magnetization import (
    SpinHamiltonian,
    generate_simulation_grids,
    get_spin_matrices,
)


@pytest.mark.parametrize("S", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_spin_matrices_algebra_and_hermiticity(S):
    Sx, Sy, Sz = get_spin_matrices(S)
    dim = int(2 * S + 1)

    np.testing.assert_allclose(Sx, Sx.conj().T, atol=1e-12)
    np.testing.assert_allclose(Sy, Sy.conj().T, atol=1e-12)
    np.testing.assert_allclose(Sz, Sz.conj().T, atol=1e-12)

    np.testing.assert_allclose(Sx @ Sy - Sy @ Sx, 1j * Sz, atol=1e-12)
    np.testing.assert_allclose(Sy @ Sz - Sz @ Sy, 1j * Sx, atol=1e-12)
    np.testing.assert_allclose(Sz @ Sx - Sx @ Sz, 1j * Sy, atol=1e-12)

    s_sq = Sx @ Sx + Sy @ Sy + Sz @ Sz
    np.testing.assert_allclose(s_sq, S * (S + 1) * np.eye(dim), atol=1e-12)


def test_spin_hamiltonian_rhombic_zfs_and_spin_quenching():
    D, E = 6.0, 1.5
    sh = SpinHamiltonian(S=1.0, D=D, E=E, g=2.0)
    energies, exp_S = sh.solve([0.0, 0.0, 0.0])

    expected_energies = [-2.0 * D / 3.0, D / 3.0 - E, D / 3.0 + E]
    np.testing.assert_allclose(energies, expected_energies, atol=1e-12)
    np.testing.assert_allclose(exp_S, np.zeros((3, 3)), atol=1e-12)


def test_spin_hamiltonian_anisotropic_zeeman_splitting():
    gx, gy, gz = 2.0, 2.2, 2.5
    Bz = 4.0
    sh = SpinHamiltonian(S=0.5, D=0.0, E=0.0, g=[gx, gy, gz])
    energies, exp_S = sh.solve([0.0, 0.0, Bz])

    half_split = 0.5 * gz * mu_b * Bz
    np.testing.assert_allclose(energies, [-half_split, half_split], atol=1e-12)
    np.testing.assert_allclose(exp_S[:, 2], [-0.5, 0.5], atol=1e-12)


def test_mcd_components_symmetry_and_zero_field():
    sh = SpinHamiltonian(S=1.0, D=4.0, E=0.0, g=[2.0, 2.0, 2.2])

    zero_comps = sh.get_mcd_components(B_mag=0.0, temp=2.0, n_theta=10, n_phi=10)
    np.testing.assert_allclose(zero_comps, [0.0, 0.0, 0.0], atol=1e-12)

    xy, yz, zx = sh.get_mcd_components(B_mag=5.0, temp=2.0, n_theta=12, n_phi=12)
    assert xy < 0.0
    assert yz < 0.0
    assert np.isclose(yz, zx, rtol=1e-12)


def test_generate_simulation_grids_shapes():
    sh = SpinHamiltonian(S=0.5, D=0.0, E=0.0, g=2.0)
    smooth_b, mag_basis, smooth_t, iso_basis = generate_simulation_grids(
        sh, unique_temps=np.array([2.0]), max_field=7.0
    )

    assert len(smooth_b) == 50
    assert np.isclose(smooth_b[-1], 7.0 * 1.05)
    assert len(mag_basis[2.0]) == 50
    assert len(smooth_t) == 100
    assert len(iso_basis) == 100
