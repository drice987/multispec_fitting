import numpy as np
import pytest

from multispec.constants import mu_b
from multispec.magnetization import (
    MagnetizationFitter,
    SpinHamiltonian,
    build_sh_parameters,
    calculate_polarizations,
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


def test_calculate_polarizations_anisotropic_and_zero():
    res = calculate_polarizations(Mxy=2.0, Myz=6.0, Mxz=3.0)
    expected = np.array([1.0, 4.0, 9.0]) / 14.0 * 100.0
    np.testing.assert_allclose([res.perc_x, res.perc_y, res.perc_z], expected, rtol=1e-9)

    zero_res = calculate_polarizations(0.0, 0.0, 0.0)
    assert (zero_res.perc_x, zero_res.perc_y, zero_res.perc_z) == (0.0, 0.0, 0.0)


def test_build_sh_parameters_symmetries_and_eta_clamping():
    iso = build_sh_parameters({"D": 4.0, "E": 1.0}, "isotropic")
    assert set(iso.keys()) == {"g", "D", "E"}
    assert iso["D"].value == 4.0
    assert iso["E"].value == 0.0
    assert iso["E"].vary is False

    axial = build_sh_parameters({"E": {"value": 1.0, "vary": True}}, "axial")
    assert set(axial.keys()) == {"gx", "gz", "D", "E"}
    assert axial["E"].value == 0.0
    assert axial["E"].vary is False

    rhombic = build_sh_parameters(
        {"D": {"value": 3.0, "vary": True}, "E": {"value": 2.0, "vary": True}},
        "rhombic",
    )
    assert set(rhombic.keys()) == {"gx", "gy", "gz", "D", "eta", "E"}
    assert np.isclose(rhombic["eta"].value, 1.0 / 3.0)
    assert np.isclose(rhombic["E"].value, 1.0)
    assert rhombic["eta"].vary is True
    assert rhombic["E"].vary is False


@pytest.mark.parametrize(
    "d_cfg, expected_d",
    [
        ({"value": 0.0, "vary": True, "min": 0.0}, 0.1),
        ({"value": 0.0, "vary": True, "max": 0.0}, -0.1),
        ({"value": -0.0005, "vary": True}, -0.1),
        ({"value": 0.0, "vary": False}, 0.0),
    ],
)
def test_build_sh_parameters_zero_d_nudging(d_cfg, expected_d):
    params = build_sh_parameters({"D": d_cfg}, "isotropic")
    assert params["D"].value == expected_d


def test_magnetization_fitter_normalization_and_zero_guard():
    sh_params = build_sh_parameters({}, "isotropic")
    fitter = MagnetizationFitter(
        S=0.5,
        temps=[2.0, 5.0],
        fields=[1.0, 7.0],
        exp_norm_dict={"Band1": [-2.0, 4.0], "DeadBand": [0.0, 0.0]},
        sh_params=sh_params,
    )

    assert fitter.scale_factors["Band1"] == 4.0
    assert fitter.scale_factors["DeadBand"] == 1.0
    np.testing.assert_allclose(fitter.exp_norm_dict["Band1"], [-0.5, 1.0])


def test_magnetization_fitter_rhombic_residual_updates_e_from_eta():
    temps = [2.0, 5.0]
    fields = [1.0, 7.0]
    sh_params = build_sh_parameters(
        {
            "D": {"value": 6.0, "vary": True},
            "E": {"value": 0.6, "vary": True},
        },
        symmetry_mode="rhombic",
    )
    fitter = MagnetizationFitter(
        S=1.0,
        temps=temps,
        fields=fields,
        exp_norm_dict={"Band1": [1.0, 2.0]},
        sh_params=sh_params,
        symmetry_mode="rhombic",
    )

    assert fitter.num_floating_sh == 2
    res = fitter.residual([6.0, 0.25, 0.5, 0.5, 0.5])
    assert res.shape == (2,)
    assert np.isclose(fitter.sh_params["E"].value, 1.5)


def test_magnetization_fitter_isotropic_recovery_and_method_validation():
    temps = [2.0, 2.0, 5.0, 5.0]
    fields = [1.0, 7.0, 1.0, 7.0]

    engine = SpinHamiltonian(S=0.5, D=0.0, E=0.0, g=2.0)
    synth_amps = [
        3.0 * sum(engine.get_mcd_components(b, t, n_theta=15, n_phi=15))
        for t, b in zip(temps, fields)
    ]

    sh_params = build_sh_parameters(
        {"g": {"value": 2.0, "vary": False}, "D": {"value": 0.0, "vary": False}},
        symmetry_mode="isotropic",
    )
    fitter = MagnetizationFitter(
        S=0.5,
        temps=temps,
        fields=fields,
        exp_norm_dict={"Band1": np.array(synth_amps)},
        sh_params=sh_params,
        symmetry_mode="isotropic",
    )

    result = fitter.run_fit(method="least_squares")
    assert result.success
    assert fitter.cost_function(result.x) < 1e-6
    assert np.isclose(result.x[0] * fitter.scale_factors["Band1"], 3.0, rtol=1e-3)

    with pytest.raises(ValueError, match="Invalid optimization method"):
        fitter.run_fit(method="invalid_solver")
