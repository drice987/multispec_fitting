import numpy as np
import pytest

from multispec.parameters import ArrayParameter, FitParameter


def test_fit_parameter_defaults():
    p = FitParameter("center", {})
    assert p.name == "center"
    assert p.value == 0.0
    assert p.min_val == -float("inf")
    assert p.max_val == float("inf")
    assert p.vary is True
    assert p.expr is None


@pytest.mark.parametrize(
    "new_val, expected",
    [
        (15200.0, 15200.0),
        (17500.0, 16000.0),
        (12000.0, 14000.0),
    ],
)
def test_fit_parameter_clamping(new_val, expected):
    p = FitParameter("center", {"value": 15000.0, "min": 14000.0, "max": 16000.0})
    p.set_value(new_val)
    assert p.value == expected


def test_fit_parameter_expression_locks_vary():
    p = FitParameter("b2_center", {"expr": "b1.center + 500", "vary": True})
    assert p.vary is False
    assert "expr='b1.center + 500'" in repr(p)
    assert p.get_value(namespace={"b1_center": 10000.0}) == 10500.0


def test_fit_parameter_missing_namespace_raises():
    p = FitParameter("bad", {"expr": "b1.center * 2"})
    with pytest.raises(RuntimeError, match="Failed to evaluate expression"):
        p.get_value(namespace=None)


def test_array_parameter_key_matching_and_field_slicing():
    raw_cfg = {
        "2": [1.0, 2.0, 3.0],
        "5.00": [0.5, 1.0, 1.5],
        "10.0004": [0.1, 0.2, 0.3],
    }
    ap = ArrayParameter(
        "b1_amps",
        raw_cfg,
        expected_keys=["2.0", "5.0", "10.0"],
        field_indices=[0, 2],
    )

    assert ap.vary is True
    assert ap.row_keys == ["2", "5.00", "10.0004"]
    assert ap.row_sizes == [2, 2, 2]
    np.testing.assert_allclose(ap.value, [1.0, 3.0, 0.5, 1.5, 0.1, 0.3])


def test_array_parameter_unfiltered_keys_and_scalar_coercion():
    raw_cfg = {
        "col_a": 2.5,
        "col_b": [1.0, 4.0],
        "min": 0.0,
        "max": 3.0,
    }
    ap = ArrayParameter("b1_amps", raw_cfg)

    assert ap.row_keys == ["col_a", "col_b"]
    assert ap.row_sizes == [1, 2]
    np.testing.assert_allclose(ap.value, [2.5, 1.0, 4.0])

    ap.set_value(np.array([-1.0, 1.5, 5.0]))
    np.testing.assert_allclose(ap.value, [0.0, 1.5, 3.0])


def test_array_parameter_missing_key_raises():
    with pytest.raises(KeyError, match="Missing amplitude array for temperature '20.0'"):
        ArrayParameter("b1_amps", {"2.0": [1.0, 2.0]}, expected_keys=["2.0", "20.0"])


def test_array_parameter_empty_raises():
    with pytest.raises(ValueError, match="No amplitude data was loaded"):
        ArrayParameter("empty_amps", {"min": 0.0, "max": 1.0})


def test_array_parameter_expression_mode():
    ap = ArrayParameter("b2_amps", {"expr": "b1.amplitudes * 0.5"})
    assert ap.vary is False
    assert ap.value is None

    ns = {"b1_amplitudes": np.array([2.0, 4.0, 6.0])}
    np.testing.assert_allclose(ap.get_value(ns), [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Cannot manually set"):
        ap.set_value(np.array([1.0, 1.0, 1.0]))
