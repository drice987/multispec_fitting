import pytest

from multispec.parameters import FitParameter


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