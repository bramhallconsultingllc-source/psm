import math

import pytest

from psm.utils import round_up_to_increment
from psm.staffing_model import StaffingModel


# -----------------------------
# Existing tests (kept)
# -----------------------------
def test_rounding_up():
    assert round_up_to_increment(2.01, 0.25) == 2.25
    assert round_up_to_increment(2.25, 0.25) == 2.25
    assert round_up_to_increment(2.26, 0.25) == 2.50


def test_xrt_fixed():
    model = StaffingModel()
    out = model.calculate(45)
    assert out["xrt_day"] == 1.0


def test_interpolation_and_rounding():
    """
    This assumes your staffing_ratios.csv has the same values you referenced.
    If that CSV changes, update this expected value accordingly.
    """
    model = StaffingModel()
    out = model.calculate(45)

    # Expect interpolation between 42 and 48:
    # provider_day between 1.25 and 1.5 → 1.375 → round UP to 1.5
    assert out["provider_day"] == 1.5


def test_total_day_is_sum():
    model = StaffingModel()
    out = model.calculate(45)

    expected = out["provider_day"] + out["psr_day"] + out["ma_day"] + out["xrt_day"]
    assert out["total_day"] == expected


# -----------------------------
# New coverage for the rewrite
# -----------------------------
REQUIRED_CALC_KEYS = {
    "visits_day",
    "provider_day",
    "psr_day",
    "ma_day",
    "xrt_day",
    "total_day",
    "patients_per_provider_day",
}


def test_calculate_has_backward_compat_keys():
    """calculate() should return the legacy dict interface keys."""
    model = StaffingModel()
    out = model.calculate(25)
    assert set(out.keys()) == REQUIRED_CALC_KEYS


def test_calculate_daily_returns_dataclass_like_object():
    """calculate_daily() should return an object with expected attributes."""
    model = StaffingModel()
    d = model.calculate_daily(25)

    # Attributes present
    for attr in [
        "visits_day",
        "provider_day",
        "psr_day",
        "ma_day",
        "xrt_day",
        "total_day",
        "patients_per_provider_day",
    ]:
        assert hasattr(d, attr)

    # Values are finite
    assert math.isfinite(d.visits_day)
    assert math.isfinite(d.provider_day)
    assert math.isfinite(d.total_day)


def test_rounding_increment_applies_to_roles():
    """Provider/PSR/MA are rounded up to the model increment; XRT stays fixed at 1.0."""
    model = StaffingModel(rounding_increment=0.25)
    d = model.calculate_daily(25)

    def is_multiple(x, inc):
        return abs((x / inc) - round(x / inc)) < 1e-9

    assert is_multiple(d.provider_day, 0.25)
    assert is_multiple(d.psr_day, 0.25)
    assert is_multiple(d.ma_day, 0.25)
    assert d.xrt_day == 1.0


def test_monotonic_provider_day_sanity():
    """
    Sanity: as visits increase, provider_day should not decrease.
    With rounding it can stay flat, but should not go down.
    """
    model = StaffingModel()
    visits = [10, 15, 20, 25, 30, 35, 40]
    prov_days = [model.calculate_daily(v).provider_day for v in visits]
    for a, b in zip(prov_days, prov_days[1:]):
        assert b >= a - 1e-9


def test_get_role_mix_ratios_outputs():
    """get_role_mix_ratios() should return expected keys and sensible nonnegative values."""
    model = StaffingModel()
    ratios = model.get_role_mix_ratios(28)

    assert set(ratios.keys()) == {"psr_per_provider", "ma_per_provider", "xrt_per_provider"}
    for k, v in ratios.items():
        assert math.isfinite(v), f"{k} is not finite"
        assert v >= 0.0, f"{k} should be >= 0"


def test_calculate_fte_needed_backward_compat_smoke():
    """calculate_fte_needed should return expected keys and finite values."""
    model = StaffingModel()
    out = model.calculate_fte_needed(
        visits_per_day=28,
        hours_of_operation_per_week=84,
        fte_hours_per_week=36,
    )

    assert set(out.keys()) == {"provider_fte", "psr_fte", "ma_fte", "xrt_fte", "total_fte"}
    for k, v in out.items():
        assert math.isfinite(v), f"{k} is not finite"
        assert v >= 0.0, f"{k} should be >= 0"


def test_calculate_support_fte_from_provider_fte_scales_up():
    """
    Capacity-aware support staffing:
    increasing provider_fte should not reduce computed support_total_fte (for same visits/day).
    """
    model = StaffingModel()
    visits = 28
    hours_week = 84
    fte_hours = 36

    s1 = model.calculate_support_fte_from_provider_fte(
        visits_per_day=visits,
        provider_fte=2.0,
        hours_of_operation_per_week=hours_week,
        fte_hours_per_week=fte_hours,
    )
    s2 = model.calculate_support_fte_from_provider_fte(
        visits_per_day=visits,
        provider_fte=3.0,
        hours_of_operation_per_week=hours_week,
        fte_hours_per_week=fte_hours,
    )

    assert "support_total_fte" in s1 and "support_total_fte" in s2
    assert s1["support_total_fte"] >= 0.0
    assert s2["support_total_fte"] >= s1["support_total_fte"] - 1e-9


def test_calculate_support_fte_from_provider_fte_has_expected_fields():
    """Make sure the support FTE helper returns a stable, predictable schema."""
    model = StaffingModel()
    out = model.calculate_support_fte_from_provider_fte(
        visits_per_day=25,
        provider_fte=2.5,
        hours_of_operation_per_week=84,
        fte_hours_per_week=36,
    )
    assert set(out.keys()) == {
        "provider_fte_input",
        "provider_day_equiv",
        "psr_fte",
        "ma_fte",
        "xrt_fte",
        "support_total_fte",
    }
