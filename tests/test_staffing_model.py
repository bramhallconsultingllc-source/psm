from psm.utils import round_up_to_increment
from psm.staffing_model import StaffingModel


def test_rounding_up():
    assert round_up_to_increment(2.01, 0.25) == 2.25
    assert round_up_to_increment(2.25, 0.25) == 2.25
    assert round_up_to_increment(2.26, 0.25) == 2.50


def test_calculate_schema():
    model = StaffingModel()
    out = model.calculate(25)

    expected_keys = {
        "visits_day",
        "provider_day",
        "psr_day",
        "ma_day",
        "xrt_day",
        "total_day",
        "patients_per_provider_day",
    }
    assert set(out.keys()) == expected_keys


def test_xrt_fixed():
    model = StaffingModel()
    out = model.calculate(45)
    assert out["xrt_day"] == 1.0


def test_total_day_is_sum():
    model = StaffingModel()
    out = model.calculate(45)

    expected = out["provider_day"] + out["psr_day"] + out["ma_day"] + out["xrt_day"]
    assert out["total_day"] == expected


def test_outputs_are_nonnegative():
    model = StaffingModel()
    out = model.calculate(25)

    assert out["visits_day"] >= 0
    assert out["provider_day"] >= 0
    assert out["psr_day"] >= 0
    assert out["ma_day"] >= 0
    assert out["xrt_day"] >= 0
    assert out["total_day"] >= 0
    # patients_per_provider_day may be 0 if your CSV has blanks, but should never be negative
    assert out["patients_per_provider_day"] >= 0


def test_provider_day_is_rounded_to_increment():
    inc = 0.25
    model = StaffingModel(rounding_increment=inc)
    out = model.calculate(45)

    # provider_day should land on an increment boundary
    assert abs((out["provider_day"] / inc) - round(out["provider_day"] / inc)) < 1e-9


def test_support_roles_are_rounded_to_increment():
    inc = 0.25
    model = StaffingModel(rounding_increment=inc)
    out = model.calculate(45)

    for k in ["psr_day", "ma_day"]:
        assert abs((out[k] / inc) - round(out[k] / inc)) < 1e-9


def test_provider_day_not_decreasing_with_volume():
    """
    Sanity: increasing visits/day should not reduce provider_day after interpolation + rounding.
    (If your CSV is not monotonic, this will catch it.)
    """
    model = StaffingModel()
    v = [10, 15, 20, 25, 30, 35, 40]
    p = [model.calculate(x)["provider_day"] for x in v]
    assert all(b >= a for a, b in zip(p, p[1:]))


def test_min_max_clamping_works():
    """
    If visits are below the table min or above the table max,
    calculate() should still return valid outputs (using endpoint rows).
    """
    model = StaffingModel()

    low = model.calculate(-5)
    high = model.calculate(10_000)

    # still returns the full schema and sane nonnegative values
    assert low["provider_day"] >= 0
    assert high["provider_day"] >= 0
    assert low["total_day"] == low["provider_day"] + low["psr_day"] + low["ma_day"] + low["xrt_day"]
    assert high["total_day"] == high["provider_day"] + high["psr_day"] + high["ma_day"] + high["xrt_day"]


def test_calculate_daily_matches_calculate():
    model = StaffingModel()
    d = model.calculate_daily(45)
    out = model.calculate(45)

    assert out["provider_day"] == d.provider_day
    assert out["psr_day"] == d.psr_day
    assert out["ma_day"] == d.ma_day
    assert out["xrt_day"] == d.xrt_day
    assert out["total_day"] == d.total_day


def test_role_mix_ratios_are_sane():
    model = StaffingModel()
    ratios = model.get_role_mix_ratios(45)

    assert ratios["psr_per_provider"] >= 0
    assert ratios["ma_per_provider"] >= 0
    assert ratios["xrt_per_provider"] >= 0


def test_role_mix_ratios_low_volume_no_crash():
    model = StaffingModel()
    ratios = model.get_role_mix_ratios(0.0)
    assert all(v >= 0 for v in ratios.values())


def test_support_fte_from_provider_fte_is_nonnegative():
    model = StaffingModel()
    out = model.calculate_support_fte_from_provider_fte(
        visits_per_day=45,
        provider_fte=2.0,
        hours_of_operation_per_week=84,
        fte_hours_per_week=36,
    )

    assert out["provider_day_equiv"] >= 0
    
    assert out["psr_fte"] >= 0
    assert out["ma_fte"] >= 0
    assert out["xrt_fte"] >= 0
    assert out["support_total_fte"] == out["psr_fte"] + out["ma_fte"] + out["xrt_fte"]

