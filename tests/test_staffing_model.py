from psm.utils import round_up_to_increment
from psm.staffing_model import StaffingModel


def test_rounding_up():
    assert round_up_to_increment(2.01, 0.25) == 2.25
    assert round_up_to_increment(2.25, 0.25) == 2.25
    assert round_up_to_increment(2.26, 0.25) == 2.50


def test_xrt_fixed():
    model = StaffingModel()
    out = model.calculate_staff_per_day(45)
    assert out["xrt_day"] == 1.0


def test_interpolation_and_rounding():
    model = StaffingModel()
    out = model.calculate_staff_per_day(45)

    # Expect interpolation between 42 and 48
    # provider interpolated ~1.375 -> rounds UP to 1.5
    assert out["provider_day"] == 1.5


def test_weekly_hours_to_fte_rounding():
    model = StaffingModel()

    weekly_hours = {
        "provider_hours_week": 42,
        "psr_hours_week": 21,
        "ma_hours_week": 60,
        "xrt_hours_week": 21,
    }

    fte = model.weekly_hours_to_fte(weekly_hours, fte_type_hours=40)

    assert fte["provider_fte"] == 1.25
    assert fte["psr_fte"] == 0.75
    assert fte["ma_fte"] == 1.5
    assert fte["xrt_fte"] == 0.75
