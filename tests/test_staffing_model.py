from psm.utils import round_up_to_increment
from psm.staffing_model import StaffingModel


def test_rounding_up():
    assert round_up_to_increment(2.01, 0.25) == 2.25
    assert round_up_to_increment(2.25, 0.25) == 2.25
    assert round_up_to_increment(2.26, 0.25) == 2.50


def test_xrt_fixed():
    model = StaffingModel()
    out = model.calculate(45)
    assert out["xrt_day"] == 1.0


def test_interpolation_and_rounding():
    model = StaffingModel()
    out = model.calculate(45)

    # Expect interpolation between 42 and 48
    # provider interpolated ~1.375 -> rounds UP to 1.5
    assert out["provider_day"] == 1.5

