import pandas as pd
from psm.utils import interpolate_staffing, round_up_to_increment


class StaffingModel:
    """
    Predictive Staffing Model (PSM)

    - Uses linear interpolation on visit volume
    - Rounds UP all roles to nearest 0.25
    - XRT is fixed at 1.0 (always)
    """

    def __init__(
        self,
        csv_path: str = "data/staffing_ratios.csv",
        round_increment: float = 0.25,
        xrt_fixed: float = 1.0,
    ):
        self.csv_path = csv_path
        self.round_increment = round_increment
        self.xrt_fixed = xrt_fixed
        self.df = pd.read_csv(csv_path)

    def calculate(self, visits_per_day: float) -> dict:
        raw = interpolate_staffing(self.df, visits_per_day)

        output = {
            "visits_per_day": visits_per_day,
            "provider_day": round_up_to_increment(raw["provider_day"], self.round_increment),
            "psr_day": round_up_to_increment(raw["psr_day"], self.round_increment),
            "ma_day": round_up_to_increment(raw["ma_day"], self.round_increment),
            "xrt_day": self.xrt_fixed,
        }

        # Total: sum of roles (rounded)
        output["total_day"] = round_up_to_increment(
            output["provider_day"] + output["psr_day"] + output["ma_day"] + output["xrt_day"],
            self.round_increment,
        )

        return output

