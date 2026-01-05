import pandas as pd
from pathlib import Path
from psm.utils import round_up_to_increment


class StaffingModel:
    """
    Staffing Model using visit/day ratios.

    Supports:
    - Linear interpolation for visits/day
    - Role rounding UP to 0.25 increments
    - XRT fixed at 1.0
    - Optional conversion to weekly hours and FTE needed
    """

    def __init__(
        self,
        ratios_path: str = "data/staffing_ratios.csv",
        rounding_increment: float = 0.25,
        xrt_fixed: float = 1.0,
    ):
        self.rounding_increment = rounding_increment
        self.xrt_fixed = xrt_fixed

        path = Path(ratios_path)
        if not path.exists():
            raise FileNotFoundError(f"Ratios file not found at: {ratios_path}")

        df = pd.read_csv(path)

        # ✅ Normalize column headers (fixes hidden spaces & casing issues)
        df.columns = df.columns.str.strip().str.upper()

        # Now safe
        df["AVE PATIENTS/DAY"]

    def calculate_staff_per_day(self, visits_per_day: float) -> dict:
        """
        Returns interpolated staffing per day rounded UP.
        """

        df = self.df
        x_col = "AVE PATIENTS/DAY"

        roles = {
            "provider_day": "PROVIDER/DAY",
            "psr_day": "PSR/DAY",
            "ma_day": "MA/DAY",
        }

        # Clamp at min/max
        if visits_per_day <= df[x_col].min():
            row = df.iloc[0]
            output = {
                k: round_up_to_increment(float(row[col]), self.rounding_increment)
                for k, col in roles.items()
            }
        elif visits_per_day >= df[x_col].max():
            row = df.iloc[-1]
            output = {
                k: round_up_to_increment(float(row[col]), self.rounding_increment)
                for k, col in roles.items()
            }
        else:
            lower = df[df[x_col] <= visits_per_day].iloc[-1]
            upper = df[df[x_col] >= visits_per_day].iloc[0]

            x0, x1 = float(lower[x_col]), float(upper[x_col])

            output = {}
            for out_key, col in roles.items():
                y0, y1 = float(lower[col]), float(upper[col])

                # Linear interpolation
                y = y0 + (y1 - y0) * ((visits_per_day - x0) / (x1 - x0))

                # Round UP to increment
                output[out_key] = round_up_to_increment(y, self.rounding_increment)

        # ✅ XRT locked
        output["xrt_day"] = self.xrt_fixed

        # Total/day
        output["total_day"] = (
            output["provider_day"]
            + output["psr_day"]
            + output["ma_day"]
            + output["xrt_day"]
        )

        return output

    def staff_day_to_weekly_hours(
        self,
        staff_per_day: dict,
        hours_of_operation_per_week: float = 84,
        days_open_per_week: int = 7,
    ) -> dict:
        """
        Converts staff/day into weekly hours.

        Assumes each staff/day covers a full operating day.
        """
        hours_per_day = hours_of_operation_per_week / days_open_per_week

        weekly_hours = {
            "provider_hours_week": staff_per_day["provider_day"] * hours_per_day * days_open_per_week,
            "psr_hours_week": staff_per_day["psr_day"] * hours_per_day * days_open_per_week,
            "ma_hours_week": staff_per_day["ma_day"] * hours_per_day * days_open_per_week,
            "xrt_hours_week": staff_per_day["xrt_day"] * hours_per_day * days_open_per_week,
        }

        weekly_hours["total_hours_week"] = sum(weekly_hours.values())
        return weekly_hours

    def weekly_hours_to_fte(
        self,
        weekly_hours: dict,
        fte_type_hours: float = 40,
    ) -> dict:
        """
        Converts weekly hours into FTE.
        Always rounds UP to 0.25 increments.
        """
        fte = {
            "provider_fte": round_up_to_increment(weekly_hours["provider_hours_week"] / fte_type_hours, self.rounding_increment),
            "psr_fte": round_up_to_increment(weekly_hours["psr_hours_week"] / fte_type_hours, self.rounding_increment),
            "ma_fte": round_up_to_increment(weekly_hours["ma_hours_week"] / fte_type_hours, self.rounding_increment),
            "xrt_fte": round_up_to_increment(weekly_hours["xrt_hours_week"] / fte_type_hours, self.rounding_increment),
        }

        fte["total_fte"] = sum(fte.values())
        return fte

    def calculate(
        self,
        visits_per_day: float,
        hours_of_operation_per_week: float = 84,
        days_open_per_week: int = 7,
        fte_type_hours: float = 40,
    ) -> dict:
        """
        Full output:
        - staff/day
        - hours/week
        - fte needed
        """
        staff = self.calculate_staff_per_day(visits_per_day)
        weekly_hours = self.staff_day_to_weekly_hours(
            staff,
            hours_of_operation_per_week=hours_of_operation_per_week,
            days_open_per_week=days_open_per_week,
        )
        fte = self.weekly_hours_to_fte(weekly_hours, fte_type_hours=fte_type_hours)

        return {**staff, **weekly_hours, **fte}
