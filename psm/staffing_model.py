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

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from psm.utils import round_up_to_increment


DEFAULT_TURNOVER_BUFFER = {
    "provider": 0.40,   # example: 40% buffer
    "psr": 0.30,
    "ma": 0.20,
    "xrt": 0.00,        # locked
}


def load_ratio_table(csv_path: str | Path) -> pd.DataFrame:
    """
    Loads staffing ratio table from CSV.
    Enforces required columns and sorts by ave_patients_day.
    """
    df = pd.read_csv(csv_path)

    required_cols = [
        "ave_patients_day",
        "patients_per_provider_day",
        "provider_day",
        "psr_day",
        "ma_day",
        "xrt_day",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in staffing table: {missing}")

    df = df.sort_values("ave_patients_day").reset_index(drop=True)
    return df


def interpolate_staffing(visits_per_day: float, df: pd.DataFrame) -> Dict[str, float]:
    """
    Returns interpolated staffing ratios for a given visits/day.

    - Uses linear interpolation between rows
    - Clamps values outside range
    - Rounds UP to nearest 0.25
    - Locks XRT at 1.0
    """

    visits = float(visits_per_day)

    # Clamp to min/max
    if visits <= df["ave_patients_day"].min():
        row = df.iloc[0]
        return _row_to_staffing(row)

    if visits >= df["ave_patients_day"].max():
        row = df.iloc[-1]
        return _row_to_staffing(row)

    # Find two nearest rows
    lower = df[df["ave_patients_day"] <= visits].iloc[-1]
    upper = df[df["ave_patients_day"] >= visits].iloc[0]

    if lower["ave_patients_day"] == upper["ave_patients_day"]:
        return _row_to_staffing(lower)

    # Linear interpolation
    x0, x1 = lower["ave_patients_day"], upper["ave_patients_day"]
    weight = (visits - x0) / (x1 - x0)

    staffing = {}
    for col in ["provider_day", "psr_day", "ma_day"]:
        y0, y1 = lower[col], upper[col]
        interpolated = y0 + weight * (y1 - y0)
        staffing[col] = round_up_to_increment(interpolated, 0.25)

    # Lock XRT at 1
    staffing["xrt_day"] = 1.0

    staffing["total_day"] = sum(staffing.values())
    return staffing


def staffing_to_weekly_hours(
    staffing: Dict[str, float],
    days_open: int,
    hours_per_shift: float
) -> Dict[str, float]:
    """
    Converts staffing/day into total hours/week per role.
    """
    weekly = {}
    for role in ["provider_day", "psr_day", "ma_day", "xrt_day"]:
        weekly[role.replace("_day", "_hours_week")] = staffing[role] * days_open * hours_per_shift
    weekly["total_hours_week"] = sum(weekly.values())
    return weekly


def weekly_hours_to_fte(
    weekly_hours: Dict[str, float],
    fte_type_hours: float = 40
) -> Dict[str, float]:
    """
    Converts weekly hours into FTE needed.
    Always rounds UP to nearest 0.25.
    """
    fte = {}
    for k, v in weekly_hours.items():
        if k.endswith("_hours_week") and k != "total_hours_week":
            base_role = k.replace("_hours_week", "")
            fte_needed = v / fte_type_hours
            fte[base_role + "_fte"] = round_up_to_increment(fte_needed, 0.25)

    fte["total_fte"] = sum(fte.values())
    return fte


def apply_turnover_buffer(
    fte: Dict[str, float],
    turnover_buffer: Dict[str, float] = DEFAULT_TURNOVER_BUFFER
) -> Dict[str, float]:
    """
    Adds turnover buffer by role and rounds UP.
    turnover_buffer should be decimals (0.20 = 20%).
    """
    adjusted = {}

    for role, fte_val in fte.items():
        if not role.endswith("_fte") or role == "total_fte":
            continue

        base_role = role.replace("_fte", "")
        buffer = turnover_buffer.get(base_role, 0.0)

        adjusted_val = fte_val * (1 + buffer)
        adjusted[base_role + "_fte_adj"] = round_up_to_increment(adjusted_val, 0.25)

    adjusted["total_fte_adj"] = sum(adjusted.values())
    return adjusted


def _row_to_staffing(row: Any) -> Dict[str, float]:
    """
    Converts a table row to staffing dict w/ rounding.
    """
    staffing = {
        "provider_day": round_up_to_increment(row["provider_day"], 0.25),
        "psr_day": round_up_to_increment(row["psr_day"], 0.25),
        "ma_day": round_up_to_increment(row["ma_day"], 0.25),
        "xrt_day": 1.0,
    }
    staffing["total_day"] = sum(staffing.values())
    return staffing
