import pandas as pd
from pathlib import Path

from psm.utils import round_up_to_increment


class StaffingModel:
    """
    StaffingModel loads staffing ratios from data/staffing_ratios.csv and
    returns interpolated staffing values for a given visits/day.

    Notes:
    - Roles are rounded UP to the nearest 0.25 (fractional staffing model).
    - XRT is fixed at 1.0.
    - Interpolation is linear between the two nearest rows.
    """

    def __init__(self, csv_path: str = None, rounding_increment: float = 0.25):
        self.rounding_increment = rounding_increment

        if csv_path is None:
            csv_path = Path(__file__).resolve().parents[1] / "data" / "staffing_ratios.csv"

        self.df = pd.read_csv(csv_path)

        # Normalize headers
        self.df.columns = self.df.columns.str.strip().str.lower()

        required_cols = [
            "ave_patients_day",
            "patients_per_provider_day",
            "provider_day",
            "psr_day",
            "ma_day",
            "xrt_day",
            "total_day",
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in staffing_ratios.csv: {missing}")

        self.df = self.df.sort_values("ave_patients_day").reset_index(drop=True)

    def calculate(self, visits_per_day: float) -> dict:
        """
        Returns interpolated staffing values for a given visits/day.
        """
        v = float(visits_per_day)

        if v <= self.df["ave_patients_day"].min():
            row = self.df.iloc[0].to_dict()
            return self._finalize(row)

        if v >= self.df["ave_patients_day"].max():
            row = self.df.iloc[-1].to_dict()
            return self._finalize(row)

        lower = self.df[self.df["ave_patients_day"] <= v].iloc[-1]
        upper = self.df[self.df["ave_patients_day"] >= v].iloc[0]

        if lower["ave_patients_day"] == upper["ave_patients_day"]:
            row = lower.to_dict()
            return self._finalize(row)

        ratio = (v - lower["ave_patients_day"]) / (
            upper["ave_patients_day"] - lower["ave_patients_day"]
        )

        interpolated = {}
        for col in self.df.columns:
            if col == "ave_patients_day":
                interpolated[col] = v
            else:
                interpolated[col] = float(lower[col] + ratio * (upper[col] - lower[col]))

        return self._finalize(interpolated)

    def _finalize(self, row: dict) -> dict:
        """
        Round up each role to nearest increment.
        XRT fixed at 1.0.
        """
        out = {}

        out["visits_day"] = float(row["ave_patients_day"])

        out["provider_day"] = round_up_to_increment(row["provider_day"], self.rounding_increment)
        out["psr_day"] = round_up_to_increment(row["psr_day"], self.rounding_increment)
        out["ma_day"] = round_up_to_increment(row["ma_day"], self.rounding_increment)

        out["xrt_day"] = 1.0

        out["total_day"] = (
            out["provider_day"]
            + out["psr_day"]
            + out["ma_day"]
            + out["xrt_day"]
        )

        out["patients_per_provider_day"] = float(row["patients_per_provider_day"])

        return out
