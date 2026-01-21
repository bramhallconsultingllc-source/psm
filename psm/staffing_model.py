# psm/staffing_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd

from psm.utils import round_up_to_increment


@dataclass(frozen=True)
class DailyStaffing:
    visits_day: float
    provider_day: float
    psr_day: float
    ma_day: float
    xrt_day: float
    total_day: float
    patients_per_provider_day: float


class StaffingModel:
    """
    Loads staffing ratios from data/staffing_ratios.csv and returns interpolated
    daily staffing outputs for a given visits/day.

    IMPORTANT (for capacity-aware PSM):
    - Use app.py capacity logic for provider targets.
    - Use this class for role mix ratios (PSR/MA/XRT) and table references.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        rounding_increment: float = 0.25,
        fixed_xrt_day: float = 1.0,
    ):
        self.rounding_increment = float(rounding_increment)
        self.fixed_xrt_day = float(fixed_xrt_day)

        if csv_path is None:
            csv_path = str(Path(__file__).resolve().parents[1] / "data" / "staffing_ratios.csv")

        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip().str.lower()

        required_cols = [
            "ave_patients_day",
            "patients_per_provider_day",
            "provider_day",
            "psr_day",
            "ma_day",
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in staffing_ratios.csv: {missing}")

        self.df = self.df.sort_values("ave_patients_day").reset_index(drop=True)

        if self.df["ave_patients_day"].isna().any():
            raise ValueError("staffing_ratios.csv contains NaN in ave_patients_day")
        if (self.df["ave_patients_day"] < 0).any():
            raise ValueError("staffing_ratios.csv contains negative ave_patients_day")

        if self.df[["provider_day", "psr_day", "ma_day", "patients_per_provider_day"]].isna().any().any():
            raise ValueError("staffing_ratios.csv contains NaN in staffing columns")

        # Only interpolate numeric ratio columns we care about (avoids non-numeric CSV columns)
        self.interp_cols: List[str] = ["provider_day", "psr_day", "ma_day", "patients_per_provider_day"]

    def debug_identity(self) -> Dict[str, str]:
        return {
            "module_file": str(Path(__file__).resolve()),
            "class": self.__class__.__name__,
            "has_get_role_mix_ratios": str(hasattr(self, "get_role_mix_ratios")),
        }

    # ---------------------------------------------------------------------
    # Core table interpolation (daily)
    # ---------------------------------------------------------------------
    def calculate(self, visits_per_day: float) -> Dict[str, float]:
        d = self.calculate_daily(visits_per_day)
        return {
            "visits_day": d.visits_day,
            "provider_day": d.provider_day,
            "psr_day": d.psr_day,
            "ma_day": d.ma_day,
            "xrt_day": d.xrt_day,
            "total_day": d.total_day,
            "patients_per_provider_day": d.patients_per_provider_day,
        }

    def calculate_daily(self, visits_per_day: float) -> DailyStaffing:
        v = float(visits_per_day)

        if v <= float(self.df["ave_patients_day"].min()):
            return self._finalize(self.df.iloc[0].to_dict(), v_override=v)

        if v >= float(self.df["ave_patients_day"].max()):
            return self._finalize(self.df.iloc[-1].to_dict(), v_override=v)

        lower = self.df[self.df["ave_patients_day"] <= v].iloc[-1]
        upper = self.df[self.df["ave_patients_day"] >= v].iloc[0]

        if float(lower["ave_patients_day"]) == float(upper["ave_patients_day"]):
            return self._finalize(lower.to_dict(), v_override=v)

        denom = float(upper["ave_patients_day"]) - float(lower["ave_patients_day"])
        ratio = (v - float(lower["ave_patients_day"])) / denom

        interpolated: Dict[str, float] = {"ave_patients_day": v}
        for col in self.interp_cols:
            interpolated[col] = float(lower[col] + ratio * (upper[col] - lower[col]))

        return self._finalize(interpolated, v_override=v)

    def _finalize(self, row: Dict[str, float], v_override: Optional[float] = None) -> DailyStaffing:
        visits_day = float(v_override if v_override is not None else row["ave_patients_day"])

        provider_day = round_up_to_increment(row["provider_day"], self.rounding_increment)
        psr_day = round_up_to_increment(row["psr_day"], self.rounding_increment)
        ma_day = round_up_to_increment(row["ma_day"], self.rounding_increment)

        xrt_day = self.fixed_xrt_day
        total_day = float(provider_day + psr_day + ma_day + xrt_day)
        pppd = float(row.get("patients_per_provider_day", 0.0))

        return DailyStaffing(
            visits_day=visits_day,
            provider_day=float(provider_day),
            psr_day=float(psr_day),
            ma_day=float(ma_day),
            xrt_day=float(xrt_day),
            total_day=float(total_day),
            patients_per_provider_day=float(pppd),
        )

    # ---------------------------------------------------------------------
    # Ratios (preferred for PSM finance realism)
    # ---------------------------------------------------------------------
    def get_role_mix_ratios(self, visits_per_day: float) -> Dict[str, float]:
        d = self.calculate_daily(visits_per_day)
        prov = max(float(d.provider_day), 0.25)
        return {
            "psr_per_provider": float(d.psr_day) / prov,
            "ma_per_provider": float(d.ma_day) / prov,
            "xrt_per_provider": float(d.xrt_day) / prov,
        }

    def suggest_patients_per_provider_day(self, visits_per_day: float, default: float = 36.0) -> float:
        d = self.calculate_daily(visits_per_day)
        return float(d.patients_per_provider_day) if float(d.patients_per_provider_day) > 0 else float(default)

    # ---------------------------------------------------------------------
    # Weekly FTE conversions
    # ---------------------------------------------------------------------
    @staticmethod
    def _weekly_fte_from_daily_staff(staff_per_day: float, hours_of_operation_per_week: float, fte_hours_per_week: float) -> float:
        how = max(float(hours_of_operation_per_week), 0.0)
        fte_hw = max(float(fte_hours_per_week), 1e-6)
        staff_day = max(float(staff_per_day), 0.0)
        return (staff_day * how) / fte_hw

    def calculate_fte_needed(self, visits_per_day: float, hours_of_operation_per_week: float, fte_hours_per_week: float = 40.0) -> Dict[str, float]:
        d = self.calculate_daily(visits_per_day)

        provider_fte = self._weekly_fte_from_daily_staff(d.provider_day, hours_of_operation_per_week, fte_hours_per_week)
        psr_fte = self._weekly_fte_from_daily_staff(d.psr_day, hours_of_operation_per_week, fte_hours_per_week)
        ma_fte = self._weekly_fte_from_daily_staff(d.ma_day, hours_of_operation_per_week, fte_hours_per_week)
        xrt_fte = self._weekly_fte_from_daily_staff(d.xrt_day, hours_of_operation_per_week, fte_hours_per_week)

        return {
            "provider_fte": float(provider_fte),
            "psr_fte": float(psr_fte),
            "ma_fte": float(ma_fte),
            "xrt_fte": float(xrt_fte),
            "total_fte": float(provider_fte + psr_fte + ma_fte + xrt_fte),
        }

    def calculate_support_fte_from_provider_fte(
        self,
        visits_per_day: float,
        provider_fte: float,
        hours_of_operation_per_week: float,
        fte_hours_per_week: float = 40.0,
    ) -> Dict[str, float]:
        ratios = self.get_role_mix_ratios(visits_per_day)

        hours_week = float(hours_of_operation_per_week)
        fte_hours = float(fte_hours_per_week)
        prov_fte = max(float(provider_fte), 0.0)

        provider_day = (prov_fte * fte_hours) / max(hours_week, 1e-9)

        psr_day = provider_day * float(ratios["psr_per_provider"])
        ma_day = provider_day * float(ratios["ma_per_provider"])
        xrt_day = provider_day * float(ratios["xrt_per_provider"])

        psr_fte = (psr_day * hours_week) / max(fte_hours, 1e-9)
        ma_fte = (ma_day * hours_week) / max(fte_hours, 1e-9)
        xrt_fte = (xrt_day * hours_week) / max(fte_hours, 1e-9)

        return {
            "provider_fte_input": float(prov_fte),
            "provider_day_equiv": float(provider_day),
            "psr_fte": float(psr_fte),
            "ma_fte": float(ma_fte),
            "xrt_fte": float(xrt_fte),
            "support_total_fte": float(psr_fte + ma_fte + xrt_fte),
        }
