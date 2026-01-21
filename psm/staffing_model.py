# psm/staffing_model.py
from typing import Dict

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from psm.utils import round_up_to_increment


@dataclass(frozen=True)
class DailyStaffing:
    """
    Daily staffing outputs from staffing_ratios.csv (per-day headcount),
    after interpolation + rounding rules.
    """
    visits_day: float
    provider_day: float
    psr_day: float
    ma_day: float
    xrt_day: float
    total_day: float
    patients_per_provider_day: float


class StaffingModel:
    """
    StaffingModel loads staffing ratios from data/staffing_ratios.csv and returns
    interpolated daily staffing outputs for a given visits/day.

    IMPORTANT DESIGN NOTE (for your PSM app):
    - This model is best used as a *ratio table* for support roles (PSR/MA/XRT) and
      as a reference for "patients_per_provider_day".
    - If you are implementing capacity-aware provider coverage in app.py, DO NOT
      use this model to drive provider FTE targets. Use your capacity logic instead,
      and use the ratios here to scale PSR/MA/XRT off provider supply/target.

    Rules:
    - Provider/PSR/MA are rounded UP to nearest rounding_increment (default 0.25).
    - XRT is fixed at 1.0 per day (can be changed if you decide to make it ratio-based).
    - Interpolation is linear between nearest rows by ave_patients_day.
    """

    def __init__(self, csv_path: Optional[str] = None, rounding_increment: float = 0.25):
        self.rounding_increment = float(rounding_increment)

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
            "xrt_day",
            "total_day",
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in staffing_ratios.csv: {missing}")

        self.df = self.df.sort_values("ave_patients_day").reset_index(drop=True)

        # Basic sanity checks
        if self.df["ave_patients_day"].isna().any():
            raise ValueError("staffing_ratios.csv contains NaN in ave_patients_day")
        if (self.df["ave_patients_day"] < 0).any():
            raise ValueError("staffing_ratios.csv contains negative ave_patients_day")

    # ---------------------------------------------------------------------
    # Core table interpolation (daily)
    # ---------------------------------------------------------------------
    def calculate(self, visits_per_day: float) -> Dict[str, float]:
        """
        Returns daily staffing values for a given visits/day (interpolated then finalized).
        Output keys match your prior interface for backward compatibility.
        """
        daily = self.calculate_daily(visits_per_day)
        return {
            "visits_day": daily.visits_day,
            "provider_day": daily.provider_day,
            "psr_day": daily.psr_day,
            "ma_day": daily.ma_day,
            "xrt_day": daily.xrt_day,
            "total_day": daily.total_day,
            "patients_per_provider_day": daily.patients_per_provider_day,
        }

    def calculate_daily(self, visits_per_day: float) -> DailyStaffing:
        """
        Strongly-typed daily staffing output.
        """
        v = float(visits_per_day)

        if v <= float(self.df["ave_patients_day"].min()):
            row = self.df.iloc[0].to_dict()
            return self._finalize(row, v_override=v)

        if v >= float(self.df["ave_patients_day"].max()):
            row = self.df.iloc[-1].to_dict()
            return self._finalize(row, v_override=v)

        lower = self.df[self.df["ave_patients_day"] <= v].iloc[-1]
        upper = self.df[self.df["ave_patients_day"] >= v].iloc[0]

        if float(lower["ave_patients_day"]) == float(upper["ave_patients_day"]):
            return self._finalize(lower.to_dict(), v_override=v)

        ratio = (v - float(lower["ave_patients_day"])) / (float(upper["ave_patients_day"]) - float(lower["ave_patients_day"]))

        interpolated: Dict[str, float] = {}
        for col in self.df.columns:
            if col == "ave_patients_day":
                interpolated[col] = v
            else:
                interpolated[col] = float(lower[col] + ratio * (upper[col] - lower[col]))

        return self._finalize(interpolated, v_override=v)

    def _finalize(self, row: Dict, v_override: Optional[float] = None) -> DailyStaffing:
        """
        Apply rounding rules and “fixed XRT” rule.
        """
        visits_day = float(v_override if v_override is not None else row["ave_patients_day"])

        provider_day = round_up_to_increment(row["provider_day"], self.rounding_increment)
        psr_day = round_up_to_increment(row["psr_day"], self.rounding_increment)
        ma_day = round_up_to_increment(row["ma_day"], self.rounding_increment)

        # Fixed XRT rule (kept from your original)
        xrt_day = 1.0

        total_day = float(provider_day + psr_day + ma_day + xrt_day)
        pppd = float(row.get("patients_per_provider_day", 0.0))  # patients per provider day

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
    # Ratios & capacity helpers (recommended for PSM)
    # ---------------------------------------------------------------------
    def get_role_mix_ratios(self, visits_per_day: float) -> Dict[str, float]:
        """
        Returns ratios of PSR/MA/XRT per 1.0 provider_day from the table.

        This is the preferred way to use StaffingModel in a capacity-aware PSM:
        - Provider FTE is computed in app.py using coverage + capacity logic
        - PSR/MA/XRT are scaled off provider FTE using these ratios
        """
        d = self.calculate_daily(visits_per_day)
        prov = max(float(d.provider_day), 0.25)
        return {
            "psr_per_provider": float(d.psr_day) / prov,
            "ma_per_provider": float(d.ma_day) / prov,
            "xrt_per_provider": float(d.xrt_day) / prov,
        }

    def suggest_patients_per_provider_day(self, visits_per_day: float, default: float = 36.0) -> float:
        """
        Returns the table-derived patients_per_provider_day (if present), else default.
        Useful if you want to prefill a UI input.
        """
        d = self.calculate_daily(visits_per_day)
        return float(d.patients_per_provider_day) if float(d.patients_per_provider_day) > 0 else float(default)

    # ---------------------------------------------------------------------
    # Weekly FTE conversions
    # ---------------------------------------------------------------------
    @staticmethod
    def _weekly_fte_from_daily_staff(staff_per_day: float, hours_of_operation_per_week: float, fte_hours_per_week: float) -> float:
        """
        Converts daily staff headcount (staff_per_day) into weekly FTE.

        Math:
          staff_hours_week = staff_per_day * hours_of_operation_per_week
          fte_needed = staff_hours_week / fte_hours_per_week
        """
        how = max(float(hours_of_operation_per_week), 0.0)
        fte_hw = max(float(fte_hours_per_week), 1e-6)
        staff_day = max(float(staff_per_day), 0.0)
        staff_hours_week = staff_day * how
        return staff_hours_week / fte_hw

    def calculate_fte_needed(
        self,
        visits_per_day: float,
        hours_of_operation_per_week: float,
        fte_hours_per_week: float = 40.0,
    ) -> Dict[str, float]:
        """
        BACKWARD-COMPAT method: converts *table-derived* daily staffing into weekly FTE.

        WARNING (PSM capacity-aware provider logic):
        - This will produce a provider_fte driven by your staffing_ratios.csv curve.
        - If your app now uses capacity-aware provider targeting, do NOT use the
          returned provider_fte to build provider targets.

        This method remains to avoid breaking older code paths.
        """
        daily = self.calculate_daily(visits_per_day)

        provider_fte = self._weekly_fte_from_daily_staff(daily.provider_day, hours_of_operation_per_week, fte_hours_per_week)
        psr_fte = self._weekly_fte_from_daily_staff(daily.psr_day, hours_of_operation_per_week, fte_hours_per_week)
        ma_fte = self._weekly_fte_from_daily_staff(daily.ma_day, hours_of_operation_per_week, fte_hours_per_week)
        xrt_fte = self._weekly_fte_from_daily_staff(daily.xrt_day, hours_of_operation_per_week, fte_hours_per_week)

        total_fte = provider_fte + psr_fte + ma_fte + xrt_fte

        return {
            "provider_fte": float(provider_fte),
            "psr_fte": float(psr_fte),
            "ma_fte": float(ma_fte),
            "xrt_fte": float(xrt_fte),
            "total_fte": float(total_fte),
        }
    def get_role_mix_ratios(self, visits_per_day: float) -> Dict[str, float]:
        """
        Returns role mix ratios relative to providers:
          PSR per provider
          MA per provider
          XRT per provider
    
        Uses the staffing_ratios.csv interpolation + rounding rules (daily FTE/day).
    
        This is intentionally used for finance realism (SWB/Visit estimation),
        NOT to drive provider target FTE (capacity-aware logic does that).
        """
        daily = self.calculate(float(visits_per_day))
    
        provider_day = max(float(daily.get("provider_day", 0.0)), 0.25)
    
        return {
            "psr_per_provider": float(daily.get("psr_day", 0.0)) / provider_day,
            "ma_per_provider": float(daily.get("ma_day", 0.0)) / provider_day,
            "xrt_per_provider": float(daily.get("xrt_day", 0.0)) / provider_day,
        }

    def calculate_support_fte_from_provider_fte(
    self,
    visits_per_day: float,
    provider_fte: float,
    hours_of_operation_per_week: float,
    fte_hours_per_week: float = 40.0,
) -> Dict[str, float]:
    """
    Capacity-aware PSM support staffing:

    - provider_fte is WEEKLY FTE (coverage + capacity logic from app)
    - support ratios come from staffing table DAILY ratios
    - We convert provider_fte -> provider_day (daily coverage headcount),
      apply ratios, then convert support_day -> support_fte.
    """
    ratios = self.get_role_mix_ratios(visits_per_day)

    hours_week = float(hours_of_operation_per_week)
    fte_hours = float(fte_hours_per_week)

    prov_fte = max(float(provider_fte), 0.0)

    # Convert weekly FTE into "provider_day" coverage headcount
    provider_day = (prov_fte * fte_hours) / max(hours_week, 1e-9)

    psr_day = provider_day * float(ratios["psr_per_provider"])
    ma_day  = provider_day * float(ratios["ma_per_provider"])
    xrt_day = provider_day * float(ratios["xrt_per_provider"])

    # Convert daily heads back into weekly FTE
    psr_fte = (psr_day * hours_week) / max(fte_hours, 1e-9)
    ma_fte  = (ma_day  * hours_week) / max(fte_hours, 1e-9)
    xrt_fte = (xrt_day * hours_week) / max(fte_hours, 1e-9)

    support_total_fte = psr_fte + ma_fte + xrt_fte

    return {
        "provider_fte_input": float(prov_fte),
        "provider_day_equiv": float(provider_day),
        "psr_fte": float(psr_fte),
        "ma_fte": float(ma_fte),
        "xrt_fte": float(xrt_fte),
        "support_total_fte": float(support_total_fte),
    }
