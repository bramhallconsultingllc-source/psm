# app_final_with_seasonal_backfill.py — Predictive Staffing Model (FINAL)
# 
# CRITICAL FIX: SEASONAL-AWARE BACKFILL
# - When a provider gives notice, look ahead 90 days (lead time)
# - If the future season needs LOWER staffing, DON'T backfill
# - This allows natural attrition to step FTE down from winter → base
#
# Example:
#   December: Provider gives notice (lead_time = 90 days)
#   Future month = March (base season)
#   Current FTE = 3.3 (winter level)
#   Target for March = 2.3 (base level)
#   Decision: DON'T backfill (we're already above March target)
#
# Two-level permanent staffing policy (Base + Winter) + flex as safety stock
# 36-month horizon • Lead-time + ramp • Hiring freeze during flu months

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from psm.staffing_model import StaffingModel

MODEL_VERSION = "2026-01-30-FINAL-seasonal-aware-backfill-v1"

st.set_page_config(page_title="PSM — FINAL (Seasonal Backfill)", layout="centered")

st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 0.75rem; padding-bottom: 2.25rem; }
      .contract { background: #f7f7f7; border: 1px solid #e6e6e6; border-radius: 10px; padding: 14px 16px; }
      .note { background: #f3f7ff; border: 1px solid #cfe0ff; border-radius: 10px; padding: 12px 14px; }
      .warn { background: #fff6e6; border: 1px solid #ffe2a8; border-radius: 10px; padding: 12px 14px; }
      .ok { background: #ecfff0; border: 1px solid #b7f0c0; border-radius: 10px; padding: 12px 14px; }
      .fix { background: #e8f5e9; border: 2px solid #4caf50; border-radius: 10px; padding: 14px 16px; }
      .divider { height: 1px; background: #eee; margin: 10px 0 14px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 PSM — FINAL (Seasonal-Aware Backfill)")
st.caption("✅ Providers give 90-day notice • Don't backfill if future season needs lower staffing • Natural step-down")

model = StaffingModel()

st.session_state.setdefault("rec_policy", None)
st.session_state.setdefault("frontier", None)

BRAND_BLACK = "#000000"
BRAND_GOLD = "#7a6200"
GRAY = "#B0B0B0"
LIGHT_GRAY = "#EAEAEA"
MID_GRAY = "#666666"

WINTER = {12, 1, 2}
SUMMER = {6, 7, 8}

N_MONTHS = 36
AVG_DAYS_PER_MONTH = 30.4

MONTH_OPTIONS = [
    ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("May", 5), ("Jun", 6),
    ("Jul", 7), ("Aug", 8), ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12)
]

HELP: dict[str, str] = {
    "visits": "Baseline average visits per day.",
    "annual_growth": "Annual growth rate.",
    "peak_factor": "Peak-to-average planning multiplier.",
    "visits_per_provider_hour": "Sustainable visits per provider hour.",
}

def wrap_month(m: int) -> int:
    m = int(m)
    while m <= 0:
        m += 12
    while m > 12:
        m -= 12
    return m

def month_name(m: int) -> str:
    return datetime(2000, int(m), 1).strftime("%b")

def lead_days_to_months(days: int, avg_days_per_month: float = AVG_DAYS_PER_MONTH) -> int:
    return max(0, int(math.ceil(float(days) / float(avg_days_per_month))))

def provider_day_equiv_from_fte(provider_fte: float, hours_week: float, fte_hours_week: float) -> float:
    return float(provider_fte) * (float(fte_hours_week) / max(float(hours_week), 1e-9))

def fte_required_for_min_perm_providers_per_day(min_providers_per_day: float, hours_week: float, fte_hours_week: float) -> float:
    return float(min_providers_per_day) * (float(hours_week) / max(float(fte_hours_week), 1e-9))

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def compute_visits_curve(months: list[int], base_year0: float, base_year1: float, base_year2: float, seasonality_pct: float) -> list[float]:
    out: list[float] = []
    for i, m in enumerate(months):
        if i < 12:
            base = base_year0
        elif i < 24:
            base = base_year1
        else:
            base = base_year2
        if m in WINTER:
            v = base * (1.0 + seasonality_pct)
        elif m in SUMMER:
            v = base * (1.0 - seasonality_pct)
        else:
            v = base
        out.append(float(v))
    return out

def apply_flu_uplift(visits_curve: list[float], months: list[int], flu_months: set[int], flu_uplift_pct: float) -> list[float]:
    out: list[float] = []
    for v, m in zip(visits_curve, months):
        out.append(float(v) * (1.0 + float(flu_uplift_pct)) if int(m) in flu_months else float(v))
    return out

def monthly_hours_from_fte(fte: float, fte_hours_per_week: float, days_in_month: int) -> float:
    return float(fte) * float(fte_hours_per_week) * (float(days_in_month) / 7.0)

def loaded_hourly_rate(base_hourly: float, benefits_load_pct: float, ot_sick_pct: float, bonus_pct: float) -> float:
    return float(base_hourly) * (1.0 + float(bonus_pct)) * (1.0 + float(benefits_load_pct)) * (1.0 + float(ot_sick_pct))

def compute_role_mix_ratios(visits_per_day: float, staffing_model: StaffingModel) -> dict[str, float]:
    v = float(visits_per_day)
    if hasattr(staffing_model, "get_role_mix_ratios"):
        return staffing_model.get_role_mix_ratios(v)
    daily = staffing_model.calculate(v)
    prov_day = max(float(daily.get("provider_day", 0.0)), 0.25)
    return {
        "psr_per_provider": float(daily.get("psr_day", 0.0)) / prov_day,
        "ma_per_provider": float(daily.get("ma_day", 0.0)) / prov_day,
        "xrt_per_provider": float(daily.get("xrt_day", 0.0)) / prov_day,
    }

def annual_swb_per_visit_from_supply(
    provider_paid_fte: list[float], provider_flex_fte: list[float], visits_per_day: list[float],
    days_in_month: list[int], fte_hours_per_week: float, role_mix: dict[str, float],
    hourly_rates: dict[str, float], benefits_load_pct: float, ot_sick_pct: float, bonus_pct: float,
    physician_supervision_hours_per_month: float = 0.0, supervisor_hours_per_month: float = 0.0,
) -> tuple[float, float, float]:
    total_swb = 0.0
    total_visits = 0.0
    
    apc_rate = loaded_hourly_rate(hourly_rates["apc"], benefits_load_pct, ot_sick_pct, bonus_pct)
    psr_rate = loaded_hourly_rate(hourly_rates["psr"], benefits_load_pct, ot_sick_pct, bonus_pct)
    ma_rate = loaded_hourly_rate(hourly_rates["ma"], benefits_load_pct, ot_sick_pct, bonus_pct)
    rt_rate = loaded_hourly_rate(hourly_rates["rt"], benefits_load_pct, ot_sick_pct, bonus_pct)
    phys_rate = loaded_hourly_rate(hourly_rates["physician"], benefits_load_pct, ot_sick_pct, bonus_pct)
    sup_rate = loaded_hourly_rate(hourly_rates["supervisor"], benefits_load_pct, ot_sick_pct, bonus_pct)

    for fte_paid, fte_flex, vpd, dim in zip(provider_paid_fte, provider_flex_fte, visits_per_day, days_in_month):
        dim_i = int(dim)
        month_visits = max(float(vpd) * float(dim_i), 1.0)
        prov_total = float(fte_paid) + float(fte_flex)
        
        psr_fte = prov_total * float(role_mix["psr_per_provider"])
        ma_fte = prov_total * float(role_mix["ma_per_provider"])
        rt_fte = prov_total * float(role_mix["xrt_per_provider"])

        prov_hours = monthly_hours_from_fte(prov_total, fte_hours_per_week, dim_i)
        psr_hours = monthly_hours_from_fte(psr_fte, fte_hours_per_week, dim_i)
        ma_hours = monthly_hours_from_fte(ma_fte, fte_hours_per_week, dim_i)
        rt_hours = monthly_hours_from_fte(rt_fte, fte_hours_per_week, dim_i)

        month_swb = (prov_hours * apc_rate + psr_hours * psr_rate + ma_hours * ma_rate + rt_hours * rt_rate
                    + float(physician_supervision_hours_per_month) * phys_rate
                    + float(supervisor_hours_per_month) * sup_rate)

        total_swb += float(month_swb)
        total_visits += float(month_visits)

    total_visits = max(total_visits, 1.0)
    return float(total_swb) / float(total_visits), float(total_swb), float(total_visits)



@dataclass(frozen=True)
class ModelParams:
    visits: float
    annual_growth: float
    seasonality_pct: float
    flu_uplift_pct: float
    flu_months: set[int]
    peak_factor: float
    visits_per_provider_hour: float
    hours_week: float
    days_open_per_week: float
    fte_hours_week: float
    annual_turnover: float
    lead_days: int
    ramp_months: int
    ramp_productivity: float
    fill_probability: float
    winter_anchor_month: int
    winter_end_month: int
    freeze_months: set[int]
    budgeted_pppd: float
    yellow_max_pppd: float
    red_start_pppd: float
    flex_max_fte_per_month: float
    flex_cost_multiplier: float
    target_swb_per_visit: float
    swb_tolerance: float
    net_revenue_per_visit: float
    visits_lost_per_provider_day_gap: float
    provider_replacement_cost: float
    turnover_yellow_mult: float
    turnover_red_mult: float
    hourly_rates: dict[str, float]
    benefits_load_pct: float
    ot_sick_pct: float
    bonus_pct: float
    physician_supervision_hours_per_month: float
    supervisor_hours_per_month: float
    min_perm_providers_per_day: float
    allow_prn_override: bool
    require_perm_under_green_no_flex: bool
    utilization_target: float
    recovery_months: int
    perm_step_epsilon: float
    idle_grace_months: int
    idle_epsilon_eff_fte: float
    elasticity_cap: float
    elasticity_stabilization_months: int
    _v: str = MODEL_VERSION


@dataclass(frozen=True)
class Policy:
    base_fte: float
    winter_fte: float


# ============================================================
# SIMULATION ENGINE WITH SEASONAL-AWARE BACKFILL (THE KEY FIX)
# ============================================================
def simulate_policy_seasonal_backfill(params: ModelParams, policy: Policy) -> dict[str, Any]:
    """
    FINAL FIX: Seasonal-Aware Backfill
    
    Key insight: When a provider gives notice (90 days), check what season
    we'll be in when they leave. If future season needs LOWER staffing,
    DON'T backfill — let natural attrition step us down.
    
    Example:
      December: Provider gives notice
      Lead time: 90 days (3 months)
      Departure: March
      Current target (Dec): 3.33 (winter)
      Future target (Mar): 2.33 (base)
      Current FTE: 3.30
      Decision: DON'T backfill (we're already above March target of 2.33)
    """
    
    today = datetime.today()
    year0 = today.year
    dates = pd.date_range(start=datetime(year0, 1, 1), periods=N_MONTHS, freq="MS")
    months = [int(d.month) for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    lead_months = lead_days_to_months(int(params.lead_days))
    monthly_turnover = float(params.annual_turnover) / 12.0
    fill_p = max(min(float(params.fill_probability), 1.0), 0.0)

    # Demand
    base_year0 = float(params.visits)
    base_year1 = base_year0 * (1.0 + float(params.annual_growth))
    base_year2 = base_year1 * (1.0 + float(params.annual_growth))

    visits_curve_base = compute_visits_curve(months, base_year0, base_year1, base_year2, float(params.seasonality_pct))
    visits_curve_flu = apply_flu_uplift(visits_curve_base, months, set(params.flu_months), float(params.flu_uplift_pct))
    visits_peak = [float(v) * float(params.peak_factor) for v in visits_curve_flu]

    role_mix = compute_role_mix_ratios(base_year1, staffing_model=model)

    # Target FTE by month
    def is_winter_month(month: int) -> bool:
        anchor = int(params.winter_anchor_month)
        end = int(params.winter_end_month)
        if anchor <= end:
            return (month >= anchor) or (month <= end)
        else:
            return (month >= anchor) or (month <= end)

    def target_fte_for_month(month: int) -> float:
        return float(policy.winter_fte) if is_winter_month(month) else float(policy.base_fte)

    def ramp_factor(age_months: int) -> float:
        rm = max(int(params.ramp_months), 0)
        if rm <= 0:
            return 1.0
        if int(age_months) < rm:
            return max(min(float(params.ramp_productivity), 1.0), 0.10)
        return 1.0

    # Initialize cohorts
    cohorts: list[dict[str, Any]] = [{"fte": float(policy.base_fte), "age": 9999}]
    pipeline: list[dict[str, Any]] = []

    paid_arr = []
    eff_arr = []
    hires_visible_arr = []
    hire_reason_arr = []
    hire_post_month_arr = []
    target_arr = []

    # Single-pass simulation with seasonal-aware backfill
    for t in range(N_MONTHS):
        current_month = int(months[t])
        
        # Apply turnover
        for c in cohorts:
            c["fte"] = max(float(c["fte"]) * (1.0 - monthly_turnover), 0.0)

        # Add hires arriving this month
        hires_this_month = [h for h in pipeline if h["arrive_month"] == t]
        total_hired = sum(h["fte"] for h in hires_this_month)
        
        if total_hired > 1e-9:
            cohorts.append({"fte": total_hired, "age": 0})
        
        # Calculate current FTE
        current_paid = sum(c["fte"] for c in cohorts)
        current_eff = sum(c["fte"] * ramp_factor(c["age"]) for c in cohorts)

        # ============================================================
        # SEASONAL-AWARE BACKFILL LOGIC (THE KEY FIX)
        # ============================================================
        if t + lead_months < N_MONTHS and current_month not in params.freeze_months:
            future_month_idx = t + lead_months
            future_month_num = int(months[future_month_idx])
            target_future = target_fte_for_month(future_month_num)
            
            # Project current FTE forward with turnover
            projected_paid = current_paid * ((1.0 - monthly_turnover) ** lead_months)
            
            # Add hires already in pipeline
            pipeline_for_future = sum(h["fte"] for h in pipeline if h["arrive_month"] == future_month_idx)
            projected_paid += pipeline_for_future
            
            # ★ KEY DECISION: Only hire if projected < target FOR THAT FUTURE SEASON ★
            # This prevents backfilling winter departures if we're stepping into base season
            if projected_paid < target_future - 0.05:
                need = target_future - projected_paid
                hire_amount = need * fill_p
                
                season_label = "winter" if is_winter_month(future_month_num) else "base"
                reason = f"Target {target_future:.2f} ({season_label}) for {month_name(future_month_num)}"
                
                pipeline.append({
                    "arrive_month": future_month_idx,
                    "fte": hire_amount,
                    "reason": reason,
                    "posted_month": current_month
                })

        # Age cohorts
        for c in cohorts:
            c["age"] += 1

        # Record state
        paid_arr.append(current_paid)
        eff_arr.append(current_eff)
        
        if hires_this_month:
            hires_visible_arr.append(total_hired)
            hire_reason_arr.append(" | ".join(h["reason"] for h in hires_this_month))
            hire_post_month_arr.append(hires_this_month[0]["posted_month"])
        else:
            hires_visible_arr.append(0.0)
            hire_reason_arr.append("")
            hire_post_month_arr.append(None)
        
        target_arr.append(target_fte_for_month(current_month))

    # Convert to numpy and calculate metrics (same as before)
    perm_paid = np.array(paid_arr, dtype=float)
    perm_eff = np.array(eff_arr, dtype=float)
    v_peak = np.array(visits_peak, dtype=float)
    v_avg = np.array(visits_curve_flu, dtype=float)
    dim = np.array(days_in_month, dtype=float)
    target_policy = np.array(target_arr, dtype=float)

    vph = max(float(params.visits_per_provider_hour), 1e-6)
    req_provider_hours_per_day = v_peak / vph
    req_eff_fte_needed = (req_provider_hours_per_day * float(params.days_open_per_week)) / max(float(params.fte_hours_week), 1e-6)

    prov_day_equiv_perm = np.array(
        [provider_day_equiv_from_fte(f, params.hours_week, params.fte_hours_week) for f in perm_eff],
        dtype=float,
    )
    load_pre = v_peak / np.maximum(prov_day_equiv_perm, 1e-6)

    flex_fte = np.zeros(N_MONTHS, dtype=float)
    load_post = np.zeros(N_MONTHS, dtype=float)

    for i in range(N_MONTHS):
        gap_fte = max(float(req_eff_fte_needed[i]) - float(perm_eff[i]), 0.0)
        flex_used = min(gap_fte, float(params.flex_max_fte_per_month))
        flex_fte[i] = flex_used

        prov_day_equiv_total = provider_day_equiv_from_fte(
            float(perm_eff[i] + flex_used), params.hours_week, params.fte_hours_week
        )
        load_post[i] = float(v_peak[i]) / max(prov_day_equiv_total, 1e-6)

    residual_gap_fte = np.maximum(req_eff_fte_needed - (perm_eff + flex_fte), 0.0)
    provider_day_gap_total = float(np.sum(residual_gap_fte * dim))

    est_visits_lost = provider_day_gap_total * float(params.visits_lost_per_provider_day_gap)
    est_margin_at_risk = est_visits_lost * float(params.net_revenue_per_visit)

    replacements_base = perm_paid * monthly_turnover
    repl_mult = np.ones(N_MONTHS, dtype=float)
    repl_mult = np.where(load_post > float(params.budgeted_pppd), float(params.turnover_yellow_mult), repl_mult)
    repl_mult = np.where(load_post > float(params.red_start_pppd), float(params.turnover_red_mult), repl_mult)
    turnover_replacement_cost = float(np.sum(replacements_base * float(params.provider_replacement_cost) * repl_mult))

    annual_swb_all, total_swb_dollars, total_visits = annual_swb_per_visit_from_supply(
        list(perm_paid), list(flex_fte), list(v_avg), list(days_in_month),
        float(params.fte_hours_week), role_mix, params.hourly_rates,
        float(params.benefits_load_pct), float(params.ot_sick_pct), float(params.bonus_pct),
        float(params.physician_supervision_hours_per_month), float(params.supervisor_hours_per_month)
    )

    green_cap = float(params.budgeted_pppd)
    red_start = float(params.red_start_pppd)
    months_yellow = int(np.sum((load_post > green_cap + 1e-9) & (load_post <= red_start + 1e-9)))
    months_red = int(np.sum(load_post > red_start + 1e-9))
    peak_load_post = float(np.max(load_post))
    peak_load_perm_only = float(np.max(load_pre))

    perm_total_fte_months = float(np.sum(perm_eff * dim))
    flex_total_fte_months = float(np.sum(flex_fte * dim))
    flex_share = flex_total_fte_months / max(perm_total_fte_months + flex_total_fte_months, 1e-9)

    hours_per_fte_per_day = float(params.fte_hours_week) / max(float(params.days_open_per_week), 1e-6)
    supplied_total_hours_per_day = (perm_eff + flex_fte) * hours_per_fte_per_day
    utilization = req_provider_hours_per_day / np.maximum(supplied_total_hours_per_day, 1e-9)

    yellow_excess = np.maximum(load_post - green_cap, 0.0)
    red_excess = np.maximum(load_post - red_start, 0.0)
    burnout_penalty = float(np.sum((yellow_excess ** 1.2) * dim) + 3.0 * np.sum((red_excess ** 2.0) * dim))

    perm_green_violation = float(np.max(np.maximum(load_pre - green_cap, 0.0)))
    perm_green_months = int(np.sum(load_pre > green_cap + 1e-9))

    total_score = (
        float(total_swb_dollars) + float(turnover_replacement_cost) +
        float(est_margin_at_risk) + 2_000.0 * burnout_penalty
    )

    # Build ledger
    rows: list[dict[str, Any]] = []
    for i in range(N_MONTHS):
        lab = dates[i].strftime("%Y-%b")
        month_visits = float(v_avg[i]) * float(days_in_month[i])
        month_net_contrib = month_visits * float(params.net_revenue_per_visit)

        month_swb_per_visit, month_swb_dollars, _ = annual_swb_per_visit_from_supply(
            [float(perm_paid[i])], [float(flex_fte[i])], [float(v_avg[i])], [int(days_in_month[i])],
            float(params.fte_hours_week), role_mix, params.hourly_rates,
            float(params.benefits_load_pct), float(params.ot_sick_pct), float(params.bonus_pct),
            float(params.physician_supervision_hours_per_month), float(params.supervisor_hours_per_month)
        )

        gap_weight = float(residual_gap_fte[i] * dim[i])
        gap_total = float(np.sum(residual_gap_fte * dim))
        month_access_risk = float(est_margin_at_risk) * (gap_weight / max(gap_total, 1e-9))
        month_ebitda_proxy = month_net_contrib - month_swb_dollars - month_access_risk

        rows.append({
            "Month": lab,
            "Visits/Day (avg)": float(v_avg[i]),
            "Total Visits (month)": month_visits,
            "SWB Dollars (month)": month_swb_dollars,
            "SWB/Visit (month)": month_swb_per_visit,
            "EBITDA Proxy (month)": month_ebitda_proxy,
            "Permanent FTE (Paid)": float(perm_paid[i]),
            "Permanent FTE (Effective)": float(perm_eff[i]),
            "Flex FTE Used": float(flex_fte[i]),
            "Required Provider FTE (effective)": float(req_eff_fte_needed[i]),
            "Utilization (Req/Supplied)": float(utilization[i]),
            "Load PPPD (post-flex)": float(load_post[i]),
            "Hires Visible (FTE)": float(hires_visible_arr[i]),
            "Hire Reason": hire_reason_arr[i],
            "Target FTE (policy)": float(target_policy[i]),
        })

    ledger = pd.DataFrame(rows)

    ledger["Year"] = ledger["Month"].str.slice(0, 4).astype(int)
    annual = ledger.groupby("Year", as_index=False).agg(
        Visits=("Total Visits (month)", "sum"),
        SWB_Dollars=("SWB Dollars (month)", "sum"),
        EBITDA_Proxy=("EBITDA Proxy (month)", "sum"),
        Avg_Perm_Paid_FTE=("Permanent FTE (Paid)", "mean"),
        Min_Perm_Paid_FTE=("Permanent FTE (Paid)", "min"),
        Max_Perm_Paid_FTE=("Permanent FTE (Paid)", "max"),
        Avg_Utilization=("Utilization (Req/Supplied)", "mean"),
    )
    annual["SWB_per_Visit"] = annual["SWB_Dollars"] / annual["Visits"].clip(lower=1.0)

    target = float(params.target_swb_per_visit)
    tol = float(params.swb_tolerance)
    annual["SWB_Deviation"] = np.maximum(np.abs(annual["SWB_per_Visit"] - target) - tol, 0.0)
    swb_band_penalty = float(np.sum((annual["SWB_Deviation"] ** 2) * 1_500_000.0))

    perm_green_penalty = 0.0
    if params.require_perm_under_green_no_flex:
        perm_green_penalty = perm_green_violation * 1_500_000.0 + perm_green_months * 150_000.0

    flex_penalty = max(flex_share - 0.10, 0.0) ** 2 * 2_000_000.0
    total_score += swb_band_penalty + perm_green_penalty + flex_penalty

    total_net_contrib = float(total_visits) * float(params.net_revenue_per_visit)
    ebitda_proxy_annual = total_net_contrib - float(total_swb_dollars) - turnover_replacement_cost - est_margin_at_risk

    return {
        "dates": list(dates),
        "months": months,
        "perm_paid": list(perm_paid),
        "perm_eff": list(perm_eff),
        "req_eff_fte_needed": list(req_eff_fte_needed),
        "utilization": list(utilization),
        "load_post": list(load_post),
        "annual_swb_per_visit": annual_swb_all,
        "total_swb_dollars": total_swb_dollars,
        "flex_share": flex_share,
        "months_red": months_red,
        "peak_load_post": peak_load_post,
        "ebitda_proxy_annual": ebitda_proxy_annual,
        "score": total_score,
        "ledger": ledger.drop(columns=["Year"]),
        "annual_summary": annual,
        "target_policy": list(target_policy),
    }



def recommend_policy(params: ModelParams, base_min: float, base_max: float, base_step: float,
                     winter_delta_max: float, winter_step: float) -> dict[str, Any]:
    candidates = []
    best = None
    for b in np.arange(base_min, base_max + 1e-9, base_step):
        for w in np.arange(b, b + winter_delta_max + 1e-9, winter_step):
            pol = Policy(base_fte=float(b), winter_fte=float(w))
            res = simulate_policy_seasonal_backfill(params, pol)
            candidates.append({
                "Base_FTE": float(b), "Winter_FTE": float(w), "Score": res["score"],
                "SWB_per_Visit": res["annual_swb_per_visit"], "EBITDA": res["ebitda_proxy_annual"],
            })
            if best is None or res["score"] < best["res"]["score"]:
                best = {"policy": pol, "res": res}
    frontier = pd.DataFrame(candidates).sort_values("Score").reset_index(drop=True)
    return {"best": best, "frontier": frontier}


@st.cache_data(show_spinner=False)
def cached_recommend(params_dict: dict, base_min: float, base_max: float, base_step: float,
                     winter_delta_max: float, winter_step: float) -> dict:
    params = ModelParams(**params_dict)
    return recommend_policy(params, base_min, base_max, base_step, winter_delta_max, winter_step)


@st.cache_data(show_spinner=False)
def cached_simulate(params_dict: dict, base_fte: float, winter_fte: float) -> dict:
    params = ModelParams(**params_dict)
    return simulate_policy_seasonal_backfill(params, Policy(base_fte=base_fte, winter_fte=winter_fte))


# SIDEBAR
with st.sidebar:
    st.markdown('<div class="fix"><b>🎯 SEASONAL BACKFILL</b><br>90-day notice → check future season<br>Don\'t backfill if stepping down</div>', unsafe_allow_html=True)
    
    st.header("Demand")
    visits = st.number_input("Avg Visits/Day", 1.0, value=36.0, step=1.0)
    annual_growth = st.number_input("Annual Growth %", 0.0, value=10.0, step=1.0) / 100.0
    peak_factor = st.slider("Peak factor", 1.0, 1.5, 1.2, 0.01)
    seasonality_pct = st.number_input("Seasonality %", 0.0, value=20.0, step=5.0) / 100.0
    flu_uplift_pct = st.number_input("Flu uplift %", 0.0, value=0.0, step=5.0) / 100.0
    flu_months = st.multiselect("Flu months", MONTH_OPTIONS, default=[("Oct",10),("Nov",11),("Dec",12),("Jan",1),("Feb",2)])
    flu_months_set = {m for _, m in flu_months} if flu_months else set()
    visits_per_provider_hour = st.slider("Visits/prov-hour", 2.0, 4.0, 3.0, 0.1)

    st.header("Clinic")
    hours_week = st.number_input("Hours/Week", 1.0, value=84.0, step=1.0)
    days_open_per_week = st.number_input("Days/Week", 1.0, 7.0, value=7.0, step=1.0)
    fte_hours_week = st.number_input("FTE Hours/Week", 1.0, value=36.0, step=1.0)

    st.header("Policy")
    min_perm_providers_per_day = st.number_input("Min prov/day", 0.0, value=1.0, step=0.25)
    allow_prn_override = st.checkbox("Allow Base < min", value=False)
    require_perm_under_green_no_flex = st.checkbox("Perm ≤ Green", value=True)

    st.header("Workforce")
    annual_turnover = st.number_input("Turnover %", 0.0, value=16.0, step=1.0) / 100.0
    lead_days = st.number_input("Lead days", 0, value=90, step=10, help="90-day notice period")
    ramp_months = st.slider("Ramp months", 0, 6, 1)
    ramp_productivity = st.slider("Ramp prod %", 30, 100, 75, 5) / 100.0
    fill_probability = st.slider("Fill prob %", 0, 100, 85, 5) / 100.0

    st.header("Risk Bands")
    budgeted_pppd = st.number_input("Green PPPD", 5.0, value=36.0, step=1.0)
    yellow_max_pppd = st.number_input("Yellow PPPD", 5.0, value=42.0, step=1.0)
    red_start_pppd = st.number_input("Red PPPD", 5.0, value=45.0, step=1.0)

    st.header("Season")
    winter_anchor_month = st.selectbox("Winter anchor", MONTH_OPTIONS, index=11)
    winter_anchor_month_num = int(winter_anchor_month[1])
    winter_end_month = st.selectbox("Winter end", MONTH_OPTIONS, index=1)
    winter_end_month_num = int(winter_end_month[1])
    freeze_months = st.multiselect("Freeze months", MONTH_OPTIONS, default=flu_months if flu_months else [])
    freeze_months_set = {m for _, m in freeze_months} if freeze_months else set()

    st.header("Flex")
    flex_max_fte_per_month = st.slider("Max flex/mo", 0.0, 10.0, 2.0, 0.25)
    flex_cost_multiplier = st.slider("Flex cost mult", 1.0, 2.0, 1.25, 0.05)

    st.header("Finance")
    target_swb_per_visit = st.number_input("Target SWB/Visit", 0.0, value=85.0, step=1.0)
    swb_tolerance = st.number_input("SWB tolerance", 0.0, value=2.0, step=0.5)
    net_revenue_per_visit = st.number_input("Net contrib/visit", 0.0, value=140.0, step=5.0)
    visits_lost_per_provider_day_gap = st.number_input("Visits lost/prov-day", 0.0, value=18.0, step=1.0)
    provider_replacement_cost = st.number_input("Replacement cost", 0.0, value=75000.0, step=5000.0)
    turnover_yellow_mult = st.slider("Yellow mult", 1.0, 3.0, 1.3, 0.05)
    turnover_red_mult = st.slider("Red mult", 1.0, 5.0, 2.0, 0.1)

    st.subheader("Wages")
    benefits_load_pct = st.number_input("Benefits %", 0.0, value=30.0, step=1.0) / 100.0
    bonus_pct = st.number_input("Bonus %", 0.0, value=10.0, step=1.0) / 100.0
    ot_sick_pct = st.number_input("OT+Sick %", 0.0, value=4.0, step=0.5) / 100.0
    physician_hr = st.number_input("Physician $/hr", 0.0, value=135.79, step=1.0)
    apc_hr = st.number_input("APP $/hr", 0.0, value=62.0, step=1.0)
    ma_hr = st.number_input("MA $/hr", 0.0, value=24.14, step=0.5)
    psr_hr = st.number_input("PSR $/hr", 0.0, value=21.23, step=0.5)
    rt_hr = st.number_input("RT $/hr", 0.0, value=31.36, step=0.5)
    supervisor_hr = st.number_input("Supervisor $/hr", 0.0, value=28.25, step=0.5)
    physician_supervision_hours_per_month = st.number_input("Phys hrs/mo", 0.0, value=0.0, step=1.0)
    supervisor_hours_per_month = st.number_input("Sup hrs/mo", 0.0, value=0.0, step=1.0)

    st.header("Optimizer")
    min_base_req = fte_required_for_min_perm_providers_per_day(min_perm_providers_per_day, hours_week, fte_hours_week)
    base_min = st.number_input("Base min", 0.0, value=0.0 if allow_prn_override else min_base_req, step=0.25)
    base_max = st.number_input("Base max", 0.0, value=6.0, step=0.25)
    base_step = st.select_slider("Base step", [0.10, 0.25, 0.50], value=0.25)
    winter_delta_max = st.number_input("Winter uplift max", 0.0, value=2.0, step=0.25)
    winter_step = st.select_slider("Winter step", [0.10, 0.25, 0.50], value=0.25)

    mode = st.radio("Mode", ["Recommend + What-If", "What-If only"], index=0)
    run_recommender = st.button("🏁 Run Recommender", use_container_width=True)

hourly_rates = {"physician": physician_hr, "apc": apc_hr, "ma": ma_hr, "psr": psr_hr, "rt": rt_hr, "supervisor": supervisor_hr}

params = ModelParams(
    visits=visits, annual_growth=annual_growth, seasonality_pct=seasonality_pct,
    flu_uplift_pct=flu_uplift_pct, flu_months=flu_months_set, peak_factor=peak_factor,
    visits_per_provider_hour=visits_per_provider_hour, hours_week=hours_week,
    days_open_per_week=days_open_per_week, fte_hours_week=fte_hours_week,
    annual_turnover=annual_turnover, lead_days=lead_days, ramp_months=ramp_months,
    ramp_productivity=ramp_productivity, fill_probability=fill_probability,
    winter_anchor_month=winter_anchor_month_num, winter_end_month=winter_end_month_num,
    freeze_months=freeze_months_set, budgeted_pppd=budgeted_pppd,
    yellow_max_pppd=yellow_max_pppd, red_start_pppd=red_start_pppd,
    flex_max_fte_per_month=flex_max_fte_per_month, flex_cost_multiplier=flex_cost_multiplier,
    target_swb_per_visit=target_swb_per_visit, swb_tolerance=swb_tolerance,
    net_revenue_per_visit=net_revenue_per_visit, visits_lost_per_provider_day_gap=visits_lost_per_provider_day_gap,
    provider_replacement_cost=provider_replacement_cost, turnover_yellow_mult=turnover_yellow_mult,
    turnover_red_mult=turnover_red_mult, hourly_rates=hourly_rates,
    benefits_load_pct=benefits_load_pct, ot_sick_pct=ot_sick_pct, bonus_pct=bonus_pct,
    physician_supervision_hours_per_month=physician_supervision_hours_per_month,
    supervisor_hours_per_month=supervisor_hours_per_month,
    min_perm_providers_per_day=min_perm_providers_per_day,
    allow_prn_override=allow_prn_override,
    require_perm_under_green_no_flex=require_perm_under_green_no_flex,
    utilization_target=0.90, recovery_months=6, perm_step_epsilon=0.25,
    idle_grace_months=3, idle_epsilon_eff_fte=0.25, elasticity_cap=1.10,
    elasticity_stabilization_months=3, _v=MODEL_VERSION
)

params_dict = {**params.__dict__, "_v": MODEL_VERSION}

# RUN RECOMMENDER
if mode == "Recommend + What-If" and run_recommender:
    with st.spinner("Running..."):
        rec = cached_recommend(params_dict, base_min, base_max, base_step, winter_delta_max, winter_step)
    st.session_state.rec_policy = rec["best"]["policy"]
    st.session_state["what_base_fte"] = float(rec["best"]["policy"].base_fte)
    st.session_state["what_winter_fte"] = float(rec["best"]["policy"].winter_fte)

rec_policy = st.session_state.rec_policy

# WHAT-IF
st.subheader("What-If Policy")
st.session_state.setdefault("what_base_fte", max(base_min, 1.0))
st.session_state.setdefault("what_winter_fte", st.session_state["what_base_fte"] + 1.0)

w1, w2 = st.columns(2)
with w1:
    what_base = st.number_input("Base FTE", 0.0 if allow_prn_override else min_base_req,
                                value=st.session_state["what_base_fte"], step=0.25, key="what_base_fte")
with w2:
    what_winter = st.number_input("Winter FTE", what_base,
                                  value=max(st.session_state.get("what_winter_fte", what_base), what_base),
                                  step=0.25, key="what_winter_fte")

R = cached_simulate(params_dict, what_base, what_winter)

# RATCHET DETECTION
st.markdown("---")
st.header("🔍 Ratchet Detection")
annual = R["annual_summary"]
if len(annual) >= 2:
    min_y1, min_y2 = annual.loc[0, "Min_Perm_Paid_FTE"], annual.loc[1, "Min_Perm_Paid_FTE"]
    min_y3 = annual.loc[2, "Min_Perm_Paid_FTE"] if len(annual) >= 3 else min_y2
    drift_y2, drift_y3 = min_y2 - min_y1, min_y3 - min_y2
    
    if abs(drift_y2) < 0.2 and abs(drift_y3) < 0.2:
        st.markdown(f'<div class="ok"><b>✅ NO RATCHET!</b><br>Y1: {min_y1:.2f} → Y2: {min_y2:.2f} (Δ{drift_y2:+.2f}) → Y3: {min_y3:.2f} (Δ{drift_y3:+.2f})<br>Policy base: {what_base:.2f}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn"><b>Minor drift detected</b><br>Y1: {min_y1:.2f} → Y2: {min_y2:.2f} (Δ{drift_y2:+.2f}) → Y3: {min_y3:.2f} (Δ{drift_y3:+.2f})</div>', unsafe_allow_html=True)

# SCORECARD
st.markdown("---")
st.header("Scorecard")
swb_y1 = annual.loc[0, "SWB_per_Visit"]
ebitda_y1, ebitda_y3 = annual.loc[0, "EBITDA_Proxy"], annual.loc[2, "EBITDA_Proxy"]
util_y1 = annual.loc[0, "Avg_Utilization"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Base / Winter", f"{what_base:.2f} / {what_winter:.2f}")
c2.metric("SWB/Visit Y1", f"${swb_y1:.2f}")
c3.metric("EBITDA Y1/Y3", f"${ebitda_y1/1000:.0f}K / ${ebitda_y3/1000:.0f}K")
c4.metric("Util Y1", f"{util_y1*100:.0f}%")

# CHARTS
st.markdown("---")
st.header("3-Year Horizon")
dates = R["dates"]
perm_paid, target_pol = np.array(R["perm_paid"]), np.array(R["target_policy"])
req_eff, load_post = np.array(R["req_eff_fte_needed"]), np.array(R["load_post"])
tick_idx = list(range(0, len(dates), 3))

fig1, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, perm_paid, linewidth=2.2, color=BRAND_BLACK, label="Paid FTE", marker='o', markersize=3)
ax.plot(dates, target_pol, linewidth=2.2, color='#1976d2', label="Target", linestyle='--', marker='s', markersize=3)
ax.plot(dates, req_eff, linewidth=2.0, color='#4caf50', label="Required", linestyle=':', marker='x', markersize=3)
ax.set_title("Supply vs Target (Seasonal-Aware Backfill)", fontsize=13, fontweight="bold")
ax.set_ylabel("Provider FTE", fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.3)
ax.set_xticks([dates[i] for i in tick_idx])
ax.set_xticklabels([dates[i].strftime("%Y-%b") for i in tick_idx], rotation=0, fontsize=9)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(dates, load_post, linewidth=2.2, color=BRAND_GOLD, marker='o', markersize=3)
ax2.axhline(budgeted_pppd, color='green', linestyle=':', label=f"Green ({budgeted_pppd:.0f})")
ax2.axhline(red_start_pppd, color='red', linestyle=':', label=f"Red ({red_start_pppd:.0f})")
ax2.set_title("Monthly Load (PPPD)", fontsize=13, fontweight="bold")
ax2.set_ylabel("PPPD", fontweight="bold")
ax2.grid(axis="y", linestyle=":", alpha=0.3)
ax2.set_xticks([dates[i] for i in tick_idx])
ax2.set_xticklabels([dates[i].strftime("%Y-%b") for i in tick_idx], rotation=0, fontsize=9)
ax2.legend(frameon=False)
plt.tight_layout()
st.pyplot(fig2)

# TABLES
st.markdown("---")
st.header("Annual Summary")
st.dataframe(annual.style.format({
    "Visits": "{:,.0f}", "SWB_per_Visit": "${:,.2f}", "EBITDA_Proxy": "${:,.0f}",
    "Min_Perm_Paid_FTE": "{:.2f}", "Max_Perm_Paid_FTE": "{:.2f}", "Avg_Utilization": "{:.0%}",
}), hide_index=True, use_container_width=True)

st.markdown("---")
st.header("Ledger")
st.dataframe(R["ledger"], hide_index=True, use_container_width=True)

# EXPORTS
st.markdown("---")
c1, c2 = st.columns(2)
c1.download_button("⬇️ Supply Chart", fig_to_png_bytes(fig1), "supply.png", "image/png")
c2.download_button("⬇️ Ledger CSV", df_to_csv_bytes(R["ledger"]), "ledger.csv", "text/csv")

st.markdown('<div class="fix"><b>🎯 SEASONAL-AWARE BACKFILL LOGIC</b><br>When provider gives 90-day notice in December, we look ahead to March. If March needs lower staffing (base season), we DON\'T backfill. This allows natural attrition to step FTE down from winter → base.</div>', unsafe_allow_html=True)
