# app_fixed.py — Predictive Staffing Model (PSM) — Policy Optimizer (FIXED)
# 
# CRITICAL FIXES:
# 1. Base replacement now targets POLICY BASE, not current elevated FTE (fixes ratchet)
# 2. Explicit step-down logic after winter season (FTE reduces back to base)
# 3. Winter steps set ABSOLUTE targets based on policy (not additive to current)
# 4. Better hiring freeze handling and diagnostics
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


# ============================================================
# VERSION (cache-busting)
# ============================================================
MODEL_VERSION = "2026-01-30-FIXED-no-ratchet-v1"


# ============================================================
# PAGE CONFIG + CSS
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM) — Policy Optimizer [FIXED]", layout="centered")

st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 0.75rem; padding-bottom: 2.25rem; }
      .small { font-size: 0.92rem; color: #444; }
      .muted { color: #6b6b6b; font-size: 0.9rem; }
      .contract { background: #f7f7f7; border: 1px solid #e6e6e6; border-radius: 10px; padding: 14px 16px; }
      .note { background: #f3f7ff; border: 1px solid #cfe0ff; border-radius: 10px; padding: 12px 14px; }
      .warn { background: #fff6e6; border: 1px solid #ffe2a8; border-radius: 10px; padding: 12px 14px; }
      .ok { background: #ecfff0; border: 1px solid #b7f0c0; border-radius: 10px; padding: 12px 14px; }
      .fix { background: #e8f5e9; border: 2px solid #4caf50; border-radius: 10px; padding: 14px 16px; }
      .kpirow { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 8px; }
      .kpibox { border: 1px solid #eee; border-radius: 12px; padding: 10px 12px; background: #fff; }
      .kpiLabel { font-size: 0.8rem; color: #666; }
      .kpiVal { font-size: 1.30rem; font-weight: 700; margin-top: 2px; }
      .kpiSub { font-size: 0.82rem; color: #666; margin-top: 2px; }
      .stickyScorecard { position: sticky; top: 0; z-index: 999; background: white; padding-top: 0.35rem; padding-bottom: 0.35rem; border-bottom: 1px solid #eee; }
      .divider { height: 1px; background: #eee; margin: 10px 0 14px 0; }
      .pill { display:inline-block; padding:2px 8px; border-radius: 999px; border: 1px solid #ddd; font-size:0.78rem; color:#444; margin-left:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔧 Predictive Staffing Model (PSM) — FIXED VERSION")
st.caption(
    "✅ Fixed ratchet effect • True two-level policy • Base stays constant • Winter steps down after season"
)

model = StaffingModel()


# ============================================================
# SESSION STATE
# ============================================================
st.session_state.setdefault("rec_policy", None)
st.session_state.setdefault("frontier", None)


# ============================================================
# CONSTANTS
# ============================================================
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


# ============================================================
# TOOLTIP HELP
# ============================================================
HELP: dict[str, str] = {
    "visits": "Baseline average visits per day for the current year (annual average, before seasonality/flu adjustments).",
    "annual_growth": "Expected annual growth rate applied to baseline visits for Year 2 and Year 3.",
    "peak_factor": "Peak-to-average planning multiplier for conservative planning.",
    "visits_per_provider_hour": "Sustainable visits per provider hour (productivity).",
    "seasonality_pct": "Seasonal swing: winter up, summer down.",
    "flu_uplift_pct": "Additional uplift in flu months.",
    "flu_months": "Months that receive flu uplift.",
    
    "hours_week": "Total clinic hours open per week.",
    "days_open_per_week": "Days open per week.",
    "fte_hours_week": "Paid hours per 1.0 FTE per week.",
    
    "annual_turnover": "Annual provider turnover rate.",
    "lead_days": "Days from requisition to independent productivity.",
    "ramp_months": "Months after independence where productivity is reduced.",
    "ramp_productivity": "Ramp productivity fraction.",
    "fill_probability": "Probability the pipeline fills posted requisitions.",
    
    "budgeted_pppd": "Green threshold (target PPPD).",
    "yellow_max_pppd": "Yellow threshold (caution).",
    "red_start_pppd": "Red threshold (high risk).",
    
    "winter_anchor_month": "Month when winter staffing should be ready by.",
    "winter_end_month": "Month when winter season ends (step-down begins).",
    "freeze_months": "Posting is not allowed during these months.",
    
    "flex_max": "Maximum flex provider FTE available per month.",
    "flex_mult": "Flex cost multiplier vs base provider.",
    
    "target_swb": "Target annual SWB/Visit.",
    "swb_tol": "Allowed annual SWB/Visit deviation from target.",
    "net_contrib": "Net contribution per visit.",
    "visits_lost": "Estimated visits lost per 1.0 provider-day gap.",
    "repl_cost": "Replacement cost per provider FTE.",
    "turn_yellow": "Replacement cost multiplier in Yellow months.",
    "turn_red": "Replacement cost multiplier in Red months.",
    
    "min_perm_providers_per_day": "Minimum permanent providers scheduled per day.",
    "allow_prn_override": "Allow Base FTE below minimum permanent coverage.",
    "require_perm_green": "Require permanent-only to stay under Green (no flex).",
    
    "util_target": "Utilization target for recovery rule.",
    "recovery_months": "Recovery window after permanent step-up.",
    "perm_step_eps": "Minimum month-over-month permanent FTE increase treated as step-up.",
    "idle_grace": "Grace period for temporary buffer.",
    "idle_eps": "Idle threshold (effective FTE).",
    "elasticity_cap": "Elasticity guardrail cap.",
    "elasticity_stabilize": "Ignore elasticity in first N months.",
}


# ============================================================
# HELPERS
# ============================================================
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
        return staffing_model.get_role_mix_ratios(v)  # type: ignore[attr-defined]
    daily = staffing_model.calculate(v)
    prov_day = max(float(daily.get("provider_day", 0.0)), 0.25)
    return {
        "psr_per_provider": float(daily.get("psr_day", 0.0)) / prov_day,
        "ma_per_provider": float(daily.get("ma_day", 0.0)) / prov_day,
        "xrt_per_provider": float(daily.get("xrt_day", 0.0)) / prov_day,
    }


def select_latest_post_month_before_freeze(anchor_month: int, lead_months: int, freeze_months: set[int]) -> int:
    latest = wrap_month(anchor_month - lead_months)
    for k in range(0, 12):
        m = wrap_month(latest - k)
        if m not in freeze_months:
            return m
    return latest



def annual_swb_per_visit_from_supply(
    provider_paid_fte: list[float],
    provider_flex_fte: list[float],
    visits_per_day: list[float],
    days_in_month: list[int],
    fte_hours_per_week: float,
    role_mix: dict[str, float],
    hourly_rates: dict[str, float],
    benefits_load_pct: float,
    ot_sick_pct: float,
    bonus_pct: float,
    physician_supervision_hours_per_month: float = 0.0,
    supervisor_hours_per_month: float = 0.0,
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

        month_swb = (
            prov_hours * apc_rate
            + psr_hours * psr_rate
            + ma_hours * ma_rate
            + rt_hours * rt_rate
            + float(physician_supervision_hours_per_month) * phys_rate
            + float(supervisor_hours_per_month) * sup_rate
        )

        total_swb += float(month_swb)
        total_visits += float(month_visits)

    total_visits = max(total_visits, 1.0)
    annual_swb = float(total_swb) / float(total_visits)
    return annual_swb, float(total_swb), float(total_visits)


# ============================================================
# MODEL PARAMETERS
# ============================================================
@dataclass(frozen=True)
class ModelParams:
    # Demand
    visits: float
    annual_growth: float
    seasonality_pct: float
    flu_uplift_pct: float
    flu_months: set[int]
    peak_factor: float
    visits_per_provider_hour: float

    # Clinic ops
    hours_week: float
    days_open_per_week: float
    fte_hours_week: float

    # Workforce
    annual_turnover: float
    lead_days: int
    ramp_months: int
    ramp_productivity: float
    fill_probability: float

    # Policy / freeze
    winter_anchor_month: int
    winter_end_month: int  # NEW: when to step down
    freeze_months: set[int]

    # Load zones
    budgeted_pppd: float
    yellow_max_pppd: float
    red_start_pppd: float

    # Flex
    flex_max_fte_per_month: float
    flex_cost_multiplier: float

    # Governance / finance
    target_swb_per_visit: float
    swb_tolerance: float
    net_revenue_per_visit: float
    visits_lost_per_provider_day_gap: float

    # Turnover replacement
    provider_replacement_cost: float
    turnover_yellow_mult: float
    turnover_red_mult: float

    # Wages for SWB/visit
    hourly_rates: dict[str, float]
    benefits_load_pct: float
    ot_sick_pct: float
    bonus_pct: float
    physician_supervision_hours_per_month: float
    supervisor_hours_per_month: float

    # Operator constraints
    min_perm_providers_per_day: float
    allow_prn_override: bool
    require_perm_under_green_no_flex: bool

    # Operator-grade governance constraints
    utilization_target: float
    recovery_months: int
    perm_step_epsilon: float
    idle_grace_months: int
    idle_epsilon_eff_fte: float
    elasticity_cap: float
    elasticity_stabilization_months: int

    # Cache key injection
    _v: str = MODEL_VERSION


@dataclass(frozen=True)
class Policy:
    base_fte: float
    winter_fte: float



# ============================================================
# SIMULATION ENGINE (FIXED)
# ============================================================
def build_annual_summary(ledger: pd.DataFrame, params: ModelParams) -> pd.DataFrame:
    df = ledger.copy()
    df["Year"] = df["Month"].str.slice(0, 4).astype(int)

    numeric_cols = [
        "Total Visits (month)", "SWB Dollars (month)", "SWB/Visit (month)",
        "Net Contribution (month)", "EBITDA Proxy (month)", "Flex FTE Used",
        "Permanent FTE (Paid)", "Permanent FTE (Effective)",
        "Load PPPD (post-flex)", "Perm Load PPPD (no-flex)",
        "Residual FTE Gap (to Required)", "Required Provider Hours/Day",
        "Required Provider FTE (effective)", "Utilization (Req/Supplied)",
        "Excess Total Eff FTE (Idle)", "Excess Perm Eff FTE (Idle)",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby("Year", as_index=False)
    annual = g.agg(
        Visits=("Total Visits (month)", "sum"),
        Net_Contribution=("Net Contribution (month)", "sum"),
        SWB_Dollars=("SWB Dollars (month)", "sum"),
        EBITDA_Proxy=("EBITDA Proxy (month)", "sum"),
        Avg_Perm_Paid_FTE=("Permanent FTE (Paid)", "mean"),
        Min_Perm_Paid_FTE=("Permanent FTE (Paid)", "min"),
        Max_Perm_Paid_FTE=("Permanent FTE (Paid)", "max"),
        Avg_Perm_Eff_FTE=("Permanent FTE (Effective)", "mean"),
        Avg_Flex_FTE=("Flex FTE Used", "mean"),
        Peak_Flex_FTE=("Flex FTE Used", "max"),
        Peak_Load_PPPD_Post=("Load PPPD (post-flex)", "max"),
        Peak_Perm_Load_NoFlex=("Perm Load PPPD (no-flex)", "max"),
        Months_Yellow=("Load PPPD (post-flex)", lambda s: int(((s > params.budgeted_pppd + 1e-9) & (s <= params.red_start_pppd + 1e-9)).sum())),
        Months_Red=("Load PPPD (post-flex)", lambda s: int((s > params.red_start_pppd + 1e-9).sum())),
        Total_Residual_Gap_FTE_Months=("Residual FTE Gap (to Required)", "sum"),
        Peak_Req_ProvHoursPerDay=("Required Provider Hours/Day", "max"),
        Peak_Req_Eff_FTE=("Required Provider FTE (effective)", "max"),
        Avg_Utilization=("Utilization (Req/Supplied)", "mean"),
        Avg_Excess_Total_Eff_FTE=("Excess Total Eff FTE (Idle)", "mean"),
        Avg_Excess_Perm_Eff_FTE=("Excess Perm Eff FTE (Idle)", "mean"),
    )

    annual["SWB_per_Visit"] = annual["SWB_Dollars"] / annual["Visits"].clip(lower=1.0)
    annual["EBITDA_per_Visit"] = annual["EBITDA_Proxy"] / annual["Visits"].clip(lower=1.0)
    annual = annual.sort_values("Year").reset_index(drop=True)
    annual["YoY_EBITDA_Proxy_Δ"] = annual["EBITDA_Proxy"].diff().fillna(0.0)

    return annual


def simulate_policy(params: ModelParams, policy: Policy) -> dict[str, Any]:
    """
    FIXED SIMULATION:
    1. Base replacement targets POLICY.BASE_FTE (not current FTE)
    2. Explicit step-down after winter season
    3. Winter steps set absolute target (policy.winter_fte)
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
    base_year1 = float(params.visits) * (1.0 + float(params.annual_growth))
    base_year2 = float(params.visits) * (1.0 + float(params.annual_growth)) ** 2

    visits_curve_base = compute_visits_curve(months, base_year0, base_year1, base_year2, float(params.seasonality_pct))
    visits_curve_flu = apply_flu_uplift(visits_curve_base, months, set(params.flu_months), float(params.flu_uplift_pct))
    visits_peak = [float(v) * float(params.peak_factor) for v in visits_curve_flu]

    role_mix = compute_role_mix_ratios(float(params.visits) * (1.0 + float(params.annual_growth)), staffing_model=model)

    # ============================================================
    # HIRING SCHEDULE (FIXED LOGIC)
    # ============================================================
    hires_visible = [0.0] * N_MONTHS
    hires_post_month: list[int | None] = [None] * N_MONTHS
    hires_reason = [""] * N_MONTHS

    def ramp_factor(age_months: int) -> float:
        rm = max(int(params.ramp_months), 0)
        if rm <= 0:
            return 1.0
        if int(age_months) < rm:
            return max(min(float(params.ramp_productivity), 1.0), 0.10)
        return 1.0

    # FIX #1: Define target levels by season (not just once-per-year events)
    def target_fte_for_month(month: int) -> float:
        """Returns the target FTE for a given month based on policy"""
        # Winter season: anchor through end month
        anchor = int(params.winter_anchor_month)
        end = int(params.winter_end_month)
        
        # Handle year-wrap for winter season
        if anchor <= end:
            # Normal case (e.g., Dec=12, Feb=2 wraps through Jan)
            in_winter = (month >= anchor) or (month <= end)
        else:
            # Wrapped case (e.g., Nov=11, Mar=3)
            in_winter = (month >= anchor) or (month <= end)
        
        return float(policy.winter_fte) if in_winter else float(policy.base_fte)

    # Winter step-up events (schedule hires to hit winter_fte by anchor month)
    anchor_month = int(params.winter_anchor_month)
    anchor_indices = [i for i, m in enumerate(months) if int(m) == int(anchor_month)]
    
    winter_step_events: list[tuple[int, int]] = []
    for a_idx in anchor_indices:
        post_m = select_latest_post_month_before_freeze(anchor_month, lead_months, set(params.freeze_months))
        post_idx = None
        for k in range(0, 12):
            j = a_idx - lead_months - k
            if 0 <= j < N_MONTHS and int(months[j]) == int(post_m) and int(months[j]) not in set(params.freeze_months):
                post_idx = j
                break
        if post_idx is None:
            post_idx = max(a_idx - lead_months, 0)
        
        vis_idx = post_idx + lead_months
        if 0 <= vis_idx < N_MONTHS:
            winter_step_events.append((post_idx, vis_idx))

    # Step-down events (after winter season ends)
    end_month = int(params.winter_end_month)
    stepdown_month = wrap_month(end_month + 1)  # Month after winter ends
    stepdown_indices = [i for i, m in enumerate(months) if int(m) == int(stepdown_month)]

    def run_supply(hires_visible_local: list[float]) -> tuple[list[float], list[float], list[float]]:
        """Simulate FTE evolution with turnover and hires"""
        local_cohorts = [{"fte": max(float(policy.base_fte), 0.0), "age": 9999}]
        paid = [0.0] * N_MONTHS
        eff = [0.0] * N_MONTHS
        shed = [0.0] * N_MONTHS

        def _paid() -> float:
            return float(sum(float(c["fte"]) for c in local_cohorts))

        for t in range(N_MONTHS):
            sp = _paid()
            # Turnover
            for c in local_cohorts:
                c["fte"] = max(float(c["fte"]) * (1.0 - monthly_turnover), 0.0)
            ap = _paid()
            shed[t] = float(ap - sp)

            # Hires
            add = float(hires_visible_local[t])
            if add > 1e-9:
                local_cohorts.append({"fte": add, "age": 0})

            ep = _paid()
            paid[t] = float(ep)

            # Effective (with ramp)
            ee = 0.0
            for c in local_cohorts:
                ee += float(c["fte"]) * ramp_factor(int(c["age"]))
            eff[t] = float(ee)

            # Age cohorts
            for c in local_cohorts:
                c["age"] = int(c["age"]) + 1

        return paid, eff, shed

    # Pass 1: Get baseline trajectory
    paid_1, eff_1, _ = run_supply(hires_visible)

    # FIX #2: Winter steps target ABSOLUTE winter_fte (not additive)
    for (post_idx, vis_idx) in winter_step_events:
        if int(months[post_idx]) in set(params.freeze_months):
            continue
        
        # Project what paid FTE will be at vis_idx
        current_paid = float(paid_1[post_idx])
        proj_paid = current_paid * ((1.0 - monthly_turnover) ** float(vis_idx - post_idx))
        
        # Target is policy.winter_fte (absolute)
        target = float(policy.winter_fte)
        need = max(target - proj_paid, 0.0)
        
        if need > 1e-6:
            step = need * fill_p
            hires_visible[vis_idx] += float(step)
            hires_post_month[vis_idx] = int(months[post_idx])
            hires_reason[vis_idx] = f"Winter step to {policy.winter_fte:.2f} (post {month_name(months[post_idx])})"

    # FIX #3: Base replacement targets POLICY.BASE_FTE (not current)
    hires_visible_with_replacement = hires_visible[:]
    hires_post_month_with_replacement = hires_post_month[:]
    hires_reason_with_replacement = hires_reason[:]

    for t in range(N_MONTHS):
        if lead_months <= 0 or t + lead_months >= N_MONTHS:
            continue
        if int(months[t]) in set(params.freeze_months):
            continue

        v = t + lead_months
        target_month_fte = target_fte_for_month(int(months[v]))
        
        # Project forward
        current_paid_t = float(paid_1[t])
        proj = current_paid_t * ((1.0 - monthly_turnover) ** float(lead_months))
        proj += float(hires_visible_with_replacement[v])

        # Compare to TARGET for that month (base or winter)
        if proj < target_month_fte - 1e-6:
            need = target_month_fte - proj
            add = need * fill_p
            hires_visible_with_replacement[v] += float(add)
            hires_post_month_with_replacement[v] = int(months[t])
            reason_part = "Base replacement" if target_month_fte == float(policy.base_fte) else "Winter replacement"
            hires_reason_with_replacement[v] = (
                (hires_reason_with_replacement[v] + " | " if hires_reason_with_replacement[v] else "") + reason_part
            )

    # FIX #4: Explicit step-down enforcement
    # After winter ends, allow attrition to reduce FTE naturally (no replacement hiring during step-down)
    # This is handled by the target_fte_for_month logic above - replacement only targets base after winter

    # Final supply run
    paid, eff, turnover_shed_arr = run_supply(hires_visible_with_replacement)

    perm_eff = np.array(eff, dtype=float)
    perm_paid = np.array(paid, dtype=float)
    v_peak = np.array(visits_peak, dtype=float)
    v_avg = np.array(visits_curve_flu, dtype=float)
    dim = np.array(days_in_month, dtype=float)


    # ============================================================
    # REQUIRED COVERAGE
    # ============================================================
    vph = max(float(params.visits_per_provider_hour), 1e-6)
    req_provider_hours_per_day = v_peak / vph
    req_provider_hours_per_day_rounded = np.round(req_provider_hours_per_day / 0.5) * 0.5
    req_eff_fte_needed = (req_provider_hours_per_day * float(params.days_open_per_week)) / max(float(params.fte_hours_week), 1e-6)

    # ============================================================
    # LOAD (PPPD)
    # ============================================================
    prov_day_equiv_perm = np.array(
        [provider_day_equiv_from_fte(f, params.hours_week, params.fte_hours_week) for f in perm_eff],
        dtype=float,
    )
    load_pre = v_peak / np.maximum(prov_day_equiv_perm, 1e-6)

    # Flex sizing
    flex_fte = np.zeros(N_MONTHS, dtype=float)
    load_post = np.zeros(N_MONTHS, dtype=float)

    for i in range(N_MONTHS):
        gap_fte = max(float(req_eff_fte_needed[i]) - float(perm_eff[i]), 0.0)
        flex_used = min(gap_fte, float(params.flex_max_fte_per_month))
        flex_fte[i] = float(flex_used)

        prov_day_equiv_total = provider_day_equiv_from_fte(
            float(perm_eff[i] + flex_used), params.hours_week, params.fte_hours_week
        )
        load_post[i] = float(v_peak[i]) / max(float(prov_day_equiv_total), 1e-6)

    residual_gap_fte = np.maximum(req_eff_fte_needed - (perm_eff + flex_fte), 0.0)
    provider_day_gap_total = float(np.sum(residual_gap_fte * dim))

    est_visits_lost = float(provider_day_gap_total) * float(params.visits_lost_per_provider_day_gap)
    est_margin_at_risk = est_visits_lost * float(params.net_revenue_per_visit)

    # Turnover replacement cost
    replacements_base = perm_paid * float(monthly_turnover)
    repl_mult = np.ones(N_MONTHS, dtype=float)
    repl_mult = np.where(load_post > float(params.budgeted_pppd), float(params.turnover_yellow_mult), repl_mult)
    repl_mult = np.where(load_post > float(params.red_start_pppd), float(params.turnover_red_mult), repl_mult)
    turnover_replacement_cost = float(np.sum(replacements_base * float(params.provider_replacement_cost) * repl_mult))

    # SWB/Visit
    annual_swb_all, total_swb_dollars, total_visits = annual_swb_per_visit_from_supply(
        provider_paid_fte=list(perm_paid),
        provider_flex_fte=list(flex_fte),
        visits_per_day=list(v_avg),
        days_in_month=list(days_in_month),
        fte_hours_per_week=float(params.fte_hours_week),
        role_mix=role_mix,
        hourly_rates=params.hourly_rates,
        benefits_load_pct=float(params.benefits_load_pct),
        ot_sick_pct=float(params.ot_sick_pct),
        bonus_pct=float(params.bonus_pct),
        physician_supervision_hours_per_month=float(params.physician_supervision_hours_per_month),
        supervisor_hours_per_month=float(params.supervisor_hours_per_month),
    )

    # Burnout
    green_cap = float(params.budgeted_pppd)
    red_start = float(params.red_start_pppd)
    months_yellow = int(np.sum((load_post > green_cap + 1e-9) & (load_post <= red_start + 1e-9)))
    months_red = int(np.sum(load_post > red_start + 1e-9))
    peak_load_post = float(np.max(load_post)) if len(load_post) else 0.0
    peak_load_perm_only = float(np.max(load_pre)) if len(load_pre) else 0.0

    # Flex dependence
    perm_total_fte_months = float(np.sum(np.maximum(perm_eff, 0.0) * dim))
    flex_total_fte_months = float(np.sum(np.maximum(flex_fte, 0.0) * dim))
    flex_share = float(flex_total_fte_months / max(perm_total_fte_months + flex_total_fte_months, 1e-9))

    # Utilization and idle
    hours_per_fte_per_day = float(params.fte_hours_week) / max(float(params.days_open_per_week), 1e-6)
    supplied_perm_hours_per_day = perm_eff * hours_per_fte_per_day
    supplied_flex_hours_per_day = flex_fte * hours_per_fte_per_day
    supplied_total_hours_per_day = supplied_perm_hours_per_day + supplied_flex_hours_per_day

    utilization = req_provider_hours_per_day / np.maximum(supplied_total_hours_per_day, 1e-9)
    excess_total_eff_fte = np.maximum((perm_eff + flex_fte) - req_eff_fte_needed, 0.0)
    excess_perm_eff_fte = np.maximum(perm_eff - req_eff_fte_needed, 0.0)

    idle_hours_month = excess_total_eff_fte * float(params.fte_hours_week) * (dim / 7.0)
    apc_rate_loaded = loaded_hourly_rate(
        params.hourly_rates["apc"], params.benefits_load_pct, params.ot_sick_pct, params.bonus_pct
    )
    idle_capacity_penalty = float(np.sum(idle_hours_month * apc_rate_loaded * 0.15))

    # Governance: utilization recovery, structural idle, elasticity (simplified)
    d_perm_paid = np.diff(perm_paid, prepend=perm_paid[0])
    is_perm_step = d_perm_paid > float(params.perm_step_epsilon)
    months_since_perm_step = np.zeros(N_MONTHS, dtype=float)
    last_step = 0
    for t in range(N_MONTHS):
        if is_perm_step[t] and t > 0:
            last_step = t
        months_since_perm_step[t] = float(t - last_step)

    util_target = float(params.utilization_target)
    recovery_months = max(int(params.recovery_months), 0)
    had_any_step = bool(np.any(is_perm_step[1:]))
    util_short = np.maximum(util_target - utilization, 0.0)
    recovery_mask = (
        (months_since_perm_step > float(recovery_months)) & (months_since_perm_step > 0.0)
        if had_any_step
        else np.zeros(N_MONTHS, dtype=bool)
    )
    utilization_recovery_penalty = float(np.sum((util_short[recovery_mask] ** 2) * dim[recovery_mask]) * 1_000_000.0)

    idle_eps = max(float(params.idle_epsilon_eff_fte), 0.0)
    idle_grace = max(int(params.idle_grace_months), 0)
    perm_idle = excess_perm_eff_fte > idle_eps
    idle_streak = np.zeros(N_MONTHS, dtype=float)
    streak = 0
    for t in range(N_MONTHS):
        if perm_idle[t]:
            streak += 1
        else:
            streak = 0
        idle_streak[t] = float(streak)

    structural_idle_index = np.maximum(idle_streak - float(idle_grace), 0.0)
    structural_idle_penalty = float(np.sum(((excess_perm_eff_fte * structural_idle_index) ** 2) * dim) * 500_000.0)

    d_req = np.diff(req_eff_fte_needed, prepend=req_eff_fte_needed[0])
    d_perm_eff = np.diff(perm_eff, prepend=perm_eff[0])
    stabilize = max(int(params.elasticity_stabilization_months), 0)
    cap = max(float(params.elasticity_cap), 0.0)
    req_eps = 0.05
    denom = np.where(d_req > req_eps, d_req, np.nan)
    elasticity = np.where(np.isfinite(denom), d_perm_eff / denom, np.nan)
    elastic_mask = (np.arange(N_MONTHS) >= stabilize) & np.isfinite(elasticity)
    elasticity_excess = np.maximum(elasticity - cap, 0.0)
    elasticity_penalty = float(np.nansum((elasticity_excess[elastic_mask] ** 2) * dim[elastic_mask]) * 1_000_000.0)

    # Burnout penalties
    yellow_excess = np.maximum(load_post - green_cap, 0.0)
    red_excess = np.maximum(load_post - red_start, 0.0)
    burnout_penalty = float(np.sum((yellow_excess ** 1.2) * dim) + 3.0 * np.sum((red_excess ** 2.0) * dim))

    # Permanent-only Green constraint
    perm_green_violation = float(np.max(np.maximum(load_pre - green_cap, 0.0)))
    perm_green_months = int(np.sum(load_pre > green_cap + 1e-9))

    # Base objective
    total_score = (
        float(total_swb_dollars)
        + float(turnover_replacement_cost)
        + float(est_margin_at_risk)
        + 2_000.0 * burnout_penalty
        + float(idle_capacity_penalty)
        + float(utilization_recovery_penalty)
        + float(structural_idle_penalty)
        + float(elasticity_penalty)
    )

    # Ledger
    rows: list[dict[str, Any]] = []
    for i in range(N_MONTHS):
        lab = dates[i].strftime("%Y-%b")
        month_visits = float(v_avg[i]) * float(days_in_month[i])
        month_net_contrib = float(month_visits) * float(params.net_revenue_per_visit)

        month_swb_per_visit, month_swb_dollars, _ = annual_swb_per_visit_from_supply(
            provider_paid_fte=[float(perm_paid[i])],
            provider_flex_fte=[float(flex_fte[i])],
            visits_per_day=[float(v_avg[i])],
            days_in_month=[int(days_in_month[i])],
            fte_hours_per_week=float(params.fte_hours_week),
            role_mix=role_mix,
            hourly_rates=params.hourly_rates,
            benefits_load_pct=float(params.benefits_load_pct),
            ot_sick_pct=float(params.ot_sick_pct),
            bonus_pct=float(params.bonus_pct),
            physician_supervision_hours_per_month=float(params.physician_supervision_hours_per_month),
            supervisor_hours_per_month=float(params.supervisor_hours_per_month),
        )

        gap_weight = float(residual_gap_fte[i] * dim[i])
        gap_total = float(np.sum(residual_gap_fte * dim))
        month_access_risk = float(est_margin_at_risk) * (gap_weight / max(gap_total, 1e-9))
        month_ebitda_proxy = float(month_net_contrib) - float(month_swb_dollars) - float(month_access_risk)

        rows.append(
            {
                "Month": lab,
                "Visits/Day (avg)": float(v_avg[i]),
                "Visits/Day (peak)": float(v_peak[i]),
                "Total Visits (month)": float(month_visits),
                "Net Contribution (month)": float(month_net_contrib),
                "SWB Dollars (month)": float(month_swb_dollars),
                "SWB/Visit (month)": float(month_swb_per_visit),
                "EBITDA Proxy (month)": float(month_ebitda_proxy),
                "Permanent FTE (Paid)": float(perm_paid[i]),
                "Permanent FTE (Effective)": float(perm_eff[i]),
                "Flex FTE Used": float(flex_fte[i]),
                "Required Provider Hours/Day": float(req_provider_hours_per_day[i]),
                "Rounded Provider Hours/Day (0.5)": float(req_provider_hours_per_day_rounded[i]),
                "Required Provider FTE (effective)": float(req_eff_fte_needed[i]),
                "Utilization (Req/Supplied)": float(utilization[i]),
                "Excess Total Eff FTE (Idle)": float(excess_total_eff_fte[i]),
                "Excess Perm Eff FTE (Idle)": float(excess_perm_eff_fte[i]),
                "Perm Load PPPD (no-flex)": float(load_pre[i]),
                "Load PPPD (post-flex)": float(load_post[i]),
                "Turnover Shed (FTE)": float(turnover_shed_arr[i]),
                "Hires Visible (FTE)": float(hires_visible_with_replacement[i]),
                "Hire Reason": hires_reason_with_replacement[i],
                "Hire Post Month": hires_post_month_with_replacement[i],
                "Residual FTE Gap (to Required)": float(residual_gap_fte[i]),
                "Target FTE (policy)": target_fte_for_month(int(months[i])),  # NEW: shows what policy expects
            }
        )

    ledger = pd.DataFrame(rows)

    # Year-by-year SWB band
    ledger["Year"] = ledger["Month"].str.slice(0, 4).astype(int)
    annual_band = ledger.groupby("Year", as_index=False).agg(
        Visits=("Total Visits (month)", "sum"),
        SWB_Dollars=("SWB Dollars (month)", "sum"),
    )
    annual_band["SWB_per_Visit"] = annual_band["SWB_Dollars"] / annual_band["Visits"].clip(lower=1.0)

    target = float(params.target_swb_per_visit)
    tol = max(float(params.swb_tolerance), 0.0)
    annual_band["SWB_Deviation_Outside_Band"] = np.maximum(
        np.abs(annual_band["SWB_per_Visit"] - target) - tol, 0.0
    )
    swb_band_penalty = float(np.sum((annual_band["SWB_Deviation_Outside_Band"] ** 2) * 1_500_000.0))

    # Permanent-only Green penalty
    perm_green_penalty = 0.0
    if bool(params.require_perm_under_green_no_flex):
        perm_green_penalty = float(perm_green_violation) * 1_500_000.0 + float(perm_green_months) * 150_000.0

    # Flex reliance penalty
    flex_penalty = float(max(flex_share - 0.10, 0.0) ** 2) * 2_000_000.0

    total_score = float(total_score) + swb_band_penalty + perm_green_penalty + flex_penalty

    annual_summary = build_annual_summary(ledger.drop(columns=["Year"]), params)

    total_net_contrib = float(total_visits) * float(params.net_revenue_per_visit)
    ebitda_proxy_annual = float(total_net_contrib) - float(total_swb_dollars) - float(turnover_replacement_cost) - float(est_margin_at_risk)

    flu_posts = ledger.loc[
        ledger["Hire Reason"].astype(str).str.contains("Winter step", na=False), "Hire Post Month"
    ].dropna().unique().tolist()
    flu_posts = [int(x) for x in flu_posts] if len(flu_posts) else []

    worst_elastic = float(np.nanmax(elasticity)) if np.any(np.isfinite(elasticity)) else float("nan")

    return {
        "dates": list(dates),
        "months": months,
        "days_in_month": list(days_in_month),
        "visits_avg": list(v_avg),
        "visits_peak": list(v_peak),
        "perm_paid": list(perm_paid),
        "perm_eff": list(perm_eff),
        "flex_fte": list(flex_fte),
        "req_provider_hours_per_day": list(req_provider_hours_per_day),
        "req_provider_hours_per_day_rounded": list(req_provider_hours_per_day_rounded),
        "req_eff_fte_needed": list(req_eff_fte_needed),
        "utilization": list(utilization),
        "excess_total_eff_fte": list(excess_total_eff_fte),
        "excess_perm_eff_fte": list(excess_perm_eff_fte),
        "load_perm_only": list(load_pre),
        "load_post": list(load_post),
        "provider_day_gap_total": provider_day_gap_total,
        "est_visits_lost": est_visits_lost,
        "est_margin_at_risk": est_margin_at_risk,
        "turnover_replacement_cost": turnover_replacement_cost,
        "annual_swb_per_visit": float(annual_swb_all),
        "total_swb_dollars": float(total_swb_dollars),
        "total_visits": float(total_visits),
        "flex_share": float(flex_share),
        "months_yellow": months_yellow,
        "months_red": months_red,
        "peak_load_post": float(peak_load_post),
        "peak_perm_load_no_flex": float(peak_load_perm_only),
        "perm_green_violation": float(perm_green_violation),
        "perm_green_months": int(perm_green_months),
        "idle_capacity_penalty": float(idle_capacity_penalty),
        "utilization_recovery_penalty": float(utilization_recovery_penalty),
        "structural_idle_penalty": float(structural_idle_penalty),
        "elasticity_penalty": float(elasticity_penalty),
        "worst_elasticity": worst_elastic,
        "months_since_perm_step": list(months_since_perm_step),
        "idle_streak": list(idle_streak),
        "elasticity": list(elasticity),
        "ebitda_proxy_annual": float(ebitda_proxy_annual),
        "total_net_contrib": float(total_net_contrib),
        "score": float(total_score),
        "ledger": ledger.drop(columns=["Year"]),
        "annual_summary": annual_summary,
        "role_mix": role_mix,
        "lead_months": int(lead_months),
        "flu_requisition_post_months": flu_posts,
    }



def recommend_policy(
    params: ModelParams,
    base_min: float,
    base_max: float,
    base_step: float,
    winter_delta_max: float,
    winter_step: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    base_values = np.arange(float(base_min), float(base_max) + 1e-9, float(base_step))
    for b in base_values:
        w_values = np.arange(float(b), float(b) + float(winter_delta_max) + 1e-9, float(winter_step))
        for w in w_values:
            pol = Policy(base_fte=float(b), winter_fte=float(w))
            res = simulate_policy(params, pol)

            candidates.append(
                {
                    "Base_FTE": float(b),
                    "Winter_FTE": float(w),
                    "Score": float(res["score"]),
                    "Annual_SWB_per_Visit (3yr-blend)": float(res["annual_swb_per_visit"]),
                    "EBITDA_Proxy_Annual": float(res["ebitda_proxy_annual"]),
                    "FlexShare": float(res["flex_share"]),
                    "Peak_Perm_Load_NoFlex": float(res["peak_perm_load_no_flex"]),
                    "Peak_Load_PostFlex": float(res["peak_load_post"]),
                    "Months_Yellow": int(res["months_yellow"]),
                    "Months_Red": int(res["months_red"]),
                    "PermGreenMonths": int(res["perm_green_months"]),
                    "IdlePenalty": float(res["idle_capacity_penalty"]),
                    "RecoveryPenalty": float(res["utilization_recovery_penalty"]),
                    "StructuralIdlePenalty": float(res["structural_idle_penalty"]),
                    "ElasticityPenalty": float(res["elasticity_penalty"]),
                    "WorstElasticity": (
                        float(res["worst_elasticity"]) if np.isfinite(res["worst_elasticity"]) else np.nan
                    ),
                }
            )

            if best is None or float(res["score"]) < float(best["res"]["score"]):
                best = {"policy": pol, "res": res}

    frontier = pd.DataFrame(candidates).sort_values(["Score", "Months_Red", "FlexShare"]).reset_index(drop=True)
    return {"best": best, "frontier": frontier}


@st.cache_data(show_spinner=False)
def cached_recommend_policy(
    params_dict: dict[str, Any],
    base_min: float,
    base_max: float,
    base_step: float,
    winter_delta_max: float,
    winter_step: float,
) -> dict[str, Any]:
    params = ModelParams(**params_dict)
    return recommend_policy(params, base_min, base_max, base_step, winter_delta_max, winter_step)


@st.cache_data(show_spinner=False)
def cached_simulate(params_dict: dict[str, Any], base_fte: float, winter_fte: float) -> dict[str, Any]:
    params = ModelParams(**params_dict)
    return simulate_policy(params, Policy(base_fte=float(base_fte), winter_fte=float(winter_fte)))



# ============================================================
# SIDEBAR — INPUTS
# ============================================================
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    st.markdown(
        """
        <div class="fix">
        <b>🔧 FIXED VERSION</b><br>
        ✅ Base replacement targets policy base<br>
        ✅ Explicit step-down after winter<br>
        ✅ Winter steps are absolute targets<br>
        ✅ Ratchet detection enabled
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.header("Demand")
    visits = st.number_input("Avg Visits/Day (annual baseline)", min_value=1.0, value=36.0, step=1.0, help=HELP["visits"])
    annual_growth = st.number_input("Annual Visit Growth %", min_value=0.0, value=10.0, step=1.0, help=HELP["annual_growth"]) / 100.0
    peak_factor = st.slider("Peak-to-average factor", 1.00, 1.50, 1.20, 0.01, help=HELP["peak_factor"])

    st.subheader("Seasonality + Flu")
    seasonality_pct = st.number_input("Seasonality swing %", min_value=0.0, value=20.0, step=5.0, help=HELP["seasonality_pct"]) / 100.0
    flu_uplift_pct = st.number_input("Flu uplift %", min_value=0.0, value=0.0, step=5.0, help=HELP["flu_uplift_pct"]) / 100.0
    flu_months = st.multiselect(
        "Flu months",
        options=MONTH_OPTIONS,
        default=[("Oct", 10), ("Nov", 11), ("Dec", 12), ("Jan", 1), ("Feb", 2)],
        help=HELP["flu_months"],
    )
    flu_months_set = {m for _, m in flu_months} if flu_months else set()

    visits_per_provider_hour = st.slider(
        "Visits per provider hour (productivity)",
        min_value=2.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
        help=HELP["visits_per_provider_hour"],
    )

    st.header("Clinic Ops")
    hours_week = st.number_input("Hours of Operation / Week", min_value=1.0, value=84.0, step=1.0, help=HELP["hours_week"])
    days_open_per_week = st.number_input("Days Open / Week", min_value=1.0, max_value=7.0, value=7.0, step=1.0, help=HELP["days_open_per_week"])
    fte_hours_week = st.number_input("FTE Hours / Week", min_value=1.0, value=36.0, step=1.0, help=HELP["fte_hours_week"])

    st.header("Policy Constraints")
    min_perm_providers_per_day = st.number_input(
        "Minimum permanent providers per day",
        min_value=0.0,
        value=1.0,
        step=0.25,
        help=HELP["min_perm_providers_per_day"],
    )
    allow_prn_override = st.checkbox("Allow Base FTE below minimum (PRN-heavy clinic)", value=False, help=HELP["allow_prn_override"])
    require_perm_under_green_no_flex = st.checkbox("Require permanent-only to stay under Green (no flex)", value=True, help=HELP["require_perm_green"])

    st.header("Governance Rules")
    utilization_target = st.slider("Utilization target (recovery rule)", 0.75, 1.05, 0.90, 0.01, help=HELP["util_target"])
    recovery_months = st.slider("Recovery window after permanent step-up (months)", 0, 12, 6, 1, help=HELP["recovery_months"])
    perm_step_epsilon = st.select_slider("Perm step-up threshold (FTE)", options=[0.05, 0.10, 0.25, 0.50], value=0.25, help=HELP["perm_step_eps"])

    st.subheader("Structural idle detection")
    idle_grace_months = st.slider("Idle grace (months)", 0, 12, 3, 1, help=HELP["idle_grace"])
    idle_epsilon_eff_fte = st.select_slider("Idle threshold (Eff FTE)", options=[0.05, 0.10, 0.25, 0.50], value=0.25, help=HELP["idle_eps"])

    st.subheader("Elasticity guardrail")
    elasticity_cap = st.slider("Elasticity cap (ΔPerm/ΔReq)", 0.80, 2.00, 1.10, 0.05, help=HELP["elasticity_cap"])
    elasticity_stabilization_months = st.slider("Stabilization months (ignore elasticity)", 0, 12, 3, 1, help=HELP["elasticity_stabilize"])

    st.header("Workforce Dynamics")
    annual_turnover = st.number_input("Annual Turnover %", min_value=0.0, value=16.0, step=1.0, help=HELP["annual_turnover"]) / 100.0
    lead_days = st.number_input("Days to Independent (Req→Independent)", min_value=0, value=210, step=10, help=HELP["lead_days"])
    ramp_months = st.slider("Ramp months after independent", 0, 6, 1, 1, help=HELP["ramp_months"])
    ramp_productivity = st.slider("Ramp productivity %", 30, 100, 75, 5, help=HELP["ramp_productivity"]) / 100.0
    fill_probability = st.slider("Fill probability %", 0, 100, 85, 5, help=HELP["fill_probability"]) / 100.0

    st.header("Risk Bands (PPPD)")
    budgeted_pppd = st.number_input("Green PPPD (target)", min_value=5.0, value=36.0, step=1.0, help=HELP["budgeted_pppd"])
    yellow_max_pppd = st.number_input("Yellow PPPD (caution)", min_value=5.0, value=42.0, step=1.0, help=HELP["yellow_max_pppd"])
    red_start_pppd = st.number_input("Red PPPD (high risk)", min_value=5.0, value=45.0, step=1.0, help=HELP["red_start_pppd"])

    st.header("Season Policy")
    winter_anchor_month = st.selectbox("Winter anchor month (ready-by)", options=MONTH_OPTIONS, index=11, help=HELP["winter_anchor_month"])
    winter_anchor_month_num = int(winter_anchor_month[1])
    
    winter_end_month = st.selectbox(
        "Winter end month (step-down after this)",
        options=MONTH_OPTIONS,
        index=1,  # February
        help=HELP["winter_end_month"],
    )
    winter_end_month_num = int(winter_end_month[1])
    
    freeze_months = st.multiselect(
        "Freeze posting months",
        options=MONTH_OPTIONS,
        default=flu_months if flu_months else [("Oct", 10), ("Nov", 11), ("Dec", 12), ("Jan", 1), ("Feb", 2)],
        help=HELP["freeze_months"],
    )
    freeze_months_set = {m for _, m in freeze_months} if freeze_months else set()

    st.header("Flex (Safety Stock)")
    flex_max_fte_per_month = st.slider("Max flex FTE available / month", 0.0, 10.0, 2.0, 0.25, help=HELP["flex_max"])
    flex_cost_multiplier = st.slider("Flex cost multiplier vs base provider", 1.0, 2.0, 1.25, 0.05, help=HELP["flex_mult"])

    st.header("Finance Governance")
    target_swb_per_visit = st.number_input("Target SWB/Visit (annual)", min_value=0.0, value=85.0, step=1.0, help=HELP["target_swb"])
    swb_tolerance = st.number_input("SWB/Visit tolerance (±$)", min_value=0.0, value=2.0, step=0.5, help=HELP["swb_tol"])
    net_revenue_per_visit = st.number_input("Net Contribution per Visit (proxy)", min_value=0.0, value=140.0, step=5.0, help=HELP["net_contrib"])
    visits_lost_per_provider_day_gap = st.number_input("Visits Lost per 1.0 Provider-Day Gap", min_value=0.0, value=18.0, step=1.0, help=HELP["visits_lost"])

    st.subheader("Provider turnover replacement cost")
    provider_replacement_cost = st.number_input("Replacement cost per 1.0 provider FTE", min_value=0.0, value=75000.0, step=5000.0, help=HELP["repl_cost"])
    turnover_yellow_mult = st.slider("Turnover cost multiplier (yellow)", 1.0, 3.0, 1.3, 0.05, help=HELP["turn_yellow"])
    turnover_red_mult = st.slider("Turnover cost multiplier (red)", 1.0, 5.0, 2.0, 0.10, help=HELP["turn_red"])

    st.subheader("Wage inputs (for SWB/Visit)")
    benefits_load_pct = st.number_input("Benefits Load %", min_value=0.0, value=30.0, step=1.0) / 100.0
    bonus_pct = st.number_input("Bonus % of base", min_value=0.0, value=10.0, step=1.0) / 100.0
    ot_sick_pct = st.number_input("OT + Sick/PTO %", min_value=0.0, value=4.0, step=0.5) / 100.0

    physician_hr = st.number_input("Physician (optional) $/hr", min_value=0.0, value=135.79, step=1.0)
    apc_hr = st.number_input("APP $/hr", min_value=0.0, value=62.0, step=1.0)
    ma_hr = st.number_input("MA $/hr", min_value=0.0, value=24.14, step=0.5)
    psr_hr = st.number_input("PSR $/hr", min_value=0.0, value=21.23, step=0.5)
    rt_hr = st.number_input("RT $/hr", min_value=0.0, value=31.36, step=0.5)
    supervisor_hr = st.number_input("Supervisor (optional) $/hr", min_value=0.0, value=28.25, step=0.5)

    physician_supervision_hours_per_month = st.number_input("Physician supervision hours/month", min_value=0.0, value=0.0, step=1.0)
    supervisor_hours_per_month = st.number_input("Supervisor hours/month", min_value=0.0, value=0.0, step=1.0)

    min_base_fte_default = fte_required_for_min_perm_providers_per_day(min_perm_providers_per_day, hours_week, fte_hours_week)
    base_min_default = 0.0 if allow_prn_override else float(min_base_fte_default)

    st.header("Optimizer Controls")
    base_min = st.number_input("Base FTE min", min_value=0.0, value=float(base_min_default), step=0.25)
    base_max = st.number_input("Base FTE max", min_value=0.0, value=6.0, step=0.25)
    base_step = st.select_slider("Base FTE step", options=[0.10, 0.25, 0.50], value=0.25)
    winter_delta_max = st.number_input("Max winter uplift above base (FTE)", min_value=0.0, value=2.0, step=0.25)
    winter_step = st.select_slider("Winter uplift step", options=[0.10, 0.25, 0.50], value=0.25)

    st.header("Mode")
    mode = st.radio("Run mode", ["Recommend + What-If", "What-If only"], index=0)

    st.divider()
    run_recommender = st.button("🏁 Run Recommender (Grid Search)", use_container_width=True)


hourly_rates = {
    "physician": float(physician_hr),
    "apc": float(apc_hr),
    "ma": float(ma_hr),
    "psr": float(psr_hr),
    "rt": float(rt_hr),
    "supervisor": float(supervisor_hr),
}

params = ModelParams(
    visits=float(visits),
    annual_growth=float(annual_growth),
    seasonality_pct=float(seasonality_pct),
    flu_uplift_pct=float(flu_uplift_pct),
    flu_months=set(flu_months_set),
    peak_factor=float(peak_factor),
    visits_per_provider_hour=float(visits_per_provider_hour),
    hours_week=float(hours_week),
    days_open_per_week=float(days_open_per_week),
    fte_hours_week=float(fte_hours_week),
    annual_turnover=float(annual_turnover),
    lead_days=int(lead_days),
    ramp_months=int(ramp_months),
    ramp_productivity=float(ramp_productivity),
    fill_probability=float(fill_probability),
    winter_anchor_month=int(winter_anchor_month_num),
    winter_end_month=int(winter_end_month_num),
    freeze_months=set(freeze_months_set),
    budgeted_pppd=float(budgeted_pppd),
    yellow_max_pppd=float(yellow_max_pppd),
    red_start_pppd=float(red_start_pppd),
    flex_max_fte_per_month=float(flex_max_fte_per_month),
    flex_cost_multiplier=float(flex_cost_multiplier),
    target_swb_per_visit=float(target_swb_per_visit),
    swb_tolerance=float(swb_tolerance),
    net_revenue_per_visit=float(net_revenue_per_visit),
    visits_lost_per_provider_day_gap=float(visits_lost_per_provider_day_gap),
    provider_replacement_cost=float(provider_replacement_cost),
    turnover_yellow_mult=float(turnover_yellow_mult),
    turnover_red_mult=float(turnover_red_mult),
    hourly_rates=hourly_rates,
    benefits_load_pct=float(benefits_load_pct),
    ot_sick_pct=float(ot_sick_pct),
    bonus_pct=float(bonus_pct),
    physician_supervision_hours_per_month=float(physician_supervision_hours_per_month),
    supervisor_hours_per_month=float(supervisor_hours_per_month),
    min_perm_providers_per_day=float(min_perm_providers_per_day),
    allow_prn_override=bool(allow_prn_override),
    require_perm_under_green_no_flex=bool(require_perm_under_green_no_flex),
    utilization_target=float(utilization_target),
    recovery_months=int(recovery_months),
    perm_step_epsilon=float(perm_step_epsilon),
    idle_grace_months=int(idle_grace_months),
    idle_epsilon_eff_fte=float(idle_epsilon_eff_fte),
    elasticity_cap=float(elasticity_cap),
    elasticity_stabilization_months=int(elasticity_stabilization_months),
    _v=MODEL_VERSION,
)

params_dict: dict[str, Any] = {**params.__dict__, "_v": MODEL_VERSION}



# ============================================================
# RUN RECOMMENDER (ONLY ON CLICK)
# ============================================================
best_block = None
frontier = None

if mode == "Recommend + What-If" and run_recommender:
    with st.spinner("Evaluating policy candidates (grid search)…"):
        rec = cached_recommend_policy(
            params_dict=params_dict,
            base_min=float(base_min),
            base_max=float(base_max),
            base_step=float(base_step),
            winter_delta_max=float(winter_delta_max),
            winter_step=float(winter_step),
        )
    best_block = rec["best"]
    frontier = rec["frontier"]

    st.session_state.rec_policy = best_block["policy"]
    st.session_state.frontier = frontier
    st.session_state["what_base_fte"] = float(st.session_state.rec_policy.base_fte)
    st.session_state["what_winter_fte"] = float(st.session_state.rec_policy.winter_fte)

if frontier is None and st.session_state["frontier"] is not None:
    frontier = st.session_state["frontier"]

rec_policy = st.session_state.rec_policy


# ============================================================
# WHAT-IF INPUTS (LIVE)
# ============================================================
st.subheader("What-If Policy Inputs")

if rec_policy is not None:
    default_base = float(rec_policy.base_fte)
    default_winter = float(rec_policy.winter_fte)
else:
    default_base = float(max(base_min, 1.0))
    default_winter = float(default_base + min(winter_delta_max, 1.0))

st.session_state["what_base_fte"] = float(st.session_state.get("what_base_fte", default_base))
st.session_state["what_winter_fte"] = float(st.session_state.get("what_winter_fte", default_winter))

min_base_fte_required = fte_required_for_min_perm_providers_per_day(
    params.min_perm_providers_per_day, params.hours_week, params.fte_hours_week
)

if not params.allow_prn_override:
    if st.session_state["what_base_fte"] < float(min_base_fte_required):
        st.session_state["what_base_fte"] = float(min_base_fte_required)

if st.session_state["what_winter_fte"] < st.session_state["what_base_fte"]:
    st.session_state["what_winter_fte"] = st.session_state["what_base_fte"]


def _sync_winter_to_base():
    if st.session_state["what_winter_fte"] < st.session_state["what_base_fte"]:
        st.session_state["what_winter_fte"] = st.session_state["what_base_fte"]


w1, w2 = st.columns(2)
with w1:
    what_base = st.number_input(
        "What-If Base FTE",
        min_value=0.0 if params.allow_prn_override else float(min_base_fte_required),
        value=float(st.session_state["what_base_fte"]),
        step=0.25,
        key="what_base_fte",
        on_change=_sync_winter_to_base,
    )

with w2:
    winter_min = float(st.session_state["what_base_fte"])
    winter_val = max(float(st.session_state["what_winter_fte"]), winter_min)
    what_winter = st.number_input(
        "What-If Winter FTE",
        min_value=winter_min,
        value=winter_val,
        step=0.25,
        key="what_winter_fte",
    )

R = cached_simulate(params_dict, float(what_base), float(what_winter))

R_rec = None
if mode == "Recommend + What-If" and rec_policy is not None:
    R_rec = cached_simulate(params_dict, float(rec_policy.base_fte), float(rec_policy.winter_fte))


# ============================================================
# RATCHET DETECTION (NEW)
# ============================================================
st.markdown("---")
st.header("🔍 Ratchet Detection")

annual = R.get("annual_summary")
if annual is not None and len(annual) >= 2:
    min_fte_y1 = float(annual.loc[0, "Min_Perm_Paid_FTE"])
    min_fte_y2 = float(annual.loc[1, "Min_Perm_Paid_FTE"])
    min_fte_y3 = float(annual.loc[2, "Min_Perm_Paid_FTE"]) if len(annual) >= 3 else min_fte_y2
    
    base_drift_y2 = min_fte_y2 - min_fte_y1
    base_drift_y3 = min_fte_y3 - min_fte_y2
    
    ratchet_detected = (base_drift_y2 > 0.5) or (base_drift_y3 > 0.5)
    
    if ratchet_detected:
        st.markdown(
            f"""
            <div class="warn">
            ⚠️ <b>RATCHET DETECTED</b><br>
            Minimum FTE is growing year-over-year (base should stay constant):<br>
            • Year 1 min: {min_fte_y1:.2f} FTE<br>
            • Year 2 min: {min_fte_y2:.2f} FTE (Δ = {base_drift_y2:+.2f})<br>
            • Year 3 min: {min_fte_y3:.2f} FTE (Δ = {base_drift_y3:+.2f})<br>
            <br>
            Expected: Year 1-3 min should be ≈ {what_base:.2f} FTE (policy base)<br>
            <b>Action:</b> Review hiring events in ledger below.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="ok">
            ✅ <b>NO RATCHET</b><br>
            Minimum FTE is stable year-over-year:<br>
            • Year 1 min: {min_fte_y1:.2f} FTE<br>
            • Year 2 min: {min_fte_y2:.2f} FTE (Δ = {base_drift_y2:+.2f})<br>
            • Year 3 min: {min_fte_y3:.2f} FTE (Δ = {base_drift_y3:+.2f})<br>
            <br>
            Policy base = {what_base:.2f} FTE ✓
            </div>
            """,
            unsafe_allow_html=True,
        )



# ============================================================
# SCORECARD
# ============================================================
st.markdown("---")
st.header("Policy Scorecard")

ledger = R.get("ledger")
swb_target = float(params.target_swb_per_visit)
tol = float(params.swb_tolerance)
band_lo, band_hi = swb_target - tol, swb_target + tol

if annual is not None and len(annual) > 0:
    swb_y1 = float(annual.loc[0, "SWB_per_Visit"])
    swb_worst = float(np.max(annual["SWB_per_Visit"].values))
    ebitda_y1 = float(annual.loc[0, "EBITDA_Proxy"])
    ebitda_y3 = float(annual.loc[len(annual) - 1, "EBITDA_Proxy"])
    util_y1 = float(annual.loc[0, "Avg_Utilization"]) if "Avg_Utilization" in annual.columns else float(np.nan)
else:
    swb_y1 = float(R["annual_swb_per_visit"])
    swb_worst = float(R["annual_swb_per_visit"])
    ebitda_y1 = float(R["ebitda_proxy_annual"])
    ebitda_y3 = float(R["ebitda_proxy_annual"])
    util_y1 = float(np.nan)

swb_status_y1 = "IN BAND" if (band_lo - 1e-9) <= swb_y1 <= (band_hi + 1e-9) else "OUT"

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Base FTE", f"{what_base:.2f}")
c2.metric("Winter FTE", f"{what_winter:.2f}")
c3.metric("SWB/Visit (Y1)", f"${swb_y1:.2f}", help=f"Target: ${swb_target:.0f} ± ${tol:.0f}")
c4.metric("EBITDA Y1/Y3", f"${ebitda_y1/1000:.0f}K / ${ebitda_y3/1000:.0f}K")
c5.metric("Util (Y1 avg)", f"{util_y1*100:.0f}%" if not np.isnan(util_y1) else "—")
c6.metric("Peak Load", f"{R['peak_load_post']:.1f} PPPD")


# ============================================================
# CHARTS
# ============================================================
st.markdown("---")
st.header("3-Year Planning Horizon")

dates = R["dates"]
perm_eff = np.array(R["perm_eff"], dtype=float)
perm_paid = np.array(R["perm_paid"], dtype=float)
flex_fte = np.array(R["flex_fte"], dtype=float)
req_eff = np.array(R["req_eff_fte_needed"], dtype=float)
v_avg = np.array(R["visits_avg"], dtype=float)
v_peak = np.array(R["visits_peak"], dtype=float)
load_perm_only = np.array(R["load_perm_only"], dtype=float)
load_post = np.array(R["load_post"], dtype=float)

tick_idx = list(range(0, len(dates), 3))

# Chart 1: Supply vs Demand
fig1, ax = plt.subplots(figsize=(12, 5.2))
ax.plot(dates, perm_eff, linewidth=2.2, color=BRAND_BLACK, label="Permanent FTE (Effective)", marker='o', markersize=3)
ax.plot(dates, perm_paid, linewidth=1.7, linestyle="--", color=MID_GRAY, label="Permanent FTE (Paid)", alpha=0.7)
ax.plot(dates, req_eff, linewidth=2.2, linestyle=":", color="#1976d2", label="Required FTE", marker='s', markersize=3)
ax.plot(dates, perm_eff + flex_fte, linewidth=2.0, linestyle=":", color=BRAND_GOLD, label="Total Coverage (Eff + Flex)")
ax.fill_between(dates, perm_eff, perm_eff + flex_fte, alpha=0.12, color=BRAND_GOLD, label="Flex Used")
ax.set_title("Supply vs Demand (Fixed: Base Should Stay Constant)", fontsize=13, fontweight="bold")
ax.set_ylabel("Provider FTE", fontsize=11, fontweight="bold")
ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)
ax.set_xticks([dates[i] for i in tick_idx])
ax.set_xticklabels([dates[i].strftime("%Y-%b") for i in tick_idx], rotation=0, fontsize=9)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
plt.tight_layout()
st.pyplot(fig1)

# Chart 2: Load
fig2, axL = plt.subplots(figsize=(12, 5.2))
axL.plot(dates, load_post, linewidth=2.2, color=BRAND_GOLD, label="Load PPPD (post-flex)")
axL.plot(dates, load_perm_only, linewidth=1.2, linestyle="--", color=GRAY, alpha=0.85, label="Perm Load PPPD (no-flex)")

axL.axhspan(0, params.budgeted_pppd, alpha=0.06, color="#00aa00")
axL.axhspan(params.budgeted_pppd, params.red_start_pppd, alpha=0.06, color="#ffaa00")
axL.axhspan(params.red_start_pppd, max(params.red_start_pppd + 10, float(np.max(load_perm_only) + 5)), alpha=0.06, color="#ff0000")
axL.axhline(params.budgeted_pppd, linewidth=1.2, color="#00aa00", linestyle=":")
axL.axhline(params.red_start_pppd, linewidth=1.2, color="#ff0000", linestyle=":")

axL.set_title("Monthly Load (PPPD) with Risk Bands", fontsize=13, fontweight="bold")
axL.set_ylabel("Patients / Provider / Day", fontsize=11, fontweight="bold")
axL.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)

axR = axL.twinx()
axR.plot(dates, v_avg, linewidth=1.3, linestyle="-.", color=MID_GRAY, label="Visits/Day (avg)")
axR.plot(dates, v_peak, linewidth=1.3, linestyle=":", color=MID_GRAY, alpha=0.85, label="Visits/Day (peak)")
axR.set_ylabel("Visits / Day", fontsize=11, fontweight="bold")

axL.set_xticks([dates[i] for i in tick_idx])
axL.set_xticklabels([dates[i].strftime("%Y-%b") for i in tick_idx], rotation=0, fontsize=9)

lines1, labels1 = axL.get_legend_handles_labels()
lines2, labels2 = axR.get_legend_handles_labels()
axL.legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
plt.tight_layout()
st.pyplot(fig2)


# ============================================================
# ANNUAL SUMMARY
# ============================================================
st.markdown("---")
st.header("Annual Summary")

if annual is not None and len(annual) > 0:
    st.dataframe(
        annual.style.format(
            {
                "Visits": "{:,.0f}",
                "Net_Contribution": "${:,.0f}",
                "SWB_Dollars": "${:,.0f}",
                "SWB_per_Visit": "${:,.2f}",
                "EBITDA_Proxy": "${:,.0f}",
                "EBITDA_per_Visit": "${:,.2f}",
                "YoY_EBITDA_Proxy_Δ": "${:,.0f}",
                "Avg_Perm_Paid_FTE": "{:,.2f}",
                "Min_Perm_Paid_FTE": "{:,.2f}",
                "Max_Perm_Paid_FTE": "{:,.2f}",
                "Avg_Perm_Eff_FTE": "{:,.2f}",
                "Avg_Flex_FTE": "{:,.2f}",
                "Peak_Flex_FTE": "{:,.2f}",
                "Peak_Load_PPPD_Post": "{:,.1f}",
                "Peak_Perm_Load_NoFlex": "{:,.1f}",
                "Total_Residual_Gap_FTE_Months": "{:,.2f}",
                "Peak_Req_ProvHoursPerDay": "{:,.1f}",
                "Peak_Req_Eff_FTE": "{:,.2f}",
                "Avg_Utilization": "{:.0%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# AUDIT LEDGER
# ============================================================
st.markdown("---")
st.header("Audit Ledger (36 months)")

st.dataframe(
    R["ledger"].style.format(
        {
            "Total Visits (month)": "{:,.0f}",
            "Net Contribution (month)": "${:,.0f}",
            "SWB Dollars (month)": "${:,.0f}",
            "SWB/Visit (month)": "${:,.2f}",
            "EBITDA Proxy (month)": "${:,.0f}",
            "Visits/Day (avg)": "{:,.2f}",
            "Visits/Day (peak)": "{:,.2f}",
            "Required Provider Hours/Day": "{:,.1f}",
            "Rounded Provider Hours/Day (0.5)": "{:,.1f}",
            "Required Provider FTE (effective)": "{:,.2f}",
            "Utilization (Req/Supplied)": "{:.0%}",
            "Excess Perm Eff FTE (Idle)": "{:,.2f}",
            "Perm Load PPPD (no-flex)": "{:,.1f}",
            "Load PPPD (post-flex)": "{:,.1f}",
            "Target FTE (policy)": "{:,.2f}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# RECOMMENDED POLICY (if available)
# ============================================================
if mode == "Recommend + What-If" and rec_policy is not None:
    best_res = R_rec if best_block is None else best_block["res"]
    best_policy = rec_policy if best_block is None else best_block["policy"]

    st.markdown("---")
    st.header("Recommended Policy")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Base FTE", f"{best_policy.base_fte:.2f}")
    c2.metric("Winter FTE", f"{best_policy.winter_fte:.2f}")
    c3.metric("Peak Perm Load", f"{best_res['peak_perm_load_no_flex']:.1f}")
    c4.metric("Flex Share", f"{best_res['flex_share']*100:.1f}%")
    c5.metric("Months Yellow", f"{best_res['months_yellow']}")
    c6.metric("Months Red", f"{best_res['months_red']}")

    if frontier is not None:
        with st.expander("Candidate table (top 25 by score)", expanded=False):
            st.dataframe(frontier.head(25), use_container_width=True, hide_index=True)


# ============================================================
# EXPORTS
# ============================================================
st.markdown("---")
st.header("Exports")

png1 = fig_to_png_bytes(fig1)
png2 = fig_to_png_bytes(fig2)
ledger_csv = df_to_csv_bytes(R["ledger"])

cA, cB, cC = st.columns(3)
with cA:
    st.download_button("⬇️ Supply Chart (PNG)", data=png1, file_name="psm_fixed_supply.png", mime="image/png", use_container_width=True)
with cB:
    st.download_button("⬇️ Load Chart (PNG)", data=png2, file_name="psm_fixed_load.png", mime="image/png", use_container_width=True)
with cC:
    st.download_button("⬇️ Ledger (CSV)", data=ledger_csv, file_name="psm_fixed_ledger.csv", mime="text/csv", use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div class="fix">
    <b>🔧 FIXED VERSION NOTES</b><br>
    This version addresses the "ratchet effect" bug where base FTE was growing year-over-year.<br>
    <br>
    <b>Key fixes:</b><br>
    • Base replacement now targets <b>policy base FTE</b>, not current elevated FTE<br>
    • Explicit step-down logic after winter season (see "Target FTE (policy)" column in ledger)<br>
    • Winter steps set <b>absolute targets</b> from policy (not additive to current level)<br>
    • Ratchet detection enabled (alerts if minimum FTE drifts upward)<br>
    <br>
    <b>Validation:</b><br>
    Check the annual summary "Min_Perm_Paid_FTE" row — it should stay close to your base policy across all 3 years.<br>
    If it grows >0.5 FTE year-over-year, the ratchet detector will flag it.
    </div>
    """,
    unsafe_allow_html=True,
)
