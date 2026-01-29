# app.py — Predictive Staffing Model (PSM) — Policy Optimizer (Client-Grade)
# Two-level permanent staffing policy (Base + Winter) + flex as safety stock
# 36-month horizon • Lead-time + ramp • Hiring freeze during flu months
# Governance: SWB/Visit affordability (annual) + burnout zones + provider turnover replacement cost
#
# NEW (Operator-grade UX):
# - Declares assumptions (explicit “what the model is assuming”)
# - Surfaces tradeoffs (permanent vs flex reliance, cost vs risk)
# - Explains risk (flags + “failure modes”)
# - Justifies decisions in plain language (executive summary)
# - Adds operational guardrails:
#     (A) Clinic-hours minimum permanent coverage (1 provider always-on coverage) by default
#     (B) Optional operator override to allow sub-floor permanent if reliable PRN/per diem exists
#     (C) Optional productivity guardrail: permanent sized to stay <= Green PPPD on AVG demand (no flex)
# - Sticky top scorecard (frozen-cells feel)

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
MODEL_VERSION = "2026-01-29-operator-grade-v2"


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM) — Policy Optimizer", layout="centered")
st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1.25rem; padding-bottom: 2.25rem; }
      .small { font-size: 0.92rem; color: #444; }
      .contract { background: #f7f7f7; border: 1px solid #e6e6e6; border-radius: 10px; padding: 14px 16px; }
      .warn { background: #fff6e6; border: 1px solid #ffe2a8; border-radius: 10px; padding: 12px 14px; }
      .ok { background: #ecfff0; border: 1px solid #b7f0c0; border-radius: 10px; padding: 12px 14px; }
      .note { background: #f3f7ff; border: 1px solid #cfe0ff; border-radius: 10px; padding: 12px 14px; }
      .kpi { background: #ffffff; border: 1px solid #eee; border-radius: 10px; padding: 10px 12px; }
      .sticky { position: sticky; top: 0; z-index: 999; background: white; padding-top: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid #eee; }
      .pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.82rem; border: 1px solid #ddd; background: #fafafa; margin-right: 6px; }
      .pill-warn { border-color: #ffe2a8; background: #fff6e6; }
      .pill-ok { border-color: #b7f0c0; background: #ecfff0; }
      .pill-bad { border-color: #ffb3b3; background: #ffecec; }
      .muted { color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Predictive Staffing Model (PSM) — Policy Optimizer")
st.caption(
    "Two-level permanent policy (Base + Winter) • Flex as safety stock • 36-month horizon • "
    "Lead-time + ramp • Flu-season hiring freeze • Annual SWB/Visit governance"
)

model = StaffingModel()


# ============================================================
# SESSION STATE (persist recommender across reruns)
# ============================================================
st.session_state.setdefault("rec_policy", None)
st.session_state.setdefault("frontier", None)
# DO NOT initialize widget keys to None; widgets will create them.


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

N_MONTHS = 36  # 3-year view
AVG_DAYS_PER_MONTH = 30.4

MONTH_OPTIONS = [
    ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("May", 5), ("Jun", 6),
    ("Jul", 7), ("Aug", 8), ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12)
]


# ============================================================
# TOOLTIP HELP (shown as "?" on hover)
# ============================================================
HELP: dict[str, str] = {
    # Demand
    "visits": "Baseline average visits per day for the current year (annual average, before seasonality/flu adjustments).",
    "annual_growth": "Expected annual growth rate applied to baseline visits for Year 2 and Year 3.",
    "peak_factor": (
        "Peak-to-average planning multiplier used to build a conservative 'peak' demand curve for staffing/load. "
        "Peak visits/day = (avg visits/day after seasonality + flu uplift) × this factor. "
        "This does not change the annual baseline input—only the peak planning curve used for load and flex sizing."
    ),
    "visits_per_provider_shift": (
        "Sweet-spot visits per provider shift (used to translate demand into required provider shifts/day). "
        "Example: 36 means ~1 provider shift/day covers ~36 visits/day."
    ),
    "seasonality_pct": "Seasonal swing applied to baseline average visits/day: winter up, summer down, spring/fall neutral.",
    "flu_uplift_pct": "Additional uplift applied to average visits/day in the selected flu months.",
    "flu_months": "Months that receive flu uplift. Often Oct–Feb.",

    # Clinic ops
    "hours_week": "Total clinic hours open per week. Used for minimum permanent coverage guardrail and load calculations.",
    "days_open_per_week": "Days open per week (informational today).",
    "fte_hours_week": "Standard paid hours per 1.0 FTE per week (e.g., 36). Used for converting FTE to hours and for minimum coverage guardrail.",

    # Workforce
    "annual_turnover": "Annual provider turnover rate. Converted to a monthly attrition rate in the simulation.",
    "lead_days": "Total days from requisition to fully independent productivity (includes hiring + training).",
    "ramp_months": "Months after independence where productivity is reduced (ramp).",
    "ramp_productivity": "Productivity during ramp months as a fraction of full productivity (e.g., 75%).",
    "fill_probability": "Probability your pipeline fills planned requisitions. Applied to hires scheduled to become visible.",

    # Load zones
    "budgeted_pppd": "Target PPPD threshold (Green). Months above Green are considered strain and can increase turnover risk.",
    "yellow_max_pppd": "Caution PPPD threshold (Yellow). Flex is used to reduce load toward this level (bounded by max flex).",
    "red_start_pppd": "High-risk PPPD threshold (Red). Months above Red are treated as burnout/instability risk.",

    # Freeze
    "winter_anchor_month": "Month when winter staffing should be ready by (the model posts earlier to meet this).",
    "freeze_months": "Posting is not allowed during these months (typically flu season).",

    # Flex
    "flex_max": "Maximum flex provider FTE available in any month (acts as safety stock).",
    "flex_mult": "Premium cost multiplier for flex providers vs base provider loaded cost.",

    # Finance
    "target_swb": "Annual target for staffing wages & benefits per visit (governance constraint).",
    "net_contrib": (
        "Net contribution (operating margin) per incremental visit. "
        "Used to estimate operating margin lost when access gaps cause lost visits."
    ),
    "visits_lost": "Estimated visits lost per 1.0 provider-day equivalent gap (access shortfall proxy).",
    "repl_cost": "Replacement cost per 1.0 provider FTE replaced (recruiting + onboarding + ramp inefficiency + temp coverage).",
    "turn_yellow": "Multiplier on replacement cost in yellow months (load above green).",
    "turn_red": "Multiplier on replacement cost in red months (load above red).",

    # Wages
    "benefits": "Benefits load applied on top of base hourly (e.g., 30%).",
    "bonus": "Bonus load applied on top of base hourly (e.g., 10%).",
    "ot_sick": "OT + sick/PTO load applied on top of base hourly (e.g., 4%).",
    "phys_hr": "Optional physician hourly rate used if supervision hours are included.",
    "apc_hr": "Provider (APP) base hourly rate.",
    "ma_hr": "MA base hourly rate.",
    "psr_hr": "PSR base hourly rate.",
    "rt_hr": "RT base hourly rate.",
    "sup_hr": "Optional supervisor hourly rate used if supervision hours are included.",
    "phys_sup_hours": "Fixed physician supervision hours per month (adds cost).",
    "sup_hours": "Fixed supervisor hours per month (adds cost).",

    # Optimizer
    "base_min": "Minimum base FTE to evaluate in the grid search (guardrails may enforce a higher minimum).",
    "base_max": "Maximum base FTE to evaluate in the grid search.",
    "base_step": "Step size for base FTE grid search.",
    "winter_delta_max": "Maximum winter uplift above base to evaluate.",
    "winter_step": "Step size for winter uplift grid search.",

    # Guardrails
    "guard_min_coverage": (
        "Operational guardrail: by default, the model will not recommend permanent staffing below "
        "the level required to cover all open clinic hours with 1 provider on-site (always-on coverage). "
        "You can override this if you have reliable PRN/per diem coverage."
    ),
    "guard_green_no_flex": (
        "Productivity guardrail: if enabled, the model enforces permanent staffing sufficient to keep "
        "AVG demand at or below Green PPPD without relying on flex. Flex remains available for peak planning and surprises."
    ),
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
    # Provider-days equivalent per day (a scaling convenience used for PPPD-based load)
    return float(provider_fte) * (float(fte_hours_week) / max(float(hours_week), 1e-9))


def fte_required_for_pppd(visits_per_day: float, pppd_target: float, hours_week: float, fte_hours_week: float) -> float:
    # Invert PPPD relation: load = visits / provider_day_equiv
    # provider_day_equiv_needed = visits / target
    # provider_day_equiv = fte * (fte_hours_week / hours_week)
    # => fte_needed = (visits/target) * (hours_week / fte_hours_week)
    v = max(float(visits_per_day), 0.0)
    t = max(float(pppd_target), 1e-6)
    return (v / t) * (float(hours_week) / max(float(fte_hours_week), 1e-6))


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def compute_visits_curve(
    months: list[int],
    base_year0: float,
    base_year1: float,
    base_year2: float,
    seasonality_pct: float,
) -> list[float]:
    """Simple seasonal swing: winter up, summer down; spring/fall neutral."""
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


def compute_role_mix_ratios(visits_per_day: float) -> dict[str, float]:
    """
    Stable ratios: lock to baseline volume rather than month-to-month noise.
    Requires staffing_model.get_role_mix_ratios(vpd) OR falls back to calculate().
    """
    v = float(visits_per_day)
    if hasattr(model, "get_role_mix_ratios"):
        return model.get_role_mix_ratios(v)  # type: ignore[attr-defined]
    daily = model.calculate(v)
    prov_day = max(float(daily.get("provider_day", 0.0)), 0.25)
    return {
        "psr_per_provider": float(daily.get("psr_day", 0.0)) / prov_day,
        "ma_per_provider": float(daily.get("ma_day", 0.0)) / prov_day,
        "xrt_per_provider": float(daily.get("xrt_day", 0.0)) / prov_day,
    }


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
    """
    Returns:
      annual_swb_per_visit, total_swb_dollars, total_visits
    SWB includes: Providers + PSR/MA/RT + optional supervision roles.
    Flex providers are treated as requiring the same support mix (conservative but practical).
    """
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


def select_latest_post_month_before_freeze(anchor_month: int, lead_months: int, freeze_months: set[int]) -> int:
    """
    Choose the latest posting month <= (anchor_month - lead_months) that is NOT in freeze months.
    Posting earlier than necessary is allowed (it simply makes hires ready early).
    """
    latest = wrap_month(anchor_month - lead_months)
    for k in range(0, 12):
        m = wrap_month(latest - k)
        if m not in freeze_months:
            return m
    return latest


# ----------------------------
# Margin-at-risk (Exposure) helpers
# ----------------------------
def policy_exposure_dollars(sim_result: dict) -> float:
    """Exposure = policy-driven components that move with staffing decisions (delta comparisons)."""
    return float(
        sim_result["permanent_provider_cost"]
        + sim_result["flex_provider_cost"]
        + sim_result["turnover_replacement_cost"]
        + sim_result["est_margin_at_risk"]
    )


def margin_at_risk_vs_recommended(what_if: dict, recommended: dict) -> dict:
    """Delta exposure and components: positive = worse than recommended (more margin at risk)."""
    wi_total = policy_exposure_dollars(what_if)
    rec_total = policy_exposure_dollars(recommended)

    return {
        "delta_total": float(wi_total - rec_total),
        "wi_total": float(wi_total),
        "rec_total": float(rec_total),
        "delta_perm_cost": float(what_if["permanent_provider_cost"] - recommended["permanent_provider_cost"]),
        "delta_flex_cost": float(what_if["flex_provider_cost"] - recommended["flex_provider_cost"]),
        "delta_turnover_cost": float(what_if["turnover_replacement_cost"] - recommended["turnover_replacement_cost"]),
        "delta_access_risk": float(what_if["est_margin_at_risk"] - recommended["est_margin_at_risk"]),
    }


def exposure_summary(sim_result: dict) -> dict:
    perm = float(sim_result["permanent_provider_cost"])
    flex = float(sim_result["flex_provider_cost"])
    turn = float(sim_result["turnover_replacement_cost"])
    access = float(sim_result["est_margin_at_risk"])
    total = perm + flex + turn + access
    return {"total": total, "perm": perm, "flex": flex, "turnover": turn, "access": access}


def compute_risk_flags(sim_result: dict[str, Any], green_cap: float, red_start: float) -> dict[str, Any]:
    ledger: pd.DataFrame = sim_result["ledger"]
    flex = pd.to_numeric(ledger["Flex FTE Used"], errors="coerce").fillna(0.0).values
    perm_eff = pd.to_numeric(ledger["Permanent FTE (Effective)"], errors="coerce").fillna(0.0).values
    load_post = pd.to_numeric(ledger["Load PPPD (post-flex)"], errors="coerce").fillna(0.0).values

    total_cov = np.maximum(perm_eff + flex, 1e-9)
    flex_share = float(np.sum(flex) / np.sum(total_cov))
    peak_flex = float(np.max(flex)) if len(flex) else 0.0

    # Consecutive red streaks
    is_red = (load_post > float(red_start) + 1e-9).astype(int)
    max_red_streak = 0
    streak = 0
    for x in is_red:
        if x == 1:
            streak += 1
            max_red_streak = max(max_red_streak, streak)
        else:
            streak = 0

    # Structural flex dependency: flex used in most months
    months_flex_used = int(np.sum(flex > 1e-6))
    structural_flex = months_flex_used >= int(0.60 * len(flex))

    # Permanent insufficiency vs Green pre-flex (if we track it)
    load_pre = pd.to_numeric(ledger["Load PPPD (pre-flex)"], errors="coerce").fillna(0.0).values
    months_perm_over_green = int(np.sum(load_pre > float(green_cap) + 1e-9))

    # Simple tier
    risk_score = 0
    if flex_share > 0.30:
        risk_score += 2
    elif flex_share > 0.15:
        risk_score += 1
    if structural_flex:
        risk_score += 2
    if max_red_streak >= 2:
        risk_score += 2
    elif max_red_streak == 1:
        risk_score += 1
    if months_perm_over_green >= 6:
        risk_score += 2
    elif months_perm_over_green >= 3:
        risk_score += 1

    if risk_score >= 6:
        tier = "High"
    elif risk_score >= 3:
        tier = "Moderate"
    else:
        tier = "Low"

    return {
        "tier": tier,
        "flex_share": flex_share,
        "peak_flex": peak_flex,
        "months_flex_used": months_flex_used,
        "structural_flex": structural_flex,
        "max_red_streak": max_red_streak,
        "months_perm_over_green": months_perm_over_green,
    }


def pill(text: str, kind: str = "ok") -> str:
    cls = {"ok": "pill pill-ok", "warn": "pill pill-warn", "bad": "pill pill-bad"}.get(kind, "pill")
    return f'<span class="{cls}">{text}</span>'


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
    visits_per_provider_shift: float

    # Clinic ops
    hours_week: float
    days_open_per_week: float
    fte_hours_week: float

    # Workforce dynamics
    annual_turnover: float
    lead_days: int
    ramp_months: int
    ramp_productivity: float
    fill_probability: float

    # Policy / freeze
    winter_anchor_month: int
    freeze_months: set[int]

    # Load zones
    budgeted_pppd: float
    yellow_max_pppd: float
    red_start_pppd: float

    # Flex
    flex_max_fte_per_month: float
    flex_cost_multiplier: float

    # Finance inputs
    target_swb_per_visit: float
    net_revenue_per_visit: float  # UI label: net contribution per visit
    visits_lost_per_provider_day_gap: float

    # Turnover replacement (providers only)
    provider_replacement_cost: float
    turnover_yellow_mult: float
    turnover_red_mult: float

    # Wages (for SWB/visit)
    hourly_rates: dict[str, float]
    benefits_load_pct: float
    ot_sick_pct: float
    bonus_pct: float
    physician_supervision_hours_per_month: float
    supervisor_hours_per_month: float

    # Guardrails
    allow_low_perm_for_prn: bool
    enforce_green_on_avg_without_flex: bool

    # Cache key injection (ignored by simulation; used only to bust Streamlit cache)
    _v: str = MODEL_VERSION


@dataclass(frozen=True)
class Policy:
    base_fte: float
    winter_fte: float


# ============================================================
# SIMULATION ENGINE
# ============================================================
def build_annual_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    df = ledger.copy()
    df["Year"] = df["Month"].str.slice(0, 4).astype(int)

    num_cols = [c for c in df.columns if c != "Month" and c != "Hire Reason"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby("Year", as_index=False)
    annual = g.agg(
        Visits=("Total Visits (month)", "sum"),
        SWB_Dollars=("SWB Dollars (month)", "sum"),
        Turnover_Cost=("Turnover Replacement Cost (month)", "sum"),
        Access_Margin_Lost=("Access Margin at Risk (month)", "sum"),
        Avg_Perm_Paid_FTE=("Permanent FTE (Paid)", "mean"),
        Avg_Perm_Eff_FTE=("Permanent FTE (Effective)", "mean"),
        Avg_Flex_FTE=("Flex FTE Used", "mean"),
        Peak_Flex_FTE=("Flex FTE Used", "max"),
        Peak_Load_PPPD_Pre=("Load PPPD (pre-flex)", "max"),
        Peak_Load_PPPD_Post=("Load PPPD (post-flex)", "max"),
        Months_Yellow=("Is_Yellow", "sum"),
        Months_Red=("Is_Red", "sum"),
        Total_Residual_Gap_FTE_Months=("Residual FTE Gap (to Yellow)", "sum"),
        Flex_FTE_Months=("Flex FTE Used", "sum"),
        Perm_FTE_Months=("Permanent FTE (Effective)", "sum"),
    )

    annual["SWB_per_Visit"] = annual["SWB_Dollars"] / annual["Visits"].clip(lower=1.0)

    # EBITDA proxy (plain language: "margin after staffing + turnover + access loss")
    # EBITDA_proxy = (net_contrib * visits) - SWB - turnover replacement - access margin lost
    # net_contrib is added later (we compute per-year in simulate and store in ledger)
    annual["Net_Contribution"] = annual["Net Contribution (month)"].fillna(0.0)
    annual["EBITDA_Proxy"] = (
        annual["Net_Contribution"]
        - annual["SWB_Dollars"]
        - annual["Turnover_Cost"]
        - annual["Access_Margin_Lost"]
    )

    # Permanent-to-flex ratio (coverage mix)
    annual["Perm_to_Flex_Ratio"] = annual["Perm_FTE_Months"] / annual["Flex_FTE_Months"].replace(0.0, np.nan)
    annual["Perm_to_Flex_Ratio"] = annual["Perm_to_Flex_Ratio"].replace([np.nan, np.inf, -np.inf], np.nan)

    return annual


def simulate_policy(params: ModelParams, policy: Policy) -> dict[str, Any]:
    today = datetime.today()
    year0 = today.year
    dates = pd.date_range(start=datetime(year0, 1, 1), periods=N_MONTHS, freq="MS")
    months = [int(d.month) for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    lead_months = lead_days_to_months(int(params.lead_days))
    monthly_turnover = float(params.annual_turnover) / 12.0
    fill_p = max(min(float(params.fill_probability), 1.0), 0.0)

    # Demand curve (avg)
    base_year0 = float(params.visits)
    base_year1 = float(params.visits) * (1.0 + float(params.annual_growth))
    base_year2 = float(params.visits) * (1.0 + float(params.annual_growth)) ** 2

    visits_curve_base = compute_visits_curve(
        months=months,
        base_year0=base_year0,
        base_year1=base_year1,
        base_year2=base_year2,
        seasonality_pct=float(params.seasonality_pct),
    )
    visits_curve_flu = apply_flu_uplift(
        visits_curve=visits_curve_base,
        months=months,
        flu_months=set(params.flu_months),
        flu_uplift_pct=float(params.flu_uplift_pct),
    )
    visits_peak = [float(v) * float(params.peak_factor) for v in visits_curve_flu]

    # Role mix locked to baseline (growth-year) volume for stability
    role_mix = compute_role_mix_ratios(float(params.visits) * (1.0 + float(params.annual_growth)))

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

    def run_supply(hires_visible_local: list[float]) -> tuple[list[float], list[float], list[float]]:
        # policy.base_fte is the starting paid cohort
        local_cohorts = [{"fte": max(float(policy.base_fte), 0.0), "age": 9999}]
        paid = [0.0] * N_MONTHS
        eff = [0.0] * N_MONTHS
        turnover_shed = [0.0] * N_MONTHS

        def _paid() -> float:
            return float(sum(float(c["fte"]) for c in local_cohorts))

        for t in range(N_MONTHS):
            sp = _paid()

            for c in local_cohorts:
                c["fte"] = max(float(c["fte"]) * (1.0 - monthly_turnover), 0.0)

            ap = _paid()
            turnover_shed[t] = float(ap - sp)  # negative

            add = float(hires_visible_local[t])
            if add > 1e-9:
                local_cohorts.append({"fte": add, "age": 0})

            ep = _paid()
            paid[t] = float(ep)

            ee = 0.0
            for c in local_cohorts:
                ee += float(c["fte"]) * ramp_factor(int(c["age"]))
            eff[t] = float(ee)

            for c in local_cohorts:
                c["age"] = int(c["age"]) + 1

        return paid, eff, turnover_shed

    # Pass 1 (no winter steps)
    paid_1, _eff_1, _shed_1 = run_supply(hires_visible)

    # Add winter steps (respect posting freeze)
    for (post_idx, vis_idx) in winter_step_events:
        if int(months[post_idx]) in set(params.freeze_months):
            continue
        expected_paid_at_vis = float(paid_1[vis_idx])
        need = max(float(policy.winter_fte) - expected_paid_at_vis, 0.0)
        if need <= 1e-6:
            continue
        step = need * fill_p
        hires_visible[vis_idx] += float(step)
        hires_post_month[vis_idx] = int(months[post_idx])
        hires_reason[vis_idx] = f"Winter step (post {month_name(months[post_idx])}) — filled @ {fill_p*100:.0f}%"

    # Replacement-to-base pipeline (post only outside freeze)
    hires_visible_pipeline = hires_visible[:]
    hires_post_month_pipeline = hires_post_month[:]
    hires_reason_pipeline = hires_reason[:]

    for t in range(N_MONTHS):
        if lead_months <= 0 or t + lead_months >= N_MONTHS:
            continue
        if int(months[t]) in set(params.freeze_months):
            continue

        v = t + lead_months
        current_paid_t = max(float(paid_1[t]), float(policy.base_fte))
        proj = current_paid_t * ((1.0 - monthly_turnover) ** float(lead_months))
        proj += float(hires_visible_pipeline[v])

        if proj < float(policy.base_fte) - 1e-6:
            need = float(policy.base_fte) - proj
            add = need * fill_p
            hires_visible_pipeline[v] += float(add)
            hires_post_month_pipeline[v] = int(months[t])
            reason = "Base replacement pipeline"
            hires_reason_pipeline[v] = (hires_reason_pipeline[v] + " | " if hires_reason_pipeline[v] else "") + reason

    # Final supply run
    paid, eff, turnover_shed_arr = run_supply(hires_visible_pipeline)

    perm_eff = np.array(eff, dtype=float)
    perm_paid = np.array(paid, dtype=float)
    v_peak = np.array(visits_peak, dtype=float)
    v_avg = np.array(visits_curve_flu, dtype=float)
    dim = np.array(days_in_month, dtype=float)

    # Required coverage (shifts/day) from peak demand
    visits_per_shift = max(float(params.visits_per_provider_shift), 1e-6)
    req_provider_shifts_per_day = v_peak / visits_per_shift
    req_provider_shifts_per_day_rounded = np.round(req_provider_shifts_per_day / 0.25) * 0.25

    # Load (PPPD) from provider-day equiv (perm only)
    prov_day_equiv_perm = np.array(
        [provider_day_equiv_from_fte(f, params.hours_week, params.fte_hours_week) for f in perm_eff],
        dtype=float,
    )
    load_pre = v_peak / np.maximum(prov_day_equiv_perm, 1e-6)

    # Flex sizing to target Yellow max (bounded)
    # Intuition: flex is safety stock to keep you out of Red / reduce strain.
    # It is NOT intended to be the primary baseline coverage.
    flex_fte = np.zeros(N_MONTHS, dtype=float)
    load_post = np.zeros(N_MONTHS, dtype=float)

    target_pppd_post = float(params.yellow_max_pppd)
    for i in range(N_MONTHS):
        # How much total effective FTE is required to hit the PPPD target?
        req_eff_fte_i = fte_required_for_pppd(
            visits_per_day=float(v_peak[i]),
            pppd_target=target_pppd_post,
            hours_week=float(params.hours_week),
            fte_hours_week=float(params.fte_hours_week),
        )

        gap_fte = max(req_eff_fte_i - float(perm_eff[i]), 0.0)
        flex_used = min(gap_fte, float(params.flex_max_fte_per_month))
        flex_fte[i] = float(flex_used)

        prov_day_equiv_total = provider_day_equiv_from_fte(
            float(perm_eff[i] + flex_used), params.hours_week, params.fte_hours_week
        )
        load_post[i] = float(v_peak[i]) / max(float(prov_day_equiv_total), 1e-6)

    # Residual gap to Yellow after flex
    req_eff_fte_needed_yellow = np.array(
        [fte_required_for_pppd(float(v_peak[i]), float(params.yellow_max_pppd), params.hours_week, params.fte_hours_week) for i in range(N_MONTHS)],
        dtype=float,
    )
    residual_gap_fte = np.maximum(req_eff_fte_needed_yellow - (perm_eff + flex_fte), 0.0)
    provider_day_gap_total = float(np.sum(residual_gap_fte * dim))

    # Lost visits & access risk (margin lost)
    est_visits_lost = float(provider_day_gap_total) * float(params.visits_lost_per_provider_day_gap)
    est_margin_at_risk = est_visits_lost * float(params.net_revenue_per_visit)

    # Turnover replacement cost (providers only) — monthly then total
    replacements_base = perm_paid * float(monthly_turnover)
    repl_mult = np.ones(N_MONTHS, dtype=float)
    repl_mult = np.where(load_post > float(params.budgeted_pppd), float(params.turnover_yellow_mult), repl_mult)
    repl_mult = np.where(load_post > float(params.red_start_pppd), float(params.turnover_red_mult), repl_mult)
    turnover_replacement_cost_month = replacements_base * float(params.provider_replacement_cost) * repl_mult
    turnover_replacement_cost = float(np.sum(turnover_replacement_cost_month))

    # SWB/Visit affordability (annual, includes flex)
    annual_swb, total_swb_dollars, total_visits = annual_swb_per_visit_from_supply(
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

    # Provider cost approximation (provider-only) for exposure
    hours_per_year = float(params.fte_hours_week) * 52.0
    apc_loaded_hr = loaded_hourly_rate(params.hourly_rates["apc"], params.benefits_load_pct, params.ot_sick_pct, params.bonus_pct)
    loaded_cost_per_provider_fte = float(apc_loaded_hr) * float(hours_per_year)
    permanent_provider_cost = float(np.sum(perm_paid * loaded_cost_per_provider_fte * (dim / 365.0)))
    flex_provider_cost = float(np.sum(flex_fte * loaded_cost_per_provider_fte * float(params.flex_cost_multiplier) * (dim / 365.0)))

    # Burnout metrics
    green_cap = float(params.budgeted_pppd)
    red_start = float(params.red_start_pppd)
    months_yellow = int(np.sum((load_post > green_cap + 1e-9) & (load_post <= red_start + 1e-9)))
    months_red = int(np.sum(load_post > red_start + 1e-9))
    peak_load = float(np.max(load_post)) if len(load_post) else 0.0

    # Penalties
    under_util = np.maximum((green_cap * 0.70) - load_post, 0.0)
    overstaff_penalty = float(np.sum(under_util * dim))

    yellow_excess = np.maximum(load_post - green_cap, 0.0)
    red_excess = np.maximum(load_post - red_start, 0.0)
    burnout_penalty = float(np.sum((yellow_excess ** 1.2) * dim) + 3.0 * np.sum((red_excess ** 2.0) * dim))

    swb_violation = max(float(annual_swb) - float(params.target_swb_per_visit), 0.0)
    swb_penalty = float(swb_violation) * 1_000_000.0

    total_score = (
        permanent_provider_cost
        + flex_provider_cost
        + turnover_replacement_cost
        + est_margin_at_risk
        + 2_000.0 * burnout_penalty
        + 500.0 * overstaff_penalty
        + swb_penalty
    )

    # ----------------------------
    # Ledger (36 months)
    # ----------------------------
    rows: list[dict[str, Any]] = []
    for i in range(N_MONTHS):
        lab = dates[i].strftime("%Y-%b")
        month_total_visits = float(v_avg[i]) * float(days_in_month[i])

        # Monthly SWB/visit + dollars (audit)
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

        # Monthly access risk (margin lost) — aligned with residual gap to Yellow
        month_provider_day_gap = float(residual_gap_fte[i]) * float(days_in_month[i])
        month_visits_lost = month_provider_day_gap * float(params.visits_lost_per_provider_day_gap)
        month_access_margin_lost = month_visits_lost * float(params.net_revenue_per_visit)

        # Monthly net contribution (counterfactual margin before staffing costs)
        month_net_contribution = float(month_total_visits) * float(params.net_revenue_per_visit)

        is_yellow = int((load_post[i] > green_cap + 1e-9) and (load_post[i] <= red_start + 1e-9))
        is_red = int(load_post[i] > red_start + 1e-9)

        rows.append(
            {
                "Month": lab,
                "Visits/Day (avg)": float(v_avg[i]),
                "Visits/Day (peak)": float(v_peak[i]),
                "Total Visits (month)": float(month_total_visits),
                "Net Contribution (month)": float(month_net_contribution),

                "SWB Dollars (month)": float(month_swb_dollars),
                "SWB/Visit (month)": float(month_swb_per_visit),

                "Permanent FTE (Paid)": float(perm_paid[i]),
                "Permanent FTE (Effective)": float(perm_eff[i]),
                "Flex FTE Used": float(flex_fte[i]),

                "Required Provider Coverage (shifts/day)": float(req_provider_shifts_per_day[i]),
                "Rounded Coverage (0.25)": float(req_provider_shifts_per_day_rounded[i]),

                "Load PPPD (pre-flex)": float(load_pre[i]),
                "Load PPPD (post-flex)": float(load_post[i]),

                "Turnover Shed (FTE)": float(turnover_shed_arr[i]),
                "Hires Visible (FTE)": float(hires_visible_pipeline[i]),
                "Hire Reason": hires_reason_pipeline[i],
                "Hire Post Month": hires_post_month_pipeline[i],

                "Residual FTE Gap (to Yellow)": float(residual_gap_fte[i]),
                "Access Margin at Risk (month)": float(month_access_margin_lost),
                "Turnover Replacement Cost (month)": float(turnover_replacement_cost_month[i]),

                "Is_Yellow": is_yellow,
                "Is_Red": is_red,
            }
        )

    ledger = pd.DataFrame(rows)
    annual_summary = build_annual_summary(ledger)

    return {
        "dates": list(dates),
        "months": months,
        "days_in_month": list(days_in_month),
        "visits_avg": list(v_avg),
        "visits_peak": list(v_peak),
        "perm_paid": list(perm_paid),
        "perm_eff": list(perm_eff),
        "flex_fte": list(flex_fte),
        "load_pre": list(load_pre),
        "load_post": list(load_post),

        "provider_day_gap_total": provider_day_gap_total,
        "est_visits_lost": est_visits_lost,
        "est_margin_at_risk": est_margin_at_risk,
        "turnover_replacement_cost": turnover_replacement_cost,

        "annual_swb_per_visit": annual_swb,
        "total_swb_dollars": total_swb_dollars,
        "total_visits": total_visits,

        "permanent_provider_cost": permanent_provider_cost,
        "flex_provider_cost": flex_provider_cost,

        "months_yellow": months_yellow,
        "months_red": months_red,
        "peak_load": peak_load,
        "score": float(total_score),

        "ledger": ledger,
        "role_mix": role_mix,
        "lead_months": int(lead_months),
        "annual_summary": annual_summary,
    }


def recommend_policy(
    params: ModelParams,
    base_min: float,
    base_max: float,
    base_step: float,
    winter_delta_max: float,
    winter_step: float,
) -> dict[str, Any]:
    """
    Transparent grid search:
      base_fte in [base_min, base_max]
      winter_fte in [base_fte, base_fte + winter_delta_max]
    Returns best result + frontier table.
    """
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
                    "Annual_SWB_per_Visit": float(res["annual_swb_per_visit"]),
                    "MarginAtRisk": float(res["est_margin_at_risk"]),
                    "TurnoverReplaceCost": float(res["turnover_replacement_cost"]),
                    "FlexCost": float(res["flex_provider_cost"]),
                    "Months_Yellow": int(res["months_yellow"]),
                    "Months_Red": int(res["months_red"]),
                    "Peak_Load_PPPD": float(res["peak_load"]),
                }
            )

            if best is None or float(res["score"]) < float(best["res"]["score"]):
                best = {"policy": pol, "res": res}

    frontier = pd.DataFrame(candidates).sort_values(["Score", "Annual_SWB_per_Visit"]).reset_index(drop=True)
    return {"best": best, "frontier": frontier}


# ============================================================
# CACHES (keyed by params_dict including "_v")
# ============================================================
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
# SIDEBAR — INPUTS (all with help="?")
# ============================================================
with st.sidebar:
    st.header("Demand")
    visits = st.number_input("Avg Visits/Day (annual baseline)", min_value=1.0, value=36.0, step=1.0, help=HELP["visits"])
    annual_growth = st.number_input("Annual Visit Growth %", min_value=0.0, value=10.0, step=1.0, help=HELP["annual_growth"]) / 100.0
    peak_factor = st.slider("Peak-to-average factor", 1.00, 1.50, 1.20, 0.01, help=HELP["peak_factor"])

    st.subheader("Seasonality + Flu")
    seasonality_pct = st.number_input("Seasonality swing % (winter up, summer down)", min_value=0.0, value=20.0, step=5.0, help=HELP["seasonality_pct"]) / 100.0
    flu_uplift_pct = st.number_input("Flu uplift % (selected months)", min_value=0.0, value=0.0, step=5.0, help=HELP["flu_uplift_pct"]) / 100.0
    flu_months = st.multiselect(
        "Flu months",
        options=MONTH_OPTIONS,
        default=[("Oct", 10), ("Nov", 11), ("Dec", 12), ("Jan", 1), ("Feb", 2)],
        help=HELP["flu_months"],
    )
    flu_months_set = {m for _, m in flu_months} if flu_months else set()

    visits_per_provider_shift = st.number_input("Visits per provider shift (sweet spot)", min_value=5.0, value=36.0, step=1.0, help=HELP["visits_per_provider_shift"])

    st.header("Clinic Ops")
    hours_week = st.number_input("Hours of Operation / Week", min_value=1.0, value=84.0, step=1.0, help=HELP["hours_week"])
    days_open_per_week = st.number_input("Days Open / Week", min_value=1.0, max_value=7.0, value=7.0, step=1.0, help=HELP["days_open_per_week"])
    fte_hours_week = st.number_input("FTE Hours / Week", min_value=1.0, value=36.0, step=1.0, help=HELP["fte_hours_week"])

    st.header("Workforce Dynamics")
    annual_turnover = st.number_input("Annual Turnover %", min_value=0.0, value=16.0, step=1.0, help=HELP["annual_turnover"]) / 100.0
    lead_days = st.number_input("Days to Independent (Req→Independent)", min_value=0, value=210, step=10, help=HELP["lead_days"])
    ramp_months = st.slider("Ramp months after independent", 0, 6, 1, 1, help=HELP["ramp_months"])
    ramp_productivity = st.slider("Ramp productivity %", 30, 100, 75, 5, help=HELP["ramp_productivity"]) / 100.0
    fill_probability = st.slider("Fill probability %", 0, 100, 85, 5, help=HELP["fill_probability"]) / 100.0

    st.header("Patients per provider per day thresholds (PPPD)")
    st.caption("These thresholds define operating zones for capacity and risk penalties.")
    budgeted_pppd = st.number_input("Target threshold (Green PPPD)", min_value=5.0, value=36.0, step=1.0, help=HELP["budgeted_pppd"])
    yellow_max_pppd = st.number_input("Caution threshold (Yellow PPPD)", min_value=5.0, value=42.0, step=1.0, help=HELP["yellow_max_pppd"])
    red_start_pppd = st.number_input("High-risk threshold (Red PPPD)", min_value=5.0, value=45.0, step=1.0, help=HELP["red_start_pppd"])

    st.header("Hiring Freeze")
    winter_anchor_month = st.selectbox("Winter anchor month (ready-by)", options=MONTH_OPTIONS, index=11, help=HELP["winter_anchor_month"])
    winter_anchor_month_num = int(winter_anchor_month[1])
    freeze_months = st.multiselect(
        "Freeze posting months (typically flu months)",
        options=MONTH_OPTIONS,
        default=flu_months if flu_months else [("Oct", 10), ("Nov", 11), ("Dec", 12), ("Jan", 1), ("Feb", 2)],
        help=HELP["freeze_months"],
    )
    freeze_months_set = {m for _, m in freeze_months} if freeze_months else set()

    st.header("Flex (Safety Stock)")
    flex_max_fte_per_month = st.slider("Max flex FTE available per month", 0.0, 10.0, 2.0, 0.25, help=HELP["flex_max"])
    flex_cost_multiplier = st.slider("Flex cost multiplier vs base provider", 1.0, 2.0, 1.25, 0.05, help=HELP["flex_mult"])

    st.header("Finance Governance")
    target_swb_per_visit = st.number_input("Target SWB/Visit (annual)", min_value=0.0, value=85.0, step=1.0, help=HELP["target_swb"])
    net_revenue_per_visit = st.number_input("Net Contribution per Visit (for access risk)", min_value=0.0, value=140.0, step=5.0, help=HELP["net_contrib"])
    visits_lost_per_provider_day_gap = st.number_input("Visits Lost per 1.0 Provider-Day Gap", min_value=0.0, value=18.0, step=1.0, help=HELP["visits_lost"])

    st.subheader("Provider turnover replacement cost (providers only)")
    provider_replacement_cost = st.number_input("Replacement cost per 1.0 provider FTE", min_value=0.0, value=75000.0, step=5000.0, help=HELP["repl_cost"])
    turnover_yellow_mult = st.slider("Turnover cost multiplier (yellow months)", 1.0, 3.0, 1.3, 0.05, help=HELP["turn_yellow"])
    turnover_red_mult = st.slider("Turnover cost multiplier (red months)", 1.0, 5.0, 2.0, 0.10, help=HELP["turn_red"])

    st.subheader("Wage inputs (for SWB/Visit)")
    benefits_load_pct = st.number_input("Benefits Load %", min_value=0.0, value=30.0, step=1.0, help=HELP["benefits"]) / 100.0
    bonus_pct = st.number_input("Bonus % of base", min_value=0.0, value=10.0, step=1.0, help=HELP["bonus"]) / 100.0
    ot_sick_pct = st.number_input("OT + Sick/PTO %", min_value=0.0, value=4.0, step=0.5, help=HELP["ot_sick"]) / 100.0

    physician_hr = st.number_input("Physician (optional) $/hr", min_value=0.0, value=135.79, step=1.0, help=HELP["phys_hr"])
    apc_hr = st.number_input("APP $/hr", min_value=0.0, value=62.0, step=1.0, help=HELP["apc_hr"])
    ma_hr = st.number_input("MA $/hr", min_value=0.0, value=24.14, step=0.5, help=HELP["ma_hr"])
    psr_hr = st.number_input("PSR $/hr", min_value=0.0, value=21.23, step=0.5, help=HELP["psr_hr"])
    rt_hr = st.number_input("RT $/hr", min_value=0.0, value=31.36, step=0.5, help=HELP["rt_hr"])
    supervisor_hr = st.number_input("Supervisor (optional) $/hr", min_value=0.0, value=28.25, step=0.5, help=HELP["sup_hr"])

    physician_supervision_hours_per_month = st.number_input("Physician supervision hours/month", min_value=0.0, value=0.0, step=1.0, help=HELP["phys_sup_hours"])
    supervisor_hours_per_month = st.number_input("Supervisor hours/month", min_value=0.0, value=0.0, step=1.0, help=HELP["sup_hours"])

    st.header("Operational Guardrails")
    allow_low_perm_for_prn = st.checkbox(
        "Allow permanent staffing below minimum coverage (I have reliable PRN/per diem)",
        value=False,
        help=HELP["guard_min_coverage"],
    )
    enforce_green_on_avg_without_flex = st.checkbox(
        "Enforce permanent staffing to stay ≤ Green PPPD on AVG demand (no flex)",
        value=True,
        help=HELP["guard_green_no_flex"],
    )

    st.header("Optimizer Controls")
    base_min = st.number_input("Base FTE min", min_value=0.0, value=1.0, step=0.25, help=HELP["base_min"])
    base_max = st.number_input("Base FTE max", min_value=0.0, value=6.0, step=0.25, help=HELP["base_max"])
    base_step = st.select_slider("Base FTE step", options=[0.05, 0.10, 0.25, 0.50], value=0.25, help=HELP["base_step"])
    winter_delta_max = st.number_input("Max winter uplift above base (FTE)", min_value=0.0, value=2.0, step=0.25, help=HELP["winter_delta_max"])
    winter_step = st.select_slider("Winter uplift step", options=[0.05, 0.10, 0.25, 0.50], value=0.25, help=HELP["winter_step"])

    st.header("Mode")
    mode = st.radio("Run mode", ["Recommend + What-If", "What-If only"], index=0)

    st.divider()
    run_recommender = st.button("🏁 Run Recommender (Grid Search)", use_container_width=True)
    st.caption("What-If updates live. Recommender runs only when you click the button.")


# ============================================================
# GUARDRAIL FLOORS (computed from inputs)
# ============================================================
# A) Clinic-hours minimum coverage floor: 1 provider always-on for all open hours.
#    If the clinic is open 84 hrs/week and 1.0 FTE = 36 hrs/week, then minimum paid FTE for always-on coverage = 84/36 = 2.33
min_perm_fte_hours_floor = float(hours_week) / max(float(fte_hours_week), 1e-6)

# B) Green-on-AVG floor (productivity expectation, no flex baseline):
#    We enforce using AVG demand (incl seasonality + flu uplift), and split into base vs high-season months.
today = datetime.today()
year0 = today.year
dates_tmp = pd.date_range(start=datetime(year0, 1, 1), periods=N_MONTHS, freq="MS")
months_tmp = [int(d.month) for d in dates_tmp]

base_year0 = float(visits)
base_year1 = float(visits) * (1.0 + float(annual_growth))
base_year2 = float(visits) * (1.0 + float(annual_growth)) ** 2
v_avg_base = compute_visits_curve(months_tmp, base_year0, base_year1, base_year2, float(seasonality_pct))
v_avg_flu = apply_flu_uplift(v_avg_base, months_tmp, set(flu_months_set), float(flu_uplift_pct))

high_season_months = set(flu_months_set) | WINTER  # conservative

req_fte_green_each_month = [
    fte_required_for_pppd(v, float(budgeted_pppd), float(hours_week), float(fte_hours_week)) for v in v_avg_flu
]
req_green_high = max([req_fte_green_each_month[i] for i, m in enumerate(months_tmp) if int(m) in high_season_months] or [0.0])
req_green_low = max([req_fte_green_each_month[i] for i, m in enumerate(months_tmp) if int(m) not in high_season_months] or [0.0])

# Apply enforcement defaults unless PRN override is checked
enforced_base_floor = 0.0
enforced_winter_floor = 0.0

if not allow_low_perm_for_prn:
    enforced_base_floor = max(enforced_base_floor, min_perm_fte_hours_floor)
    enforced_winter_floor = max(enforced_winter_floor, min_perm_fte_hours_floor)

if enforce_green_on_avg_without_flex:
    enforced_base_floor = max(enforced_base_floor, float(req_green_low))
    enforced_winter_floor = max(enforced_winter_floor, float(req_green_high))

# Winter must be >= base by policy definition
enforced_winter_floor = max(enforced_winter_floor, enforced_base_floor)


# ============================================================
# BUILD PARAMS (inject _v for cache busting)
# ============================================================
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
    visits_per_provider_shift=float(visits_per_provider_shift),

    hours_week=float(hours_week),
    days_open_per_week=float(days_open_per_week),
    fte_hours_week=float(fte_hours_week),

    annual_turnover=float(annual_turnover),
    lead_days=int(lead_days),
    ramp_months=int(ramp_months),
    ramp_productivity=float(ramp_productivity),
    fill_probability=float(fill_probability),

    winter_anchor_month=int(winter_anchor_month_num),
    freeze_months=set(freeze_months_set),

    budgeted_pppd=float(budgeted_pppd),
    yellow_max_pppd=float(yellow_max_pppd),
    red_start_pppd=float(red_start_pppd),

    flex_max_fte_per_month=float(flex_max_fte_per_month),
    flex_cost_multiplier=float(flex_cost_multiplier),

    target_swb_per_visit=float(target_swb_per_visit),
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

    allow_low_perm_for_prn=bool(allow_low_perm_for_prn),
    enforce_green_on_avg_without_flex=bool(enforce_green_on_avg_without_flex),

    _v=MODEL_VERSION,
)

# IMPORTANT: use params_dict everywhere you call cached_* so the cache key includes _v
params_dict: dict[str, Any] = {**params.__dict__, "_v": MODEL_VERSION}


# ============================================================
# RUN RECOMMENDER (ONLY ON CLICK) + WHAT-IF (ALWAYS LIVE)
# ============================================================
best_block = None
frontier = None

# Enforce grid search bounds quietly (but transparently explained in assumptions)
base_min_eff = max(float(base_min), float(enforced_base_floor))
base_max_eff = max(float(base_max), base_min_eff)
winter_delta_max_eff = float(winter_delta_max)

if mode == "Recommend + What-If" and run_recommender:
    with st.spinner("Evaluating policy candidates (grid search)…"):
        rec = cached_recommend_policy(
            params_dict=params_dict,
            base_min=float(base_min_eff),
            base_max=float(base_max_eff),
            base_step=float(base_step),
            winter_delta_max=float(winter_delta_max_eff),
            winter_step=float(winter_step),
        )
    best_block = rec["best"]
    frontier = rec["frontier"]

    st.session_state.rec_policy = best_block["policy"]
    st.session_state.frontier = frontier

    # Snap What-If controls to new recommendation
    st.session_state.what_base_fte = float(st.session_state.rec_policy.base_fte)
    st.session_state.what_winter_fte = float(st.session_state.rec_policy.winter_fte)

if frontier is None and st.session_state["frontier"] is not None:
    frontier = st.session_state["frontier"]

rec_policy = st.session_state.rec_policy


# ============================================================
# Recommended baseline for comparisons (compute BEFORE using it)
# ============================================================
R_rec = None
if mode == "Recommend + What-If" and rec_policy is not None:
    R_rec = cached_simulate(params_dict, float(rec_policy.base_fte), float(rec_policy.winter_fte))


# ============================================================
# What-If selection (LIVE) with guardrail-aware defaults
# ============================================================
st.subheader("What-If Policy Inputs")

if rec_policy is not None:
    default_base = float(rec_policy.base_fte)
    default_winter = float(rec_policy.winter_fte)
else:
    default_base = float(max(base_min_eff, 0.0))
    default_winter = float(max(default_base + min(winter_delta_max_eff, 1.0), enforced_winter_floor))

# Initialize state once
st.session_state.setdefault("what_base_fte", float(default_base))
st.session_state.setdefault("what_winter_fte", float(default_winter))

# Enforce relational rule (winter >= base)
if float(st.session_state["what_winter_fte"]) < float(st.session_state["what_base_fte"]):
    st.session_state["what_winter_fte"] = float(st.session_state["what_base_fte"])

min_base_input = 0.0 if allow_low_perm_for_prn else float(enforced_base_floor)
min_winter_input = float(enforced_winter_floor) if not allow_low_perm_for_prn else float(st.session_state["what_base_fte"])

w1, w2 = st.columns(2)
with w1:
    what_base = st.number_input(
        "What-If Base FTE",
        min_value=float(min_base_input),
        value=float(st.session_state["what_base_fte"]),
        step=0.25,
        key="what_base_fte",
        help=(
            "Annual base permanent provider FTE level for the What-If scenario. "
            + ("Guardrails enforce a minimum unless PRN override is enabled." if not allow_low_perm_for_prn else "PRN override enabled: lower permanent allowed.")
        ),
    )
with w2:
    what_winter_min = max(float(what_base), float(min_winter_input))
    what_winter = st.number_input(
        "What-If Winter FTE",
        min_value=float(what_winter_min),
        value=float(st.session_state["what_winter_fte"]),
        step=0.25,
        key="what_winter_fte",
        help="Winter target permanent provider FTE level for the What-If scenario (must be ≥ base).",
    )

R = cached_simulate(params_dict, float(what_base), float(what_winter))


# ============================================================
# TOP STICKY SCORECARD (frozen-cells feel)
# ============================================================
annual = R.get("annual_summary")
year1_row = annual.iloc[0] if annual is not None and len(annual) > 0 else None

risk = compute_risk_flags(R, green_cap=float(budgeted_pppd), red_start=float(red_start_pppd))
risk_tier = str(risk["tier"])

feasible = float(R["annual_swb_per_visit"]) <= float(params.target_swb_per_visit) + 1e-9
risk_badge = pill(f"Risk: {risk_tier}", "bad" if risk_tier == "High" else ("warn" if risk_tier == "Moderate" else "ok"))
swb_badge = pill("SWB/Visit: OK" if feasible else "SWB/Visit: Over target", "ok" if feasible else "warn")

y1_swb = float(year1_row["SWB_per_Visit"]) if year1_row is not None else float(R["annual_swb_per_visit"])
y1_swb_dollars = float(year1_row["SWB_Dollars"]) if year1_row is not None else float(R["total_swb_dollars"]) / 3.0
y1_ebitda = float(year1_row["EBITDA_Proxy"]) if year1_row is not None else float("nan")

st.markdown('<div class="sticky">', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="small">
  {swb_badge}
  {risk_badge}
  <span class="pill">Guardrails: {"ON" if (not allow_low_perm_for_prn or enforce_green_on_avg_without_flex) else "OFF"}</span>
  <span class="pill muted">Model v: {MODEL_VERSION}</span>
</div>
""",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("SWB/Visit (Yr 1)", f"${y1_swb:,.2f}", f"Target ${params.target_swb_per_visit:,.2f}")
k2.metric("Staffing Expense (Yr 1)", f"${y1_swb_dollars:,.0f}")
k3.metric("EBITDA Proxy (Yr 1)", f"${y1_ebitda:,.0f}" if not np.isnan(y1_ebitda) else "—")
k4.metric("Avg Visits/Day (Yr 1)", f"{float(year1_row['Visits']/365.0):.1f}" if year1_row is not None else "—")
k5.metric("Peak Load (PPPD)", f"{R['peak_load']:.1f}")
k6.metric("Flex Share (36 mo)", f"{risk['flex_share']*100:.0f}%")
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DECLARE ASSUMPTIONS (plain language)
# ============================================================
with st.expander("Assumptions (what this model is assuming)", expanded=False):
    st.markdown(
        f"""
- **Planning horizon:** 36 months starting **{datetime.today().strftime("%Y-%b")}**.
- **Demand:** baseline **{visits:.1f} visits/day**, growth **{annual_growth*100:.1f}%/yr**, seasonality **±{seasonality_pct*100:.0f}%**, flu uplift **+{flu_uplift_pct*100:.0f}%** for {sorted(list(flu_months_set)) if flu_months_set else "no flu months"}.
- **Peak planning:** peak visits/day uses factor **{peak_factor:.2f}×** on the avg+flu curve.
- **Productivity:** sweet spot **{visits_per_provider_shift:.0f} visits/provider shift**.
- **Workforce:** lead time **{lead_days} days (~{lead_days_to_months(int(lead_days))} months)**, ramp **{ramp_months} months @ {ramp_productivity*100:.0f}%**, fill probability **{fill_probability*100:.0f}%**, turnover **{annual_turnover*100:.0f}%/yr**.
- **Freeze:** no posting in months {sorted(list(freeze_months_set)) if freeze_months_set else "none"}; winter anchor **{month_name(winter_anchor_month_num)}**.
- **Flex:** max **{flex_max_fte_per_month:.2f} FTE/month**, premium **{flex_cost_multiplier:.2f}×**.
- **Governance:** SWB/Visit target **${target_swb_per_visit:.2f}**; access loss converts gaps to visits lost (**{visits_lost_per_provider_day_gap:.1f} visits per provider-day gap**) and margin lost (**${net_revenue_per_visit:.0f}/visit**).
""".strip()
    )

    guard_lines = []
    guard_lines.append(f"- **Minimum permanent coverage (clinic hours):** {min_perm_fte_hours_floor:.2f} FTE (computed as hours/week ÷ FTE hours/week).")
    if enforce_green_on_avg_without_flex:
        guard_lines.append(f"- **Green-on-AVG demand floor:** base ≥ {req_green_low:.2f} FTE; winter ≥ {req_green_high:.2f} FTE (computed from AVG demand to keep ≤ Green PPPD).")
    if allow_low_perm_for_prn:
        guard_lines.append("- **PRN override is ON:** you may staff below the minimum coverage floor if you have reliable PRN/per diem coverage.")
    else:
        guard_lines.append("- **PRN override is OFF:** permanent staffing below minimum coverage is blocked in optimizer and constrained in What-If.")
    st.markdown("\n".join(guard_lines))


# ============================================================
# Recommended block UI (if available)
# ============================================================
if mode == "Recommend + What-If" and rec_policy is not None:
    if best_block is None:
        best_res = cached_simulate(params_dict, float(rec_policy.base_fte), float(rec_policy.winter_fte))
        best_policy = rec_policy
    else:
        best_policy = best_block["policy"]
        best_res = best_block["res"]

    st.markdown("---")
    st.header("Recommended Permanent Staffing Policy")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base FTE (Annual)", f"{best_policy.base_fte:.2f}")
    c2.metric("Winter FTE (Target)", f"{best_policy.winter_fte:.2f}")
    c3.metric("Annual SWB/Visit", f"${best_res['annual_swb_per_visit']:.2f}")
    c4.metric("Access Risk (Margin at risk)", f"${best_res['est_margin_at_risk']:,.0f}")

    st.markdown(
        """
<div class="contract">
  <b>Policy contract (what the recommender is doing)</b>
  <ul class="small" style="margin-top:8px;">
    <li><b>Two-level permanent staffing:</b> Annual base + winter target (achieved by posting earlier than anchor month if needed).</li>
    <li><b>Freeze months:</b> No posting during freeze. Replacements only restore toward <b>annual base</b> outside freeze.</li>
    <li><b>Flex:</b> Used as safety stock to reduce peak-month load toward <b>Yellow max</b>, bounded by max flex capacity.</li>
    <li><b>Governance:</b> Optimization penalizes red-zone exposure, turnover replacement cost, access margin lost, and SWB/Visit over target.</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    if frontier is not None:
        with st.expander("Candidate table (top 25 by score)", expanded=False):
            st.dataframe(frontier.head(25), use_container_width=True, hide_index=True)

elif mode == "What-If only":
    st.markdown("---")
    st.header("What-If (no recommendation run)")


# ============================================================
# SURFACE TRADEOFFS + EXPLAIN RISK (plain language)
# ============================================================
st.markdown("---")
st.header("Tradeoffs & Risk (What this policy is buying you, and what it’s costing you)")

risk_kind = "bad" if risk_tier == "High" else ("warn" if risk_tier == "Moderate" else "ok")
risk_text = (
    "High risk: this plan depends heavily on flex and/or experiences red-zone streaks."
    if risk_tier == "High"
    else ("Moderate risk: some reliance on flex or short red exposures. Monitor closely."
          if risk_tier == "Moderate"
          else "Low risk: flex is used as true safety stock and red exposures are limited.")
)

t1, t2, t3 = st.columns(3)
t1.metric("Risk Tier", risk_tier)
t2.metric("Flex months used", f"{risk['months_flex_used']} / {N_MONTHS}")
t3.metric("Max red streak", f"{risk['max_red_streak']}")

st.markdown(
    f"""
- **Flex reliance:** about **{risk['flex_share']*100:.0f}%** of total coverage over 36 months comes from flex (higher = more operational fragility).
- **Permanent sufficiency:** permanent-only load exceeds **Green** in **{risk['months_perm_over_green']}** months (pre-flex).
- **Red streaks:** longest continuous red run is **{risk['max_red_streak']}** month(s).
- **Plain-language read:** {risk_text}
""".strip()
)

with st.expander("Failure modes (what would make this plan break)", expanded=False):
    st.markdown(
        f"""
This policy becomes unstable if any of the following happen:

1) **Flex doesn’t show up** (availability below your assumption) while the plan is using flex in **{risk['months_flex_used']}** months.  
2) **Turnover is higher than modeled** (currently {annual_turnover*100:.0f}%/yr) during Yellow/Red periods, which amplifies replacement cost.  
3) **Demand exceeds peak planning** (peak factor {peak_factor:.2f}× + flu uplift) for multiple months.  
4) **Hiring lead time expands** (currently {lead_days} days), delaying winter readiness and extending capacity gaps.
""".strip()
    )


# ============================================================
# CHARTS
# ============================================================
st.markdown("---")
st.header("3-Year Planning Horizon (Policy View)")

dates = R["dates"]
perm_eff = np.array(R["perm_eff"], dtype=float)
perm_paid = np.array(R["perm_paid"], dtype=float)
flex_fte = np.array(R["flex_fte"], dtype=float)
vis_avg = np.array(R["visits_avg"], dtype=float)
vis_peak = np.array(R["visits_peak"], dtype=float)
load_pre = np.array(R["load_pre"], dtype=float)
load_post = np.array(R["load_post"], dtype=float)

tick_idx = list(range(0, len(dates), 3))

# --- Chart 1: Supply + Flex (FTE) ---
fig1, ax = plt.subplots(figsize=(12, 5.2))
ax.plot(dates, perm_eff, linewidth=2.2, color=BRAND_BLACK, label="Permanent FTE (Effective)")
ax.plot(dates, perm_paid, linewidth=1.7, linestyle="--", color=MID_GRAY, label="Permanent FTE (Paid)")
ax.plot(dates, perm_eff + flex_fte, linewidth=2.0, linestyle=":", color=BRAND_GOLD, label="Total Coverage (Effective + Flex)")
ax.fill_between(dates, perm_eff, perm_eff + flex_fte, alpha=0.12, color=BRAND_GOLD, label="Flex Used")

ax.set_title("Permanent Policy + Flex Usage (36 months)", fontsize=13, fontweight="bold")
ax.set_ylabel("Provider FTE", fontsize=11, fontweight="bold")
ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)
ax.set_xticks([dates[i] for i in tick_idx])
ax.set_xticklabels([dates[i].strftime("%Y-%b") for i in tick_idx], rotation=0, fontsize=9)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
plt.tight_layout()
st.pyplot(fig1)

# --- Chart 2: Load PPPD with bands + Demand overlay ---
fig2, axL = plt.subplots(figsize=(12, 5.2))
axL.plot(dates, load_post, linewidth=2.2, color=BRAND_GOLD, label="Load PPPD (post-flex)")
axL.plot(dates, load_pre, linewidth=1.2, linestyle="--", color=GRAY, alpha=0.8, label="Load PPPD (pre-flex)")

axL.axhspan(0, params.budgeted_pppd, alpha=0.06, color="#00aa00")
axL.axhspan(params.budgeted_pppd, params.red_start_pppd, alpha=0.06, color="#ffaa00")
axL.axhspan(params.red_start_pppd, max(params.red_start_pppd + 10, float(np.max(load_pre) + 5)), alpha=0.06, color="#ff0000")
axL.axhline(params.budgeted_pppd, linewidth=1.2, color="#00aa00", linestyle=":")
axL.axhline(params.red_start_pppd, linewidth=1.2, color="#ff0000", linestyle=":")

axL.set_title("Monthly Load (PPPD) with Risk Bands", fontsize=13, fontweight="bold")
axL.set_ylabel("Patients / Provider / Day", fontsize=11, fontweight="bold")
axL.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)

axR = axL.twinx()
axR.plot(dates, vis_avg, linewidth=1.3, linestyle="-.", color=MID_GRAY, label="Visits/Day (avg, incl flu)")
axR.plot(dates, vis_peak, linewidth=1.3, linestyle=":", color=MID_GRAY, alpha=0.85, label="Visits/Day (peak adj)")
axR.set_ylabel("Visits / Day", fontsize=11, fontweight="bold")

axL.set_xticks([dates[i] for i in tick_idx])
axL.set_xticklabels([dates[i].strftime("%Y-%b") for i in tick_idx], rotation=0, fontsize=9)

lines1, labels1 = axL.get_legend_handles_labels()
lines2, labels2 = axR.get_legend_handles_labels()
axL.legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
plt.tight_layout()
st.pyplot(fig2)


# ============================================================
# FINANCE SUMMARY (Annualized, plus recommended comparison)
# ============================================================
st.markdown("---")
st.header("Finance & Risk Summary (Annualized)")

f1, f2, f3, f4 = st.columns(4)
f1.metric("Permanent Provider Cost", f"${R['permanent_provider_cost']:,.0f}")
f2.metric("Flex Provider Cost", f"${R['flex_provider_cost']:,.0f}")
f3.metric("Provider Turnover Replacement Cost", f"${R['turnover_replacement_cost']:,.0f}")
f4.metric("Margin at Risk (Access)", f"${R['est_margin_at_risk']:,.0f}")

if feasible:
    st.success(f"Annual SWB/Visit is feasible: ${R['annual_swb_per_visit']:.2f} ≤ ${params.target_swb_per_visit:.2f}")
else:
    st.warning(f"Annual SWB/Visit exceeds target: ${R['annual_swb_per_visit']:.2f} > ${params.target_swb_per_visit:.2f}")

st.caption(
    "Note: SWB/Visit includes providers + support roles (via role mix) and treats flex providers as requiring the same support mix (conservative). "
    "Turnover replacement cost is provider-only and amplified in yellow/red load months."
)

if R_rec is not None:
    st.markdown("### Policy Exposure (Recommended vs What-If)")
    ex_wi = exposure_summary(R)
    ex_rec = exposure_summary(R_rec)
    ex_delta = float(ex_wi["total"] - ex_rec["total"])

    a1, a2, a3 = st.columns(3)
    a1.metric("Recommended Exposure (Annual)", f"${ex_rec['total']:,.0f}")
    a2.metric("What-If Exposure (Annual)", f"${ex_wi['total']:,.0f}")
    a3.metric("Δ Exposure (What-If − Rec)", f"${ex_delta:,.0f}", help="Positive means the What-If policy increases margin exposure vs recommended.")


# ============================================================
# ANNUAL SUMMARY (Rollup) + EBITDA YOY
# ============================================================
st.markdown("---")
st.header("Annual Summary (YOY)")

annual = R.get("annual_summary")
if annual is None or len(annual) == 0:
    st.caption("Run contains no annual summary.")
else:
    show = annual.copy()
    show["Visits"] = show["Visits"].astype(float)
    show["SWB_Dollars"] = show["SWB_Dollars"].astype(float)
    show["Turnover_Cost"] = show["Turnover_Cost"].astype(float)
    show["Access_Margin_Lost"] = show["Access_Margin_Lost"].astype(float)
    show["EBITDA_Proxy"] = show["EBITDA_Proxy"].astype(float)

    st.dataframe(
        show.style.format({
            "Visits": "{:,.0f}",
            "SWB_Dollars": "${:,.0f}",
            "SWB_per_Visit": "${:,.2f}",
            "Turnover_Cost": "${:,.0f}",
            "Access_Margin_Lost": "${:,.0f}",
            "EBITDA_Proxy": "${:,.0f}",
            "Avg_Perm_Paid_FTE": "{:,.2f}",
            "Avg_Perm_Eff_FTE": "{:,.2f}",
            "Avg_Flex_FTE": "{:,.2f}",
            "Peak_Flex_FTE": "{:,.2f}",
            "Peak_Load_PPPD_Pre": "{:,.1f}",
            "Peak_Load_PPPD_Post": "{:,.1f}",
            "Perm_to_Flex_Ratio": "{:,.1f}",
            "Total_Residual_Gap_FTE_Months": "{:,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# EXECUTIVE SUMMARY (plain language justification)
# ============================================================
st.markdown("---")
st.header("Executive Summary (Plain Language)")

annual = R.get("annual_summary")
y1 = annual.iloc[0] if annual is not None and len(annual) > 0 else None
y2 = annual.iloc[1] if annual is not None and len(annual) > 1 else None
y3 = annual.iloc[2] if annual is not None and len(annual) > 2 else None

key_takeaways = []
key_takeaways.append(f"- **Policy selected:** Base **{what_base:.2f} FTE**, Winter **{what_winter:.2f} FTE**.")
key_takeaways.append(f"- **SWB/Visit:** Year 1 ≈ **${float(y1['SWB_per_Visit']):.2f}** (target **${params.target_swb_per_visit:.2f}**)."
                     if y1 is not None else f"- **SWB/Visit (36 mo avg):** **${R['annual_swb_per_visit']:.2f}** (target **${params.target_swb_per_visit:.2f}**).")
key_takeaways.append(f"- **Risk:** **{risk_tier}** (flex share **{risk['flex_share']*100:.0f}%**, max red streak **{risk['max_red_streak']}**).")
key_takeaways.append(f"- **Capacity:** {R['months_red']} red months and {R['months_yellow']} yellow months over 36 months; peak load **{R['peak_load']:.1f} PPPD**.")
key_takeaways.append(f"- **Access loss:** estimated margin at risk over 36 months ≈ **${R['est_margin_at_risk']:,.0f}** (driven by residual gaps after flex).")

guardrail_takeaways = []
if not allow_low_perm_for_prn:
    guardrail_takeaways.append(f"- **Minimum permanent coverage enforced:** ≥ **{enforced_base_floor:.2f} base FTE** (clinic-hours and/or green-on-avg floor).")
else:
    guardrail_takeaways.append("- **PRN override enabled:** permanent may be below minimum coverage; risk increases if PRN does not show up.")
if enforce_green_on_avg_without_flex:
    guardrail_takeaways.append("- **Productivity expectation enforced:** permanent is sized to keep AVG demand ≤ Green without flex. Flex remains for peak planning / surprises.")
else:
    guardrail_takeaways.append("- **Productivity expectation not enforced:** plan may rely on flex to stay out of strain zones.")

st.markdown("\n".join(key_takeaways))
st.markdown("#### Guardrails & why they exist")
st.markdown("\n".join(guardrail_takeaways))

st.markdown("#### What you are trading off")
st.markdown(
    f"""
- If you **raise permanent FTE**, you usually reduce **flex dependency** and **red exposure**, but you increase fixed staffing expense.
- If you **lower permanent FTE**, you usually improve fixed-cost efficiency, but you increase **operational fragility** (flex/PRN reliability) and **turnover risk** (more yellow/red).
- This plan’s “lean-ness” is mostly expressed as **flex share = {risk['flex_share']*100:.0f}%**. If you want less risk, start by reducing that.
""".strip()
)


# ============================================================
# LEDGER (audit)
# ============================================================
st.markdown("---")
st.header("Audit Ledger (36 months)")

st.dataframe(
    R["ledger"].style.format({
        "Total Visits (month)": "{:,.0f}",
        "Net Contribution (month)": "${:,.0f}",
        "SWB Dollars (month)": "${:,.0f}",
        "SWB/Visit (month)": "${:,.2f}",
        "Access Margin at Risk (month)": "${:,.0f}",
        "Turnover Replacement Cost (month)": "${:,.0f}",
        "Visits/Day (avg)": "{:,.2f}",
        "Visits/Day (peak)": "{:,.2f}",
        "Required Provider Coverage (shifts/day)": "{:,.2f}",
        "Rounded Coverage (0.25)": "{:,.2f}",
        "Load PPPD (pre-flex)": "{:,.1f}",
        "Load PPPD (post-flex)": "{:,.1f}",
    }),
    hide_index=True,
    use_container_width=True,
)


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
    st.download_button("⬇️ Supply Chart (PNG)", data=png1, file_name="psm_policy_supply.png", mime="image/png", use_container_width=True)
with cB:
    st.download_button("⬇️ Load Chart (PNG)", data=png2, file_name="psm_policy_load.png", mime="image/png", use_container_width=True)
with cC:
    st.download_button("⬇️ Ledger (CSV)", data=ledger_csv, file_name="psm_policy_ledger.csv", mime="text/csv", use_container_width=True)
