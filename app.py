# app.py — Predictive Staffing Model (PSM) — Policy Optimizer (Operator-Grade)
# Two-level permanent staffing policy (Base + Winter) + flex as safety stock
# 36-month horizon • Lead-time + ramp • Hiring freeze during flu months
#
# Operator goals:
# 1) Maintain SWB/Visit near target (default: target ± $2), year-by-year
# 2) Protect provider experience (minimize Yellow/Red)
# 3) Avoid structural dependence on flex (permanent-first)
# 4) Explain assumptions, tradeoffs, and risk in plain language
#
# Key upgrades:
# - Sticky top scorecard (frozen cells feel)
# - SWB/Visit governance is a target BAND (± tolerance), applied per year
# - Optional constraint: permanent-only load must stay under Green (no flex)
# - Minimum permanent coverage default: >= 1 provider shift/day (converted to FTE)
# - PRN-heavy override allowed (operator can set min below)
# - Fix Streamlit "value below min" errors for linked inputs
# - Adds EBITDA proxy + YoY change to annual rollup
# - Adds flu requisition post month(s) surfaced from ledger

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
# PAGE CONFIG + CSS
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM) — Policy Optimizer", layout="centered")

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
      .kpirow { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 8px; }
      .kpibox { border: 1px solid #eee; border-radius: 12px; padding: 10px 12px; background: #fff; }
      .kpiLabel { font-size: 0.8rem; color: #666; }
      .kpiVal { font-size: 1.35rem; font-weight: 700; margin-top: 2px; }
      .kpiSub { font-size: 0.82rem; color: #666; margin-top: 2px; }
      .stickyScorecard { position: sticky; top: 0; z-index: 999; background: white; padding-top: 0.35rem; padding-bottom: 0.35rem; border-bottom: 1px solid #eee; }
      .divider { height: 1px; background: #eee; margin: 10px 0 14px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Predictive Staffing Model (PSM) — Policy Optimizer")
st.caption(
    "Permanent-first policy (Base + Winter) • Flex as safety stock • 36-month horizon • "
    "Lead-time + ramp • Flu-season posting freeze • SWB/Visit governance band"
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
    "peak_factor": (
        "Peak-to-average planning multiplier used to build a conservative 'peak' demand curve for staffing/load. "
        "Peak visits/day = (avg visits/day after seasonality + flu uplift) × this factor. "
        "This only affects peak planning curves (load + flex sizing)."
    ),
    "visits_per_provider_shift": (
        "Sweet-spot visits per provider shift. Used for 'required provider coverage' (shifts/day). "
        "Example: 36 => ~1 shift/day supports 36 peak visits/day."
    ),
    "seasonality_pct": "Seasonal swing applied to baseline visits/day: winter up, summer down; spring/fall neutral.",
    "flu_uplift_pct": "Additional uplift applied to avg visits/day in selected flu months.",
    "flu_months": "Months that receive flu uplift (often Oct–Feb).",

    "hours_week": "Total clinic hours open per week. Used to convert provider shifts/day into FTE.",
    "days_open_per_week": "Days open per week. Used for minimum coverage translation.",
    "fte_hours_week": "Paid hours per 1.0 FTE per week (e.g., 36). Used to translate FTE into paid hours.",

    "annual_turnover": "Annual provider turnover rate converted to monthly attrition in the simulation.",
    "lead_days": "Days from requisition to independent productivity (hiring + training).",
    "ramp_months": "Months after independence where productivity is reduced (ramp).",
    "ramp_productivity": "Ramp productivity fraction (e.g., 75%).",
    "fill_probability": "Probability the pipeline fills posted requisitions (applied to hires).",

    "budgeted_pppd": "Green threshold (target PPPD). Used for risk bands and permanent-only constraint if enabled.",
    "yellow_max_pppd": "Yellow threshold (caution). Used for penalties and reporting.",
    "red_start_pppd": "Red threshold (high risk). Months above this are treated as burnout risk.",

    "winter_anchor_month": "Month when winter staffing should be ready by (model posts earlier to meet this).",
    "freeze_months": "Posting is not allowed during these months.",

    "flex_max": "Maximum flex provider FTE available in any month (safety stock cap).",
    "flex_mult": "Flex cost multiplier vs base provider loaded cost.",

    "target_swb": "Target annual SWB/Visit. Optimizer aims to stay within target ± tolerance, per year.",
    "swb_tol": "Allowed annual SWB/Visit deviation from target (default $2). Applied per year.",
    "net_contrib": (
        "Net contribution per visit (proxy for EBITDA contribution before staffing and other fixed expenses). "
        "Used to estimate EBITDA proxy and access risk."
    ),
    "visits_lost": "Estimated visits lost per 1.0 provider-day gap (access shortfall proxy).",
    "repl_cost": "Replacement cost per provider FTE replaced (recruit + onboarding + ramp inefficiency + temp coverage).",
    "turn_yellow": "Replacement cost multiplier in Yellow months.",
    "turn_red": "Replacement cost multiplier in Red months.",

    "min_perm_shifts": (
        "Minimum permanent coverage in provider shifts/day. Default 1.0 = at least one permanent provider covering daily. "
        "Converted to FTE using clinic hours and FTE hours/week."
    ),
    "allow_prn_override": "If enabled, the operator can set Base FTE below the minimum permanent coverage (PRN-heavy clinics).",
    "require_perm_green": "If enabled, the model penalizes/filters policies where permanent-only load exceeds Green (no flex).",

    "benefits": "Benefits load applied on top of base hourly (e.g., 30%).",
    "bonus": "Bonus load applied on top of base hourly (e.g., 10%).",
    "ot_sick": "OT + sick/PTO load applied on top of base hourly (e.g., 4%).",
    "phys_hr": "Optional physician hourly rate (if supervision hours included).",
    "apc_hr": "Provider (APP) base hourly rate.",
    "ma_hr": "MA base hourly rate.",
    "psr_hr": "PSR base hourly rate.",
    "rt_hr": "RT base hourly rate.",
    "sup_hr": "Supervisor hourly rate (if supervision hours included).",
    "phys_sup_hours": "Fixed physician supervision hours/month (adds cost).",
    "sup_hours": "Fixed supervisor hours/month (adds cost).",

    "base_min": "Minimum base FTE to evaluate in the grid search (auto-defaults to minimum permanent coverage unless PRN override).",
    "base_max": "Maximum base FTE to evaluate in the grid search.",
    "base_step": "Step size for base FTE grid search.",
    "winter_delta_max": "Maximum winter uplift above base to evaluate.",
    "winter_step": "Step size for winter uplift grid search.",
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
    # Provider-days equivalent per day (scaling used for PPPD load).
    return float(provider_fte) * (float(fte_hours_week) / max(float(hours_week), 1e-9))


def fte_required_for_provider_shifts_per_day(shifts_per_day: float, hours_week: float, fte_hours_week: float) -> float:
    # In this model: required shifts/day -> required effective FTE using the same conversion used elsewhere.
    # req_eff_fte = shifts_per_day * (hours_week / fte_hours_week)
    return float(shifts_per_day) * (float(hours_week) / max(float(fte_hours_week), 1e-9))


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


def select_latest_post_month_before_freeze(anchor_month: int, lead_months: int, freeze_months: set[int]) -> int:
    latest = wrap_month(anchor_month - lead_months)
    for k in range(0, 12):
        m = wrap_month(latest - k)
        if m not in freeze_months:
            return m
    return latest


def policy_exposure_dollars(sim_result: dict[str, Any]) -> float:
    return float(
        sim_result["total_swb_dollars"]
        + sim_result["turnover_replacement_cost"]
        + sim_result["est_margin_at_risk"]
    )


def exposure_summary(sim_result: dict[str, Any]) -> dict[str, float]:
    swb = float(sim_result["total_swb_dollars"])
    turn = float(sim_result["turnover_replacement_cost"])
    access = float(sim_result["est_margin_at_risk"])
    total = swb + turn + access
    return {"total": total, "swb": swb, "turnover": turn, "access": access}


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

    # Workforce
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
    min_perm_provider_shifts_per_day: float
    allow_prn_override: bool
    require_perm_under_green_no_flex: bool

    # Cache key injection
    _v: str = MODEL_VERSION


@dataclass(frozen=True)
class Policy:
    base_fte: float
    winter_fte: float


# ============================================================
# SIMULATION ENGINE
# ============================================================
def build_annual_summary(ledger: pd.DataFrame, params: ModelParams) -> pd.DataFrame:
    df = ledger.copy()
    df["Year"] = df["Month"].str.slice(0, 4).astype(int)

    # Ensure numeric
    numeric_cols = [
        "Total Visits (month)",
        "SWB Dollars (month)",
        "SWB/Visit (month)",
        "Net Contribution (month)",
        "EBITDA Proxy (month)",
        "Flex FTE Used",
        "Permanent FTE (Paid)",
        "Permanent FTE (Effective)",
        "Load PPPD (pre-flex)",
        "Load PPPD (post-flex)",
        "Perm Load PPPD (no-flex)",
        "Residual FTE Gap (to Sweet Spot)",
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
        Avg_Perm_Eff_FTE=("Permanent FTE (Effective)", "mean"),
        Avg_Flex_FTE=("Flex FTE Used", "mean"),
        Peak_Flex_FTE=("Flex FTE Used", "max"),
        Peak_Load_PPPD_Post=("Load PPPD (post-flex)", "max"),
        Peak_Perm_Load_NoFlex=("Perm Load PPPD (no-flex)", "max"),
        Months_Yellow=("Load PPPD (post-flex)", lambda s: int(((s > params.budgeted_pppd + 1e-9) & (s <= params.red_start_pppd + 1e-9)).sum())),
        Months_Red=("Load PPPD (post-flex)", lambda s: int((s > params.red_start_pppd + 1e-9).sum())),
        Total_Residual_Gap_FTE_Months=("Residual FTE Gap (to Sweet Spot)", "sum"),
    )

    annual["SWB_per_Visit"] = annual["SWB_Dollars"] / annual["Visits"].clip(lower=1.0)
    annual["EBITDA_per_Visit"] = annual["EBITDA_Proxy"] / annual["Visits"].clip(lower=1.0)

    # YoY deltas
    annual = annual.sort_values("Year").reset_index(drop=True)
    annual["YoY_EBITDA_Proxy_Δ"] = annual["EBITDA_Proxy"].diff().fillna(0.0)

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

    # Demand
    base_year0 = float(params.visits)
    base_year1 = float(params.visits) * (1.0 + float(params.annual_growth))
    base_year2 = float(params.visits) * (1.0 + float(params.annual_growth)) ** 2

    visits_curve_base = compute_visits_curve(months, base_year0, base_year1, base_year2, float(params.seasonality_pct))
    visits_curve_flu = apply_flu_uplift(visits_curve_base, months, set(params.flu_months), float(params.flu_uplift_pct))
    visits_peak = [float(v) * float(params.peak_factor) for v in visits_curve_flu]

    # Role mix locked to growth-year baseline
    role_mix = compute_role_mix_ratios(float(params.visits) * (1.0 + float(params.annual_growth)), staffing_model=model)

    # Hiring schedule
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
        local_cohorts = [{"fte": max(float(policy.base_fte), 0.0), "age": 9999}]
        paid = [0.0] * N_MONTHS
        eff = [0.0] * N_MONTHS
        shed = [0.0] * N_MONTHS

        def _paid() -> float:
            return float(sum(float(c["fte"]) for c in local_cohorts))

        for t in range(N_MONTHS):
            sp = _paid()
            for c in local_cohorts:
                c["fte"] = max(float(c["fte"]) * (1.0 - monthly_turnover), 0.0)
            ap = _paid()
            shed[t] = float(ap - sp)  # negative

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

        return paid, eff, shed

    # Pass 1 (no winter steps)
    paid_1, _, _ = run_supply(hires_visible)

    # Winter steps
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
            hires_reason_pipeline[v] = (hires_reason_pipeline[v] + " | " if hires_reason_pipeline[v] else "") + "Base replacement"

    # Final supply run
    paid, eff, turnover_shed_arr = run_supply(hires_visible_pipeline)

    perm_eff = np.array(eff, dtype=float)
    perm_paid = np.array(paid, dtype=float)
    v_peak = np.array(visits_peak, dtype=float)
    v_avg = np.array(visits_curve_flu, dtype=float)
    dim = np.array(days_in_month, dtype=float)

    # Required coverage (shifts/day)
    visits_per_shift = max(float(params.visits_per_provider_shift), 1e-6)
    req_provider_shifts_per_day = v_peak / visits_per_shift
    req_provider_shifts_per_day_rounded = np.round(req_provider_shifts_per_day / 0.25) * 0.25

    # Load computation
    prov_day_equiv_perm = np.array([provider_day_equiv_from_fte(f, params.hours_week, params.fte_hours_week) for f in perm_eff], dtype=float)
    load_pre = v_peak / np.maximum(prov_day_equiv_perm, 1e-6)  # pre-flex load (permanent-only)

    # Flex sizing to sweet-spot coverage
    flex_fte = np.zeros(N_MONTHS, dtype=float)
    load_post = np.zeros(N_MONTHS, dtype=float)

    for i in range(N_MONTHS):
        # required effective FTE for coverage
        req_eff_fte_i = float(req_provider_shifts_per_day[i]) * (float(params.hours_week) / max(float(params.fte_hours_week), 1e-6))
        gap_fte = max(req_eff_fte_i - float(perm_eff[i]), 0.0)
        flex_used = min(gap_fte, float(params.flex_max_fte_per_month))
        flex_fte[i] = float(flex_used)

        prov_day_equiv_total = provider_day_equiv_from_fte(float(perm_eff[i] + flex_used), params.hours_week, params.fte_hours_week)
        load_post[i] = float(v_peak[i]) / max(float(prov_day_equiv_total), 1e-6)

    # Residual gap to sweet-spot after flex
    req_eff_fte_needed = req_provider_shifts_per_day * (float(params.hours_week) / max(float(params.fte_hours_week), 1e-6))
    residual_gap_fte = np.maximum(req_eff_fte_needed - (perm_eff + flex_fte), 0.0)
    provider_day_gap_total = float(np.sum(residual_gap_fte * dim))

    # Lost visits and access margin at risk
    est_visits_lost = float(provider_day_gap_total) * float(params.visits_lost_per_provider_day_gap)
    est_margin_at_risk = est_visits_lost * float(params.net_revenue_per_visit)

    # Turnover replacement cost
    replacements_base = perm_paid * float(monthly_turnover)
    repl_mult = np.ones(N_MONTHS, dtype=float)
    repl_mult = np.where(load_post > float(params.budgeted_pppd), float(params.turnover_yellow_mult), repl_mult)
    repl_mult = np.where(load_post > float(params.red_start_pppd), float(params.turnover_red_mult), repl_mult)
    turnover_replacement_cost = float(np.sum(replacements_base * float(params.provider_replacement_cost) * repl_mult))

    # SWB/Visit affordability (annual, includes flex)
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

    # Annualized/year-by-year SWB band penalty will be computed from the ledger later.

    # Burnout metrics
    green_cap = float(params.budgeted_pppd)
    red_start = float(params.red_start_pppd)
    months_yellow = int(np.sum((load_post > green_cap + 1e-9) & (load_post <= red_start + 1e-9)))
    months_red = int(np.sum(load_post > red_start + 1e-9))
    peak_load_post = float(np.max(load_post)) if len(load_post) else 0.0
    peak_load_perm_only = float(np.max(load_pre)) if len(load_pre) else 0.0

    # Flex dependence (operator risk signal)
    perm_total_fte_months = float(np.sum(np.maximum(perm_eff, 0.0) * dim))
    flex_total_fte_months = float(np.sum(np.maximum(flex_fte, 0.0) * dim))
    flex_share = float(flex_total_fte_months / max(perm_total_fte_months + flex_total_fte_months, 1e-9))

    # Penalties: Under/over, burnout, access risk, turnover risk
    under_util = np.maximum((green_cap * 0.70) - load_post, 0.0)
    overstaff_penalty = float(np.sum(under_util * dim))

    yellow_excess = np.maximum(load_post - green_cap, 0.0)
    red_excess = np.maximum(load_post - red_start, 0.0)
    burnout_penalty = float(np.sum((yellow_excess ** 1.2) * dim) + 3.0 * np.sum((red_excess ** 2.0) * dim))

    # Permanent-only Green constraint (A)
    perm_green_violation = float(np.max(np.maximum(load_pre - green_cap, 0.0)))
    perm_green_months = int(np.sum(load_pre > green_cap + 1e-9))

    # Score baseline components (policy exposure)
    # Note: total_swb_dollars already includes provider + support roles + supervision (if entered).
    total_score = (
        float(total_swb_dollars)
        + float(turnover_replacement_cost)
        + float(est_margin_at_risk)
        + 2_000.0 * burnout_penalty
        + 500.0 * overstaff_penalty
    )

    # We apply SWB band penalties by YEAR (strongly).
    # The ledger will compute year SWB/Visit and we penalize deviation beyond tolerance for each year.
    # Placeholder (added after ledger built).

    # Ledger (monthly)
    rows: list[dict[str, Any]] = []
    for i in range(N_MONTHS):
        lab = dates[i].strftime("%Y-%b")
        month_visits = float(v_avg[i]) * float(days_in_month[i])
        month_net_contrib = float(month_visits) * float(params.net_revenue_per_visit)

        # Monthly SWB dollars (recalc per month for audit)
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

        # EBITDA proxy (monthly): contribution - staffing - turnover - access risk allocation
        # Access risk is already in contribution units ($). Allocate to month proportional to lost visits share.
        # For simplicity: allocate by residual gap share.
        # If no gaps, alloc is 0.
        # This keeps annual EBITDA proxy consistent with summary math.
        # (We will also compute an annual EBITDA proxy using totals, but this is useful for rollups.)
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

                "Required Provider Coverage (shifts/day)": float(req_provider_shifts_per_day[i]),
                "Rounded Coverage (0.25)": float(req_provider_shifts_per_day_rounded[i]),

                "Perm Load PPPD (no-flex)": float(load_pre[i]),
                "Load PPPD (post-flex)": float(load_post[i]),

                "Turnover Shed (FTE)": float(turnover_shed_arr[i]),
                "Hires Visible (FTE)": float(hires_visible_pipeline[i]),
                "Hire Reason": hires_reason_pipeline[i],
                "Hire Post Month": hires_post_month_pipeline[i],

                "Residual FTE Gap (to Sweet Spot)": float(residual_gap_fte[i]),
            }
        )

    ledger = pd.DataFrame(rows)

    # Year-by-year SWB band penalties (operator requirement: keep near target each year)
    ledger["Year"] = ledger["Month"].str.slice(0, 4).astype(int)
    annual = ledger.groupby("Year", as_index=False).agg(
        Visits=("Total Visits (month)", "sum"),
        SWB_Dollars=("SWB Dollars (month)", "sum"),
    )
    annual["SWB_per_Visit"] = annual["SWB_Dollars"] / annual["Visits"].clip(lower=1.0)

    target = float(params.target_swb_per_visit)
    tol = max(float(params.swb_tolerance), 0.0)

    # Penalty only outside the band; squared so big misses hurt a lot.
    annual["SWB_Deviation_Outside_Band"] = np.maximum(np.abs(annual["SWB_per_Visit"] - target) - tol, 0.0)
    swb_band_penalty = float(np.sum((annual["SWB_Deviation_Outside_Band"] ** 2) * 2_000_000.0))

    # Permanent-only Green constraint penalty (if enabled)
    perm_green_penalty = 0.0
    if bool(params.require_perm_under_green_no_flex):
        # Big penalty for any permanent-only exceedance beyond Green
        perm_green_penalty = float(perm_green_violation) * 2_500_000.0 + float(perm_green_months) * 250_000.0

    # Flex reliance penalty (structural dependence)
    # Penalize high flex share. Encourage flex to remain "safety stock", not baseline coverage.
    flex_penalty = float(max(flex_share - 0.10, 0.0) ** 2) * 3_000_000.0  # no penalty up to 10% share

    total_score = float(total_score) + swb_band_penalty + perm_green_penalty + flex_penalty

    annual_summary = build_annual_summary(ledger, params)

    # Annual totals for EBITDA proxy
    total_net_contrib = float(total_visits) * float(params.net_revenue_per_visit)
    # Annual EBITDA proxy: contribution - SWB - turnover - access risk
    ebitda_proxy_annual = float(total_net_contrib) - float(total_swb_dollars) - float(turnover_replacement_cost) - float(est_margin_at_risk)

    # Flu requisition post months
    flu_posts = ledger.loc[ledger["Hire Reason"].astype(str).str.contains("Winter step", na=False), "Hire Post Month"].dropna().unique().tolist()
    flu_posts = [int(x) for x in flu_posts] if len(flu_posts) else []

    return {
        "dates": list(dates),
        "months": months,
        "days_in_month": list(days_in_month),

        "visits_avg": list(v_avg),
        "visits_peak": list(v_peak),

        "perm_paid": list(perm_paid),
        "perm_eff": list(perm_eff),
        "flex_fte": list(flex_fte),

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
                }
            )

            if best is None or float(res["score"]) < float(best["res"]["score"]):
                best = {"policy": pol, "res": res}

    frontier = pd.DataFrame(candidates).sort_values(["Score", "Months_Red", "FlexShare"]).reset_index(drop=True)
    return {"best": best, "frontier": frontier}


# ============================================================
# CACHES (keyed by params_dict including "_v")
# ============================================================
@st.cache_data(show_spinner=False)
def cached_recommend_policy(params_dict: dict[str, Any], base_min: float, base_max: float, base_step: float, winter_delta_max: float, winter_step: float) -> dict[str, Any]:
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
    st.header("Demand")
    visits = st.number_input("Avg Visits/Day (annual baseline)", min_value=1.0, value=36.0, step=1.0, help=HELP["visits"])
    annual_growth = st.number_input("Annual Visit Growth %", min_value=0.0, value=10.0, step=1.0, help=HELP["annual_growth"]) / 100.0
    peak_factor = st.slider("Peak-to-average factor", 1.00, 1.50, 1.20, 0.01, help=HELP["peak_factor"])

    st.subheader("Seasonality + Flu")
    seasonality_pct = st.number_input("Seasonality swing %", min_value=0.0, value=20.0, step=5.0, help=HELP["seasonality_pct"]) / 100.0
    flu_uplift_pct = st.number_input("Flu uplift %", min_value=0.0, value=0.0, step=5.0, help=HELP["flu_uplift_pct"]) / 100.0
    flu_months = st.multiselect("Flu months", options=MONTH_OPTIONS, default=[("Oct", 10), ("Nov", 11), ("Dec", 12), ("Jan", 1), ("Feb", 2)], help=HELP["flu_months"])
    flu_months_set = {m for _, m in flu_months} if flu_months else set()

    visits_per_provider_shift = st.number_input("Visits per provider shift (sweet spot)", min_value=5.0, value=36.0, step=1.0, help=HELP["visits_per_provider_shift"])

    st.header("Clinic Ops")
    hours_week = st.number_input("Hours of Operation / Week", min_value=1.0, value=84.0, step=1.0, help=HELP["hours_week"])
    days_open_per_week = st.number_input("Days Open / Week", min_value=1.0, max_value=7.0, value=7.0, step=1.0, help=HELP["days_open_per_week"])
    fte_hours_week = st.number_input("FTE Hours / Week", min_value=1.0, value=36.0, step=1.0, help=HELP["fte_hours_week"])

    st.header("Operator Constraints (Permanent-first)")
    min_perm_provider_shifts_per_day = st.number_input(
        "Minimum permanent coverage (provider shifts/day)",
        min_value=0.0,
        value=1.0,
        step=0.25,
        help=HELP["min_perm_shifts"],
    )
    allow_prn_override = st.checkbox("Allow Base FTE below minimum (PRN-heavy clinic)", value=False, help=HELP["allow_prn_override"])
    require_perm_under_green_no_flex = st.checkbox("Require permanent-only to stay under Green (no flex)", value=True, help=HELP["require_perm_green"])

    st.header("Workforce Dynamics")
    annual_turnover = st.number_input("Annual Turnover %", min_value=0.0, value=16.0, step=1.0, help=HELP["annual_turnover"]) / 100.0
    lead_days = st.number_input("Days to Independent (Req→Independent)", min_value=0, value=210, step=10, help=HELP["lead_days"])
    ramp_months = st.slider("Ramp months after independent", 0, 6, 1, 1, help=HELP["ramp_months"])
    ramp_productivity = st.slider("Ramp productivity %", 30, 100, 75, 5, help=HELP["ramp_productivity"]) / 100.0
    fill_probability = st.slider("Fill probability %", 0, 100, 85, 5, help=HELP["fill_probability"]) / 100.0

    st.header("Risk Bands (PPPD)")
    st.caption("PPPD thresholds define operating zones and risk penalties.")
    budgeted_pppd = st.number_input("Green PPPD (target)", min_value=5.0, value=36.0, step=1.0, help=HELP["budgeted_pppd"])
    yellow_max_pppd = st.number_input("Yellow PPPD (caution)", min_value=5.0, value=42.0, step=1.0, help=HELP["yellow_max_pppd"])
    red_start_pppd = st.number_input("Red PPPD (high risk)", min_value=5.0, value=45.0, step=1.0, help=HELP["red_start_pppd"])

    st.header("Hiring Freeze")
    winter_anchor_month = st.selectbox("Winter anchor month (ready-by)", options=MONTH_OPTIONS, index=11, help=HELP["winter_anchor_month"])
    winter_anchor_month_num = int(winter_anchor_month[1])
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

    st.subheader("Provider turnover replacement cost (providers only)")
    provider_replacement_cost = st.number_input("Replacement cost per 1.0 provider FTE", min_value=0.0, value=75000.0, step=5000.0, help=HELP["repl_cost"])
    turnover_yellow_mult = st.slider("Turnover cost multiplier (yellow)", 1.0, 3.0, 1.3, 0.05, help=HELP["turn_yellow"])
    turnover_red_mult = st.slider("Turnover cost multiplier (red)", 1.0, 5.0, 2.0, 0.10, help=HELP["turn_red"])

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

    # Minimum base FTE default derived from minimum permanent shifts/day
    min_base_fte_default = fte_required_for_provider_shifts_per_day(min_perm_provider_shifts_per_day, hours_week, fte_hours_week)
    # If PRN override is allowed, we can go lower (but risk will be explained + penalized)
    base_min_default = 0.0 if allow_prn_override else float(min_base_fte_default)

    st.header("Optimizer Controls")
    base_min = st.number_input("Base FTE min", min_value=0.0, value=float(base_min_default), step=0.25, help=HELP["base_min"])
    base_max = st.number_input("Base FTE max", min_value=0.0, value=6.0, step=0.25, help=HELP["base_max"])
    base_step = st.select_slider("Base FTE step", options=[0.10, 0.25, 0.50], value=0.25, help=HELP["base_step"])
    winter_delta_max = st.number_input("Max winter uplift above base (FTE)", min_value=0.0, value=2.0, step=0.25, help=HELP["winter_delta_max"])
    winter_step = st.select_slider("Winter uplift step", options=[0.10, 0.25, 0.50], value=0.25, help=HELP["winter_step"])

    st.header("Mode")
    mode = st.radio("Run mode", ["Recommend + What-If", "What-If only"], index=0)

    st.divider()
    run_recommender = st.button("🏁 Run Recommender (Grid Search)", use_container_width=True)
    st.caption("What-If updates live. Recommender runs only when you click the button.")


# ============================================================
# BUILD PARAMS (inject _v)
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

    min_perm_provider_shifts_per_day=float(min_perm_provider_shifts_per_day),
    allow_prn_override=bool(allow_prn_override),
    require_perm_under_green_no_flex=bool(require_perm_under_green_no_flex),

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

    # Snap What-If controls to new recommendation (safe set)
    st.session_state["what_base_fte"] = float(st.session_state.rec_policy.base_fte)
    st.session_state["what_winter_fte"] = float(st.session_state.rec_policy.winter_fte)

if frontier is None and st.session_state["frontier"] is not None:
    frontier = st.session_state["frontier"]

rec_policy = st.session_state.rec_policy


# ============================================================
# WHAT-IF INPUTS (LIVE) — robust min/max handling
# ============================================================
st.subheader("What-If Policy Inputs")

# Default what-if values
if rec_policy is not None:
    default_base = float(rec_policy.base_fte)
    default_winter = float(rec_policy.winter_fte)
else:
    default_base = float(max(base_min, 1.0))
    default_winter = float(default_base + min(winter_delta_max, 1.0))

# Initialize
st.session_state["what_base_fte"] = float(st.session_state.get("what_base_fte", default_base))
st.session_state["what_winter_fte"] = float(st.session_state.get("what_winter_fte", default_winter))

# Enforce minimum base coverage unless PRN override
min_base_fte_required = fte_required_for_provider_shifts_per_day(params.min_perm_provider_shifts_per_day, params.hours_week, params.fte_hours_week)
if not params.allow_prn_override:
    if st.session_state["what_base_fte"] < float(min_base_fte_required):
        st.session_state["what_base_fte"] = float(min_base_fte_required)

# Always enforce winter >= base before widgets render (prevents StreamlitValueBelowMinError)
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
        help=(
            "Annual base permanent provider FTE level for the What-If scenario."
            + ("" if params.allow_prn_override else f" Minimum enforced = {min_base_fte_required:.2f} FTE (based on minimum permanent coverage).")
        ),
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
        help="Winter target permanent provider FTE level (must be ≥ base).",
    )

# Simulate live what-if
R = cached_simulate(params_dict, float(what_base), float(what_winter))

# Recommended baseline (for comparison)
R_rec = None
if mode == "Recommend + What-If" and rec_policy is not None:
    R_rec = cached_simulate(params_dict, float(rec_policy.base_fte), float(rec_policy.winter_fte))


# ============================================================
# STICKY TOP SCORECARD (frozen rows)
# ============================================================
annual = R.get("annual_summary")
if annual is not None and len(annual) > 0:
    # Use first year as "current", last year as "forward"
    swb_y1 = float(annual.loc[0, "SWB_per_Visit"])
    ebitda_y1 = float(annual.loc[0, "EBITDA_Proxy"])
    ebitda_y3 = float(annual.loc[len(annual) - 1, "EBITDA_Proxy"])
    ebitda_yoy = float(annual.loc[len(annual) - 1, "YoY_EBITDA_Proxy_Δ"]) if "YoY_EBITDA_Proxy_Δ" in annual.columns else 0.0
else:
    swb_y1 = float(R["annual_swb_per_visit"])
    ebitda_y1 = float(R["ebitda_proxy_annual"])
    ebitda_y3 = float(R["ebitda_proxy_annual"])
    ebitda_yoy = 0.0

flu_posts = R.get("flu_requisition_post_months", [])
flu_post_label = ", ".join([month_name(m) for m in flu_posts]) if flu_posts else "—"

swb_target = float(params.target_swb_per_visit)
tol = float(params.swb_tolerance)
band_lo, band_hi = swb_target - tol, swb_target + tol
swb_status = "IN BAND" if (band_lo - 1e-9) <= swb_y1 <= (band_hi + 1e-9) else "OUT OF BAND"

st.markdown('<div class="stickyScorecard">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="kpirow">
      <div class="kpibox">
        <div class="kpiLabel">SWB/Visit (Year 1)</div>
        <div class="kpiVal">${swb_y1:,.2f}</div>
        <div class="kpiSub">Target ${swb_target:,.0f} ± ${tol:,.0f} ({swb_status})</div>
      </div>
      <div class="kpibox">
        <div class="kpiLabel">Annual Staffing Expense (3yr blend)</div>
        <div class="kpiVal">${R["total_swb_dollars"]:,.0f}</div>
        <div class="kpiSub">SWB dollars (providers + support + supervision)</div>
      </div>
      <div class="kpibox">
        <div class="kpiLabel">EBITDA Proxy (Year 1)</div>
        <div class="kpiVal">${ebitda_y1:,.0f}</div>
        <div class="kpiSub">Contribution − SWB − turnover − access risk</div>
      </div>
      <div class="kpibox">
        <div class="kpiLabel">Peak Load (Post-flex)</div>
        <div class="kpiVal">{R["peak_load_post"]:.1f}</div>
        <div class="kpiSub">PPPD (risk bands)</div>
      </div>
      <div class="kpibox">
        <div class="kpiLabel">Permanent→Flex Reliance</div>
        <div class="kpiVal">{(1.0 - float(R["flex_share"])) * 100:.0f}% / {float(R["flex_share"]) * 100:.0f}%</div>
        <div class="kpiSub">Permanent share / Flex share (FTE-month weighted)</div>
      </div>
      <div class="kpibox">
        <div class="kpiLabel">Flu Requisition Post Month</div>
        <div class="kpiVal">{flu_post_label}</div>
        <div class="kpiSub">From winter-step postings</div>
      </div>
    </div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# OPERATOR MESSAGES: ASSUMPTIONS + FLAGS
# ============================================================
min_base_req = fte_required_for_provider_shifts_per_day(params.min_perm_provider_shifts_per_day, params.hours_week, params.fte_hours_week)

assumptions_msg = f"""
<div class="note">
  <b>Assumptions (what the model believes right now)</b>
  <ul class="small" style="margin-top:8px;">
    <li><b>Minimum permanent coverage:</b> {params.min_perm_provider_shifts_per_day:.2f} provider shifts/day ⇒ <b>{min_base_req:.2f} FTE</b> minimum base (unless PRN override is enabled).</li>
    <li><b>SWB/Visit governance:</b> Target <b>${params.target_swb_per_visit:.0f} ± ${params.swb_tolerance:.0f}</b>, enforced <b>year-by-year</b> (not just blended).</li>
    <li><b>Permanent-first rule (optional):</b> Permanent-only load must stay ≤ Green without flex = <b>{'ON' if params.require_perm_under_green_no_flex else 'OFF'}</b>.</li>
    <li><b>Flex is safety stock:</b> capped at <b>{params.flex_max_fte_per_month:.2f} FTE/month</b>; optimizer penalizes structural flex dependence.</li>
  </ul>
</div>
"""
st.markdown(assumptions_msg, unsafe_allow_html=True)

# Risk callouts
risk_lines = []
if params.allow_prn_override and float(what_base) < float(min_base_req) - 1e-9:
    risk_lines.append("Base FTE is below minimum permanent coverage — this assumes reliable PRN coverage. Flex dependence and no-show risk rise.")
if params.require_perm_under_green_no_flex and int(R["perm_green_months"]) > 0:
    risk_lines.append(f"Permanent-only exceeds Green in {int(R['perm_green_months'])} month(s) — policy fails the 'no-flex-for-green' expectation.")
if float(R["flex_share"]) > 0.10:
    risk_lines.append(f"Flex share is {float(R['flex_share'])*100:.1f}% — above the recommended safety-stock range (≤ 10%).")
if int(R["months_red"]) > 0:
    risk_lines.append(f"{int(R['months_red'])} month(s) in Red — meaningful burnout / access risk.")
if int(R["months_yellow"]) > 6:
    risk_lines.append(f"{int(R['months_yellow'])} month(s) in Yellow — sustained strain risk.")

if risk_lines:
    st.markdown(
        "<div class='warn'><b>Risk flags</b><ul class='small' style='margin-top:8px;'>"
        + "".join([f"<li>{x}</li>" for x in risk_lines])
        + "</ul></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div class='ok'><b>Risk flags</b><div class='small' style='margin-top:6px;'>No major flags detected under the current assumptions.</div></div>", unsafe_allow_html=True)


# ============================================================
# RECOMMENDED POLICY (if available)
# ============================================================
if mode == "Recommend + What-If" and rec_policy is not None:
    if best_block is None:
        best_res = R_rec
        best_policy = rec_policy
    else:
        best_policy = best_block["policy"]
        best_res = best_block["res"]

    st.markdown("---")
    st.header("Recommended Permanent Staffing Policy")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base FTE", f"{best_policy.base_fte:.2f}")
    c2.metric("Winter FTE", f"{best_policy.winter_fte:.2f}")
    c3.metric("Peak Perm Load (No-flex)", f"{best_res['peak_perm_load_no_flex']:.1f}")
    c4.metric("Flex Share", f"{best_res['flex_share']*100:.1f}%")

    st.markdown(
        """
        <div class="contract">
          <b>What the recommender is optimizing for</b>
          <ul class="small" style="margin-top:8px;">
            <li><b>Keep SWB/Visit in band</b> year-by-year (target ± tolerance).</li>
            <li><b>Protect providers</b> by minimizing Yellow/Red exposure.</li>
            <li><b>Permanent-first</b>: penalizes structural flex reliance; optional permanent-only ≤ Green rule.</li>
            <li><b>Right-time hiring</b>: honors lead time and freeze months while stepping up for winter readiness.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if frontier is not None:
        with st.expander("Candidate table (top 25 by score)", expanded=False):
            st.dataframe(frontier.head(25), use_container_width=True, hide_index=True)


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
load_perm_only = np.array(R["load_perm_only"], dtype=float)
load_post = np.array(R["load_post"], dtype=float)

tick_idx = list(range(0, len(dates), 3))

# Chart 1: Supply + Flex
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

# Chart 2: Load with bands + demand overlay
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
# FINANCE + ANNUAL SUMMARY
# ============================================================
st.markdown("---")
st.header("Finance & Risk Summary")

f1, f2, f3, f4 = st.columns(4)
f1.metric("SWB Dollars (3yr blend)", f"${R['total_swb_dollars']:,.0f}")
f2.metric("Turnover Replacement Cost", f"${R['turnover_replacement_cost']:,.0f}")
f3.metric("Access Margin at Risk", f"${R['est_margin_at_risk']:,.0f}")
f4.metric("EBITDA Proxy (annual)", f"${R['ebitda_proxy_annual']:,.0f}")

band_ok = (swb_target - tol - 1e-9) <= swb_y1 <= (swb_target + tol + 1e-9)
if band_ok:
    st.success(f"Year-1 SWB/Visit is within band: ${swb_y1:.2f} (target ${swb_target:.0f} ± ${tol:.0f})")
else:
    st.warning(f"Year-1 SWB/Visit is out of band: ${swb_y1:.2f} (target ${swb_target:.0f} ± ${tol:.0f})")

st.caption(
    "EBITDA Proxy = (Visits × Net Contribution/Visit) − SWB dollars − turnover replacement cost − access margin at risk. "
    "This is a decision-support proxy (not GAAP)."
)

st.markdown("---")
st.header("Annual Summary (Rollup)")

annual = R.get("annual_summary")
if annual is None or len(annual) == 0:
    st.caption("No annual summary available.")
else:
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
                "Avg_Perm_Eff_FTE": "{:,.2f}",
                "Avg_Flex_FTE": "{:,.2f}",
                "Peak_Flex_FTE": "{:,.2f}",
                "Peak_Load_PPPD_Post": "{:,.1f}",
                "Peak_Perm_Load_NoFlex": "{:,.1f}",
                "Total_Residual_Gap_FTE_Months": "{:,.2f}",
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
            "Required Provider Coverage (shifts/day)": "{:,.2f}",
            "Rounded Coverage (0.25)": "{:,.2f}",
            "Perm Load PPPD (no-flex)": "{:,.1f}",
            "Load PPPD (post-flex)": "{:,.1f}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# EXECUTIVE SUMMARY (plain language)
# ============================================================
st.markdown("---")
st.header("Executive Summary (Plain Language)")

summary_points = []

summary_points.append(
    f"**Minimum permanent coverage** is set to **{params.min_perm_provider_shifts_per_day:.2f} shift(s)/day**, "
    f"which translates to **{min_base_req:.2f} FTE** given clinic hours. "
    + ("PRN override is enabled, so Base FTE may be set below this (higher operational risk)." if params.allow_prn_override else "Base FTE is constrained to stay at/above this level.")
)

summary_points.append(
    f"**SWB/Visit governance** aims to keep each year within **${params.target_swb_per_visit:.0f} ± ${params.swb_tolerance:.0f}**. "
    f"Year 1 SWB/Visit is **${swb_y1:.2f}** ({'in band' if band_ok else 'out of band'})."
)

summary_points.append(
    f"**Provider experience / burnout risk**: {int(R['months_yellow'])} Yellow month(s), {int(R['months_red'])} Red month(s), "
    f"with peak post-flex load **{R['peak_load_post']:.1f} PPPD**."
)

summary_points.append(
    f"**Flex reliance**: flex share is **{float(R['flex_share'])*100:.1f}%** of total coverage (FTE-month weighted). "
    "This tool treats flex as safety stock and penalizes structural dependence."
)

if params.require_perm_under_green_no_flex:
    summary_points.append(
        f"**Permanent-only Green expectation** is ON. Permanent-only exceeded Green in **{int(R['perm_green_months'])} month(s)** "
        f"(peak permanent-only load **{R['peak_perm_load_no_flex']:.1f} PPPD**)."
    )
else:
    summary_points.append(
        "Permanent-only Green expectation is OFF. Flex may be used to keep post-flex load within acceptable bands."
    )

summary_points.append(
    f"**Financial proxy**: annual EBITDA proxy is **${R['ebitda_proxy_annual']:,.0f}**, computed as "
    "Contribution − SWB − turnover replacement − access margin at risk."
)

summary_points.append(
    f"**Flu requisition timing**: winter-step posting month(s) detected = **{flu_post_label}** (if present)."
)

st.markdown("\n".join([f"- {p}" for p in summary_points]))


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
