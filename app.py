# Bramhall Co. — Predictive Staffing Model (PSM)
# "predict. perform. prosper."

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from psm.staffing_model import StaffingModel

MODEL_VERSION = "2026-01-30-bramhall-v1"

# ============================================================
# BRAND IDENTITY
# ============================================================
GOLD = "#7a6200"
BLACK = "#000000"
CREAM = "#faf8f3"
LIGHT_GOLD = "#d4c17f"
DARK_GOLD = "#5c4a00"
GOLD_MUTED = "#a89968"

WINTER = {12, 1, 2}
SUMMER = {6, 7, 8}
N_MONTHS = 36
AVG_DAYS_PER_MONTH = 30.4

MONTH_OPTIONS: List[Tuple[str, int]] = [
    ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4),
    ("May", 5), ("Jun", 6), ("Jul", 7), ("Aug", 8),
    ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12),
]

# ============================================================
# POSTURE MAPS (module scope; used in multiple sections)
# ============================================================
POSTURE_LABEL = {
    1: "Very Lean",
    2: "Lean",
    3: "Balanced",
    4: "Safe",
    5: "Very Safe",
}
POSTURE_TEXT = {
    1: "Very Lean: Minimum permanent staff. Higher utilization, more volatility, more flex/visit-loss risk.",
    2: "Lean: Cost-efficient posture. Limited buffer. Requires strong flex/PRN execution.",
    3: "Balanced: Standard posture. Reasonable buffer for peaks and normal absenteeism.",
    4: "Safe: Proactive staffing. Protects access and quality. Higher SWB/visit.",
    5: "Very Safe: Maximum stability. Highest cost. Best for high-acuity/high-reliability expectations.",
}
POSTURE_BASE_COVERAGE_MULT = {1: 0.92, 2: 0.96, 3: 1.00, 4: 1.04, 5: 1.08}
POSTURE_WINTER_BUFFER_ADD  = {1: 0.00, 2: 0.02, 3: 0.04, 4: 0.06, 5: 0.08}
POSTURE_FLEX_CAP_MULT      = {1: 1.35, 2: 1.15, 3: 1.00, 4: 0.90, 5: 0.80}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Predictive Staffing Model | Bramhall Co.",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# EMBEDDED LOGO + CSS
# ============================================================
# ✅ Paste your existing giant LOGO_B64 value here:
LOGO_B64 = "<PASTE_YOUR_EXISTING_LOGO_B64_HERE>"

# ✅ Paste your existing INTRO_CSS string here (keep as-is):
INTRO_CSS = "<PASTE_YOUR_EXISTING_INTRO_CSS_HERE>"

st.markdown(INTRO_CSS, unsafe_allow_html=True)

# ============================================================
# INTRO
# ============================================================
st.markdown("<div class='intro-container'>", unsafe_allow_html=True)
st.markdown(
    f'<img src="data:image/png;base64,{LOGO_B64}" class="intro-logo" />',
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class='intro-line-wrapper'><div class='intro-line'></div></div>
<div class='intro-text'>
  <h2>Predictive Staffing Model</h2>
  <p class='intro-tagline'>predict. perform. prosper.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# ============================================================
# MODEL + HELPERS
# ============================================================
model = StaffingModel()

def month_name(m: int) -> str:
    return datetime(2000, int(m), 1).strftime("%b")

def lead_days_to_months(days: int, avg: float = AVG_DAYS_PER_MONTH) -> int:
    return max(0, int(math.ceil(float(days) / float(avg))))

def provider_day_equiv_from_fte(fte: float, hrs_wk: float, fte_hrs: float) -> float:
    return float(fte) * (float(fte_hrs) / max(float(hrs_wk), 1e-9))

def compute_visits_curve(months: List[int], y0: float, y1: float, y2: float, seas: float) -> List[float]:
    out: List[float] = []
    for i, m in enumerate(months):
        base = y0 if i < 12 else y1 if i < 24 else y2
        if m in WINTER:
            v = base * (1 + seas)
        elif m in SUMMER:
            v = base * (1 - seas)
        else:
            v = base
        out.append(float(v))
    return out

def apply_flu_uplift(visits: List[float], months: List[int], flu_months: Set[int], uplift: float) -> List[float]:
    return [float(v) * (1 + uplift) if m in flu_months else float(v) for v, m in zip(visits, months)]

def monthly_hours_from_fte(fte: float, fte_hrs: float, days: int) -> float:
    return float(fte) * float(fte_hrs) * (float(days) / 7.0)

def loaded_hourly_rate(base: float, ben: float, ot: float, bon: float) -> float:
    return float(base) * (1 + bon) * (1 + ben) * (1 + ot)

def compute_role_mix_ratios(vpd: float, mdl: StaffingModel) -> Dict[str, float]:
    if hasattr(mdl, "get_role_mix_ratios"):
        return mdl.get_role_mix_ratios(vpd)
    daily = mdl.calculate(vpd)
    prov = max(float(daily.get("provider_day", 0.0)), 0.25)
    return {
        "psr_per_provider": float(daily.get("psr_day", 0.0)) / prov,
        "ma_per_provider": float(daily.get("ma_day", 0.0)) / prov,
        "xrt_per_provider": float(daily.get("xrt_day", 0.0)) / prov,
    }

def annual_swb_per_visit_from_supply(
    prov_paid: List[float],
    prov_flex: List[float],
    vpd: List[float],
    dim: List[int],
    fte_hrs: float,
    role_mix: Dict[str, float],
    rates: Dict[str, float],
    ben: float,
    ot: float,
    bon: float,
    phys_hrs: float = 0.0,
    sup_hrs: float = 0.0,
) -> Tuple[float, float, float]:
    total_swb, total_vis = 0.0, 0.0
    apc_r = loaded_hourly_rate(rates["apc"], ben, ot, bon)
    psr_r = loaded_hourly_rate(rates["psr"], ben, ot, bon)
    ma_r = loaded_hourly_rate(rates["ma"], ben, ot, bon)
    rt_r = loaded_hourly_rate(rates["rt"], ben, ot, bon)
    phys_r = loaded_hourly_rate(rates["physician"], ben, ot, bon)
    sup_r = loaded_hourly_rate(rates["supervisor"], ben, ot, bon)

    for paid, flex, v, d in zip(prov_paid, prov_flex, vpd, dim):
        mv = max(float(v) * float(d), 1.0)
        pt = float(paid) + float(flex)

        psr_fte = pt * role_mix["psr_per_provider"]
        ma_fte = pt * role_mix["ma_per_provider"]
        rt_fte = pt * role_mix["xrt_per_provider"]

        ph = monthly_hours_from_fte(pt, fte_hrs, int(d))
        psr_h = monthly_hours_from_fte(psr_fte, fte_hrs, int(d))
        ma_h = monthly_hours_from_fte(ma_fte, fte_hrs, int(d))
        rt_h = monthly_hours_from_fte(rt_fte, fte_hrs, int(d))

        swb = ph * apc_r + psr_h * psr_r + ma_h * ma_r + rt_h * rt_r + phys_hrs * phys_r + sup_hrs * sup_r
        total_swb += swb
        total_vis += mv

    return total_swb / max(total_vis, 1.0), total_swb, total_vis

# ============================================================
# DATA MODELS
# ============================================================
@dataclass(frozen=True)
class ModelParams:
    visits: float
    annual_growth: float
    seasonality_pct: float
    flu_uplift_pct: float
    flu_months: Set[int]
    peak_factor: float
    visits_per_provider_hour: float
    hours_week: float
    days_open_per_week: float
    fte_hours_week: float

    annual_turnover: float
    turnover_notice_days: int
    hiring_runway_days: int

    ramp_months: int
    ramp_productivity: float
    fill_probability: float

    winter_anchor_month: int
    winter_end_month: int
    freeze_months: Set[int]

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

    hourly_rates: Dict[str, float]
    benefits_load_pct: float
    ot_sick_pct: float
    bonus_pct: float
    physician_supervision_hours_per_month: float
    supervisor_hours_per_month: float

    min_perm_providers_per_day: float
    allow_prn_override: bool
    require_perm_under_green_no_flex: bool

    _v: str = MODEL_VERSION

@dataclass(frozen=True)
class Policy:
    base_coverage_pct: float
    winter_coverage_pct: float

# ============================================================
# SIMULATION ENGINE (now supports current_fte_override)
# ============================================================
def simulate_policy(
    params: ModelParams,
    policy: Policy,
    current_fte_override: Optional[float] = None,
) -> Dict[str, object]:
    today = datetime.today()
    dates = pd.date_range(start=datetime(today.year, 1, 1), periods=N_MONTHS, freq="MS")
    months = [int(d.month) for d in dates]
    dim = [pd.Period(d, "M").days_in_month for d in dates]

    hiring_lead_mo = lead_days_to_months(params.hiring_runway_days)
    mo_turn = params.annual_turnover / 12.0
    fill_p = float(np.clip(params.fill_probability, 0.0, 1.0))

    # Demand
    y0 = params.visits
    y1 = y0 * (1 + params.annual_growth)
    y2 = y1 * (1 + params.annual_growth)

    v_base = compute_visits_curve(months, y0, y1, y2, params.seasonality_pct)
    v_flu = apply_flu_uplift(v_base, months, params.flu_months, params.flu_uplift_pct)
    v_peak = [v * params.peak_factor for v in v_flu]

    role_mix = compute_role_mix_ratios(y1, model)

    vph = max(params.visits_per_provider_hour, 1e-6)
    req_hr_day = np.array([v / vph for v in v_peak], dtype=float)
    req_eff = (req_hr_day * params.days_open_per_week) / max(params.fte_hours_week, 1e-6)

    def is_winter(m: int) -> bool:
        a, e = params.winter_anchor_month, params.winter_end_month
        if a <= e:
            return a <= m <= e
        return (m >= a) or (m <= e)

    def target_fte_for_month(idx: int) -> float:
        idx = min(max(idx, 0), len(req_eff) - 1)
        base_required = float(req_eff[idx])
        return base_required * (policy.winter_coverage_pct if is_winter(months[idx]) else policy.base_coverage_pct)

    def ramp_factor(age: int) -> float:
        rm = max(int(params.ramp_months), 0)
        return params.ramp_productivity if (rm > 0 and age < rm) else 1.0

    # ✅ MODIFIED INIT (Enhancement 2)
    initial_staff = float(current_fte_override) if current_fte_override is not None else float(target_fte_for_month(0))
    cohorts = [{"fte": initial_staff, "age": 9999}]
    pipeline: List[Dict[str, object]] = []

    paid_arr: List[float] = []
    eff_arr: List[float] = []
    hires_arr: List[float] = []
    hire_reason_arr: List[str] = []
    target_arr: List[float] = []
    req_arr: List[float] = []

    for t in range(N_MONTHS):
        cur_mo = months[t]

        # Turnover
        for c in cohorts:
            c["fte"] = max(float(c["fte"]) * (1 - mo_turn), 0.0)

        # Arriving hires
        arriving = [h for h in pipeline if int(h["arrive"]) == t]
        total_hired = float(sum(float(h["fte"]) for h in arriving))
        if total_hired > 1e-9:
            cohorts.append({"fte": total_hired, "age": 0})

        # Current supply
        cur_paid = float(sum(float(c["fte"]) for c in cohorts))
        cur_eff = float(sum(float(c["fte"]) * ramp_factor(int(c["age"])) for c in cohorts))

        # Peak-aware hiring (freeze months prevent posting)
        can_post = cur_mo not in params.freeze_months
        if can_post and (t + hiring_lead_mo < N_MONTHS):
            future_idx = t + hiring_lead_mo

            horizon_end = min(future_idx + 6, N_MONTHS)
            peak_target = target_fte_for_month(future_idx)
            peak_month_idx = future_idx
            for check_idx in range(future_idx + 1, horizon_end):
                check_target = target_fte_for_month(check_idx)
                if check_target > peak_target:
                    peak_target = check_target
                    peak_month_idx = check_idx

            projected_paid = cur_paid * ((1 - mo_turn) ** hiring_lead_mo)
            for h in pipeline:
                arr = int(h["arrive"])
                if t < arr <= peak_month_idx:
                    months_from_arrival_to_peak = peak_month_idx - arr
                    projected_paid += float(h["fte"]) * ((1 - mo_turn) ** months_from_arrival_to_peak)

            hiring_gap = peak_target - projected_paid
            if hiring_gap > 0.05:
                hire_amount = hiring_gap * fill_p
                arrival_m = months[future_idx]
                peak_m = months[peak_month_idx]
                season_label = "winter" if is_winter(peak_m) else "base"
                if peak_month_idx > future_idx:
                    reason = (
                        f"Post {month_name(cur_mo)} for {month_name(arrival_m)} arrival → "
                        f"Staff for {month_name(peak_m)} peak ({peak_target:.2f} {season_label}). Gap: {hiring_gap:.2f}"
                    )
                else:
                    reason = (
                        f"Post {month_name(cur_mo)} for {month_name(arrival_m)} arrival: "
                        f"need {peak_target:.2f} ({season_label}). Gap: {hiring_gap:.2f}"
                    )

                pipeline.append({"req_posted": t, "arrive": future_idx, "fte": hire_amount, "reason": reason})

        for c in cohorts:
            c["age"] = int(c["age"]) + 1

        paid_arr.append(cur_paid)
        eff_arr.append(cur_eff)
        hires_arr.append(total_hired)
        hire_reason_arr.append(" | ".join(str(h["reason"]) for h in arriving) if arriving else "")
        target_arr.append(target_fte_for_month(t))
        req_arr.append(float(req_eff[t]))

    p_paid = np.array(paid_arr, dtype=float)
    p_eff = np.array(eff_arr, dtype=float)
    v_pk = np.array(v_peak, dtype=float)
    v_av = np.array(v_flu, dtype=float)
    d = np.array(dim, dtype=float)
    tgt_pol = np.array(target_arr, dtype=float)
    req_eff_arr = np.array(req_arr, dtype=float)

    # PPPD load (post-flex)
    flex_fte = np.zeros(N_MONTHS, dtype=float)
    load_post = np.zeros(N_MONTHS, dtype=float)
    for i in range(N_MONTHS):
        gap = max(req_eff_arr[i] - p_eff[i], 0.0)
        flex_used = min(gap, float(params.flex_max_fte_per_month))
        flex_fte[i] = flex_used
        pde_tot = provider_day_equiv_from_fte(p_eff[i] + flex_used, params.hours_week, params.fte_hours_week)
        load_post[i] = v_pk[i] / max(pde_tot, 1e-6)

    residual_gap = np.maximum(req_eff_arr - (p_eff + flex_fte), 0.0)
    prov_day_gap = float(np.sum(residual_gap * d))
    est_visits_lost = prov_day_gap * params.visits_lost_per_provider_day_gap
    est_margin_risk = est_visits_lost * params.net_revenue_per_visit

    repl = p_paid * mo_turn
    repl_mult = np.ones(N_MONTHS, dtype=float)
    repl_mult = np.where(load_post > params.budgeted_pppd, params.turnover_yellow_mult, repl_mult)
    repl_mult = np.where(load_post > params.red_start_pppd, params.turnover_red_mult, repl_mult)
    turn_cost = float(np.sum(repl * params.provider_replacement_cost * repl_mult))

    swb_all, swb_tot, vis_tot = annual_swb_per_visit_from_supply(
        list(p_paid),
        list(flex_fte),
        list(v_av),
        list(dim),
        params.fte_hours_week,
        role_mix,
        params.hourly_rates,
        params.benefits_load_pct,
        params.ot_sick_pct,
        params.bonus_pct,
        params.physician_supervision_hours_per_month,
        params.supervisor_hours_per_month,
    )

    hrs_per_fte_day = params.fte_hours_week / max(params.days_open_per_week, 1e-6)
    sup_tot_hrs = (p_eff + flex_fte) * hrs_per_fte_day
    util = (v_pk / params.visits_per_provider_hour) / np.maximum(sup_tot_hrs, 1e-9)

    yellow_ex = np.maximum(load_post - params.budgeted_pppd, 0.0)
    red_ex = np.maximum(load_post - params.red_start_pppd, 0.0)
    burn_pen = float(np.sum((yellow_ex**1.2) * d) + 3.0 * float(np.sum((red_ex**2.0) * d)))

    perm_total = float(np.sum(p_eff * d))
    flex_total = float(np.sum(flex_fte * d))
    flex_share = flex_total / max(perm_total + flex_total, 1e-9)

    score = swb_tot + turn_cost + est_margin_risk + 2000.0 * burn_pen

    # Ledger
    rows: List[Dict[str, object]] = []
    for i in range(N_MONTHS):
        lab = dates[i].strftime("%Y-%b")
        mv = float(v_av[i] * dim[i])
        mnc = mv * params.net_revenue_per_visit

        mswb_pv, mswb, _ = annual_swb_per_visit_from_supply(
            [float(p_paid[i])],
            [float(flex_fte[i])],
            [float(v_av[i])],
            [int(dim[i])],
            params.fte_hours_week,
            role_mix,
            params.hourly_rates,
            params.benefits_load_pct,
            params.ot_sick_pct,
            params.bonus_pct,
            params.physician_supervision_hours_per_month,
            params.supervisor_hours_per_month,
        )

        gw = float(residual_gap[i] * dim[i])
        gt = float(np.sum(residual_gap * d))
        mar = est_margin_risk * (gw / max(gt, 1e-9))
        mebitda = mnc - mswb - mar

        rows.append(
            {
                "Month": lab,
                "Visits/Day (avg)": float(v_av[i]),
                "Total Visits (month)": mv,
                "SWB Dollars (month)": float(mswb),
                "SWB/Visit (month)": float(mswb_pv),
                "EBITDA Proxy (month)": float(mebitda),
                "Permanent FTE (Paid)": float(p_paid[i]),
                "Permanent FTE (Effective)": float(p_eff[i]),
                "Flex FTE Used": float(flex_fte[i]),
                "Required Provider FTE (effective)": float(req_eff_arr[i]),
                "Utilization (Req/Supplied)": float(util[i]),
                "Load PPPD (post-flex)": float(load_post[i]),
                "Hires Visible (FTE)": float(hires_arr[i]),
                "Hire Reason": str(hire_reason_arr[i]),
                "Target FTE (policy)": float(tgt_pol[i]),
            }
        )

    ledger = pd.DataFrame(rows)
    ledger["Year"] = ledger["Month"].str[:4].astype(int)

    annual = ledger.groupby("Year", as_index=False).agg(
        Visits=("Total Visits (month)", "sum"),
        SWB_Dollars=("SWB Dollars (month)", "sum"),
        EBITDA_Proxy=("EBITDA Proxy (month)", "sum"),
        Min_Perm_Paid_FTE=("Permanent FTE (Paid)", "min"),
        Max_Perm_Paid_FTE=("Permanent FTE (Paid)", "max"),
        Avg_Utilization=("Utilization (Req/Supplied)", "mean"),
    )
    annual["SWB_per_Visit"] = annual["SWB_Dollars"] / annual["Visits"].clip(lower=1.0)

    tgt = params.target_swb_per_visit
    tol = params.swb_tolerance
    annual["SWB_Dev"] = np.maximum(np.abs(annual["SWB_per_Visit"] - tgt) - tol, 0.0)
    swb_pen = float(np.sum((annual["SWB_Dev"] ** 2) * 1_500_000.0))
    flex_pen = (max(flex_share - 0.10, 0.0) ** 2) * 2_000_000.0
    score += swb_pen + flex_pen

    ebitda_ann = vis_tot * params.net_revenue_per_visit - swb_tot - turn_cost - est_margin_risk
    mo_red = int(np.sum(load_post > params.red_start_pppd))
    pk_load = float(np.max(load_post))

    return {
        "dates": list(dates),
        "months": months,
        "perm_paid": list(p_paid),
        "perm_eff": list(p_eff),
        "req_eff_fte_needed": list(req_eff_arr),
        "utilization": list(util),
        "load_post": list(load_post),
        "annual_swb_per_visit": float(swb_all),
        "flex_share": float(flex_share),
        "months_red": mo_red,
        "peak_load_post": pk_load,
        "ebitda_proxy_annual": float(ebitda_ann),
        "score": float(score),
        "ledger": ledger.drop(columns=["Year"]),
        "annual_summary": annual,
        "target_policy": list(tgt_pol),
    }

# ============================================================
# CACHED SIMULATION (Enhancement 3)
# ============================================================
@st.cache_data(show_spinner=False)
def cached_simulate(
    params_dict: Dict[str, Any],
    policy_dict: Dict[str, float],
    current_fte_override: Optional[float] = None,
) -> Dict[str, object]:
    p = dict(params_dict)
    p["flu_months"] = set(p.get("flu_months", []))
    p["freeze_months"] = set(p.get("freeze_months", []))
    params = ModelParams(**p)
    policy = Policy(**policy_dict)
    return simulate_policy(params, policy, current_fte_override)

# ============================================================
# SIDEBAR
# ============================================================
def build_sidebar() -> Tuple[ModelParams, Policy, Dict[str, Any], bool]:
    with st.sidebar:
        st.markdown(
            f"""
            <div style='background: white; padding: 1.5rem; border-radius: 12px;
                        border: 2px solid {GOLD}; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05);'>
                <div style='font-weight: 700; font-size: 1.1rem; color: {GOLD}; margin-bottom: 0.75rem;
                            font-family: "Cormorant Garamond", serif;'>
                    🎯 Intelligent Cost-Driven Staffing
                </div>
                <div style='font-size: 0.85rem; color: #555; line-height: 1.6;'>
                    Peak-aware planning with hiring runway + optional posting freezes.
                    Set a target utilization, or use <strong>"Suggest Optimal"</strong> to
                    find utilization that best matches your SWB/visit target.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"<h3 style='color: {GOLD}; font-size: 1.1rem; margin-bottom: 1rem;'>📊 Core Settings</h3>", unsafe_allow_html=True)

        visits = st.number_input("**Average Visits/Day**", 1.0, value=36.0, step=1.0)
        annual_growth = st.number_input("**Annual Growth %**", 0.0, value=10.0, step=1.0) / 100.0

        c1, c2 = st.columns(2)
        with c1:
            seasonality_pct = st.number_input("Seasonality %", 0.0, value=20.0, step=5.0) / 100.0
        with c2:
            peak_factor = st.number_input("Peak Factor", 1.0, value=1.2, step=0.1)

        annual_turnover = st.number_input("**Turnover %**", 0.0, value=16.0, step=1.0) / 100.0

        # ============================================================
        # ENHANCEMENT 1: Current State Input
        # ============================================================
        st.markdown("#### Current Staffing")
        use_current_state = st.checkbox(
            "Start from actual current FTE (not policy target)",
            value=False,
            help="Check this to model from your actual current staffing level instead of policy target",
        )
        current_fte = None
        if use_current_state:
            current_fte = st.number_input(
                "Current FTE on Staff",
                0.1,
                value=3.5,
                step=0.1,
                help="Your actual current provider FTE (paid headcount)",
            )

        st.markdown("**Lead Times**")
        c1, c2 = st.columns(2)
        with c1:
            turnover_notice_days = st.number_input("Turnover Notice", 0, value=90, step=10)
        with c2:
            hiring_runway_days = st.number_input("Hiring Runway", 0, value=210, step=10)

        with st.expander("ℹ️ **Hiring Timeline Breakdown**", expanded=False):
            st.markdown(
                """
**Hiring Runway** = Time from req posted until provider is productive

Example breakdown for 210 days:
- Recruitment: 90 days
- Credentialing: 90 days
- Onboarding: 30 days
- Total: 210 days (~7 months)
"""
            )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        with st.expander("⚙️ **Advanced Settings**", expanded=False):
            st.markdown("**Clinic Operations**")
            hours_week = st.number_input("Clinic Hours/Week", 1.0, value=84.0, step=1.0)
            days_open_per_week = st.number_input("Days Open/Week", 1.0, 7.0, value=7.0, step=1.0)
            fte_hours_week = st.number_input("FTE Hours/Week", 1.0, value=36.0, step=1.0)
            visits_per_provider_hour = st.slider("Visits/Provider-Hour", 2.0, 4.0, 3.0, 0.1)

            st.markdown("**Workforce**")
            ramp_months = st.slider("Ramp-up Months", 0, 6, 1)
            ramp_productivity = st.slider("Ramp Productivity %", 30, 100, 75, 5) / 100.0
            fill_probability = st.slider("Fill Probability %", 0, 100, 85, 5) / 100.0

            st.markdown("**Risk Thresholds (PPPD)**")
            budgeted_pppd = st.number_input("Green Threshold", 5.0, value=36.0, step=1.0)
            yellow_max_pppd = st.number_input("Yellow Threshold", 5.0, value=42.0, step=1.0)
            red_start_pppd = st.number_input("Red Threshold", 5.0, value=45.0, step=1.0)

            st.markdown("**Seasonal Configuration**")
            flu_uplift_pct = st.number_input("Flu Season Uplift %", 0.0, value=0.0, step=5.0) / 100.0
            flu_months = st.multiselect(
                "Flu Months",
                MONTH_OPTIONS,
                default=[("Oct", 10), ("Nov", 11), ("Dec", 12), ("Jan", 1), ("Feb", 2)],
            )
            flu_months_set = {m for _, m in flu_months} if flu_months else set()

            winter_anchor_month = st.selectbox("Winter Start Month", MONTH_OPTIONS, index=11)
            winter_anchor_month_num = int(winter_anchor_month[1])
            winter_end_month = st.selectbox("Winter End Month", MONTH_OPTIONS, index=1)
            winter_end_month_num = int(winter_end_month[1])

            freeze_months = st.multiselect(
                "Hiring Freeze Months (no req posting)",
                MONTH_OPTIONS,
                default=[],
            )
            freeze_months_set = {m for _, m in freeze_months} if freeze_months else set()

            st.markdown("**Policy Constraints**")
            min_perm_providers_per_day = st.number_input("Min Providers/Day", 0.0, value=1.0, step=0.25)
            allow_prn_override = st.checkbox("Allow Base < Minimum", value=False)
            require_perm_under_green_no_flex = st.checkbox("Require Perm ≤ Green", value=True)
            flex_max_fte_per_month = st.slider("Max Flex FTE/Month", 0.0, 10.0, 2.0, 0.25)
            flex_cost_multiplier = st.slider("Flex Cost Multiplier", 1.0, 2.0, 1.25, 0.05)

        with st.expander("💰 **Financial Parameters**", expanded=False):
            target_swb_per_visit = st.number_input("Target SWB/Visit ($)", 0.0, value=85.0, step=1.0)
            swb_tolerance = st.number_input("SWB Tolerance ($)", 0.0, value=2.0, step=0.5)
            net_revenue_per_visit = st.number_input("Net Contribution/Visit ($)", 0.0, value=140.0, step=5.0)
            visits_lost_per_provider_day_gap = st.number_input("Visits Lost/Provider-Day Gap", 0.0, value=18.0, step=1.0)
            provider_replacement_cost = st.number_input("Replacement Cost ($)", 0.0, value=75000.0, step=5000.0)
            turnover_yellow_mult = st.slider("Turnover Mult (Yellow)", 1.0, 3.0, 1.3, 0.05)
            turnover_red_mult = st.slider("Turnover Mult (Red)", 1.0, 5.0, 2.0, 0.1)

            benefits_load_pct = st.number_input("Benefits Load %", 0.0, value=30.0, step=1.0) / 100.0
            bonus_pct = st.number_input("Bonus %", 0.0, value=10.0, step=1.0) / 100.0
            ot_sick_pct = st.number_input("OT+Sick %", 0.0, value=4.0, step=0.5) / 100.0

            c1, c2 = st.columns(2)
            with c1:
                physician_hr = st.number_input("Physician ($/hr)", 0.0, value=135.79, step=1.0)
                apc_hr = st.number_input("APP ($/hr)", 0.0, value=62.0, step=1.0)
                ma_hr = st.number_input("MA ($/hr)", 0.0, value=24.14, step=0.5)
            with c2:
                psr_hr = st.number_input("PSR ($/hr)", 0.0, value=21.23, step=0.5)
                rt_hr = st.number_input("RT ($/hr)", 0.0, value=31.36, step=0.5)
                supervisor_hr = st.number_input("Supervisor ($/hr)", 0.0, value=28.25, step=0.5)

            physician_supervision_hours_per_month = st.number_input("Physician Supervision (hrs/mo)", 0.0, value=0.0, step=1.0)
            supervisor_hours_per_month = st.number_input("Supervisor Hours (hrs/mo)", 0.0, value=0.0, step=1.0)

        st.markdown(f"<div style='height: 2px; background: {LIGHT_GOLD}; margin: 2rem 0;'></div>", unsafe_allow_html=True)

        # ✅ FIX: Use select_slider so format_func is supported (prevents your TypeError)
        st.markdown(f"<h3 style='color: {GOLD}; font-size: 1.1rem; margin-bottom: 1rem;'>🏛 Staffing Risk Posture</h3>", unsafe_allow_html=True)
        risk_posture = st.select_slider(
            "**Lean ↔ Safe**",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: POSTURE_LABEL.get(x, str(x)),
            help="Controls how much permanent staffing buffer you carry vs relying on flex and absorbing volatility.",
        )
        st.caption(POSTURE_TEXT[int(risk_posture)])

        st.markdown(f"<h3 style='color: {GOLD}; font-size: 1.1rem; margin-bottom: 1rem;'>🎯 Smart Staffing Policy</h3>", unsafe_allow_html=True)

        if "target_utilization" not in st.session_state:
            st.session_state.target_utilization = 92

        target_utilization = st.slider("**Target Utilization %**", 80, 98, value=int(st.session_state.target_utilization), step=2)

        # ✅ Keep only ONE winter buffer slider
        winter_buffer_pct = st.slider("**Winter Buffer %**", 0, 10, 3, 1) / 100.0

        base_coverage_from_util = 1.0 / (target_utilization / 100.0)
        posture_mult = POSTURE_BASE_COVERAGE_MULT[int(risk_posture)]
        posture_winter_add = POSTURE_WINTER_BUFFER_ADD[int(risk_posture)]
        base_coverage_pct = base_coverage_from_util * posture_mult
        winter_coverage_pct = base_coverage_pct * (1 + winter_buffer_pct + posture_winter_add)

        # Flex cap posture scaling
        flex_max_fte_effective = float(flex_max_fte_per_month) * POSTURE_FLEX_CAP_MULT[int(risk_posture)]

        # Suggest Optimal (now uses the already-defined flex_max_fte_effective)
        if st.button("🎯 Suggest Optimal", use_container_width=True):
            with st.spinner("Finding optimal staffing..."):
                hourly_rates_temp = {
                    "physician": physician_hr,
                    "apc": apc_hr,
                    "ma": ma_hr,
                    "psr": psr_hr,
                    "rt": rt_hr,
                    "supervisor": supervisor_hr,
                }

                params_temp = ModelParams(
                    visits=visits,
                    annual_growth=annual_growth,
                    seasonality_pct=seasonality_pct,
                    flu_uplift_pct=flu_uplift_pct,
                    flu_months=flu_months_set,
                    peak_factor=peak_factor,
                    visits_per_provider_hour=visits_per_provider_hour,
                    hours_week=hours_week,
                    days_open_per_week=days_open_per_week,
                    fte_hours_week=fte_hours_week,
                    annual_turnover=annual_turnover,
                    turnover_notice_days=turnover_notice_days,
                    hiring_runway_days=hiring_runway_days,
                    ramp_months=ramp_months,
                    ramp_productivity=ramp_productivity,
                    fill_probability=fill_probability,
                    winter_anchor_month=winter_anchor_month_num,
                    winter_end_month=winter_end_month_num,
                    freeze_months=freeze_months_set,
                    budgeted_pppd=budgeted_pppd,
                    yellow_max_pppd=yellow_max_pppd,
                    red_start_pppd=red_start_pppd,
                    flex_max_fte_per_month=flex_max_fte_effective,
                    flex_cost_multiplier=flex_cost_multiplier,
                    target_swb_per_visit=target_swb_per_visit,
                    swb_tolerance=swb_tolerance,
                    net_revenue_per_visit=net_revenue_per_visit,
                    visits_lost_per_provider_day_gap=visits_lost_per_provider_day_gap,
                    provider_replacement_cost=provider_replacement_cost,
                    turnover_yellow_mult=turnover_yellow_mult,
                    turnover_red_mult=turnover_red_mult,
                    hourly_rates=hourly_rates_temp,
                    benefits_load_pct=benefits_load_pct,
                    ot_sick_pct=ot_sick_pct,
                    bonus_pct=bonus_pct,
                    physician_supervision_hours_per_month=physician_supervision_hours_per_month,
                    supervisor_hours_per_month=supervisor_hours_per_month,
                    min_perm_providers_per_day=min_perm_providers_per_day,
                    allow_prn_override=allow_prn_override,
                    require_perm_under_green_no_flex=require_perm_under_green_no_flex,
                    _v=MODEL_VERSION,
                )

                best_util = 90
                best_diff = 1e18
                results_cache: Dict[int, float] = {}

                for test_util in range(86, 99, 2):
                    test_coverage = 1.0 / (test_util / 100.0)
                    test_winter = test_coverage * (1 + winter_buffer_pct + posture_winter_add)
                    test_policy = Policy(base_coverage_pct=test_coverage, winter_coverage_pct=test_winter)
                    test_result = simulate_policy(params_temp, test_policy, current_fte)
                    test_swb = float(test_result["annual_swb_per_visit"])
                    results_cache[test_util] = test_swb

                    diff = abs(test_swb - target_swb_per_visit)
                    if diff < best_diff:
                        best_util, best_diff = test_util, diff

                st.session_state.target_utilization = best_util
                st.success(f"✅ Best Match: **{best_util}%** utilization (SWB/visit ≈ ${results_cache[best_util]:.2f})")
                st.rerun()

        run_simulation = st.button("🚀 Run Simulation", use_container_width=True, type="primary")

        hourly_rates = {
            "physician": physician_hr,
            "apc": apc_hr,
            "ma": ma_hr,
            "psr": psr_hr,
            "rt": rt_hr,
            "supervisor": supervisor_hr,
        }

        params = ModelParams(
            visits=visits,
            annual_growth=annual_growth,
            seasonality_pct=seasonality_pct,
            flu_uplift_pct=flu_uplift_pct,
            flu_months=flu_months_set,
            peak_factor=peak_factor,
            visits_per_provider_hour=visits_per_provider_hour,
            hours_week=hours_week,
            days_open_per_week=days_open_per_week,
            fte_hours_week=fte_hours_week,
            annual_turnover=annual_turnover,
            turnover_notice_days=turnover_notice_days,
            hiring_runway_days=hiring_runway_days,
            ramp_months=ramp_months,
            ramp_productivity=ramp_productivity,
            fill_probability=fill_probability,
            winter_anchor_month=winter_anchor_month_num,
            winter_end_month=winter_end_month_num,
            freeze_months=freeze_months_set,
            budgeted_pppd=budgeted_pppd,
            yellow_max_pppd=yellow_max_pppd,
            red_start_pppd=red_start_pppd,
            flex_max_fte_per_month=flex_max_fte_effective,
            flex_cost_multiplier=flex_cost_multiplier,
            target_swb_per_visit=target_swb_per_visit,
            swb_tolerance=swb_tolerance,
            net_revenue_per_visit=net_revenue_per_visit,
            visits_lost_per_provider_day_gap=visits_lost_per_provider_day_gap,
            provider_replacement_cost=provider_replacement_cost,
            turnover_yellow_mult=turnover_yellow_mult,
            turnover_red_mult=turnover_red_mult,
            hourly_rates=hourly_rates,
            benefits_load_pct=benefits_load_pct,
            ot_sick_pct=ot_sick_pct,
            bonus_pct=bonus_pct,
            physician_supervision_hours_per_month=physician_supervision_hours_per_month,
            supervisor_hours_per_month=supervisor_hours_per_month,
            min_perm_providers_per_day=min_perm_providers_per_day,
            allow_prn_override=allow_prn_override,
            require_perm_under_green_no_flex=require_perm_under_green_no_flex,
            _v=MODEL_VERSION,
        )

        policy = Policy(base_coverage_pct=float(base_coverage_pct), winter_coverage_pct=float(winter_coverage_pct))

        ui = {
            "target_utilization": int(target_utilization),
            "winter_buffer_pct": float(winter_buffer_pct),
            "risk_posture": int(risk_posture),
            "base_coverage_from_util": float(base_coverage_from_util),
            "flex_max_fte_per_month": float(flex_max_fte_per_month),
            "current_fte": current_fte,
        }

        return params, policy, ui, bool(run_simulation)

params, policy, ui, run_simulation = build_sidebar()

# ============================================================
# RUN / LOAD RESULTS
# ============================================================
params_dict = {**params.__dict__}
policy_dict = {"base_coverage_pct": policy.base_coverage_pct, "winter_coverage_pct": policy.winter_coverage_pct}
current_fte = ui.get("current_fte", None)

if run_simulation:
    with st.spinner("🔍 Running simulation..."):
        R = cached_simulate(params_dict, policy_dict, current_fte)
    st.session_state["simulation_result"] = R
    st.success("✅ Simulation complete!")
else:
    if "simulation_result" not in st.session_state:
        R = cached_simulate(params_dict, policy_dict, current_fte)
        st.session_state["simulation_result"] = R
    R = st.session_state["simulation_result"]

# ============================================================
# EXEC HELPERS
# ============================================================
def pack_exec_metrics(res: dict) -> dict:
    annual = res["annual_summary"]
    y1_swb = float(annual.loc[0, "SWB_per_Visit"])
    y1_ebitda = float(annual.loc[0, "EBITDA_Proxy"])
    flex_share = float(res["flex_share"])
    red_months = int(res["months_red"])
    peak_load = float(res["peak_load_post"])
    margin = float(res["ebitda_proxy_annual"])
    return {
        "SWB/Visit (Y1)": y1_swb,
        "EBITDA Proxy (Y1)": y1_ebitda,
        "EBITDA Proxy (3yr total)": margin,
        "Red Months": red_months,
        "Flex Share": flex_share,
        "Peak Load (PPPD)": peak_load,
    }

# ============================================================
# SMART HIRING INSIGHTS + EXPORT (Enhancement 4)
# ============================================================
st.markdown("## 🧠 Smart Hiring Insights")
ledger = R["ledger"]
upcoming_hires = ledger[ledger["Hires Visible (FTE)"] > 0.05].head(12)

if len(upcoming_hires) > 0:
    st.markdown("### 📋 Next 12 Months Hiring Plan")
    st.dataframe(upcoming_hires[["Month", "Hires Visible (FTE)", "Hire Reason"]], use_container_width=True, hide_index=True)

    total_hires_12mo = float(upcoming_hires["Hires Visible (FTE)"].sum())
    avg_fte = float(ledger.head(12)["Permanent FTE (Paid)"].mean())

    st.markdown("### 📥 Export Hiring Plan")
    hiring_export = upcoming_hires[["Month", "Hires Visible (FTE)", "Hire Reason"]].copy()
    hiring_export.columns = ["Month", "FTE to Hire", "Hiring Rationale"]
    hiring_export["FTE to Hire"] = hiring_export["FTE to Hire"].map(lambda x: f"{float(x):.2f}")

    total_row = pd.DataFrame([{
        "Month": "TOTAL (12 months)",
        "FTE to Hire": f"{total_hires_12mo:.2f}",
        "Hiring Rationale": f"Total hiring volume: {(total_hires_12mo/max(avg_fte,1e-9))*100:.0f}% of avg staff",
    }])
    hiring_export = pd.concat([hiring_export, total_row], ignore_index=True)
    hiring_csv = hiring_export.to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.download_button(
            "⬇️ Download Hiring Action Plan (CSV)",
            hiring_csv,
            "hiring_action_plan.csv",
            "text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "⬇️ Download Full Ledger (CSV)",
            ledger.to_csv(index=False).encode("utf-8"),
            "staffing_ledger_full.csv",
            "text/csv",
            use_container_width=True,
        )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# SCENARIO COMPARISON (Enhancement 5)
# ============================================================
st.markdown("## 🔀 Scenario Comparison")
st.markdown("Compare how changing growth, turnover, and utilization affects outcomes.")

with st.expander("🎯 **Run Scenario Analysis**", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📊 Base Case (Current)**")
        st.info(
            f"Utilization: {ui['target_utilization']}%  \n"
            f"Growth: {params.annual_growth*100:.0f}%  \n"
            f"Turnover: {params.annual_turnover*100:.0f}%"
        )

    with col2:
        st.markdown("**🔺 Optimistic Scenario**")
        opt_growth = st.number_input("Growth %", 0.0, value=15.0, step=1.0, key="opt_growth") / 100.0
        opt_turnover = st.number_input("Turnover %", 0.0, value=12.0, step=1.0, key="opt_turn") / 100.0
        opt_util = st.slider("Utilization %", 80, 98, 94, 2, key="opt_util")

    with col3:
        st.markdown("**🔻 Conservative Scenario**")
        cons_growth = st.number_input("Growth %", 0.0, value=5.0, step=1.0, key="cons_growth") / 100.0
        cons_turnover = st.number_input("Turnover %", 0.0, value=20.0, step=1.0, key="cons_turn") / 100.0
        cons_util = st.slider("Utilization %", 80, 98, 88, 2, key="cons_util")

    if st.button("🔄 Run Scenario Comparison", use_container_width=True):
        with st.spinner("Running scenarios..."):
            base_metrics = pack_exec_metrics(R)

            opt_params = ModelParams(**{**params.__dict__, "annual_growth": opt_growth, "annual_turnover": opt_turnover})
            opt_cov = 1.0 / (opt_util / 100.0)
            opt_policy = Policy(base_coverage_pct=opt_cov, winter_coverage_pct=opt_cov * 1.03)
            R_opt = simulate_policy(opt_params, opt_policy, current_fte)
            opt_metrics = pack_exec_metrics(R_opt)

            cons_params = ModelParams(**{**params.__dict__, "annual_growth": cons_growth, "annual_turnover": cons_turnover})
            cons_cov = 1.0 / (cons_util / 100.0)
            cons_policy = Policy(base_coverage_pct=cons_cov, winter_coverage_pct=cons_cov * 1.03)
            R_cons = simulate_policy(cons_params, cons_policy, current_fte)
            cons_metrics = pack_exec_metrics(R_cons)

            df_scenarios = pd.DataFrame([
                {"Scenario": "Base Case", **base_metrics},
                {"Scenario": "Optimistic", **opt_metrics},
                {"Scenario": "Conservative", **cons_metrics},
            ])

            st.markdown("### 📊 Scenario Results")
            st.dataframe(
                df_scenarios.style.format({
                    "SWB/Visit (Y1)": lambda x: f"${x:.2f}",
                    "EBITDA Proxy (Y1)": lambda x: f"${x:,.0f}",
                    "EBITDA Proxy (3yr total)": lambda x: f"${x:,.0f}",
                    "Flex Share": lambda x: f"{x*100:.1f}%",
                    "Peak Load (PPPD)": lambda x: f"{x:.1f}",
                }),
                hide_index=True,
                use_container_width=True,
            )

            ebitda_range = float(df_scenarios["EBITDA Proxy (3yr total)"].max() - df_scenarios["EBITDA Proxy (3yr total)"].min())
            st.info(f"**Financial Range:** 3-year EBITDA varies by **${ebitda_range:,.0f}** across scenarios.")

            st.download_button(
                "⬇️ Download Scenario Comparison (CSV)",
                df_scenarios.to_csv(index=False).encode("utf-8"),
                "scenario_comparison.csv",
                "text/csv",
                use_container_width=True,
            )

# ============================================================
# SENSITIVITY ANALYSIS (Enhancement 6)
# ============================================================
with st.expander("📈 **Sensitivity Analysis**", expanded=False):
    st.markdown("See how sensitive results are to key assumptions.")

    if st.button("🔬 Run Sensitivity Analysis", key="sensitivity"):
        with st.spinner("Analyzing sensitivity..."):
            base_swb = float(R["annual_swb_per_visit"])
            base_ebitda = float(R["ebitda_proxy_annual"])

            turnover_tests = [0.12, 0.16, 0.20, 0.24]
            turnover_rows: List[Dict[str, float]] = []

            for turn in turnover_tests:
                test_params = ModelParams(**{**params.__dict__, "annual_turnover": float(turn)})
                test_result = simulate_policy(test_params, policy, current_fte)
                turnover_rows.append({
                    "Turnover": float(turn),
                    "SWB/Visit": float(test_result["annual_swb_per_visit"]),
                    "EBITDA": float(test_result["ebitda_proxy_annual"]),
                    "Δ SWB": float(test_result["annual_swb_per_visit"]) - base_swb,
                    "Δ EBITDA": float(test_result["ebitda_proxy_annual"]) - base_ebitda,
                })

            df_turn = pd.DataFrame(turnover_rows)
            st.dataframe(
                df_turn.style.format({
                    "Turnover": "{:.0%}",
                    "SWB/Visit": "${:.2f}",
                    "EBITDA": "${:,.0f}",
                    "Δ SWB": "${:+.2f}",
                    "Δ EBITDA": "${:+,.0f}",
                }),
                hide_index=True,
                use_container_width=True,
            )

            approx_per_4pt = abs(df_turn.loc[df_turn.index[-1], "Δ EBITDA"] - df_turn.loc[df_turn.index[0], "Δ EBITDA"]) / 3.0
            st.caption(f"💡 **Insight:** Every ~4% increase in turnover costs ~**${approx_per_4pt:,.0f}** in 3-year EBITDA (rough).")

# ============================================================
# CHARTS (keep your existing Plotly sections if you want)
# ============================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("## 📊 3-Year Financial Projection")

dates = R["dates"]
perm_paid = np.array(R["perm_paid"], dtype=float)
target_pol = np.array(R["target_policy"], dtype=float)
req_eff = np.array(R["req_eff_fte_needed"], dtype=float)
util = np.array(R["utilization"], dtype=float)
load_post = np.array(R["load_post"], dtype=float)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=dates, y=perm_paid, mode="lines+markers", name="Paid FTE", line=dict(color=GOLD, width=3)))
fig1.add_trace(go.Scatter(x=dates, y=target_pol, mode="lines+markers", name="Target (policy)", line=dict(color=BLACK, width=2, dash="dash")))
fig1.add_trace(go.Scatter(x=dates, y=req_eff, mode="lines", name="Required FTE", line=dict(color=LIGHT_GOLD, width=2, dash="dot")))
fig1.update_layout(height=450, hovermode="x unified", paper_bgcolor="white", plot_bgcolor="rgba(250, 248, 243, 0.3)")
st.plotly_chart(fig1, use_container_width=True)

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(x=dates, y=util * 100, mode="lines+markers", name="Utilization %"), secondary_y=False)
fig2.add_trace(go.Scatter(x=dates, y=load_post, mode="lines+markers", name="Load PPPD"), secondary_y=True)
fig2.update_layout(height=450, hovermode="x unified", paper_bgcolor="white", plot_bgcolor="rgba(250, 248, 243, 0.3)")
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TABLES
# ============================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("## 📈 Detailed Results")

annual = R["annual_summary"]
tab1, tab2 = st.tabs(["📊 Annual Summary", "📋 Monthly Ledger"])
with tab1:
    st.dataframe(annual, hide_index=True, use_container_width=True)
with tab2:
    st.dataframe(ledger, hide_index=True, use_container_width=True, height=420)

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div style='text-align: center; padding: 2rem 0; color: {GOLD_MUTED};'>
  <div style='font-size: 0.9rem; font-style: italic; font-family: "Cormorant Garamond", serif;'>
    predict. perform. prosper.
  </div>
  <div style='font-size: 0.75rem; margin-top: 0.5rem; color: #999;'>
    Bramhall Co. | Predictive Staffing Model v{MODEL_VERSION.split('-')[-1]}
  </div>
</div>
""",
    unsafe_allow_html=True,
)
