# Bramhall Co. — Predictive Staffing Model (PSM)
# Elegant, branded interface with sophisticated design
# "predict. perform. prosper."

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Tuple

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
    ("Jan", 1),
    ("Feb", 2),
    ("Mar", 3),
    ("Apr", 4),
    ("May", 5),
    ("Jun", 6),
    ("Jul", 7),
    ("Aug", 8),
    ("Sep", 9),
    ("Oct", 10),
    ("Nov", 11),
    ("Dec", 12),
]

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
# EMBEDDED LOGO (No file dependency)
# ============================================================
LOGO_B64 = "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"

# ============================================================
# CSS
# ============================================================
INTRO_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

h1, h2, h3, h4 {{
    font-family: 'Cormorant Garamond', serif !important;
    color: {BLACK} !important;
    letter-spacing: 0.015em;
    font-weight: 600 !important;
}}

body, p, div, span, label {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #2c2c2c;
}}

.intro-container {{
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem 0;
}}

.intro-logo {{
    max-width: 220px !important;
    width: 100% !important;
    height: auto !important;
    margin: 0 auto !important;
    display: block;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
}}

@media (max-width: 600px) {{
    .intro-logo {{
        max-width: 200px !important;
        width: 200px !important;
        margin-top: 0.6rem !important;
    }}
}}
@media (max-width: 400px) {{
    .intro-logo {{
        max-width: 180px !important;
        width: 180px !important;
        margin-top: 0.6rem !important;
    }}
}}

.intro-line-wrapper {{
    display: flex;
    justify-content: center;
    margin: 1.5rem 0 1rem;
}}
.intro-line {{
    width: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, {GOLD} 50%, transparent 100%);
    animation: lineGrow 1.6s ease-out forwards;
}}
.intro-text {{
    opacity: 0;
    transform: translateY(6px);
    animation: fadeInUp 1.4s ease-out forwards;
    animation-delay: 1.0s;
    text-align: center;
}}
.intro-text h2 {{
    font-size: 2.2rem;
    font-weight: 600;
    color: {BLACK};
    margin-bottom: 0.5rem;
    font-family: 'Cormorant Garamond', serif;
}}
.intro-tagline {{
    font-size: 1.1rem;
    font-style: italic;
    color: {GOLD};
    font-family: 'Cormorant Garamond', serif;
    letter-spacing: 0.1em;
    margin-top: 0.5rem;
}}

@keyframes lineGrow {{
    0%   {{ width: 0; }}
    100% {{ width: 360px; }}
}}
@keyframes fadeInUp {{
    0%   {{ opacity: 0; transform: translateY(6px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

.scorecard-hero {{
    background: white;
    border: 3px solid {GOLD};
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin: 2.5rem 0;
    box-shadow: 0 8px 32px rgba(122, 98, 0, 0.12);
    position: relative;
}}
.scorecard-hero::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 6px;
    height: 100%;
    background: linear-gradient(180deg, {GOLD} 0%, {DARK_GOLD} 100%);
    border-radius: 16px 0 0 16px;
}}
.scorecard-title {{
    font-size: 1.4rem;
    font-weight: 600;
    color: {BLACK};
    margin: 0 0 2rem 0;
    padding-bottom: 1rem;
    border-bottom: 2px solid {CREAM};
    font-family: 'Cormorant Garamond', serif;
    letter-spacing: 0.03em;
}}
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 2rem;
}}
.metric-card {{
    background: linear-gradient(135deg, {CREAM} 0%, white 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid {GOLD};
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}
.metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(122, 98, 0, 0.15);
    border-left-color: {DARK_GOLD};
}}
.metric-label {{
    font-size: 0.75rem;
    font-weight: 600;
    color: {DARK_GOLD};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.75rem;
}}
.metric-value {{
    font-size: 2.25rem;
    font-weight: 700;
    color: {BLACK};
    font-family: 'Cormorant Garamond', serif;
    line-height: 1;
    margin-bottom: 0.5rem;
}}
.metric-detail {{
    font-size: 0.8rem;
    color: #666;
    font-weight: 400;
}}

.status-card {{
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1.5rem 0;
    border-left: 5px solid;
    animation: slideIn 0.4s ease-out;
}}
@keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-20px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}
.status-success {{
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left-color: #28a745;
}}
.status-warning {{
    background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
    border-left-color: {GOLD};
}}
.status-content {{
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}}
.status-icon {{
    font-size: 2rem;
    line-height: 1;
}}
.status-title {{
    font-weight: 600;
    font-size: 1.15rem;
    margin-bottom: 0.5rem;
    color: {BLACK};
}}
.status-message {{
    font-size: 0.95rem;
    color: #444;
    line-height: 1.5;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {CREAM} 0%, #ffffff 100%);
    border-right: 1px solid {LIGHT_GOLD};
}}

.stButton > button {{
    background: linear-gradient(135deg, {GOLD} 0%, {DARK_GOLD} 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.85rem 2.5rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.85rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(122, 98, 0, 0.25);
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {DARK_GOLD} 0%, {BLACK} 100%);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(122, 98, 0, 0.35);
}}

.stDownloadButton > button {{
    background: white;
    color: {GOLD};
    border: 2px solid {GOLD};
    border-radius: 8px;
    padding: 0.65rem 1.5rem;
    font-weight: 500;
    transition: all 0.3s ease;
}}
.stDownloadButton > button:hover {{
    background: {GOLD};
    color: white;
    border-color: {GOLD};
}}

.streamlit-expanderHeader {{
    background: {CREAM};
    border-radius: 10px;
    font-weight: 500;
    border: 1px solid {LIGHT_GOLD};
    transition: all 0.3s ease;
}}
.streamlit-expanderHeader:hover {{
    background: white;
    border-color: {GOLD};
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 12px;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    background: {CREAM};
    border-radius: 10px 10px 0 0;
    padding: 1rem 2rem;
    font-weight: 500;
    border: 1px solid {LIGHT_GOLD};
    border-bottom: none;
    transition: all 0.3s ease;
}}
.stTabs [aria-selected="true"] {{
    background: white;
    border-color: {GOLD};
    color: {BLACK};
    font-weight: 600;
}}

.divider {{
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, {GOLD} 50%, transparent 100%);
    margin: 3rem 0;
}}
</style>
"""
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
    # coverage multipliers applied to required effective FTE
    base_coverage_pct: float
    winter_coverage_pct: float

# ============================================================
# SIMULATION ENGINE (peak-aware, lead-time aware, optional hiring freeze)
# ============================================================
def simulate_policy(params: ModelParams, policy: Policy) -> Dict[str, object]:
    today = datetime.today()
    dates = pd.date_range(start=datetime(today.year, 1, 1), periods=N_MONTHS, freq="MS")
    months = [int(d.month) for d in dates]
    dim = [pd.Period(d, "M").days_in_month for d in dates]

    hiring_lead_mo = lead_days_to_months(params.hiring_runway_days)
    mo_turn = params.annual_turnover / 12.0
    fill_p = float(np.clip(params.fill_probability, 0.0, 1.0))

    # Demand (baseline + growth + seasonality + flu uplift + peak factor)
    y0 = params.visits
    y1 = y0 * (1 + params.annual_growth)
    y2 = y1 * (1 + params.annual_growth)

    v_base = compute_visits_curve(months, y0, y1, y2, params.seasonality_pct)
    v_flu = apply_flu_uplift(v_base, months, params.flu_months, params.flu_uplift_pct)
    v_peak = [v * params.peak_factor for v in v_flu]

    role_mix = compute_role_mix_ratios(y1, model)

    # Required effective provider FTE (DEMAND-DRIVEN)
    vph = max(params.visits_per_provider_hour, 1e-6)
    req_hr_day = np.array([v / vph for v in v_peak], dtype=float)
    req_eff = (req_hr_day * params.days_open_per_week) / max(params.fte_hours_week, 1e-6)

    def is_winter(m: int) -> bool:
        a, e = params.winter_anchor_month, params.winter_end_month
        # handles wrap-around periods (e.g., Nov → Feb)
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

    # Initialize with a sensible starting point
    initial_target = target_fte_for_month(0)
    cohorts = [{"fte": initial_target, "age": 9999}]
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

        # Peak-aware hiring (respect freeze months for posting)
        can_post = cur_mo not in params.freeze_months
        if can_post and (t + hiring_lead_mo < N_MONTHS):
            future_idx = t + hiring_lead_mo

            # Look 6 months past arrival for peak planning
            horizon_end = min(future_idx + 6, N_MONTHS)
            peak_target = target_fte_for_month(future_idx)
            peak_month_idx = future_idx
            for check_idx in range(future_idx + 1, horizon_end):
                check_target = target_fte_for_month(check_idx)
                if check_target > peak_target:
                    peak_target = check_target
                    peak_month_idx = check_idx

            # Project paid FTE forward to peak (attrition + existing pipeline)
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

        # Age cohorts
        for c in cohorts:
            c["age"] = int(c["age"]) + 1

        # Record
        paid_arr.append(cur_paid)
        eff_arr.append(cur_eff)
        hires_arr.append(total_hired)
        hire_reason_arr.append(" | ".join(str(h["reason"]) for h in arriving) if arriving else "")
        target_arr.append(target_fte_for_month(t))
        req_arr.append(float(req_eff[t]))

    # Arrays
    p_paid = np.array(paid_arr, dtype=float)
    p_eff = np.array(eff_arr, dtype=float)
    v_pk = np.array(v_peak, dtype=float)
    v_av = np.array(v_flu, dtype=float)
    d = np.array(dim, dtype=float)
    tgt_pol = np.array(target_arr, dtype=float)
    req_eff_arr = np.array(req_arr, dtype=float)

    # PPPD load (post-flex)
    pde_perm = np.array(
        [provider_day_equiv_from_fte(f, params.hours_week, params.fte_hours_week) for f in p_eff], dtype=float
    )
    # Flex coverage
    flex_fte = np.zeros(N_MONTHS, dtype=float)
    load_post = np.zeros(N_MONTHS, dtype=float)
    for i in range(N_MONTHS):
        gap = max(req_eff_arr[i] - p_eff[i], 0.0)
        flex_used = min(gap, float(params.flex_max_fte_per_month))
        flex_fte[i] = flex_used
        pde_tot = provider_day_equiv_from_fte(p_eff[i] + flex_used, params.hours_week, params.fte_hours_week)
        load_post[i] = v_pk[i] / max(pde_tot, 1e-6)

    # Residual gap → visits lost → margin risk
    residual_gap = np.maximum(req_eff_arr - (p_eff + flex_fte), 0.0)
    prov_day_gap = float(np.sum(residual_gap * d))
    est_visits_lost = prov_day_gap * params.visits_lost_per_provider_day_gap
    est_margin_risk = est_visits_lost * params.net_revenue_per_visit

    # Turnover cost (stress multipliers based on load)
    repl = p_paid * mo_turn
    repl_mult = np.ones(N_MONTHS, dtype=float)
    repl_mult = np.where(load_post > params.budgeted_pppd, params.turnover_yellow_mult, repl_mult)
    repl_mult = np.where(load_post > params.red_start_pppd, params.turnover_red_mult, repl_mult)
    turn_cost = float(np.sum(repl * params.provider_replacement_cost * repl_mult))

    # SWB
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

    # Utilization (Req/Supplied hours)
    hrs_per_fte_day = params.fte_hours_week / max(params.days_open_per_week, 1e-6)
    sup_tot_hrs = (p_eff + flex_fte) * hrs_per_fte_day
    util = (v_pk / params.visits_per_provider_hour) / np.maximum(sup_tot_hrs, 1e-9)

    # Penalties
    yellow_ex = np.maximum(load_post - params.budgeted_pppd, 0.0)
    red_ex = np.maximum(load_post - params.red_start_pppd, 0.0)
    burn_pen = float(np.sum((yellow_ex**1.2) * d) + 3.0 * float(np.sum((red_ex**2.0) * d)))

    perm_total = float(np.sum(p_eff * d))
    flex_total = float(np.sum(flex_fte * d))
    flex_share = flex_total / max(perm_total + flex_total, 1e-9)

    score = swb_tot + turn_cost + est_margin_risk + 2000.0 * burn_pen

    # Ledger (monthly)
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

    # SWB band penalty + flex share penalty
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

@st.cache_data(show_spinner=False)
def cached_simulate(params_dict: Dict[str, object], policy_dict: Dict[str, float]) -> Dict[str, object]:
    params = ModelParams(**params_dict)
    policy = Policy(**policy_dict)
    return simulate_policy(params, policy)

# ============================================================
# SIDEBAR (returns params, policy, target utilization, run flag)
# ============================================================
def build_sidebar() -> Tuple[ModelParams, Policy, int, float, float, bool, Dict[str, float]]:
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

        visits = st.number_input("**Average Visits/Day**", 1.0, value=36.0, step=1.0, help="Baseline daily patient volume")
        annual_growth = st.number_input("**Annual Growth %**", 0.0, value=10.0, step=1.0, help="Expected year-over-year growth") / 100.0

        c1, c2 = st.columns(2)
        with c1:
            seasonality_pct = st.number_input("Seasonality %", 0.0, value=20.0, step=5.0) / 100.0
        with c2:
            peak_factor = st.number_input("Peak Factor", 1.0, value=1.2, step=0.1)

        annual_turnover = st.number_input("**Turnover %**", 0.0, value=16.0, step=1.0, help="Annual provider turnover rate") / 100.0

        st.markdown("**Lead Times**")
        c1, c2 = st.columns(2)
        with c1:
            turnover_notice_days = st.number_input("Turnover Notice", 0, value=90, step=10, help="Days from resignation to departure")
        with c2:
            hiring_runway_days = st.number_input("Hiring Runway", 0, value=210, step=10, help="Total days: req posted → productive provider")

        with st.expander("ℹ️ **Hiring Timeline Breakdown**", expanded=False):
            st.markdown(
                """
**Hiring Runway** = Time from req posted until provider is productive

Example breakdown for 210 days:
- **Recruitment:** 90 days (post req → signed offer)
- **Credentialing:** 90 days (offer → first day)
- **Onboarding:** 30 days (first day → independent)
- **Total:** 210 days (~7 months)
"""
            )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        with st.expander("⚙️ **Advanced Settings**", expanded=False):
            st.markdown("**Clinic Operations**")
            hours_week = st.number_input("Clinic Hours/Week", 1.0, value=84.0, step=1.0)
            days_open_per_week = st.number_input("Days Open/Week", 1.0, 7.0, value=7.0, step=1.0)
            fte_hours_week = st.number_input("FTE Hours/Week", 1.0, value=36.0, step=1.0)
            visits_per_provider_hour = st.slider("Visits/Provider-Hour", 2.0, 4.0, 3.0, 0.1)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Workforce**")
            ramp_months = st.slider("Ramp-up Months", 0, 6, 1)
            ramp_productivity = st.slider("Ramp Productivity %", 30, 100, 75, 5) / 100.0
            fill_probability = st.slider("Fill Probability %", 0, 100, 85, 5) / 100.0

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Risk Thresholds (PPPD)**")
            budgeted_pppd = st.number_input("Green Threshold", 5.0, value=36.0, step=1.0)
            yellow_max_pppd = st.number_input("Yellow Threshold", 5.0, value=42.0, step=1.0)
            red_start_pppd = st.number_input("Red Threshold", 5.0, value=45.0, step=1.0)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
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
                help="If selected, the model will not post new reqs during these months.",
            )
            freeze_months_set = {m for _, m in freeze_months} if freeze_months else set()

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Policy Constraints**")
            min_perm_providers_per_day = st.number_input("Min Providers/Day", 0.0, value=1.0, step=0.25)
            allow_prn_override = st.checkbox("Allow Base < Minimum", value=False)
            require_perm_under_green_no_flex = st.checkbox("Require Perm ≤ Green", value=True)
            flex_max_fte_per_month = st.slider("Max Flex FTE/Month", 0.0, 10.0, 2.0, 0.25)
            flex_cost_multiplier = st.slider("Flex Cost Multiplier", 1.0, 2.0, 1.25, 0.05)

        with st.expander("💰 **Financial Parameters**", expanded=False):
            st.markdown("**Targets & Constraints**")
            target_swb_per_visit = st.number_input("Target SWB/Visit ($)", 0.0, value=85.0, step=1.0)
            swb_tolerance = st.number_input("SWB Tolerance ($)", 0.0, value=2.0, step=0.5)
            net_revenue_per_visit = st.number_input("Net Contribution/Visit ($)", 0.0, value=140.0, step=5.0)
            visits_lost_per_provider_day_gap = st.number_input("Visits Lost/Provider-Day Gap", 0.0, value=18.0, step=1.0)
            provider_replacement_cost = st.number_input("Replacement Cost ($)", 0.0, value=75000.0, step=5000.0)
            turnover_yellow_mult = st.slider("Turnover Mult (Yellow)", 1.0, 3.0, 1.3, 0.05)
            turnover_red_mult = st.slider("Turnover Mult (Red)", 1.0, 5.0, 2.0, 0.1)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Compensation**")
            benefits_load_pct = st.number_input("Benefits Load %", 0.0, value=30.0, step=1.0) / 100.0
            bonus_pct = st.number_input("Bonus %", 0.0, value=10.0, step=1.0) / 100.0
            ot_sick_pct = st.number_input("OT+Sick %", 0.0, value=4.0, step=0.5) / 100.0

            st.markdown("**Hourly Rates**")
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

        # === RISK POSTURE (Lean ↔ Safe) ===
        st.markdown(f"<h3 style='color: {GOLD}; font-size: 1.1rem; margin-bottom: 1rem;'>🏛 Staffing Risk Posture</h3>", unsafe_allow_html=True)
        
        POSTURE_LABEL = {
            1: "Very Lean",
            2: "Lean",
            3: "Balanced",
            4: "Safe",
            5: "Very Safe",
        }
        
        risk_posture = st.slider(
            "**Lean ↔ Safe**",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            format_func=lambda x: POSTURE_LABEL.get(x, str(x)),
            help="Controls how much permanent staffing buffer you carry vs relying on flex and absorbing volatility."
        )
        
        POSTURE_TEXT = {
            1: "Very Lean: Minimum permanent staff. Higher utilization, more volatility, more flex/visit-loss risk.",
            2: "Lean: Cost-efficient posture. Limited buffer. Requires strong flex/PRN execution.",
            3: "Balanced: Standard posture. Reasonable buffer for peaks and normal absenteeism.",
            4: "Safe: Proactive staffing. Protects access and quality. Higher SWB/visit.",
            5: "Very Safe: Maximum stability. Highest cost. Best for high-acuity/high-reliability expectations.",
        }
        st.caption(POSTURE_TEXT[risk_posture])
        
        # How posture changes levers (tight + explainable)
        POSTURE_BASE_COVERAGE_MULT = {1: 0.92, 2: 0.96, 3: 1.00, 4: 1.04, 5: 1.08}
        POSTURE_WINTER_BUFFER_ADD  = {1: 0.00, 2: 0.02, 3: 0.04, 4: 0.06, 5: 0.08}
        POSTURE_FLEX_CAP_MULT      = {1: 1.35, 2: 1.15, 3: 1.00, 4: 0.90, 5: 0.80}

        
        st.markdown(f"<h3 style='color: {GOLD}; font-size: 1.1rem; margin-bottom: 1rem;'>🎯 Smart Staffing Policy</h3>", unsafe_allow_html=True)
        st.markdown(
            """
**Manual Control with Smart Suggestions:** Set your target utilization, or let the model suggest
the utilization that best matches your SWB/visit target.
"""
        )

        if "target_utilization" not in st.session_state:
            st.session_state.target_utilization = 92

        c1, c2 = st.columns([2, 1])
        with c1:
            target_utilization = st.slider(
                "**Target Utilization %**",
                80,
                98,
                value=int(st.session_state.target_utilization),
                step=2,
                help="Higher utilization = lower cost but less buffer. 90–95% is common for most clinics.",
            )

        # Winter buffer is needed by the optimizer, so define it BEFORE the button logic
        winter_buffer_pct = st.slider(
            "**Winter Buffer %**",
            0,
            10,
            3,
            1,
            help="Additional buffer for winter demand uncertainty (typically 3–5%)",
        ) / 100.0

        with c2:
            st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
            if st.button("🎯 Suggest Optimal", help="Find utilization that best matches your SWB/visit target", use_container_width=True):
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
                        test_winter = test_coverage * (1 + winter_buffer_pct)

                        test_policy = Policy(base_coverage_pct=test_coverage, winter_coverage_pct=test_winter)
                        test_result = simulate_policy(params_temp, test_policy)
                        test_swb = float(test_result["annual_swb_per_visit"])

                        results_cache[test_util] = test_swb
                        diff = abs(test_swb - target_swb_per_visit)
                        if diff < best_diff:
                            best_util, best_diff = test_util, diff

                    st.session_state.target_utilization = best_util

                    st.success(
                        f"""
✅ **Best Match:** {best_util}% utilization  
- **Estimated SWB/visit:** ${results_cache[best_util]:.2f}  
- **Target:** ${target_swb_per_visit:.0f} ± ${swb_tolerance:.0f}  
- **Difference:** ${abs(results_cache[best_util] - target_swb_per_visit):.2f}

Slider updated to **{best_util}%**. Click **Run Simulation** to apply.
"""
                    )

                    with st.expander("📊 Optimization Details"):
                        for util_test in sorted(results_cache.keys()):
                            icon = "✅" if util_test == best_util else "○"
                            st.text(f"{icon} {util_test}% → ${results_cache[util_test]:.2f} SWB/visit")

                st.rerun()

        # Calculate coverage from utilization (baseline)
        base_coverage_from_util = 1.0 / (target_utilization / 100.0)
        
        # Winter buffer control (user-facing)
        winter_buffer_pct = st.slider(
            "**Winter Buffer %**",
            0, 10, 3, 1,
            help="Additional buffer for winter demand uncertainty (typically 3-5%)"
        ) / 100.0
        
        # Apply posture adjustments (Balanced = 1.00 / +0.04 default)
        posture_mult = POSTURE_BASE_COVERAGE_MULT[risk_posture]
        posture_winter_add = POSTURE_WINTER_BUFFER_ADD[risk_posture]
        
        # Final coverages (policy)
        base_coverage_pct = base_coverage_from_util * posture_mult
        winter_coverage_pct = base_coverage_pct * (1 + winter_buffer_pct + posture_winter_add)
        
        # Flex cap posture scaling (Lean allows more flex, Safe expects perm)
        flex_max_fte_effective = flex_max_fte_per_month * POSTURE_FLEX_CAP_MULT[risk_posture]


        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
<div style='background: white; padding: 1rem; border-radius: 8px; border: 2px solid {LIGHT_GOLD};'>
  <div style='font-size: 0.75rem; color: {DARK_GOLD}; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.1em; margin-bottom: 0.5rem;'>
    Base Period Policy
  </div>
  <div style='font-size: 2rem; font-weight: 700; color: {GOLD}; font-family: "Cormorant Garamond", serif;'>
    {base_coverage_pct*100:.0f}%
  </div>
  <div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>
    Coverage ({target_utilization}% util target)
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
<div style='background: white; padding: 1rem; border-radius: 8px; border: 2px solid {GOLD};'>
  <div style='font-size: 0.75rem; color: {DARK_GOLD}; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.1em; margin-bottom: 0.5rem;'>
    Winter Period Policy
  </div>
  <div style='font-size: 2rem; font-weight: 700; color: {GOLD}; font-family: "Cormorant Garamond", serif;'>
    {winter_coverage_pct*100:.0f}%
  </div>
  <div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>
    Base + {winter_buffer_pct*100:.0f}% buffer
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        if target_utilization >= 98:
            st.warning(
                """
⚠️ **98%+ Utilization:** Minimal buffer. You will likely need fractional FTE, per diem, or flex coverage
to manage spikes, absences, and variability.
"""
            )
        elif target_utilization <= 85:
            st.info(
                """
ℹ️ **≤85% Utilization:** Strong buffer, higher cost. SWB/visit may exceed target.
Use **Suggest Optimal** to find a better cost match.
"""
            )
        else:
            st.success(
                f"""
✅ **{target_utilization}% Utilization:** Balanced efficiency and buffer.
Run the simulation to compare against your **${target_swb_per_visit:.0f} ± ${swb_tolerance:.0f}** SWB/visit target.
"""
            )

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

        policy = Policy(base_coverage_pct=base_coverage_pct, winter_coverage_pct=winter_coverage_pct)

        return params, policy, int(target_utilization), float(winter_buffer_pct), float(base_coverage_pct), bool(run_simulation), hourly_rates

params, policy, target_utilization, winter_buffer_pct, base_coverage_pct, run_simulation, hourly_rates = build_sidebar()

# ============================================================
# RUN / LOAD RESULTS
# ============================================================
params_dict = {**params.__dict__}
policy_dict = {"base_coverage_pct": policy.base_coverage_pct, "winter_coverage_pct": policy.winter_coverage_pct}

if run_simulation:
    with st.spinner("🔍 Running simulation..."):
        R = cached_simulate(params_dict, policy_dict)
    st.session_state["simulation_result"] = R
    st.success("✅ Simulation complete!")
else:
    if "simulation_result" not in st.session_state:
        R = cached_simulate(params_dict, policy_dict)
        st.session_state["simulation_result"] = R
    R = st.session_state["simulation_result"]

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# POLICY HEALTH CHECK (Ratchet / drift)
# ============================================================
st.markdown("## 🔍 Policy Health Check")

annual = R["annual_summary"]
if len(annual) >= 2:
    min_y1 = float(annual.loc[0, "Min_Perm_Paid_FTE"])
    min_y2 = float(annual.loc[1, "Min_Perm_Paid_FTE"])
    min_y3 = float(annual.loc[2, "Min_Perm_Paid_FTE"]) if len(annual) >= 3 else min_y2

    drift_y2 = min_y2 - min_y1
    drift_y3 = min_y3 - min_y2

    if abs(drift_y2) < 0.2 and abs(drift_y3) < 0.2:
        st.markdown(
            f"""
<div class="status-card status-success">
  <div class="status-content">
    <div class="status-icon">✅</div>
    <div class="status-text">
      <div class="status-title">No Ratchet Detected</div>
      <div class="status-message">
        Base FTE is stable across all 3 years:<br>
        <strong>Year 1:</strong> {min_y1:.2f} FTE →
        <strong>Year 2:</strong> {min_y2:.2f} FTE (Δ{drift_y2:+.2f}) →
        <strong>Year 3:</strong> {min_y3:.2f} FTE (Δ{drift_y3:+.2f})<br>
        Policy: {policy.base_coverage_pct*100:.0f}% base coverage, {policy.winter_coverage_pct*100:.0f}% winter coverage
      </div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
<div class="status-card status-warning">
  <div class="status-content">
    <div class="status-icon">⚠️</div>
    <div class="status-text">
      <div class="status-title">Minor Drift Detected</div>
      <div class="status-message">
        <strong>Year 1:</strong> {min_y1:.2f} →
        <strong>Year 2:</strong> {min_y2:.2f} (Δ{drift_y2:+.2f}) →
        <strong>Year 3:</strong> {min_y3:.2f} (Δ{drift_y3:+.2f})<br>
        Expected: ±0.2 FTE/year. Consider adjusting turnover, fill probability, or hiring runway.
      </div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# HERO SCORECARD
# ============================================================
swb_y1 = float(annual.loc[0, "SWB_per_Visit"])
ebitda_y1 = float(annual.loc[0, "EBITDA_Proxy"])
ebitda_y3 = float(annual.loc[len(annual) - 1, "EBITDA_Proxy"])
util_y1 = float(annual.loc[0, "Avg_Utilization"])
util_y3 = float(annual.loc[len(annual) - 1, "Avg_Utilization"])
min_y1 = float(annual.loc[0, "Min_Perm_Paid_FTE"])
max_y1 = float(annual.loc[0, "Max_Perm_Paid_FTE"])

st.markdown(
    f"""
<div class="scorecard-hero">
  <div class="scorecard-title">Policy Performance Scorecard</div>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Staffing Policy</div>
      <div class="metric-value">{target_utilization:.0f}% Target</div>
      <div class="metric-detail">Coverage: {policy.base_coverage_pct*100:.0f}% base / {policy.winter_coverage_pct*100:.0f}% winter</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">SWB per Visit (Y1)</div>
      <div class="metric-value">${swb_y1:.2f}</div>
      <div class="metric-detail">Target: ${params.target_swb_per_visit:.0f} ± ${params.swb_tolerance:.0f}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">EBITDA Proxy</div>
      <div class="metric-value">${ebitda_y1/1000:.0f}K</div>
      <div class="metric-detail">Year 1 / Year {len(annual)}: ${ebitda_y3/1000:.0f}K</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Utilization</div>
      <div class="metric-value">{util_y1*100:.0f}%</div>
      <div class="metric-detail">Year 1 / Year {len(annual)}: {util_y3*100:.0f}%</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">FTE Range (Y1)</div>
      <div class="metric-value">{min_y1:.1f}-{max_y1:.1f}</div>
      <div class="metric-detail">Min–Max across months</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Peak Load</div>
      <div class="metric-value">{float(R['peak_load_post']):.1f}</div>
      <div class="metric-detail">PPPD (post-flex)</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# SMART HIRING INSIGHTS
# ============================================================
st.markdown("## 🧠 Smart Hiring Insights")

ledger = R["ledger"]
upcoming_hires = ledger[ledger["Hires Visible (FTE)"] > 0.05].head(12)

if len(upcoming_hires) > 0:
    st.markdown(
        f"""
<div style='background: linear-gradient(135deg, {CREAM} 0%, white 100%);
            padding: 1.5rem; border-radius: 12px; border-left: 4px solid {GOLD};
            margin-bottom: 1.5rem;'>
  <div style='font-weight: 600; font-size: 1.1rem; color: {BLACK}; margin-bottom: 1rem;'>
    📋 Next 12 Months Hiring Plan
  </div>
  <div style='font-size: 0.9rem; color: #444; line-height: 1.6;'>
    The model identified <b>{len(upcoming_hires)} hiring events</b> in the next year.
    These are timed to meet peak demand with required lead time.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Upcoming Hiring Events:**")
        for _, row in upcoming_hires.head(5).iterrows():
            month = row["Month"]
            hires = float(row["Hires Visible (FTE)"])
            reason = str(row["Hire Reason"] or "")
            st.markdown(
                f"""
<div style='background: white; padding: 0.75rem; margin: 0.5rem 0;
            border-radius: 8px; border-left: 3px solid {GOLD};'>
  <div style='font-weight: 600; color: {BLACK};'>{month}: +{hires:.2f} FTE</div>
  <div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>{reason[:160]}{"..." if len(reason) > 160 else ""}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    with c2:
        total_hires_12mo = float(upcoming_hires["Hires Visible (FTE)"].sum())
        avg_fte = float(ledger.head(12)["Permanent FTE (Paid)"].mean())

        st.markdown(
            f"""
<div style='background: white; padding: 1.25rem; border-radius: 12px;
            border: 2px solid {LIGHT_GOLD}; text-align: center;'>
  <div style='font-size: 0.75rem; color: {DARK_GOLD}; font-weight: 600;
              text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;'>
    12-Month Hiring Volume
  </div>
  <div style='font-size: 2.5rem; font-weight: 700; color: {GOLD};
              font-family: "Cormorant Garamond", serif; line-height: 1;'>
    {total_hires_12mo:.1f}
  </div>
  <div style='font-size: 0.85rem; color: #666; margin-top: 0.5rem;'>
    FTE to hire ({(total_hires_12mo/max(avg_fte,1e-9))*100:.0f}% of avg staff)
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# CHARTS
# ============================================================
st.markdown("## 📊 3-Year Financial Projection")

dates = R["dates"]
perm_paid = np.array(R["perm_paid"], dtype=float)
target_pol = np.array(R["target_policy"], dtype=float)
req_eff = np.array(R["req_eff_fte_needed"], dtype=float)
util = np.array(R["utilization"], dtype=float)
load_post = np.array(R["load_post"], dtype=float)

# Supply vs Target
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=dates,
        y=perm_paid,
        mode="lines+markers",
        name="Paid FTE",
        line=dict(color=GOLD, width=3),
        marker=dict(size=5, color=GOLD),
        hovertemplate="<b>%{x|%Y-%b}</b><br>Paid FTE: %{y:.2f}<extra></extra>",
    )
)
fig1.add_trace(
    go.Scatter(
        x=dates,
        y=target_pol,
        mode="lines+markers",
        name="Target (policy)",
        line=dict(color=BLACK, width=2, dash="dash"),
        marker=dict(size=5, symbol="square", color=BLACK),
        hovertemplate="<b>%{x|%Y-%b}</b><br>Target: %{y:.2f}<extra></extra>",
    )
)
fig1.add_trace(
    go.Scatter(
        x=dates,
        y=req_eff,
        mode="lines",
        name="Required FTE",
        line=dict(color=LIGHT_GOLD, width=2, dash="dot"),
        hovertemplate="<b>%{x|%Y-%b}</b><br>Required: %{y:.2f}<extra></extra>",
    )
)
fig1.update_layout(
    title={
        "text": "<b>Supply vs Target FTE</b><br><sup>Base should stay stable year-over-year</sup>",
        "font": {"size": 20, "family": "Cormorant Garamond, serif", "color": BLACK},
        "x": 0.5,
        "xanchor": "center",
    },
    xaxis={"title": "", "showgrid": True, "gridcolor": "rgba(0,0,0,0.05)"},
    yaxis={"title": "Provider FTE", "showgrid": True, "gridcolor": "rgba(0,0,0,0.05)"},
    hovermode="x unified",
    plot_bgcolor="rgba(250, 248, 243, 0.3)",
    paper_bgcolor="white",
    height=450,
    font={"family": "IBM Plex Sans, sans-serif", "size": 12},
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "bgcolor": "rgba(255,255,255,0.8)",
        "bordercolor": LIGHT_GOLD,
        "borderwidth": 1,
    },
)
st.plotly_chart(fig1, use_container_width=True)

# Utilization & Load
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(
    go.Scatter(
        x=dates,
        y=util * 100,
        mode="lines+markers",
        name="Utilization %",
        line=dict(color="#2ecc71", width=3),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%Y-%b}</b><br>Utilization: %{y:.1f}%<extra></extra>",
    ),
    secondary_y=False,
)
fig2.add_trace(
    go.Scatter(
        x=dates,
        y=load_post,
        mode="lines+markers",
        name="Load PPPD",
        line=dict(color=GOLD, width=3),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%Y-%b}</b><br>Load: %{y:.1f} PPPD<extra></extra>",
    ),
    secondary_y=True,
)

fig2.add_hline(
    y=target_utilization,
    line_dash="dot",
    line_color="green",
    secondary_y=False,
    annotation_text=f"{target_utilization:.0f}% Target",
    annotation_position="right",
)
fig2.add_hline(
    y=params.budgeted_pppd,
    line_dash="dot",
    line_color="green",
    secondary_y=True,
    annotation_text=f"Green ({params.budgeted_pppd:.0f})",
    annotation_position="right",
)
fig2.add_hline(
    y=params.red_start_pppd,
    line_dash="dot",
    line_color="red",
    secondary_y=True,
    annotation_text=f"Red ({params.red_start_pppd:.0f})",
    annotation_position="right",
)

fig2.update_layout(
    title={
        "text": "<b>Utilization & Provider Load</b><br><sup>Keep utilization near target and load under green threshold</sup>",
        "font": {"size": 20, "family": "Cormorant Garamond, serif", "color": BLACK},
        "x": 0.5,
        "xanchor": "center",
    },
    xaxis={"title": "", "showgrid": True, "gridcolor": "rgba(0,0,0,0.05)"},
    hovermode="x unified",
    plot_bgcolor="rgba(250, 248, 243, 0.3)",
    paper_bgcolor="white",
    height=450,
    font={"family": "IBM Plex Sans, sans-serif", "size": 12},
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "bgcolor": "rgba(255,255,255,0.8)",
        "bordercolor": LIGHT_GOLD,
        "borderwidth": 1,
    },
)
fig2.update_yaxes(title_text="<b>Utilization (%)</b>", secondary_y=False)
fig2.update_yaxes(title_text="<b>Load (PPPD)</b>", secondary_y=True)
st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# TABLES
# ============================================================
st.markdown("## 📈 Detailed Results")

tab1, tab2 = st.tabs(["📊 Annual Summary", "📋 Monthly Ledger"])

with tab1:
    st.markdown("### Annual Performance by Year")
    st.dataframe(
        annual.style.format(
            {
                "Visits": "{:,.0f}",
                "SWB_per_Visit": "${:,.2f}",
                "SWB_Dollars": "${:,.0f}",
                "EBITDA_Proxy": "${:,.0f}",
                "Min_Perm_Paid_FTE": "{:.2f}",
                "Max_Perm_Paid_FTE": "{:.2f}",
                "Avg_Utilization": "{:.1%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with tab2:
    st.markdown("### Month-by-Month Audit Trail")
    st.dataframe(
        ledger.style.format(
            {
                "Visits/Day (avg)": "{:.1f}",
                "Total Visits (month)": "{:,.0f}",
                "SWB/Visit (month)": "${:.2f}",
                "SWB Dollars (month)": "${:,.0f}",
                "EBITDA Proxy (month)": "${:,.0f}",
                "Permanent FTE (Paid)": "{:.2f}",
                "Target FTE (policy)": "{:.2f}",
                "Utilization (Req/Supplied)": "{:.1%}",
                "Load PPPD (post-flex)": "{:.1f}",
                "Hires Visible (FTE)": "{:.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
        height=400,
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# EXPORTS
# ============================================================
st.markdown("## 💾 Export Results")

def fig_to_bytes(fig: go.Figure) -> bytes:
    return fig.to_image(format="png", engine="kaleido")

c1, c2, c3 = st.columns(3)

with c1:
    try:
        png1 = fig_to_bytes(fig1)
        st.download_button("⬇️ Supply Chart (PNG)", png1, "supply_vs_target.png", "image/png", use_container_width=True)
    except Exception:
        st.info("Install kaleido for image export: `pip install kaleido`")

with c2:
    try:
        png2 = fig_to_bytes(fig2)
        st.download_button("⬇️ Utilization Chart (PNG)", png2, "utilization_load.png", "image/png", use_container_width=True)
    except Exception:
        pass

with c3:
    csv_data = ledger.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Ledger (CSV)", csv_data, "staffing_ledger.csv", "text/csv", use_container_width=True)

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
