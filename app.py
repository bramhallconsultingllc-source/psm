# Bramhall Co. — Predictive Staffing Model (PSM)
# Elegant, branded interface with sophisticated design
# "predict. perform. prosper."

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any

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
# Keep your existing base64 string here exactly as-is.
LOGO_B64 = """PASTE_YOUR_EXISTING_LOGO_B64_HERE"""

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

    # Demand
    y0 = params.visits
    y1 = y0 * (1 + params.annual_growth)
    y2 = y1 * (1 + params.annual_growth)

    v_base = compute_visits_curve(months, y0, y1, y2, params.seasonality_pct)
    v_flu = apply_flu_uplift(v_base, months, params.flu_months, params.flu_uplift_pct)
    v_peak = [v * params.peak_factor for v in v_flu]

    role_mix = compute_role_mix_ratios(y1, model)

    # Required effective provider FTE
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

        # Posting freeze respected here
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

    # Flex coverage
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

    # Monthly ledger
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
                "Hires Visible (FTE)": float(float(hires_arr[i])),
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
# CACHE-SAFE PARAM SERIALIZATION
# ============================================================
def params_to_cache_dict(p: ModelParams) -> Dict[str, Any]:
    d = dict(p.__dict__)
    d["flu_months"] = sorted(list(p.flu_months))
    d["freeze_months"] = sorted(list(p.freeze_months))
    return d


def policy_to_cache_dict(pol: Policy) -> Dict[str, float]:
    return {"base_coverage_pct": float(pol.base_coverage_pct), "winter_coverage_pct": float(pol.winter_coverage_pct)}


@st.cache_data(show_spinner=False)
def cached_simulate(params_dict: Dict[str, Any], policy_dict: Dict[str, float]) -> Dict[str, object]:
    # reconstruct sets
    params_dict = dict(params_dict)
    params_dict["flu_months"] = set(params_dict.get("flu_months", []))
    params_dict["freeze_months"] = set(params_dict.get("freeze_months", []))
    params = ModelParams(**params_dict)
    policy = Policy(**policy_dict)
    return simulate_policy(params, policy)


# ============================================================
# SIDEBAR
# ============================================================
POSTURE_LABEL = {1: "Very Lean", 2: "Lean", 3: "Balanced", 4: "Safe", 5: "Very Safe"}
POSTURE_TEXT = {
    1: "Very Lean: Minimum permanent staff. Higher utilization, more volatility, more flex/visit-loss risk.",
    2: "Lean: Cost-efficient posture. Limited buffer. Requires strong flex/PRN execution.",
    3: "Balanced: Standard posture. Reasonable buffer for peaks and normal absenteeism.",
    4: "Safe: Proactive staffing. Protects access and quality. Higher SWB/visit.",
    5: "Very Safe: Maximum stability. Highest cost. Best for high-acuity/high-reliability expectations.",
}

# How posture changes levers (tight + explainable)
POSTURE_BASE_COVERAGE_MULT = {1: 0.92, 2: 0.96, 3: 1.00, 4: 1.04, 5: 1.08}
POSTURE_WINTER_BUFFER_ADD = {1: 0.00, 2: 0.02, 3: 0.04, 4: 0.06, 5: 0.08}
POSTURE_FLEX_CAP_MULT = {1: 1.35, 2: 1.15, 3: 1.00, 4: 0.90, 5: 0.80}


def build_sidebar() -> Tuple[ModelParams, Policy, Dict[str, Any]]:
    """
    Returns:
      params, policy, ui (dict with useful intermediate values for later sections)
    """
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

        # Defaults that Advanced Settings will set
        hours_week = 84.0
        days_open_per_week = 7.0
        fte_hours_week = 36.0
        visits_per_provider_hour = 3.0

        ramp_months = 1
        ramp_productivity = 0.75
        fill_probability = 0.85

        budgeted_pppd = 36.0
        yellow_max_pppd = 42.0
        red_start_pppd = 45.0

        flu_uplift_pct = 0.0
        flu_months_set: Set[int] = {10, 11, 12, 1, 2}

        winter_anchor_month_num = 12
        winter_end_month_num = 2
        freeze_months_set: Set[int] = set()

        min_perm_providers_per_day = 1.0
        allow_prn_override = False
        require_perm_under_green_no_flex = True
        flex_max_fte_per_month = 2.0
        flex_cost_multiplier = 1.25

        with st.expander("⚙️ **Advanced Settings**", expanded=False):
            st.markdown("**Clinic Operations**")
            hours_week = st.number_input("Clinic Hours/Week", 1.0, value=float(hours_week), step=1.0)
            days_open_per_week = st.number_input("Days Open/Week", 1.0, 7.0, value=float(days_open_per_week), step=1.0)
            fte_hours_week = st.number_input("FTE Hours/Week", 1.0, value=float(fte_hours_week), step=1.0)
            visits_per_provider_hour = st.slider("Visits/Provider-Hour", 2.0, 4.0, float(visits_per_provider_hour), 0.1)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Workforce**")
            ramp_months = st.slider("Ramp-up Months", 0, 6, int(ramp_months))
            ramp_productivity = st.slider("Ramp Productivity %", 30, 100, int(ramp_productivity * 100), 5) / 100.0
            fill_probability = st.slider("Fill Probability %", 0, 100, int(fill_probability * 100), 5) / 100.0

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Risk Thresholds (PPPD)**")
            budgeted_pppd = st.number_input("Green Threshold", 5.0, value=float(budgeted_pppd), step=1.0)
            yellow_max_pppd = st.number_input("Yellow Threshold", 5.0, value=float(yellow_max_pppd), step=1.0)
            red_start_pppd = st.number_input("Red Threshold", 5.0, value=float(red_start_pppd), step=1.0)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Seasonal Configuration**")
            flu_uplift_pct = st.number_input("Flu Season Uplift %", 0.0, value=float(flu_uplift_pct), step=5.0) / 100.0
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
            min_perm_providers_per_day = st.number_input("Min Providers/Day", 0.0, value=float(min_perm_providers_per_day), step=0.25)
            allow_prn_override = st.checkbox("Allow Base < Minimum", value=bool(allow_prn_override))
            require_perm_under_green_no_flex = st.checkbox("Require Perm ≤ Green", value=bool(require_perm_under_green_no_flex))
            flex_max_fte_per_month = st.slider("Max Flex FTE/Month", 0.0, 10.0, float(flex_max_fte_per_month), 0.25)
            flex_cost_multiplier = st.slider("Flex Cost Multiplier", 1.0, 2.0, float(flex_cost_multiplier), 0.05)

        # Financial parameters defaults
        target_swb_per_visit = 85.0
        swb_tolerance = 2.0
        net_revenue_per_visit = 140.0
        visits_lost_per_provider_day_gap = 18.0
        provider_replacement_cost = 75000.0
        turnover_yellow_mult = 1.3
        turnover_red_mult = 2.0

        benefits_load_pct = 0.30
        bonus_pct = 0.10
        ot_sick_pct = 0.04

        physician_hr = 135.79
        apc_hr = 62.0
        ma_hr = 24.14
        psr_hr = 21.23
        rt_hr = 31.36
        supervisor_hr = 28.25
        physician_supervision_hours_per_month = 0.0
        supervisor_hours_per_month = 0.0

        with st.expander("💰 **Financial Parameters**", expanded=False):
            st.markdown("**Targets & Constraints**")
            target_swb_per_visit = st.number_input("Target SWB/Visit ($)", 0.0, value=float(target_swb_per_visit), step=1.0)
            swb_tolerance = st.number_input("SWB Tolerance ($)", 0.0, value=float(swb_tolerance), step=0.5)
            net_revenue_per_visit = st.number_input("Net Contribution/Visit ($)", 0.0, value=float(net_revenue_per_visit), step=5.0)
            visits_lost_per_provider_day_gap = st.number_input("Visits Lost/Provider-Day Gap", 0.0, value=float(visits_lost_per_provider_day_gap), step=1.0)
            provider_replacement_cost = st.number_input("Replacement Cost ($)", 0.0, value=float(provider_replacement_cost), step=5000.0)
            turnover_yellow_mult = st.slider("Turnover Mult (Yellow)", 1.0, 3.0, float(turnover_yellow_mult), 0.05)
            turnover_red_mult = st.slider("Turnover Mult (Red)", 1.0, 5.0, float(turnover_red_mult), 0.1)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Compensation**")
            benefits_load_pct = st.number_input("Benefits Load %", 0.0, value=float(benefits_load_pct * 100), step=1.0) / 100.0
            bonus_pct = st.number_input("Bonus %", 0.0, value=float(bonus_pct * 100), step=1.0) / 100.0
            ot_sick_pct = st.number_input("OT+Sick %", 0.0, value=float(ot_sick_pct * 100), step=0.5) / 100.0

            st.markdown("**Hourly Rates**")
            c1, c2 = st.columns(2)
            with c1:
                physician_hr = st.number_input("Physician ($/hr)", 0.0, value=float(physician_hr), step=1.0)
                apc_hr = st.number_input("APP ($/hr)", 0.0, value=float(apc_hr), step=1.0)
                ma_hr = st.number_input("MA ($/hr)", 0.0, value=float(ma_hr), step=0.5)
            with c2:
                psr_hr = st.number_input("PSR ($/hr)", 0.0, value=float(psr_hr), step=0.5)
                rt_hr = st.number_input("RT ($/hr)", 0.0, value=float(rt_hr), step=0.5)
                supervisor_hr = st.number_input("Supervisor ($/hr)", 0.0, value=float(supervisor_hr), step=0.5)

            physician_supervision_hours_per_month = st.number_input(
                "Physician Supervision (hrs/mo)", 0.0, value=float(physician_supervision_hours_per_month), step=1.0
            )
            supervisor_hours_per_month = st.number_input("Supervisor Hours (hrs/mo)", 0.0, value=float(supervisor_hours_per_month), step=1.0)

        st.markdown(f"<div style='height: 2px; background: {LIGHT_GOLD}; margin: 2rem 0;'></div>", unsafe_allow_html=True)

        # === RISK POSTURE (Lean ↔ Safe) ===
        st.markdown(f"<h3 style='color: {GOLD}; font-size: 1.1rem; margin-bottom: 1rem;'>🏛 Staffing Risk Posture</h3>", unsafe_allow_html=True)

        # FIX: st.slider does NOT support format_func (that was your crash).
        # Use select_slider instead (supports format_func).
        risk_posture = st.select_slider(
            "**Lean ↔ Safe**",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: POSTURE_LABEL.get(int(x), str(x)),
            help="Controls how much permanent staffing buffer you carry vs relying on flex and absorbing volatility.",
        )
        st.caption(POSTURE_TEXT[int(risk_posture)])

        st.markdown(f"<h3 style='color: {GOLD}; font-size: 1.1rem; margin-bottom: 1rem;'>🎯 Smart Staffing Policy</h3>", unsafe_allow_html=True)
        st.markdown(
            """
**Manual Control with Smart Suggestions:** Set your target utilization, or let the model suggest
the utilization that best matches your SWB/visit target.
"""
        )

        if "target_utilization" not in st.session_state:
            st.session_state.target_utilization = 92

        target_utilization = st.slider(
            "**Target Utilization %**",
            80,
            98,
            value=int(st.session_state.target_utilization),
            step=2,
            help="Higher utilization = lower cost but less buffer. 90–95% is common for most clinics.",
        )

        # Winter buffer control (single definition; no duplicates)
        winter_buffer_pct = st.slider(
            "**Winter Buffer %**",
            0,
            10,
            3,
            1,
            help="Additional buffer for winter demand uncertainty (typically 3–5%)",
        ) / 100.0

        # Calculate coverage from utilization (baseline)
        base_coverage_from_util = 1.0 / (target_utilization / 100.0)

        # Apply posture adjustments
        posture_mult = POSTURE_BASE_COVERAGE_MULT[int(risk_posture)]
        posture_winter_add = POSTURE_WINTER_BUFFER_ADD[int(risk_posture)]

        base_coverage_pct = base_coverage_from_util * posture_mult
        winter_coverage_pct = base_coverage_pct * (1 + winter_buffer_pct + posture_winter_add)

        # Flex cap posture scaling (Lean allows more flex, Safe expects perm)
        flex_max_fte_effective = float(flex_max_fte_per_month * POSTURE_FLEX_CAP_MULT[int(risk_posture)])

        # Suggest optimal (uses defined vars; no forward refs)
        c1, c2 = st.columns([2, 1])
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

                    # Use CURRENT posture-adjusted flex cap during optimization
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
                        test_cov_from_util = 1.0 / (test_util / 100.0)
                        test_base = test_cov_from_util * posture_mult
                        test_winter = test_base * (1 + winter_buffer_pct + posture_winter_add)

                        test_policy = Policy(base_coverage_pct=float(test_base), winter_coverage_pct=float(test_winter))
                        test_result = simulate_policy(params_temp, test_policy)
                        test_swb = float(test_result["annual_swb_per_visit"])

                        results_cache[test_util] = test_swb
                        diff = abs(test_swb - target_swb_per_visit)
                        if diff < best_diff:
                            best_util, best_diff = test_util, diff

                    st.session_state.target_utilization = int(best_util)

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

        # Coverage cards
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
    Base + {(winter_buffer_pct + posture_winter_add)*100:.0f}% buffer
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

        policy = Policy(base_coverage_pct=float(base_coverage_pct), winter_coverage_pct=float(winter_coverage_pct))

        ui = {
            "run_simulation": bool(run_simulation),
            "target_utilization": int(target_utilization),
            "winter_buffer_pct": float(winter_buffer_pct),
            "risk_posture": int(risk_posture),
            "base_coverage_from_util": float(base_coverage_from_util),
            "flex_max_fte_base": float(flex_max_fte_per_month),
        }
        return params, policy, ui


params, policy, ui = build_sidebar()

# ============================================================
# RUN / LOAD RESULTS
# ============================================================
params_dict = params_to_cache_dict(params)
policy_dict = policy_to_cache_dict(policy)

if ui["run_simulation"]:
    with st.spinner("🔍 Running simulation..."):
        R = cached_simulate(params_dict, policy_dict)
    st.session_state["simulation_result"] = R
    st.success("✅ Simulation complete!")
else:
    if "simulation_result" not in st.session_state:
        R = cached_simulate(params_dict, policy_dict)
        st.session_state["simulation_result"] = R
    R = st.session_state["simulation_result"]

# ============================================================
# EXEC VIEW: Lean vs Safe Tradeoffs (Exec View)
# ============================================================
st.markdown("## ⚖️ Lean vs Safe Tradeoffs (Exec View)")

def build_policy_for_posture(posture_level: int) -> Tuple[ModelParams, Policy]:
    posture_mult = POSTURE_BASE_COVERAGE_MULT[int(posture_level)]
    posture_winter_add = POSTURE_WINTER_BUFFER_ADD[int(posture_level)]
    posture_flex_mult = POSTURE_FLEX_CAP_MULT[int(posture_level)]

    base_cov = ui["base_coverage_from_util"] * posture_mult
    winter_cov = base_cov * (1 + ui["winter_buffer_pct"] + posture_winter_add)

    params_alt = ModelParams(
        **{
            **params.__dict__,
            "flex_max_fte_per_month": float(ui["flex_max_fte_base"] * posture_flex_mult),
        }
    )
    pol_alt = Policy(base_coverage_pct=float(base_cov), winter_coverage_pct=float(winter_cov))
    return params_alt, pol_alt

lean_params, lean_policy = build_policy_for_posture(2)
safe_params, safe_policy = build_policy_for_posture(4)

with st.spinner("Comparing Lean vs Safe..."):
    R_lean = simulate_policy(lean_params, lean_policy)
    R_safe = simulate_policy(safe_params, safe_policy)

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

m_cur = pack_exec_metrics(R)
m_lean = pack_exec_metrics(R_lean)
m_safe = pack_exec_metrics(R_safe)

df_exec = pd.DataFrame(
    [
        {"Scenario": "Lean", **m_lean},
        {"Scenario": "Current", **m_cur},
        {"Scenario": "Safe", **m_safe},
    ]
)

def fmt_money(x): return f"${x:,.0f}"
def fmt_money2(x): return f"${x:,.2f}"
def fmt_pct(x): return f"{x*100:.1f}%"
def fmt_num1(x): return f"{x:.1f}"

st.dataframe(
    df_exec.style.format(
        {
            "SWB/Visit (Y1)": fmt_money2,
            "EBITDA Proxy (Y1)": fmt_money,
            "EBITDA Proxy (3yr total)": fmt_money,
            "Flex Share": fmt_pct,
            "Peak Load (PPPD)": fmt_num1,
        }
    ),
    hide_index=True,
    use_container_width=True,
)

def delta(a, b):  # b - a
    return b - a

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Lean → Current (What you save / what you accept)")
    st.write(f"- **SWB/Visit (Y1):** {fmt_money2(m_lean['SWB/Visit (Y1)'])} → {fmt_money2(m_cur['SWB/Visit (Y1)'])} (Δ {fmt_money2(delta(m_lean['SWB/Visit (Y1)'], m_cur['SWB/Visit (Y1)']))})")
    st.write(f"- **Red Months:** {m_lean['Red Months']} → {m_cur['Red Months']} (Δ {delta(m_lean['Red Months'], m_cur['Red Months']):+d})")
    st.write(f"- **Flex Share:** {fmt_pct(m_lean['Flex Share'])} → {fmt_pct(m_cur['Flex Share'])} (Δ {fmt_pct(delta(m_lean['Flex Share'], m_cur['Flex Share']))})")
    st.write(f"- **3yr EBITDA Proxy:** {fmt_money(m_lean['EBITDA Proxy (3yr total)'])} → {fmt_money(m_cur['EBITDA Proxy (3yr total)'])} (Δ {fmt_money(delta(m_lean['EBITDA Proxy (3yr total)'], m_cur['EBITDA Proxy (3yr total)']))})")

with col2:
    st.markdown("### Current → Safe (What you buy / what it costs)")
    st.write(f"- **SWB/Visit (Y1):** {fmt_money2(m_cur['SWB/Visit (Y1)'])} → {fmt_money2(m_safe['SWB/Visit (Y1)'])} (Δ {fmt_money2(delta(m_cur['SWB/Visit (Y1)'], m_safe['SWB/Visit (Y1)']))})")
    st.write(f"- **Red Months:** {m_cur['Red Months']} → {m_safe['Red Months']} (Δ {delta(m_cur['Red Months'], m_safe['Red Months']):+d})")
    st.write(f"- **Flex Share:** {fmt_pct(m_cur['Flex Share'])} → {fmt_pct(m_safe['Flex Share'])} (Δ {fmt_pct(delta(m_cur['Flex Share'], m_safe['Flex Share']))})")
    st.write(f"- **3yr EBITDA Proxy:** {fmt_money(m_cur['EBITDA Proxy (3yr total)'])} → {fmt_money(m_safe['EBITDA Proxy (3yr total)'])} (Δ {fmt_money(delta(m_cur['EBITDA Proxy (3yr total)'], m_safe['EBITDA Proxy (3yr total)']))})")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# POLICY HEALTH CHECK
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
      <div class="metric-value">{ui["target_utilization"]:.0f}% Target</div>
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
    y=ui["target_utilization"],
    line_dash="dot",
    line_color="green",
    secondary_y=False,
    annotation_text=f"{ui['target_utilization']:.0f}% Target",
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
