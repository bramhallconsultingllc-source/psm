# Bramhall Co. — Predictive Staffing Model
# Elegant, branded interface with sophisticated design
# "predict. perform. prosper."

from __future__ import annotations
import io, math, base64
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from psm.staffing_model import StaffingModel

MODEL_VERSION = "2026-01-30-bramhall-v1"

# ============================================================
# BRAMHALL CO. BRAND IDENTITY
# ============================================================
GOLD = "#7a6200"
BLACK = "#000000"
CREAM = "#faf8f3"
LIGHT_GOLD = "#d4c17f"
DARK_GOLD = "#5c4a00"
GOLD_MUTED = "#a89968"

# Page Configuration
st.set_page_config(
    page_title="Predictive Staffing Model | Bramhall Co.",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sophisticated Custom Styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
    
    /* ===== GLOBAL RESETS ===== */
    .block-container {{
        padding: 1.5rem 2rem 3rem 2rem;
        max-width: 1600px;
    }}
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4 {{
        font-family: 'Cormorant Garamond', serif !important;
        color: {BLACK} !important;
        letter-spacing: 0.015em;
        font-weight: 600 !important;
    }}
    
    h1 {{ font-size: 2.75rem !important; margin-bottom: 0.5rem !important; }}
    h2 {{ font-size: 2rem !important; margin: 2rem 0 1rem 0 !important; }}
    h3 {{ font-size: 1.5rem !important; margin: 1.5rem 0 1rem 0 !important; }}
    
    p, div, span, label {{
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: #2c2c2c;
    }}
    
    /* ===== HERO HEADER ===== */
    .hero-container {{
        background: linear-gradient(135deg, {BLACK} 0%, {DARK_GOLD} 95%, {GOLD} 100%);
        padding: 3rem 2.5rem;
        border-radius: 20px;
        margin: -1rem 0 2.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }}
    
    .hero-container::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: 
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 2px,
                rgba(255,255,255,0.015) 2px,
                rgba(255,255,255,0.015) 4px
            ),
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(255,255,255,0.015) 2px,
                rgba(255,255,255,0.015) 4px
            );
        pointer-events: none;
    }}
    
    .logo-container {{
        display: flex;
        align-items: flex-start;
        gap: 1.5rem;
        position: relative;
        z-index: 1;
    }}
    
    .logo-img {{
        height: 90px;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
    }}
    
    .hero-text {{
        flex: 1;
    }}
    
    .hero-title {{
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0 0 0.75rem 0;
        font-family: 'Cormorant Garamond', serif;
        letter-spacing: 0.02em;
    }}
    
    .hero-tagline {{
        font-size: 1.1rem;
        color: {LIGHT_GOLD};
        font-style: italic;
        font-family: 'Cormorant Garamond', serif;
        letter-spacing: 0.1em;
        margin: 0;
    }}
    
    /* ===== SCORECARD (HERO METRICS) ===== */
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
    
    /* ===== STATUS CARDS ===== */
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
    
    .status-error {{
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left-color: #dc3545;
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
    
    .status-text {{
        flex: 1;
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
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {CREAM} 0%, #ffffff 100%);
        border-right: 1px solid {LIGHT_GOLD};
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        font-size: 0.9rem;
    }}
    
    /* ===== BUTTONS ===== */
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
    
    .stButton > button:active {{
        transform: translateY(-1px);
    }}
    
    /* ===== DOWNLOAD BUTTONS ===== */
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
    
    /* ===== EXPANDERS ===== */
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
    
    /* ===== TABS ===== */
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
    
    /* ===== SECTION DIVIDERS ===== */
    .divider {{
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, {GOLD} 50%, transparent 100%);
        margin: 3rem 0;
    }}
    
    .divider-simple {{
        height: 1px;
        background: {LIGHT_GOLD};
        margin: 2rem 0;
        opacity: 0.5;
    }}
    
    /* ===== DATAFRAME STYLING ===== */
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid {LIGHT_GOLD};
    }}
    
    /* ===== METRIC WIDGETS ===== */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: {GOLD};
        font-family: 'Cormorant Garamond', serif;
    }}
    
    /* ===== INPUT STYLING ===== */
    .stNumberInput input, .stSelectbox select, .stSlider {{
        border-color: {LIGHT_GOLD} !important;
    }}
    
    .stNumberInput input:focus, .stSelectbox select:focus {{
        border-color: {GOLD} !important;
        box-shadow: 0 0 0 2px rgba(122, 98, 0, 0.1) !important;
    }}
</style>
""", unsafe_allow_html=True)

# Load logo
with open("/home/claude/logo_encoded.txt", "r") as f:
    LOGO_B64 = f.read()

# Hero Header with Logo
st.markdown(f"""
<div class="hero-container">
    <div class="logo-container">
        <img src="data:image/png;base64,{LOGO_B64}" class="logo-img" alt="Bramhall Co.">
        <div class="hero-text">
            <div class="hero-title">Predictive Staffing Model</div>
            <div class="hero-tagline">predict. perform. prosper.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

model = StaffingModel()
st.session_state.setdefault("rec_policy", None)

WINTER = {12, 1, 2}
SUMMER = {6, 7, 8}
N_MONTHS = 36
AVG_DAYS_PER_MONTH = 30.4

MONTH_OPTIONS = [
    ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("May", 5), ("Jun", 6),
    ("Jul", 7), ("Aug", 8), ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12)
]

# Helper functions
def wrap_month(m: int) -> int:
    m = int(m)
    while m <= 0: m += 12
    while m > 12: m -= 12
    return m

def month_name(m: int) -> str:
    return datetime(2000, int(m), 1).strftime("%b")

def lead_days_to_months(days: int, avg: float = AVG_DAYS_PER_MONTH) -> int:
    return max(0, int(math.ceil(float(days) / float(avg))))

def provider_day_equiv_from_fte(fte: float, hrs_wk: float, fte_hrs: float) -> float:
    return float(fte) * (float(fte_hrs) / max(float(hrs_wk), 1e-9))

def fte_required_for_min_perm(min_prov: float, hrs_wk: float, fte_hrs: float) -> float:
    return float(min_prov) * (float(hrs_wk) / max(float(fte_hrs), 1e-9))

def compute_visits_curve(months: list, y0: float, y1: float, y2: float, seas: float) -> list:
    out = []
    for i, m in enumerate(months):
        base = y0 if i < 12 else y1 if i < 24 else y2
        if m in WINTER: v = base * (1 + seas)
        elif m in SUMMER: v = base * (1 - seas)
        else: v = base
        out.append(float(v))
    return out

def apply_flu_uplift(visits: list, months: list, flu_months: set, uplift: float) -> list:
    return [float(v) * (1 + uplift) if m in flu_months else float(v) for v, m in zip(visits, months)]

def monthly_hours_from_fte(fte: float, fte_hrs: float, days: int) -> float:
    return float(fte) * float(fte_hrs) * (float(days) / 7.0)

def loaded_hourly_rate(base: float, ben: float, ot: float, bon: float) -> float:
    return float(base) * (1 + bon) * (1 + ben) * (1 + ot)

def compute_role_mix_ratios(vpd: float, mdl: StaffingModel) -> dict:
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
    prov_paid: list, prov_flex: list, vpd: list, dim: list, fte_hrs: float,
    role_mix: dict, rates: dict, ben: float, ot: float, bon: float,
    phys_hrs: float = 0.0, sup_hrs: float = 0.0
) -> tuple:
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

@dataclass(frozen=True)
class ModelParams:
    visits: float; annual_growth: float; seasonality_pct: float; flu_uplift_pct: float
    flu_months: set; peak_factor: float; visits_per_provider_hour: float
    hours_week: float; days_open_per_week: float; fte_hours_week: float
    annual_turnover: float; lead_days: int; ramp_months: int; ramp_productivity: float
    fill_probability: float; winter_anchor_month: int; winter_end_month: int
    freeze_months: set; budgeted_pppd: float; yellow_max_pppd: float; red_start_pppd: float
    flex_max_fte_per_month: float; flex_cost_multiplier: float
    target_swb_per_visit: float; swb_tolerance: float; net_revenue_per_visit: float
    visits_lost_per_provider_day_gap: float; provider_replacement_cost: float
    turnover_yellow_mult: float; turnover_red_mult: float; hourly_rates: dict
    benefits_load_pct: float; ot_sick_pct: float; bonus_pct: float
    physician_supervision_hours_per_month: float; supervisor_hours_per_month: float
    min_perm_providers_per_day: float; allow_prn_override: bool
    require_perm_under_green_no_flex: bool
    _v: str = MODEL_VERSION

@dataclass(frozen=True)
class Policy:
    base_fte: float
    winter_fte: float

# Simulation engine with seasonal-aware backfill
def simulate_policy(params: ModelParams, policy: Policy) -> dict:
    today = datetime.today()
    dates = pd.date_range(start=datetime(today.year, 1, 1), periods=N_MONTHS, freq="MS")
    months = [int(d.month) for d in dates]
    dim = [pd.Period(d, "M").days_in_month for d in dates]
    
    lead_mo = lead_days_to_months(params.lead_days)
    mo_turn = params.annual_turnover / 12.0
    fill_p = max(min(params.fill_probability, 1.0), 0.0)
    
    # Demand
    y0 = params.visits
    y1 = y0 * (1 + params.annual_growth)
    y2 = y1 * (1 + params.annual_growth)
    v_base = compute_visits_curve(months, y0, y1, y2, params.seasonality_pct)
    v_flu = apply_flu_uplift(v_base, months, params.flu_months, params.flu_uplift_pct)
    v_peak = [v * params.peak_factor for v in v_flu]
    
    role_mix = compute_role_mix_ratios(y1, model)
    
    def is_winter(m: int) -> bool:
        a, e = params.winter_anchor_month, params.winter_end_month
        return (m >= a) or (m <= e) if a <= e else (m >= a) or (m <= e)
    
    def target_fte(m: int) -> float:
        return policy.winter_fte if is_winter(m) else policy.base_fte
    
    def ramp_factor(age: int) -> float:
        rm = max(params.ramp_months, 0)
        return params.ramp_productivity if age < rm and rm > 0 else 1.0
    
    cohorts = [{"fte": policy.base_fte, "age": 9999}]
    pipeline = []
    paid_arr, eff_arr, hires_arr, hire_reason_arr, target_arr = [], [], [], [], []
    
    for t in range(N_MONTHS):
        cur_mo = months[t]
        
        # Turnover
        for c in cohorts:
            c["fte"] = max(c["fte"] * (1 - mo_turn), 0.0)
        
        # Add hires
        arriving = [h for h in pipeline if h["arrive"] == t]
        total_hired = sum(h["fte"] for h in arriving)
        if total_hired > 1e-9:
            cohorts.append({"fte": total_hired, "age": 0})
        
        # Current state
        cur_paid = sum(c["fte"] for c in cohorts)
        cur_eff = sum(c["fte"] * ramp_factor(c["age"]) for c in cohorts)
        
        # Seasonal-aware backfill decision
        if t + lead_mo < N_MONTHS and cur_mo not in params.freeze_months:
            fut_idx = t + lead_mo
            fut_mo = months[fut_idx]
            tgt_fut = target_fte(fut_mo)
            
            proj = cur_paid * ((1 - mo_turn) ** lead_mo)
            proj += sum(h["fte"] for h in pipeline if h["arrive"] == fut_idx)
            
            if proj < tgt_fut - 0.05:
                need = (tgt_fut - proj) * fill_p
                season_lbl = "winter" if is_winter(fut_mo) else "base"
                pipeline.append({
                    "arrive": fut_idx,
                    "fte": need,
                    "reason": f"Target {tgt_fut:.2f} ({season_lbl}) for {month_name(fut_mo)}"
                })
        
        # Age cohorts
        for c in cohorts:
            c["age"] += 1
        
        # Record
        paid_arr.append(cur_paid)
        eff_arr.append(cur_eff)
        hires_arr.append(total_hired)
        hire_reason_arr.append(" | ".join(h["reason"] for h in arriving) if arriving else "")
        target_arr.append(target_fte(cur_mo))
    
    # Convert to numpy
    p_paid = np.array(paid_arr)
    p_eff = np.array(eff_arr)
    v_pk = np.array(v_peak)
    v_av = np.array(v_flu)
    d = np.array(dim)
    tgt_pol = np.array(target_arr)
    
    # Required coverage
    vph = max(params.visits_per_provider_hour, 1e-6)
    req_hr_day = v_pk / vph
    req_eff = (req_hr_day * params.days_open_per_week) / max(params.fte_hours_week, 1e-6)
    
    # PPPD load
    pde_perm = np.array([provider_day_equiv_from_fte(f, params.hours_week, params.fte_hours_week) for f in p_eff])
    load_pre = v_pk / np.maximum(pde_perm, 1e-6)
    
    # Flex
    flex_fte = np.zeros(N_MONTHS)
    load_post = np.zeros(N_MONTHS)
    for i in range(N_MONTHS):
        gap = max(req_eff[i] - p_eff[i], 0.0)
        flex_used = min(gap, params.flex_max_fte_per_month)
        flex_fte[i] = flex_used
        pde_tot = provider_day_equiv_from_fte(p_eff[i] + flex_used, params.hours_week, params.fte_hours_week)
        load_post[i] = v_pk[i] / max(pde_tot, 1e-6)
    
    # Metrics
    residual_gap = np.maximum(req_eff - (p_eff + flex_fte), 0.0)
    prov_day_gap = float(np.sum(residual_gap * d))
    est_visits_lost = prov_day_gap * params.visits_lost_per_provider_day_gap
    est_margin_risk = est_visits_lost * params.net_revenue_per_visit
    
    repl = p_paid * mo_turn
    repl_mult = np.ones(N_MONTHS)
    repl_mult = np.where(load_post > params.budgeted_pppd, params.turnover_yellow_mult, repl_mult)
    repl_mult = np.where(load_post > params.red_start_pppd, params.turnover_red_mult, repl_mult)
    turn_cost = float(np.sum(repl * params.provider_replacement_cost * repl_mult))
    
    # SWB
    swb_all, swb_tot, vis_tot = annual_swb_per_visit_from_supply(
        list(p_paid), list(flex_fte), list(v_av), list(dim), params.fte_hours_week,
        role_mix, params.hourly_rates, params.benefits_load_pct, params.ot_sick_pct,
        params.bonus_pct, params.physician_supervision_hours_per_month, params.supervisor_hours_per_month
    )
    
    # Risk bands
    mo_red = int(np.sum(load_post > params.red_start_pppd))
    pk_load = float(np.max(load_post))
    
    # Utilization
    hrs_per_fte_day = params.fte_hours_week / max(params.days_open_per_week, 1e-6)
    sup_tot_hrs = (p_eff + flex_fte) * hrs_per_fte_day
    util = req_hr_day / np.maximum(sup_tot_hrs, 1e-9)
    
    # Penalties (simplified)
    yellow_ex = np.maximum(load_post - params.budgeted_pppd, 0.0)
    red_ex = np.maximum(load_post - params.red_start_pppd, 0.0)
    burn_pen = float(np.sum((yellow_ex ** 1.2) * d) + 3.0 * np.sum((red_ex ** 2.0) * d))
    
    perm_total = float(np.sum(p_eff * d))
    flex_total = float(np.sum(flex_fte * d))
    flex_share = flex_total / max(perm_total + flex_total, 1e-9)
    
    score = swb_tot + turn_cost + est_margin_risk + 2000.0 * burn_pen
    
    # Ledger
    rows = []
    for i in range(N_MONTHS):
        lab = dates[i].strftime("%Y-%b")
        mv = v_av[i] * dim[i]
        mnc = mv * params.net_revenue_per_visit
        
        mswb_pv, mswb, _ = annual_swb_per_visit_from_supply(
            [p_paid[i]], [flex_fte[i]], [v_av[i]], [dim[i]], params.fte_hours_week,
            role_mix, params.hourly_rates, params.benefits_load_pct, params.ot_sick_pct,
            params.bonus_pct, params.physician_supervision_hours_per_month, params.supervisor_hours_per_month
        )
        
        gw = residual_gap[i] * d[i]
        gt = np.sum(residual_gap * d)
        mar = est_margin_risk * (gw / max(gt, 1e-9))
        mebitda = mnc - mswb - mar
        
        rows.append({
            "Month": lab, "Visits/Day (avg)": v_av[i], "Total Visits (month)": mv,
            "SWB Dollars (month)": mswb, "SWB/Visit (month)": mswb_pv,
            "EBITDA Proxy (month)": mebitda, "Permanent FTE (Paid)": p_paid[i],
            "Permanent FTE (Effective)": p_eff[i], "Flex FTE Used": flex_fte[i],
            "Required Provider FTE (effective)": req_eff[i], "Utilization (Req/Supplied)": util[i],
            "Load PPPD (post-flex)": load_post[i], "Hires Visible (FTE)": hires_arr[i],
            "Hire Reason": hire_reason_arr[i], "Target FTE (policy)": tgt_pol[i],
        })
    
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
    
    # SWB band penalty
    tgt = params.target_swb_per_visit
    tol = params.swb_tolerance
    annual["SWB_Dev"] = np.maximum(np.abs(annual["SWB_per_Visit"] - tgt) - tol, 0.0)
    swb_pen = float(np.sum((annual["SWB_Dev"] ** 2) * 1_500_000.0))
    
    flex_pen = max(flex_share - 0.10, 0.0) ** 2 * 2_000_000.0
    score += swb_pen + flex_pen
    
    ebitda_ann = vis_tot * params.net_revenue_per_visit - swb_tot - turn_cost - est_margin_risk
    
    return {
        "dates": list(dates), "months": months, "perm_paid": list(p_paid), "perm_eff": list(p_eff),
        "req_eff_fte_needed": list(req_eff), "utilization": list(util), "load_post": list(load_post),
        "annual_swb_per_visit": swb_all, "flex_share": flex_share, "months_red": mo_red,
        "peak_load_post": pk_load, "ebitda_proxy_annual": ebitda_ann, "score": score,
        "ledger": ledger.drop(columns=["Year"]), "annual_summary": annual, "target_policy": list(tgt_pol),
    }

def recommend_policy(params: ModelParams, base_min: float, base_max: float, base_step: float,
                     winter_delta_max: float, winter_step: float) -> dict:
    candidates = []
    best = None
    for b in np.arange(base_min, base_max + 1e-9, base_step):
        for w in np.arange(b, b + winter_delta_max + 1e-9, winter_step):
            pol = Policy(base_fte=float(b), winter_fte=float(w))
            res = simulate_policy(params, pol)
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
    return simulate_policy(params, Policy(base_fte=base_fte, winter_fte=winter_fte))

# ============================================================
# SIDEBAR WITH COLLAPSIBLE SECTIONS
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style='background: {CREAM}; padding: 1.25rem; border-radius: 12px; 
                border-left: 4px solid {GOLD}; margin-bottom: 1.5rem;'>
        <div style='font-weight: 600; font-size: 0.95rem; color: {BLACK}; margin-bottom: 0.5rem;'>
            🎯 Seasonal-Aware Backfill
        </div>
        <div style='font-size: 0.8rem; color: #555; line-height: 1.5;'>
            Providers give 90-day notice. Model checks future season before backfilling. 
            Enables natural step-down from winter to base.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 **Demand Settings**", expanded=True):
        visits = st.number_input("Avg Visits/Day", 1.0, value=36.0, step=1.0)
        annual_growth = st.number_input("Annual Growth %", 0.0, value=10.0, step=1.0) / 100.0
        peak_factor = st.slider("Peak factor", 1.0, 1.5, 1.2, 0.01)
        seasonality_pct = st.number_input("Seasonality %", 0.0, value=20.0, step=5.0) / 100.0
        flu_uplift_pct = st.number_input("Flu uplift %", 0.0, value=0.0, step=5.0) / 100.0
        flu_months = st.multiselect("Flu months", MONTH_OPTIONS, default=[("Oct",10),("Nov",11),("Dec",12),("Jan",1),("Feb",2)])
        flu_months_set = {m for _, m in flu_months} if flu_months else set()
        visits_per_provider_hour = st.slider("Visits/prov-hour", 2.0, 4.0, 3.0, 0.1)
    
    with st.expander("🏥 **Clinic Operations**", expanded=False):
        hours_week = st.number_input("Hours/Week", 1.0, value=84.0, step=1.0)
        days_open_per_week = st.number_input("Days/Week", 1.0, 7.0, value=7.0, step=1.0)
        fte_hours_week = st.number_input("FTE Hours/Week", 1.0, value=36.0, step=1.0)
    
    with st.expander("📋 **Staffing Policy**", expanded=False):
        min_perm_providers_per_day = st.number_input("Min prov/day", 0.0, value=1.0, step=0.25)
        allow_prn_override = st.checkbox("Allow Base < min", value=False)
        require_perm_under_green_no_flex = st.checkbox("Perm ≤ Green", value=True)
    
    with st.expander("👥 **Workforce Dynamics**", expanded=False):
        annual_turnover = st.number_input("Turnover %", 0.0, value=16.0, step=1.0) / 100.0
        lead_days = st.number_input("Lead days (notice period)", 0, value=90, step=10)
        ramp_months = st.slider("Ramp months", 0, 6, 1)
        ramp_productivity = st.slider("Ramp prod %", 30, 100, 75, 5) / 100.0
        fill_probability = st.slider("Fill prob %", 0, 100, 85, 5) / 100.0
    
    with st.expander("⚡ **Risk Bands (PPPD)**", expanded=False):
        budgeted_pppd = st.number_input("Green PPPD", 5.0, value=36.0, step=1.0)
        yellow_max_pppd = st.number_input("Yellow PPPD", 5.0, value=42.0, step=1.0)
        red_start_pppd = st.number_input("Red PPPD", 5.0, value=45.0, step=1.0)
    
    with st.expander("❄️ **Seasonal Settings**", expanded=False):
        winter_anchor_month = st.selectbox("Winter anchor", MONTH_OPTIONS, index=11)
        winter_anchor_month_num = int(winter_anchor_month[1])
        winter_end_month = st.selectbox("Winter end", MONTH_OPTIONS, index=1)
        winter_end_month_num = int(winter_end_month[1])
        freeze_months = st.multiselect("Freeze months", MONTH_OPTIONS, default=flu_months if flu_months else [])
        freeze_months_set = {m for _, m in freeze_months} if freeze_months else set()
    
    with st.expander("🔄 **Flex Staffing**", expanded=False):
        flex_max_fte_per_month = st.slider("Max flex/mo", 0.0, 10.0, 2.0, 0.25)
        flex_cost_multiplier = st.slider("Flex cost mult", 1.0, 2.0, 1.25, 0.05)
    
    with st.expander("💰 **Financial Targets**", expanded=False):
        target_swb_per_visit = st.number_input("Target SWB/Visit", 0.0, value=85.0, step=1.0)
        swb_tolerance = st.number_input("SWB tolerance", 0.0, value=2.0, step=0.5)
        net_revenue_per_visit = st.number_input("Net contrib/visit", 0.0, value=140.0, step=5.0)
        visits_lost_per_provider_day_gap = st.number_input("Visits lost/prov-day", 0.0, value=18.0, step=1.0)
        provider_replacement_cost = st.number_input("Replacement cost", 0.0, value=75000.0, step=5000.0)
        turnover_yellow_mult = st.slider("Yellow mult", 1.0, 3.0, 1.3, 0.05)
        turnover_red_mult = st.slider("Red mult", 1.0, 5.0, 2.0, 0.1)
    
    with st.expander("💵 **Compensation**", expanded=False):
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
    
    st.markdown('<div class="divider-simple"></div>', unsafe_allow_html=True)
    
    with st.expander("🎯 **Optimizer Settings**", expanded=True):
        min_base_req = fte_required_for_min_perm(min_perm_providers_per_day, hours_week, fte_hours_week)
        base_min = st.number_input("Base min", 0.0, value=0.0 if allow_prn_override else min_base_req, step=0.25)
        base_max = st.number_input("Base max", 0.0, value=6.0, step=0.25)
        base_step = st.select_slider("Base step", [0.10, 0.25, 0.50], value=0.25)
        winter_delta_max = st.number_input("Winter uplift max", 0.0, value=2.0, step=0.25)
        winter_step = st.select_slider("Winter step", [0.10, 0.25, 0.50], value=0.25)
    
    mode = st.radio("Mode", ["Recommend + What-If", "What-If only"], index=0)
    run_recommender = st.button("🏁 Run Optimization", use_container_width=True)

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
    _v=MODEL_VERSION
)

params_dict = {**params.__dict__, "_v": MODEL_VERSION}

# RUN RECOMMENDER
if mode == "Recommend + What-If" and run_recommender:
    with st.spinner("🔍 Optimizing policy..."):
        rec = cached_recommend(params_dict, base_min, base_max, base_step, winter_delta_max, winter_step)
    st.session_state.rec_policy = rec["best"]["policy"]
    st.session_state["what_base_fte"] = float(rec["best"]["policy"].base_fte)
    st.session_state["what_winter_fte"] = float(rec["best"]["policy"].winter_fte)
    st.success("✅ Optimization complete!")

rec_policy = st.session_state.rec_policy

# ============================================================
# WHAT-IF POLICY INPUTS
# ============================================================
st.markdown("## 🎯 Policy Configuration")

st.session_state.setdefault("what_base_fte", max(base_min, 2.0))
st.session_state.setdefault("what_winter_fte", st.session_state["what_base_fte"] + 1.0)

col1, col2, col3 = st.columns([2, 2, 3])
with col1:
    what_base = st.number_input(
        "**Base FTE** (non-winter)", 
        0.0 if allow_prn_override else min_base_req,
        value=st.session_state["what_base_fte"], 
        step=0.25, 
        key="what_base_fte"
    )
with col2:
    what_winter = st.number_input(
        "**Winter FTE** (Dec-Feb)", 
        what_base,
        value=max(st.session_state.get("what_winter_fte", what_base), what_base),
        step=0.25, 
        key="what_winter_fte"
    )
with col3:
    uplift = what_winter - what_base
    st.markdown(f"""
    <div style='padding: 1rem; background: {CREAM}; border-radius: 8px; margin-top: 1.8rem;'>
        <div style='font-size: 0.8rem; color: {DARK_GOLD}; font-weight: 600; margin-bottom: 0.25rem;'>
            SEASONAL UPLIFT
        </div>
        <div style='font-size: 1.75rem; font-weight: 700; color: {GOLD}; font-family: "Cormorant Garamond", serif;'>
            +{uplift:.2f} FTE
        </div>
    </div>
    """, unsafe_allow_html=True)

# Run simulation
R = cached_simulate(params_dict, what_base, what_winter)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# RATCHET DETECTION STATUS
# ============================================================
st.markdown("## 🔍 Policy Health Check")

annual = R["annual_summary"]
if len(annual) >= 2:
    min_y1, min_y2 = annual.loc[0, "Min_Perm_Paid_FTE"], annual.loc[1, "Min_Perm_Paid_FTE"]
    min_y3 = annual.loc[2, "Min_Perm_Paid_FTE"] if len(annual) >= 3 else min_y2
    drift_y2, drift_y3 = min_y2 - min_y1, min_y3 - min_y2
    
    if abs(drift_y2) < 0.2 and abs(drift_y3) < 0.2:
        st.markdown(f"""
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
                        Policy base target: {what_base:.2f} FTE
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-card status-warning">
            <div class="status-content">
                <div class="status-icon">⚠️</div>
                <div class="status-text">
                    <div class="status-title">Minor Drift Detected</div>
                    <div class="status-message">
                        <strong>Year 1:</strong> {min_y1:.2f} → 
                        <strong>Year 2:</strong> {min_y2:.2f} (Δ{drift_y2:+.2f}) → 
                        <strong>Year 3:</strong> {min_y3:.2f} (Δ{drift_y3:+.2f})<br>
                        Expected: ±0.2 FTE/year. Consider adjusting turnover or fill probability.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# HERO SCORECARD
# ============================================================
swb_y1 = annual.loc[0, "SWB_per_Visit"]
ebitda_y1, ebitda_y3 = annual.loc[0, "EBITDA_Proxy"], annual.loc[2, "EBITDA_Proxy"]
util_y1, util_y3 = annual.loc[0, "Avg_Utilization"], annual.loc[2, "Avg_Utilization"]
min_y1, max_y1 = annual.loc[0, "Min_Perm_Paid_FTE"], annual.loc[0, "Max_Perm_Paid_FTE"]

st.markdown(f"""
<div class="scorecard-hero">
    <div class="scorecard-title">Policy Performance Scorecard</div>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Policy Levels</div>
            <div class="metric-value">{what_base:.1f} / {what_winter:.1f}</div>
            <div class="metric-detail">Base / Winter FTE</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">SWB per Visit (Y1)</div>
            <div class="metric-value">${swb_y1:.2f}</div>
            <div class="metric-detail">Target: ${target_swb_per_visit:.0f} ± ${swb_tolerance:.0f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">EBITDA Proxy</div>
            <div class="metric-value">${ebitda_y1/1000:.0f}K</div>
            <div class="metric-detail">Year 1 / Year 3: ${ebitda_y3/1000:.0f}K</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Utilization</div>
            <div class="metric-value">{util_y1*100:.0f}%</div>
            <div class="metric-detail">Year 1 / Year 3: {util_y3*100:.0f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">FTE Range (Y1)</div>
            <div class="metric-value">{min_y1:.1f}-{max_y1:.1f}</div>
            <div class="metric-detail">Min-Max across months</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Peak Load</div>
            <div class="metric-value">{R['peak_load_post']:.1f}</div>
            <div class="metric-detail">PPPD (post-flex)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# INTERACTIVE PLOTLY CHARTS
# ============================================================
st.markdown("## 📊 3-Year Financial Projection")

dates = R["dates"]
perm_paid = np.array(R["perm_paid"])
target_pol = np.array(R["target_policy"])
req_eff = np.array(R["req_eff_fte_needed"])
util = np.array(R["utilization"])
load_post = np.array(R["load_post"])

# Chart 1: Supply vs Target (Plotly)
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=dates, y=perm_paid,
    mode='lines+markers',
    name='Paid FTE',
    line=dict(color=GOLD, width=3),
    marker=dict(size=5, color=GOLD),
    hovertemplate='<b>%{x|%Y-%b}</b><br>Paid FTE: %{y:.2f}<extra></extra>'
))

fig1.add_trace(go.Scatter(
    x=dates, y=target_pol,
    mode='lines+markers',
    name='Target (policy)',
    line=dict(color=BLACK, width=2, dash='dash'),
    marker=dict(size=5, symbol='square', color=BLACK),
    hovertemplate='<b>%{x|%Y-%b}</b><br>Target: %{y:.2f}<extra></extra>'
))

fig1.add_trace(go.Scatter(
    x=dates, y=req_eff,
    mode='lines',
    name='Required FTE',
    line=dict(color=LIGHT_GOLD, width=2, dash='dot'),
    hovertemplate='<b>%{x|%Y-%b}</b><br>Required: %{y:.2f}<extra></extra>'
))

fig1.update_layout(
    title=dict(
        text='<b>Supply vs Target FTE</b><br><sup>Base should stay constant year-over-year</sup>',
        font=dict(size=20, family='Cormorant Garamond, serif', color=BLACK),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='',
        showgrid=True,
        gridcolor='rgba(0,0,0,0.05)'
    ),
    yaxis=dict(
        title='Provider FTE',
        titlefont=dict(size=14, family='IBM Plex Sans, sans-serif'),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.05)'
    ),
    hovermode='x unified',
    plot_bgcolor='rgba(250, 248, 243, 0.3)',
    paper_bgcolor='white',
    height=450,
    font=dict(family='IBM Plex Sans, sans-serif', size=12),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=LIGHT_GOLD,
        borderwidth=1
    )
)

st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Utilization & Load (Dual axis)
fig2 = make_subplots(specs=[[{"secondary_y": True}]])

fig2.add_trace(
    go.Scatter(
        x=dates, y=util*100,
        mode='lines+markers',
        name='Utilization %',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=5),
        hovertemplate='<b>%{x|%Y-%b}</b><br>Utilization: %{y:.1f}%<extra></extra>'
    ),
    secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=dates, y=load_post,
        mode='lines+markers',
        name='Load PPPD',
        line=dict(color=GOLD, width=3),
        marker=dict(size=5),
        hovertemplate='<b>%{x|%Y-%b}</b><br>Load: %{y:.1f} PPPD<extra></extra>'
    ),
    secondary_y=True
)

fig2.add_hline(y=90, line_dash="dot", line_color="green", secondary_y=False,
               annotation_text="90% Target", annotation_position="right")
fig2.add_hline(y=budgeted_pppd, line_dash="dot", line_color="green", secondary_y=True,
               annotation_text=f"Green ({budgeted_pppd:.0f})", annotation_position="right")
fig2.add_hline(y=red_start_pppd, line_dash="dot", line_color="red", secondary_y=True,
               annotation_text=f"Red ({red_start_pppd:.0f})", annotation_position="right")

fig2.update_layout(
    title=dict(
        text='<b>Utilization & Provider Load</b><br><sup>Target: 85-95% utilization, load under green threshold</sup>',
        font=dict(size=20, family='Cormorant Garamond, serif', color=BLACK),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(title='', showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
    hovermode='x unified',
    plot_bgcolor='rgba(250, 248, 243, 0.3)',
    paper_bgcolor='white',
    height=450,
    font=dict(family='IBM Plex Sans, sans-serif', size=12),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=LIGHT_GOLD,
        borderwidth=1
    )
)

fig2.update_yaxes(title_text="<b>Utilization (%)</b>", secondary_y=False, 
                  titlefont=dict(family='IBM Plex Sans, sans-serif', size=14))
fig2.update_yaxes(title_text="<b>Load (PPPD)</b>", secondary_y=True,
                  titlefont=dict(family='IBM Plex Sans, sans-serif', size=14))

st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# DATA TABLES (Tabs)
# ============================================================
st.markdown("## 📈 Detailed Results")

tab1, tab2 = st.tabs(["📊 Annual Summary", "📋 Monthly Ledger"])

with tab1:
    st.markdown("### Annual Performance by Year")
    st.dataframe(
        annual.style.format({
            "Visits": "{:,.0f}",
            "SWB_per_Visit": "${:,.2f}",
            "SWB_Dollars": "${:,.0f}",
            "EBITDA_Proxy": "${:,.0f}",
            "Min_Perm_Paid_FTE": "{:.2f}",
            "Max_Perm_Paid_FTE": "{:.2f}",
            "Avg_Utilization": "{:.1%}",
        }),
        hide_index=True,
        use_container_width=True
    )

with tab2:
    st.markdown("### Month-by-Month Audit Trail")
    st.dataframe(
        R["ledger"].style.format({
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
        }),
        hide_index=True,
        use_container_width=True,
        height=400
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# EXPORTS
# ============================================================
st.markdown("## 💾 Export Results")

def fig_to_bytes(fig):
    return fig.to_image(format="png", engine="kaleido")

col1, col2, col3 = st.columns(3)

with col1:
    try:
        png1 = fig_to_bytes(fig1)
        st.download_button(
            "⬇️ Supply Chart (PNG)",
            png1,
            "supply_vs_target.png",
            "image/png",
            use_container_width=True
        )
    except:
        st.info("Install kaleido for image export: `pip install kaleido`")

with col2:
    try:
        png2 = fig_to_bytes(fig2)
        st.download_button(
            "⬇️ Utilization Chart (PNG)",
            png2,
            "utilization_load.png",
            "image/png",
            use_container_width=True
        )
    except:
        pass

with col3:
    csv_data = R["ledger"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Ledger (CSV)",
        csv_data,
        "staffing_ledger.csv",
        "text/csv",
        use_container_width=True
    )

# Footer
st.markdown('<div class="divider-simple"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; padding: 2rem 0; color: {GOLD_MUTED};'>
    <div style='font-size: 0.9rem; font-style: italic; font-family: "Cormorant Garamond", serif;'>
        predict. perform. prosper.
    </div>
    <div style='font-size: 0.75rem; margin-top: 0.5rem; color: #999;'>
        Bramhall Co. | Predictive Staffing Model v{MODEL_VERSION.split('-')[-1]}
    </div>
</div>
""", unsafe_allow_html=True)
