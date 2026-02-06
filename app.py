# Bramhall Co. — Predictive Staffing Model (PSM)
# "predict. perform. prosper."

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from psm.staffing_model import StaffingModel

MODEL_VERSION = "2026-01-30-bramhall-v1"

# ============================================================
# BRAND
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

POSTURE_LABEL = {
    1: "Very Lean",
    2: "Lean",
    3: "Balanced",
    4: "Safe",
    5: "Very Safe",
}
POSTURE_TEXT = {
    1: "Very Lean: aggressive cost strategy; high risk of red months and visit loss.",
    2: "Lean: cost-focused posture; moderate flex reliance.",
    3: "Balanced: standard posture; 100% coverage of required FTE.",
    4: "Safe: proactive staffing; protects access/quality; higher cost.",
    5: "Very Safe: maximum stability; highest cost; lowest volatility.",
}
POSTURE_BASE_COVERAGE_MULT = {1: 0.80, 2: 0.88, 3: 1.00, 4: 1.04, 5: 1.08}
POSTURE_WINTER_BUFFER_ADD  = {1: 0.00, 2: 0.01, 3: 0.04, 4: 0.06, 5: 0.08}
POSTURE_FLEX_CAP_MULT      = {1: 2.00, 2: 1.50, 3: 1.00, 4: 0.90, 5: 0.80}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Predictive Staffing Model | Bramhall Co.",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS (KEEP THIS AS ONE STRING — NO STRAY CSS OUTSIDE IT)
# ============================================================
def inject_css() -> None:
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, p, div, label, li, input, textarea, button {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      color: #2c2c2c;
      line-height: 1.55;
    }}

    h1, h2, h3, h4, h5, h6 {{
      font-family: 'Inter', sans-serif !important;
      font-weight: 650 !important;
      color: #141414 !important;
      letter-spacing: -0.02em;
      line-height: 1.25;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
      font-size: 1.9rem !important;
      font-weight: 650 !important;
      font-family: 'IBM Plex Mono', monospace !important;
      color: #151515 !important;
    }}
    [data-testid="stMetricLabel"] {{
      font-size: 0.78rem !important;
      font-weight: 650 !important;
      color: #666 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.08em !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
      gap: 2rem;
      border-bottom: 1px solid #e6e6e6;
    }}
    .stTabs [data-baseweb="tab"] {{
      padding: 0.75rem 0;
      font-weight: 520;
      color: #666;
      border-bottom: 2px solid transparent;
    }}
    .stTabs [aria-selected="true"] {{
      color: {GOLD};
      border-bottom-color: {GOLD};
    }}

    /* Simple section divider */
    .divider {{
      height: 1px;
      background: #ececec;
      margin: 2.25rem 0;
    }}

    /* Intro */
    .intro {{
      text-align: center;
      padding: 2.2rem 0 1.6rem 0;
      border-bottom: 1px solid #ececec;
      margin-bottom: 1.8rem;
    }}
    .tagline {{
      font-size: 0.92rem;
      color: {GOLD};
      font-weight: 550;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      margin-top: 0.35rem;
    }}

    /* Hero scorecard */
    .hero {{
      border: 1px solid #ececec;
      border-left: 4px solid {GOLD};
      border-radius: 12px;
      padding: 1.1rem 1.1rem;
      background: white;
    }}
    .hero-title {{
      font-weight: 700;
      color: {GOLD};
      margin-bottom: 0.85rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.8rem;
    }}
    .card {{
      border: 1px solid #efefef;
      border-radius: 12px;
      padding: 0.75rem 0.85rem;
      background: #fbfbfb;
      border-left: 4px solid #dcdcdc;
      min-height: 96px;
    }}
    .kpi-label {{
      font-size: 0.68rem;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      font-weight: 700;
      color: #777;
      margin-bottom: 0.25rem;
    }}
    .kpi-value {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.35rem;
      font-weight: 650;
      color: #111;
      margin-bottom: 0.15rem;
    }}
    .kpi-detail {{
      font-size: 0.78rem;
      color: #666;
      line-height: 1.25;
    }}

    @media (max-width: 1200px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# ============================================================
# INTRO
# ============================================================
st.markdown(
    f"""
    <div class="intro">
      <h2 style="margin-bottom: 0.2rem;">Predictive Staffing Model</h2>
      <div class="tagline">Predict. Perform. Prosper.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    min_permanent_fte: float
    allow_prn_override: bool
    require_perm_under_green_no_flex: bool

    _v: str = MODEL_VERSION

@dataclass(frozen=True)
class Policy:
    base_coverage_pct: float
    winter_coverage_pct: float

# ============================================================
# SIMULATION
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
        policy_target = base_required * (policy.winter_coverage_pct if is_winter(months[idx]) else policy.base_coverage_pct)
        return max(policy_target, float(params.min_permanent_fte))

    def ramp_factor(age: int) -> float:
        rm = max(int(params.ramp_months), 0)
        return params.ramp_productivity if (rm > 0 and age < rm) else 1.0

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

        for c in cohorts:
            c["fte"] = max(float(c["fte"]) * (1 - mo_turn), 0.0)

        arriving = [h for h in pipeline if int(h["arrive"]) == t]
        total_hired = float(sum(float(h["fte"]) for h in arriving))
        if total_hired > 1e-9:
            cohorts.append({"fte": total_hired, "age": 0})

        cur_paid = float(sum(float(c["fte"]) for c in cohorts))
        cur_eff = float(sum(float(c["fte"]) * ramp_factor(int(c["age"])) for c in cohorts))

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

    flex_fte = np.zeros(N_MONTHS, dtype=float)
    load_post = np.zeros(N_MONTHS, dtype=float)
    load_pre = np.zeros(N_MONTHS, dtype=float)

    for i in range(N_MONTHS):
        gap = max(req_eff_arr[i] - p_eff[i], 0.0)
        flex_used = min(gap, float(params.flex_max_fte_per_month))
        flex_fte[i] = flex_used

        pde_perm = provider_day_equiv_from_fte(p_eff[i], params.hours_week, params.fte_hours_week)
        load_pre[i] = v_pk[i] / max(pde_perm, 1e-6)

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

    return {
        "dates": list(dates),
        "months": months,
        "perm_paid": list(p_paid),
        "perm_eff": list(p_eff),
        "req_eff_fte_needed": list(req_eff_arr),
        "utilization": list(util),
        "load_post": list(load_post),
        "load_pre": list(load_pre),
        "annual_swb_per_visit": float(swb_all),
        "flex_share": float(flex_share),
        "months_red": mo_red,
        "peak_load_post": float(np.max(load_post)),
        "peak_load_pre": float(np.max(load_pre)),
        "ebitda_proxy_annual": float(ebitda_ann),
        "score": float(score),
        "ledger": ledger.drop(columns=["Year"]),
        "annual_summary": annual,
        "target_policy": list(tgt_pol),
    }

@st.cache_data(show_spinner=False)
def cached_simulate(params_dict: Dict[str, Any], policy_dict: Dict[str, float], current_fte_override: Optional[float]) -> Dict[str, object]:
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
            <div style="background: white; padding: 1.1rem; border-radius: 10px;
                        border: 1px solid {LIGHT_GOLD}; margin-bottom: 1.25rem;">
              <div style="font-weight: 700; color: {GOLD};">Intelligent Cost-Driven Staffing</div>
              <div style="font-size: 0.85rem; color:#666; margin-top: 0.35rem;">
                Peak-aware planning with hiring runway and optional posting freezes.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.checkbox("Seeing errors? Clear cache", value=False):
            st.cache_data.clear()
            st.success("Cache cleared.")
            st.rerun()

        st.markdown("### Core Settings")
        visits = st.number_input("Average Visits/Day", 1.0, value=36.0, step=1.0)
        annual_growth = st.number_input("Annual Growth %", 0.0, value=10.0, step=1.0) / 100.0

        c1, c2 = st.columns(2)
        with c1:
            seasonality_pct = st.number_input("Seasonality %", 0.0, value=20.0, step=5.0) / 100.0
        with c2:
            peak_factor = st.number_input("Peak Factor", 1.0, value=1.0, step=0.1)

        annual_turnover = st.number_input("Turnover %", 0.0, value=16.0, step=1.0) / 100.0

        st.markdown("### Current Staffing")
        use_current_state = st.checkbox("Start from actual current FTE (not policy target)", value=False)
        current_fte = None
        if use_current_state:
            current_fte = st.number_input("Current FTE on Staff", 0.1, value=3.5, step=0.1)

        st.markdown("### Lead Times")
        c1, c2 = st.columns(2)
        with c1:
            turnover_notice_days = st.number_input("Turnover Notice (days)", 0, value=90, step=10)
        with c2:
            hiring_runway_days = st.number_input("Hiring Runway (days)", 0, value=210, step=10)

        with st.expander("Advanced Settings", expanded=False):
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

            freeze_months = st.multiselect("Hiring Freeze Months", MONTH_OPTIONS, default=[])
            freeze_months_set = {m for _, m in freeze_months} if freeze_months else set()

            st.markdown("**Policy Constraints**")
            min_permanent_fte = st.number_input("Min Permanent FTE", 0.0, value=2.33, step=0.1)
            min_perm_providers_per_day = st.number_input("Min Providers/Day", 0.0, value=1.0, step=0.25)
            allow_prn_override = st.checkbox("Allow Base < Minimum", value=False)
            require_perm_under_green_no_flex = st.checkbox("Require Perm ≤ Green", value=True)
            flex_max_fte_per_month = st.slider("Max Flex FTE/Month", 0.0, 10.0, 2.0, 0.25)
            flex_cost_multiplier = st.slider("Flex Cost Multiplier", 1.0, 2.0, 1.25, 0.05)
        # defaults when expander closed (Streamlit still defines them only if run)
        # so we must ensure they exist:
        if "hours_week" not in locals():
            hours_week, days_open_per_week, fte_hours_week = 84.0, 7.0, 36.0
            visits_per_provider_hour = 3.0
            ramp_months, ramp_productivity, fill_probability = 1, 0.75, 0.85
            budgeted_pppd, yellow_max_pppd, red_start_pppd = 36.0, 42.0, 45.0
            flu_uplift_pct, flu_months_set = 0.0, {10, 11, 12, 1, 2}
            winter_anchor_month_num, winter_end_month_num = 12, 2
            freeze_months_set = set()
            min_permanent_fte, min_perm_providers_per_day = 2.33, 1.0
            allow_prn_override, require_perm_under_green_no_flex = False, True
            flex_max_fte_per_month, flex_cost_multiplier = 2.0, 1.25

        with st.expander("Financial Parameters", expanded=False):
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

        if "target_swb_per_visit" not in locals():
            target_swb_per_visit, swb_tolerance = 85.0, 2.0
            net_revenue_per_visit, visits_lost_per_provider_day_gap = 140.0, 18.0
            provider_replacement_cost, turnover_yellow_mult, turnover_red_mult = 75000.0, 1.3, 2.0
            benefits_load_pct, bonus_pct, ot_sick_pct = 0.30, 0.10, 0.04
            physician_hr, apc_hr, ma_hr = 135.79, 62.0, 24.14
            psr_hr, rt_hr, supervisor_hr = 21.23, 31.36, 28.25
            physician_supervision_hours_per_month, supervisor_hours_per_month = 0.0, 0.0

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown("### Staffing Risk Posture")
        risk_posture = st.select_slider(
            "Lean ↔ Safe",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: POSTURE_LABEL.get(x, str(x)),
        )
        st.caption(POSTURE_TEXT[int(risk_posture)])

        st.markdown("### Smart Staffing Policy")
        if "target_utilization" not in st.session_state:
            st.session_state.target_utilization = 92
        target_utilization = st.slider("Target Utilization %", 80, 98, value=int(st.session_state.target_utilization), step=2)
        winter_buffer_pct = st.slider("Winter Buffer %", 0, 10, 3, 1) / 100.0

        base_coverage_from_util = 1.0 / (target_utilization / 100.0)
        posture_mult = POSTURE_BASE_COVERAGE_MULT[int(risk_posture)]
        posture_winter_add = POSTURE_WINTER_BUFFER_ADD[int(risk_posture)]
        base_coverage_pct = base_coverage_from_util * posture_mult
        winter_coverage_pct = base_coverage_pct * (1 + winter_buffer_pct + posture_winter_add)

        flex_max_fte_effective = float(flex_max_fte_per_month) * POSTURE_FLEX_CAP_MULT[int(risk_posture)]

        run_simulation = st.button("Run Simulation", use_container_width=True, type="primary")

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
            min_permanent_fte=min_permanent_fte,
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
    with st.spinner("Running simulation..."):
        R = cached_simulate(params_dict, policy_dict, current_fte)
    st.session_state["simulation_result"] = R
else:
    if "simulation_result" not in st.session_state:
        R = cached_simulate(params_dict, policy_dict, current_fte)
        st.session_state["simulation_result"] = R
    R = st.session_state["simulation_result"]

# ============================================================
# HERO SCORECARD (render in-page so injected CSS applies)
# ============================================================
annual = R["annual_summary"]

# Guard: annual may not have rows if something upstream fails
if annual is None or len(annual) == 0:
    st.error("Annual summary is empty — unable to render scorecard.")
else:
    swb_y1 = float(annual.loc[0, "SWB_per_Visit"])
    ebitda_y1 = float(annual.loc[0, "EBITDA_Proxy"])
    util_y1 = float(annual.loc[0, "Avg_Utilization"])
    min_y1 = float(annual.loc[0, "Min_Perm_Paid_FTE"])
    max_y1 = float(annual.loc[0, "Max_Perm_Paid_FTE"])
    fte_range = max_y1 - min_y1
    fte_avg = (max_y1 + min_y1) / 2.0
    fte_volatility = fte_range / max(fte_avg, 1.0)

    peak_pre = float(R.get("peak_load_pre", R.get("peak_load_post", 0.0)))
    peak_post = float(R.get("peak_load_post", peak_pre))
    red_months = int(R["months_red"])
    flex_share = float(R["flex_share"])

    def swb_color():
        diff = abs(swb_y1 - params.target_swb_per_visit)
        if diff <= params.swb_tolerance:
            return "#27ae60", "On Target"
        if diff <= params.swb_tolerance * 2:
            return "#f39c12", "Close"
        return "#e74c3c", "Off Target"

    def util_color():
        diff = abs(util_y1 * 100 - ui["target_utilization"])
        if diff <= 2:
            return "#27ae60", "On Target"
        if diff <= 5:
            return "#f39c12", "Close"
        return "#e74c3c", "Off Target"

    def peak_color():
        if peak_pre <= params.budgeted_pppd:
            return "#27ae60", "Green"
        if peak_pre <= params.yellow_max_pppd:
            return "#f1c40f", "Yellow"
        if peak_pre <= params.red_start_pppd:
            return "#f39c12", "Orange"
        return "#e74c3c", "Red"

    def fte_color():
        if fte_volatility <= 0.10:
            return "#27ae60", "Stable"
        if fte_volatility <= 0.20:
            return "#f1c40f", "Moderate"
        return "#e74c3c", "Volatile"

    swb_c, swb_s = swb_color()
    util_c, util_s = util_color()
    peak_c, peak_s = peak_color()
    fte_c, fte_s = fte_color()

    hero_html = f"""
    <div class="hero">
      <div class="hero-title">Policy Performance Scorecard</div>
      <div class="grid">
        <div class="card" style="border-left-color:{GOLD}; background:white;">
          <div class="kpi-label">Staffing Policy</div>
          <div class="kpi-value">{ui["target_utilization"]}% Target</div>
          <div class="kpi-detail">
            Coverage: {policy.base_coverage_pct*100:.0f}% base / {policy.winter_coverage_pct*100:.0f}% winter<br/>
            Posture: {POSTURE_LABEL[int(ui["risk_posture"])]}
          </div>
        </div>

        <div class="card" style="border-left-color:{swb_c};">
          <div class="kpi-label">SWB per Visit (Y1)</div>
          <div class="kpi-value" style="color:{swb_c};">${swb_y1:.2f}</div>
          <div class="kpi-detail">Target ${params.target_swb_per_visit:.0f} ± ${params.swb_tolerance:.0f}
            <b style="color:{swb_c};">({swb_s})</b>
          </div>
        </div>

        <div class="card" style="border-left-color:{util_c};">
          <div class="kpi-label">Utilization (Y1)</div>
          <div class="kpi-value" style="color:{util_c};">{util_y1*100:.0f}%</div>
          <div class="kpi-detail">Target {ui["target_utilization"]}%
            <b style="color:{util_c};">({util_s})</b>
          </div>
        </div>

        <div class="card" style="border-left-color:{peak_c};">
          <div class="kpi-label">Peak Load (PPPD)</div>
          <div class="kpi-value" style="color:{peak_c};">{peak_pre:.1f}</div>
          <div class="kpi-detail">Pre-flex <b style="color:{peak_c};">({peak_s})</b><br/>Post-flex: {peak_post:.1f}</div>
        </div>

        <div class="card" style="border-left-color:{fte_c};">
          <div class="kpi-label">FTE Range (Y1)</div>
          <div class="kpi-value" style="color:{fte_c};">{min_y1:.2f}–{max_y1:.2f}</div>
          <div class="kpi-detail">Volatility {fte_volatility*100:.0f}% <b style="color:{fte_c};">({fte_s})</b></div>
        </div>

        <div class="card" style="border-left-color:#666;">
          <div class="kpi-label">Flex Share</div>
          <div class="kpi-value">{flex_share*100:.1f}%</div>
          <div class="kpi-detail">Share of total provider-days (perm + flex)</div>
        </div>

        <div class="card" style="border-left-color:{'#e74c3c' if red_months>0 else '#27ae60'};">
          <div class="kpi-label">Red Months</div>
          <div class="kpi-value">{red_months}</div>
          <div class="kpi-detail">Months above red PPPD threshold</div>
        </div>

        <div class="card" style="border-left-color:{GOLD_MUTED};">
          <div class="kpi-label">Model Version</div>
          <div class="kpi-value" style="font-size:1.05rem;">{MODEL_VERSION}</div>
          <div class="kpi-detail">36-month horizon</div>
        </div>
      </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

# ============================================================
# CHARTS
# ============================================================
st.markdown("## Financial Projections")
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

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# TABLES
# ============================================================
st.markdown("## Detailed Results")
ledger = R["ledger"]
annual = R["annual_summary"]

tab1, tab2 = st.tabs(["Annual Summary", "Monthly Ledger"])
with tab1:
    st.dataframe(annual, hide_index=True, use_container_width=True)
with tab2:
    st.dataframe(ledger, hide_index=True, use_container_width=True, height=420)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div style="text-align:center; padding: 1.5rem 0; color:{GOLD_MUTED};">
  <div style="font-size:0.9rem; font-style:italic;">predict. perform. prosper.</div>
  <div style="font-size:0.75rem; margin-top:0.35rem; color:#999;">
    Bramhall Co. | Predictive Staffing Model v{MODEL_VERSION.split('-')[-1]}
  </div>
</div>
""",
    unsafe_allow_html=True,
)
