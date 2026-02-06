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
import streamlit.components.v1 as components
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

LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

# Seasonality helpers
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
# POSTURE MAPS
# ============================================================
POSTURE_LABEL = {
    1: "Very Lean",
    2: "Lean",
    3: "Balanced",
    4: "Safe",
    5: "Very Safe",
}
POSTURE_TEXT = {
    1: "Very Lean: Staff to ~80% of required (heavy flex/PRN). Highest risk.",
    2: "Lean: Staff to ~88% of required. Moderate flex reliance.",
    3: "Balanced: 100% of required. Standard buffer for normal variation.",
    4: "Safe: 104% coverage. Protects access/quality; higher SWB/visit.",
    5: "Very Safe: 108% coverage. Highest cost; maximum stability.",
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
# CSS (ALL CSS MUST LIVE INSIDE THIS STRING)
# ============================================================
BRAND_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:FILL,GRAD,opsz,wght@0,0,24,400');

html, body, p, div, label, li, input, textarea, button {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: #2c2c2c;
  line-height: 1.55;
}}

h1, h2, h3, h4, h5, h6 {{
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  color: #1a1a1a !important;
  letter-spacing: -0.02em;
  line-height: 1.25;
}}

.material-symbols-outlined,
[data-testid="stIconMaterial"] {{
  font-family: "Material Symbols Outlined" !important;
  font-variation-settings: "opsz" 24, "wght" 400, "FILL" 0, "GRAD" 0 !important;
  font-feature-settings: "liga" 1 !important;
  line-height: 1 !important;
  vertical-align: middle !important;
}}

.intro-container {{
  text-align: center;
  margin-bottom: 2rem;
  padding: 2.25rem 0 1.75rem 0;
  border-bottom: 1px solid #e8e8e8;
}}

.intro-tagline {{
  font-size: 0.92rem;
  color: {GOLD};
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-top: 0.6rem;
}}

.intro-logo {{
  height: 52px;
  width: 52px;
  opacity: 0.95;
}}

[data-testid="stMetricValue"] {{
  font-size: 2rem !important;
  font-weight: 600 !important;
  font-family: 'IBM Plex Mono', monospace !important;
  color: #1a1a1a !important;
}}

[data-testid="stMetricLabel"] {{
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: #666 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}}

[data-testid="stMetricDelta"] {{
  font-size: 0.85rem !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 2rem;
  border-bottom: 1px solid #e0e0e0;
}}

.stTabs [data-baseweb="tab"] {{
  padding: 0.75rem 0;
  font-weight: 500;
  color: #666;
  border-bottom: 2px solid transparent;
}}

.stTabs [aria-selected="true"] {{
  color: {GOLD};
  border-bottom-color: {GOLD};
}}

.divider {{
  height: 1px;
  background: #e8e8e8;
  margin: 2rem 0;
}}

.scorecard-hero {{
  background: white;
  border: 1px solid #eee;
  border-left: 6px solid {GOLD};
  border-radius: 10px;
  padding: 1.25rem 1.25rem 1rem 1.25rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}

.scorecard-title {{
  font-weight: 650;
  color: {GOLD};
  letter-spacing: 0.02em;
  margin-bottom: 1rem;
}}

.metrics-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
}}

.metric-card {{
  border: 1px solid #eee;
  border-left: 5px solid {LIGHT_GOLD};
  border-radius: 10px;
  padding: 0.9rem 0.95rem;
  background: #fff;
}}

.metric-label {{
  font-size: 0.72rem;
  font-weight: 650;
  color: #777;
  text-transform: uppercase;
  letter-spacing: 0.09em;
}}

.metric-value {{
  margin-top: 0.35rem;
  font-size: 1.55rem;
  font-family: 'IBM Plex Mono', monospace;
  font-weight: 600;
  color: #111;
}}

.metric-detail {{
  margin-top: 0.35rem;
  font-size: 0.82rem;
  color: #666;
  line-height: 1.35;
}}

@media (max-width: 1200px) {{
  .metrics-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}

@media (max-width: 650px) {{
  .metrics-grid {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}
</style>
"""
st.markdown(BRAND_CSS, unsafe_allow_html=True)

# ============================================================
# JS (ALL JS MUST LIVE INSIDE components.html)
# ============================================================
components.html(
    """
<script>
function cleanArrows() {
  const expanders = document.querySelectorAll('[data-testid="stExpander"] summary');
  expanders.forEach(summary => {
    const walker = document.createTreeWalker(summary, NodeFilter.SHOW_TEXT);
    const toRemove = [];
    let node;
    while (node = walker.nextNode()) {
      const t = (node.textContent || "").trim();
      if (t.includes('_arrow_right') || t.includes('_arrow') || t.startsWith('_')) {
        toRemove.push(node);
      }
    }
    toRemove.forEach(n => n.parentNode && n.parentNode.removeChild(n));
  });
}
cleanArrows();
const obs = new MutationObserver(() => setTimeout(cleanArrows, 80));
obs.observe(document.body, { childList: true, subtree: true });
</script>
""",
    height=0,
)

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
<div class='intro-text'>
  <h2>Predictive Staffing Model</h2>
  <p class='intro-tagline'>Predict. Perform. Prosper.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

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
# SIMULATION ENGINE
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

    # Use avg year-2 volume to estimate role mix
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

        # attrition
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

            # Look forward 6 months from arrival and staff to the peak target
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

        # Pre-flex load
        pde_perm = provider_day_equiv_from_fte(p_eff[i], params.hours_week, params.fte_hours_week)
        load_pre[i] = v_pk[i] / max(pde_perm, 1e-6)

        # Post-flex load
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
                "Load PPPD (pre-flex)": float(load_pre[i]),
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
    pk_load_post = float(np.max(load_post))
    pk_load_pre = float(np.max(load_pre))

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
        "peak_load_post": pk_load_post,
        "peak_load_pre": pk_load_pre,
        "ebitda_proxy_annual": float(ebitda_ann),
        "score": float(score),
        "ledger": ledger.drop(columns=["Year"]),
        "annual_summary": annual,
        "target_policy": list(tgt_pol),
    }

# ============================================================
# CACHED SIMULATION
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
            <div style='background: white; padding: 1.25rem; border-radius: 10px;
                        border: 1px solid {LIGHT_GOLD}; margin-bottom: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);'>
                <div style='font-weight: 650; font-size: 1rem; color: {GOLD}; margin-bottom: 0.6rem;'>
                    Intelligent Cost-Driven Staffing
                </div>
                <div style='font-size: 0.85rem; color: #666; line-height: 1.55;'>
                    Peak-aware planning with hiring runway and optional posting freezes.
                    Use <strong>Suggest Optimal</strong> to match your SWB/visit target.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.checkbox("Seeing errors? Clear cache", value=False):
            st.cache_data.clear()
            st.success("Cache cleared.")
            st.rerun()

        st.markdown(f"<h3 style='color:{GOLD}; margin-top:0.25rem;'>Core Settings</h3>", unsafe_allow_html=True)
        visits = st.number_input("Average Visits/Day", 1.0, value=36.0, step=1.0)
        annual_growth = st.number_input("Annual Growth %", 0.0, value=10.0, step=1.0) / 100.0

        c1, c2 = st.columns(2)
        with c1:
            seasonality_pct = st.number_input("Seasonality %", 0.0, value=20.0, step=5.0) / 100.0
        with c2:
            peak_factor = st.number_input("Peak Factor", 1.0, value=1.0, step=0.1)

        annual_turnover = st.number_input("Turnover %", 0.0, value=16.0, step=1.0) / 100.0

        st.markdown("#### Current Staffing")
        use_current_state = st.checkbox(
            "Start from actual current FTE (not policy target)",
            value=False,
        )
        current_fte = None
        if use_current_state:
            current_fte = st.number_input("Current FTE on Staff", 0.1, value=3.5, step=0.1)

        st.markdown("**Lead Times**")
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

            st.markdown("**Workforce Ratios (for Workforce Planning)**")
            supervisor_ratio = st.number_input("Providers per Supervisor", 1.0, value=5.0, step=0.5)
            physician_supervisor_ratio = st.number_input("Providers per Physician Supervisor", 1.0, value=10.0, step=1.0)
            rt_fte_fixed = st.number_input("X-Ray Tech FTE (Fixed)", 0.0, value=1.0, step=0.25)

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

        # Defaults if advanced sections never opened (Streamlit still defines them, but keep safe)
        locals_ = locals()

        # Staffing risk posture + policy
        st.markdown(f"<div style='height: 6px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:{GOLD};'>🏛 Staffing Risk Posture</h3>", unsafe_allow_html=True)
        risk_posture = st.select_slider(
            "Lean ↔ Safe",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: POSTURE_LABEL.get(x, str(x)),
        )
        st.caption(POSTURE_TEXT[int(risk_posture)])

        if "target_utilization" not in st.session_state:
            st.session_state.target_utilization = 92

        st.markdown(f"<h3 style='color:{GOLD};'>Smart Staffing Policy</h3>", unsafe_allow_html=True)
        target_utilization = st.slider("Target Utilization %", 80, 98, value=int(st.session_state.target_utilization), step=2)
        winter_buffer_pct = st.slider("Winter Buffer %", 0, 10, 3, 1) / 100.0

        base_coverage_from_util = 1.0 / (target_utilization / 100.0)
        posture_mult = POSTURE_BASE_COVERAGE_MULT[int(risk_posture)]
        posture_winter_add = POSTURE_WINTER_BUFFER_ADD[int(risk_posture)]
        base_coverage_pct = base_coverage_from_util * posture_mult
        winter_coverage_pct = base_coverage_pct * (1 + winter_buffer_pct + posture_winter_add)

        # Flex cap adjusted by posture
        flex_max_fte_effective = float(locals_.get("flex_max_fte_per_month", 2.0)) * POSTURE_FLEX_CAP_MULT[int(risk_posture)]

        # Suggest Optimal
        if st.button("Suggest Optimal", use_container_width=True):
            hourly_rates_temp = {
                "physician": locals_.get("physician_hr", 135.79),
                "apc": locals_.get("apc_hr", 62.0),
                "ma": locals_.get("ma_hr", 24.14),
                "psr": locals_.get("psr_hr", 21.23),
                "rt": locals_.get("rt_hr", 31.36),
                "supervisor": locals_.get("supervisor_hr", 28.25),
            }
            params_temp = ModelParams(
                visits=visits,
                annual_growth=annual_growth,
                seasonality_pct=seasonality_pct,
                flu_uplift_pct=locals_.get("flu_uplift_pct", 0.0),
                flu_months=locals_.get("flu_months_set", set()),
                peak_factor=peak_factor,
                visits_per_provider_hour=locals_.get("visits_per_provider_hour", 3.0),
                hours_week=locals_.get("hours_week", 84.0),
                days_open_per_week=locals_.get("days_open_per_week", 7.0),
                fte_hours_week=locals_.get("fte_hours_week", 36.0),
                annual_turnover=annual_turnover,
                turnover_notice_days=turnover_notice_days,
                hiring_runway_days=hiring_runway_days,
                ramp_months=locals_.get("ramp_months", 1),
                ramp_productivity=locals_.get("ramp_productivity", 0.75),
                fill_probability=locals_.get("fill_probability", 0.85),
                winter_anchor_month=locals_.get("winter_anchor_month_num", 12),
                winter_end_month=locals_.get("winter_end_month_num", 2),
                freeze_months=locals_.get("freeze_months_set", set()),
                budgeted_pppd=locals_.get("budgeted_pppd", 36.0),
                yellow_max_pppd=locals_.get("yellow_max_pppd", 42.0),
                red_start_pppd=locals_.get("red_start_pppd", 45.0),
                flex_max_fte_per_month=flex_max_fte_effective,
                flex_cost_multiplier=locals_.get("flex_cost_multiplier", 1.25),
                target_swb_per_visit=locals_.get("target_swb_per_visit", 85.0),
                swb_tolerance=locals_.get("swb_tolerance", 2.0),
                net_revenue_per_visit=locals_.get("net_revenue_per_visit", 140.0),
                visits_lost_per_provider_day_gap=locals_.get("visits_lost_per_provider_day_gap", 18.0),
                provider_replacement_cost=locals_.get("provider_replacement_cost", 75000.0),
                turnover_yellow_mult=locals_.get("turnover_yellow_mult", 1.3),
                turnover_red_mult=locals_.get("turnover_red_mult", 2.0),
                hourly_rates=hourly_rates_temp,
                benefits_load_pct=locals_.get("benefits_load_pct", 0.30),
                ot_sick_pct=locals_.get("ot_sick_pct", 0.04),
                bonus_pct=locals_.get("bonus_pct", 0.10),
                physician_supervision_hours_per_month=locals_.get("physician_supervision_hours_per_month", 0.0),
                supervisor_hours_per_month=locals_.get("supervisor_hours_per_month", 0.0),
                min_perm_providers_per_day=locals_.get("min_perm_providers_per_day", 1.0),
                min_permanent_fte=locals_.get("min_permanent_fte", 2.33),
                allow_prn_override=locals_.get("allow_prn_override", False),
                require_perm_under_green_no_flex=locals_.get("require_perm_under_green_no_flex", True),
                _v=MODEL_VERSION,
            )

            best_util = 90
            best_diff = 1e18
            results_cache: Dict[int, float] = {}
            for test_util in range(86, 99, 2):
                test_cov = 1.0 / (test_util / 100.0)
                test_winter = test_cov * (1 + winter_buffer_pct + posture_winter_add)
                test_policy = Policy(base_coverage_pct=test_cov, winter_coverage_pct=test_winter)
                test_result = simulate_policy(params_temp, test_policy, current_fte)
                test_swb = float(test_result["annual_swb_per_visit"])
                results_cache[test_util] = test_swb
                diff = abs(test_swb - locals_.get("target_swb_per_visit", 85.0))
                if diff < best_diff:
                    best_util, best_diff = test_util, diff

            st.session_state.target_utilization = best_util
            st.success(f"Best Match: {best_util}% utilization (SWB/visit ≈ ${results_cache[best_util]:.2f})")
            st.rerun()

        run_simulation = st.button("Run Simulation", use_container_width=True, type="primary")

        hourly_rates = {
            "physician": locals_.get("physician_hr", 135.79),
            "apc": locals_.get("apc_hr", 62.0),
            "ma": locals_.get("ma_hr", 24.14),
            "psr": locals_.get("psr_hr", 21.23),
            "rt": locals_.get("rt_hr", 31.36),
            "supervisor": locals_.get("supervisor_hr", 28.25),
        }

        params = ModelParams(
            visits=visits,
            annual_growth=annual_growth,
            seasonality_pct=seasonality_pct,
            flu_uplift_pct=locals_.get("flu_uplift_pct", 0.0),
            flu_months=locals_.get("flu_months_set", set()),
            peak_factor=peak_factor,
            visits_per_provider_hour=locals_.get("visits_per_provider_hour", 3.0),
            hours_week=locals_.get("hours_week", 84.0),
            days_open_per_week=locals_.get("days_open_per_week", 7.0),
            fte_hours_week=locals_.get("fte_hours_week", 36.0),
            annual_turnover=annual_turnover,
            turnover_notice_days=turnover_notice_days,
            hiring_runway_days=hiring_runway_days,
            ramp_months=locals_.get("ramp_months", 1),
            ramp_productivity=locals_.get("ramp_productivity", 0.75),
            fill_probability=locals_.get("fill_probability", 0.85),
            winter_anchor_month=locals_.get("winter_anchor_month_num", 12),
            winter_end_month=locals_.get("winter_end_month_num", 2),
            freeze_months=locals_.get("freeze_months_set", set()),
            budgeted_pppd=locals_.get("budgeted_pppd", 36.0),
            yellow_max_pppd=locals_.get("yellow_max_pppd", 42.0),
            red_start_pppd=locals_.get("red_start_pppd", 45.0),
            flex_max_fte_per_month=flex_max_fte_effective,
            flex_cost_multiplier=locals_.get("flex_cost_multiplier", 1.25),
            target_swb_per_visit=locals_.get("target_swb_per_visit", 85.0),
            swb_tolerance=locals_.get("swb_tolerance", 2.0),
            net_revenue_per_visit=locals_.get("net_revenue_per_visit", 140.0),
            visits_lost_per_provider_day_gap=locals_.get("visits_lost_per_provider_day_gap", 18.0),
            provider_replacement_cost=locals_.get("provider_replacement_cost", 75000.0),
            turnover_yellow_mult=locals_.get("turnover_yellow_mult", 1.3),
            turnover_red_mult=locals_.get("turnover_red_mult", 2.0),
            hourly_rates=hourly_rates,
            benefits_load_pct=locals_.get("benefits_load_pct", 0.30),
            ot_sick_pct=locals_.get("ot_sick_pct", 0.04),
            bonus_pct=locals_.get("bonus_pct", 0.10),
            physician_supervision_hours_per_month=locals_.get("physician_supervision_hours_per_month", 0.0),
            supervisor_hours_per_month=locals_.get("supervisor_hours_per_month", 0.0),
            min_perm_providers_per_day=locals_.get("min_perm_providers_per_day", 1.0),
            min_permanent_fte=locals_.get("min_permanent_fte", 2.33),
            allow_prn_override=locals_.get("allow_prn_override", False),
            require_perm_under_green_no_flex=locals_.get("require_perm_under_green_no_flex", True),
            _v=MODEL_VERSION,
        )

        policy = Policy(base_coverage_pct=float(base_coverage_pct), winter_coverage_pct=float(winter_coverage_pct))

        ui = {
            "target_utilization": int(target_utilization),
            "winter_buffer_pct": float(winter_buffer_pct),
            "risk_posture": int(risk_posture),
            "base_coverage_from_util": float(base_coverage_from_util),
            "flex_max_fte_per_month": float(locals_.get("flex_max_fte_per_month", 2.0)),
            "current_fte": current_fte,
            "supervisor_ratio": float(locals_.get("supervisor_ratio", 5.0)),
            "physician_supervisor_ratio": float(locals_.get("physician_supervisor_ratio", 10.0)),
            "rt_fte_fixed": float(locals_.get("rt_fte_fixed", 1.0)),
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
    st.success("Simulation complete.")
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
    peak_load_post = float(res["peak_load_post"])
    peak_load_pre = float(res["peak_load_pre"])
    margin = float(res["ebitda_proxy_annual"])
    return {
        "SWB/Visit (Y1)": y1_swb,
        "EBITDA Proxy (Y1)": y1_ebitda,
        "EBITDA Proxy (3yr total)": margin,
        "Red Months": red_months,
        "Flex Share": flex_share,
        "Peak Load (PPPD)": peak_load_post,
        "Peak Load Pre-Flex": peak_load_pre,
    }

def build_policy_for_posture(posture_level: int) -> Tuple[ModelParams, Policy]:
    posture_mult = POSTURE_BASE_COVERAGE_MULT[int(posture_level)]
    posture_winter_add = POSTURE_WINTER_BUFFER_ADD[int(posture_level)]
    posture_flex_mult = POSTURE_FLEX_CAP_MULT[int(posture_level)]

    base_cov = ui["base_coverage_from_util"] * posture_mult
    winter_cov = base_cov * (1 + ui["winter_buffer_pct"] + posture_winter_add)

    params_alt = ModelParams(**{**params.__dict__, "flex_max_fte_per_month": float(ui["flex_max_fte_per_month"] * posture_flex_mult)})
    pol_alt = Policy(base_coverage_pct=float(base_cov), winter_coverage_pct=float(winter_cov))
    return params_alt, pol_alt

# ============================================================
# LEAN VS SAFE TRADEOFFS
# ============================================================
st.markdown("## Lean vs. Safe Tradeoffs")
st.markdown("**Executive decision framework**")

lean_params, lean_policy = build_policy_for_posture(2)
safe_params, safe_policy = build_policy_for_posture(4)

with st.spinner("Comparing Lean vs Safe..."):
    R_lean = simulate_policy(lean_params, lean_policy, current_fte)
    R_safe = simulate_policy(safe_params, safe_policy, current_fte)

m_cur = pack_exec_metrics(R)
m_lean = pack_exec_metrics(R_lean)
m_safe = pack_exec_metrics(R_safe)

df_exec = pd.DataFrame([
    {"Scenario": "Lean", **m_lean},
    {"Scenario": "Current", **m_cur},
    {"Scenario": "Safe", **m_safe},
])

def fmt_money(x): return f"${x:,.0f}"
def fmt_money2(x): return f"${x:,.2f}"
def fmt_pct(x): return f"{x*100:.1f}%"
def fmt_num1(x): return f"{x:.1f}"
def delta(a, b): return b - a

st.dataframe(
    df_exec.style.format({
        "SWB/Visit (Y1)": fmt_money2,
        "EBITDA Proxy (Y1)": fmt_money,
        "EBITDA Proxy (3yr total)": fmt_money,
        "Flex Share": fmt_pct,
        "Peak Load (PPPD)": fmt_num1,
        "Peak Load Pre-Flex": fmt_num1,
    }),
    hide_index=True,
    use_container_width=True,
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Lean → Current")
    st.write(f"- SWB/Visit: {fmt_money2(m_lean['SWB/Visit (Y1)'])} → {fmt_money2(m_cur['SWB/Visit (Y1)'])} (Δ {fmt_money2(delta(m_lean['SWB/Visit (Y1)'], m_cur['SWB/Visit (Y1)']))})")
    st.write(f"- Red Months: {m_lean['Red Months']} → {m_cur['Red Months']} (Δ {delta(m_lean['Red Months'], m_cur['Red Months']):+d})")
    st.write(f"- Flex Share: {fmt_pct(m_lean['Flex Share'])} → {fmt_pct(m_cur['Flex Share'])} (Δ {fmt_pct(delta(m_lean['Flex Share'], m_cur['Flex Share']))})")
    st.write(f"- 3yr EBITDA Proxy: {fmt_money(m_lean['EBITDA Proxy (3yr total)'])} → {fmt_money(m_cur['EBITDA Proxy (3yr total)'])} (Δ {fmt_money(delta(m_lean['EBITDA Proxy (3yr total)'], m_cur['EBITDA Proxy (3yr total)']))})")
with c2:
    st.markdown("### Current → Safe")
    st.write(f"- SWB/Visit: {fmt_money2(m_cur['SWB/Visit (Y1)'])} → {fmt_money2(m_safe['SWB/Visit (Y1)'])} (Δ {fmt_money2(delta(m_cur['SWB/Visit (Y1)'], m_safe['SWB/Visit (Y1)']))})")
    st.write(f"- Red Months: {m_cur['Red Months']} → {m_safe['Red Months']} (Δ {delta(m_cur['Red Months'], m_safe['Red Months']):+d})")
    st.write(f"- Flex Share: {fmt_pct(m_cur['Flex Share'])} → {fmt_pct(m_safe['Flex Share'])} (Δ {fmt_pct(delta(m_cur['Flex Share'], m_safe['Flex Share']))})")
    st.write(f"- 3yr EBITDA Proxy: {fmt_money(m_cur['EBITDA Proxy (3yr total)'])} → {fmt_money(m_safe['EBITDA Proxy (3yr total)'])} (Δ {fmt_money(delta(m_cur['EBITDA Proxy (3yr total)'], m_safe['EBITDA Proxy (3yr total)']))})")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# POLICY HEALTH CHECK
# ============================================================
st.markdown("## Policy Health Check")
st.markdown("**Staffing stability validation**")

annual = R["annual_summary"]
if annual is None or len(annual) < 2:
    st.info("Need at least 2 years of annual output to run drift check.")
else:
    mins = [float(x) for x in annual["Min_Perm_Paid_FTE"].tolist()]
    years = [int(y) for y in annual["Year"].tolist()]
    drifts = [mins[i] - mins[i - 1] for i in range(1, len(mins))]

    DRIFT_OK = 0.20
    DRIFT_WARN = 0.35
    worst = max(abs(d) for d in drifts)

    chain = f"{years[0]}: {mins[0]:.2f} FTE"
    for i in range(1, len(mins)):
        chain += f" → {years[i]}: {mins[i]:.2f} (Δ{drifts[i-1]:+0.2f})"

    if worst <= DRIFT_OK:
        st.success(f"**No Ratchet Detected**\n\n{chain}")
    elif worst <= DRIFT_WARN:
        st.warning(f"**Minor Drift Detected**\n\n{chain}")
    else:
        st.error(
            f"**Ratchet Risk Detected**\n\n{chain}\n\n"
            f"Try: lower turnover, raise fill probability, shorten runway, or use winter buffer instead of lifting base."
        )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# HERO SCORECARD
# ============================================================
annual = R["annual_summary"]
ledger = R["ledger"]

swb_y1 = float(annual.loc[0, "SWB_per_Visit"])
ebitda_y1 = float(annual.loc[0, "EBITDA_Proxy"])
ebitda_y3 = float(annual.loc[len(annual) - 1, "EBITDA_Proxy"])
util_y1 = float(annual.loc[0, "Avg_Utilization"])
util_y3 = float(annual.loc[len(annual) - 1, "Avg_Utilization"])
min_y1 = float(annual.loc[0, "Min_Perm_Paid_FTE"])
max_y1 = float(annual.loc[0, "Max_Perm_Paid_FTE"])

peak_load_post = float(R["peak_load_post"])
peak_load_pre = float(R["peak_load_pre"])

# Color coding
swb_target = params.target_swb_per_visit
swb_tol = params.swb_tolerance
swb_diff = abs(swb_y1 - swb_target)
if swb_diff <= swb_tol:
    swb_color, swb_status = "#27ae60", "On Target"
elif swb_diff <= swb_tol * 2:
    swb_color, swb_status = "#f39c12", "Close"
else:
    swb_color, swb_status = "#e74c3c", "Off Target"

target_util = ui["target_utilization"]
util_diff = abs(util_y1 * 100 - target_util)
if util_diff <= 2:
    util_color, util_status = "#27ae60", "On Target"
elif util_diff <= 5:
    util_color, util_status = "#f39c12", "Close"
else:
    util_color, util_status = "#e74c3c", "Off Target"

if peak_load_pre <= params.budgeted_pppd:
    peak_color, peak_status = "#27ae60", "Green"
elif peak_load_pre <= params.yellow_max_pppd:
    peak_color, peak_status = "#f1c40f", "Yellow"
elif peak_load_pre <= params.red_start_pppd:
    peak_color, peak_status = "#f39c12", "Orange"
else:
    peak_color, peak_status = "#e74c3c", "Red"

if ebitda_y1 > 0:
    ebitda_color, ebitda_status = "#27ae60", "Positive"
elif ebitda_y1 > -50000:
    ebitda_color, ebitda_status = "#f39c12", "Marginal"
else:
    ebitda_color, ebitda_status = "#e74c3c", "Negative"

fte_range = max_y1 - min_y1
fte_avg = (max_y1 + min_y1) / 2.0
fte_vol = fte_range / max(fte_avg, 1.0)
if fte_vol <= 0.10:
    fte_color, fte_status = "#27ae60", "Stable"
elif fte_vol <= 0.20:
    fte_color, fte_status = "#f1c40f", "Moderate"
else:
    fte_color, fte_status = "#e74c3c", "Volatile"

months_red = int(R["months_red"])
risk_posture = int(ui["risk_posture"])
if risk_posture <= 2 and months_red > 3:
    risk_color, burnout_risk = "#e74c3c", "High"
elif risk_posture <= 3 and months_red > 3:
    risk_color, burnout_risk = "#f39c12", "Moderate"
else:
    risk_color, burnout_risk = "#27ae60", "Low"

st.markdown(
    f"""
<div class="scorecard-hero">
  <div class="scorecard-title">Policy Performance Scorecard</div>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Staffing Policy</div>
      <div class="metric-value">{ui["target_utilization"]:.0f}% Target</div>
      <div class="metric-detail">
        Coverage: {policy.base_coverage_pct*100:.0f}% base / {policy.winter_coverage_pct*100:.0f}% winter<br>
        Posture: {POSTURE_LABEL[risk_posture]}
      </div>
    </div>

    <div class="metric-card" style="border-left-color:{swb_color};">
      <div class="metric-label">SWB / Visit (Y1)</div>
      <div class="metric-value" style="color:{swb_color};">${swb_y1:.2f}</div>
      <div class="metric-detail">Target ${swb_target:.0f} ± ${swb_tol:.0f} — <b style="color:{swb_color};">{swb_status}</b></div>
    </div>

    <div class="metric-card" style="border-left-color:{ebitda_color};">
      <div class="metric-label">EBITDA Proxy</div>
      <div class="metric-value" style="color:{ebitda_color};">${ebitda_y1/1000:.0f}K</div>
      <div class="metric-detail">Year 1 / Year {len(annual)}: ${ebitda_y3/1000:.0f}K — <b style="color:{ebitda_color};">{ebitda_status}</b></div>
    </div>

    <div class="metric-card" style="border-left-color:{util_color};">
      <div class="metric-label">Utilization</div>
      <div class="metric-value" style="color:{util_color};">{util_y1*100:.0f}%</div>
      <div class="metric-detail">Target {target_util}% — <b style="color:{util_color};">{util_status}</b><br>Year {len(annual)}: {util_y3*100:.0f}%</div>
    </div>

    <div class="metric-card" style="border-left-color:{fte_color};">
      <div class="metric-label">FTE Range (Y1)</div>
      <div class="metric-value" style="color:{fte_color};">{min_y1:.1f}-{max_y1:.1f}</div>
      <div class="metric-detail">Volatility {fte_vol*100:.0f}% — <b style="color:{fte_color};">{fte_status}</b></div>
    </div>

    <div class="metric-card" style="border-left-color:{peak_color};">
      <div class="metric-label">Peak Load (Pre-Flex)</div>
      <div class="metric-value" style="color:{peak_color};">{peak_load_pre:.1f}</div>
      <div class="metric-detail">PPPD — <b style="color:{peak_color};">{peak_status}</b><br>Post-flex: {peak_load_post:.1f}</div>
    </div>

    <div class="metric-card" style="border-left-color:{risk_color};">
      <div class="metric-label">Burnout Risk</div>
      <div class="metric-value" style="color:{risk_color};">{burnout_risk}</div>
      <div class="metric-detail">{months_red} red months / 36</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# HIRING INSIGHTS + EXPORT
# ============================================================
st.markdown("## Hiring Insights")
st.markdown("**Actionable recruitment roadmap**")

upcoming_hires = ledger[ledger["Hires Visible (FTE)"] > 0.05].head(12)

if len(upcoming_hires) == 0:
    st.info("No hiring actions in the next 12 months under current settings.")
else:
    total_hires_12mo = float(upcoming_hires["Hires Visible (FTE)"].sum())
    avg_fte = float(ledger.head(12)["Permanent FTE (Paid)"].mean())

    st.markdown("### Actionable Hiring Plan (Rolling 90-day windows)")

    hire_events = []
    for _, row in upcoming_hires.iterrows():
        month_date = pd.to_datetime(row["Month"])
        hire_events.append({
            "date": month_date,
            "fte": float(row["Hires Visible (FTE)"]),
            "reason": str(row["Hire Reason"]),
            "month_label": str(row["Month"]),
        })

    hiring_actions = []
    i = 0
    while i < len(hire_events):
        window_start = hire_events[i]["date"]
        window_end = window_start + pd.Timedelta(days=90)

        window_fte = 0.0
        window_months = []
        window_reasons = []
        j = i
        while j < len(hire_events) and hire_events[j]["date"] < window_end:
            window_fte += hire_events[j]["fte"]
            window_months.append(hire_events[j]["month_label"])
            window_reasons.append(hire_events[j]["reason"])
            j += 1

        post_by_date = hire_events[i]["date"] - pd.Timedelta(days=int(params.hiring_runway_days))
        today = pd.Timestamp.today()
        days_until_post = int((post_by_date - today).days)

        if days_until_post < 0:
            urgency = "🔴 OVERDUE"
        elif days_until_post <= 30:
            urgency = "🟠 POST NOW"
        elif days_until_post <= 60:
            urgency = "🟡 SOON"
        else:
            urgency = "🟢 PLAN"

        # Action label
        if window_fte >= 0.75:
            num_full = int(window_fte + 0.001)
            remainder = window_fte - num_full
            if remainder >= 0.75:
                hire_action = f"Hire {num_full + 1} provider(s)"
            elif remainder >= 0.25:
                hire_action = f"Hire {num_full} + 0.5 FTE"
            else:
                hire_action = f"Hire {max(num_full,1)} provider(s)"
        elif window_fte >= 0.25:
            hire_action = "Hire 0.5 FTE"
        else:
            hire_action = "Monitor (fractional)"

        period = window_months[0] if len(window_months) == 1 else f"{window_months[0]} – {window_months[-1]}"
        primary_reason = window_reasons[0].split("→")[-1].split(".")[0].strip() if window_reasons else "Staffing need"

        hiring_actions.append({
            "Post Req By": post_by_date.strftime("%b %Y"),
            "Urgency": urgency,
            "Arrival Window": period,
            "Hiring Action": hire_action,
            "FTE": f"{window_fte:.2f}",
            "Purpose": primary_reason,
            "_days_until": days_until_post,
        })
        i = j

    df_actions = pd.DataFrame(hiring_actions)

    urgent = df_actions[df_actions["_days_until"] <= 30]
    overdue = df_actions[df_actions["_days_until"] < 0]
    if len(overdue) > 0:
        st.error(f"🔴 URGENT: {len(overdue)} overdue posting(s).")
    elif len(urgent) > 0:
        st.warning(f"🟠 ACTION: {len(urgent)} posting(s) due within 30 days.")
    else:
        st.success("No urgent postings. All actions have sufficient lead time.")

    def highlight(row):
        u = row["Urgency"]
        if "OVERDUE" in u:
            return ["background-color: #fadbd8"] * len(row)
        if "POST NOW" in u:
            return ["background-color: #fdebd0"] * len(row)
        if "SOON" in u:
            return ["background-color: #fef5e7"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_actions[["Post Req By", "Urgency", "Arrival Window", "Hiring Action", "FTE", "Purpose"]].style.apply(highlight, axis=1),
        hide_index=True,
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Hiring Actions", f"{len(df_actions)}")
    with c2:
        st.metric("~Full-Time Hires", f"{int(total_hires_12mo)}")
    with c3:
        hiring_intensity = (total_hires_12mo / max(avg_fte, 1e-9)) * 100
        st.metric("Hiring Intensity", f"{hiring_intensity:.0f}%")

    st.markdown("### Export")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Hiring Action Plan (CSV)",
            df_actions.drop(columns=["_days_until"]).to_csv(index=False).encode("utf-8"),
            "hiring_actions.csv",
            "text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Detailed Monthly Plan (CSV)",
            upcoming_hires[["Month", "Hires Visible (FTE)", "Hire Reason"]].to_csv(index=False).encode("utf-8"),
            "hiring_plan_detailed.csv",
            "text/csv",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "Full Staffing Ledger (CSV)",
            ledger.to_csv(index=False).encode("utf-8"),
            "staffing_ledger_full.csv",
            "text/csv",
            use_container_width=True,
        )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# WORKFORCE PLANNING (ALL POSITIONS)
# ============================================================
st.markdown("## Workforce Planning")
st.markdown("**Comprehensive position requirements**")

with st.expander("View Complete Staffing Plan (All Positions)", expanded=False):
    dates = R["dates"]
    perm_eff = R["perm_eff"]
    flex_fte = [float(ledger.loc[i, "Flex FTE Used"]) for i in range(len(ledger))]
    v_av = [float(ledger.loc[i, "Visits/Day (avg)"]) for i in range(len(ledger))]

    y1_avg_visits = float(ledger.head(12)["Visits/Day (avg)"].mean())
    role_mix = compute_role_mix_ratios(y1_avg_visits, model)

    position_rows = []
    for i in range(len(dates)):
        month_label = dates[i].strftime("%Y-%b")
        provider_perm = float(perm_eff[i])
        provider_flex = float(flex_fte[i])
        provider_total = provider_perm + provider_flex

        psr_fte = provider_total * role_mix["psr_per_provider"]
        ma_fte = provider_total * role_mix["ma_per_provider"]
        xrt_fte = float(ui.get("rt_fte_fixed", 1.0))

        supervisor_ratio = float(ui.get("supervisor_ratio", 5.0))
        physician_supervisor_ratio = float(ui.get("physician_supervisor_ratio", 10.0))
        supervisor_fte = max(0.5, provider_total / max(supervisor_ratio, 1e-9))
        physician_supervisor_fte = max(0.25, provider_total / max(physician_supervisor_ratio, 1e-9))

        total_workforce = provider_total + psr_fte + ma_fte + xrt_fte + supervisor_fte + physician_supervisor_fte

        position_rows.append({
            "Month": month_label,
            "Visits/Day": float(v_av[i]),
            "Provider (Permanent)": provider_perm,
            "Provider (Flex/PRN)": provider_flex,
            "Provider (Total)": provider_total,
            "PSR": psr_fte,
            "MA": ma_fte,
            "X-Ray Tech": xrt_fte,
            "Supervisor": supervisor_fte,
            "Physician Supervisor": physician_supervisor_fte,
            "Total Workforce": total_workforce,
        })

    df_positions = pd.DataFrame(position_rows)
    df_positions["Year"] = df_positions["Month"].str[:4].astype(int)

    st.dataframe(
        df_positions.drop(columns=["Year"]).style.format("{:.2f}", subset=[c for c in df_positions.columns if c not in ["Month"]]),
        hide_index=True,
        use_container_width=True,
        height=520,
    )

    annual_positions = df_positions.groupby("Year").mean(numeric_only=True).round(2)
    st.markdown("### Annual Averages")
    st.dataframe(annual_positions, use_container_width=True)

    st.download_button(
        "Export Workforce Plan (CSV)",
        df_positions.drop(columns=["Year"]).to_csv(index=False).encode("utf-8"),
        "workforce_positions_36mo.csv",
        "text/csv",
        use_container_width=True,
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# SCENARIO COMPARISON
# ============================================================
st.markdown("## Scenario Comparison")
st.markdown("**Compare outcomes under different assumptions**")

with st.expander("Run Scenario Analysis", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Base Case\n\nUtil: {ui['target_utilization']}%\nGrowth: {params.annual_growth*100:.0f}%\nTurnover: {params.annual_turnover*100:.0f}%")
    with col2:
        opt_growth = st.number_input("Optimistic Growth %", 0.0, value=15.0, step=1.0) / 100.0
        opt_turnover = st.number_input("Optimistic Turnover %", 0.0, value=12.0, step=1.0) / 100.0
        opt_util = st.slider("Optimistic Utilization %", 80, 98, 94, 2)
    with col3:
        cons_growth = st.number_input("Conservative Growth %", 0.0, value=5.0, step=1.0) / 100.0
        cons_turnover = st.number_input("Conservative Turnover %", 0.0, value=20.0, step=1.0) / 100.0
        cons_util = st.slider("Conservative Utilization %", 80, 98, 88, 2)

    if st.button("Run Scenario Comparison", use_container_width=True):
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

        df_scen = pd.DataFrame([
            {"Scenario": "Base", **base_metrics},
            {"Scenario": "Optimistic", **opt_metrics},
            {"Scenario": "Conservative", **cons_metrics},
        ])
        st.dataframe(df_scen, hide_index=True, use_container_width=True)
        st.download_button(
            "Download Scenario Comparison (CSV)",
            df_scen.to_csv(index=False).encode("utf-8"),
            "scenario_comparison.csv",
            "text/csv",
            use_container_width=True,
        )

# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================
with st.expander("Sensitivity Analysis", expanded=False):
    if st.button("Run Sensitivity Analysis", use_container_width=True):
        base_swb = float(R["annual_swb_per_visit"])
        base_ebitda = float(R["ebitda_proxy_annual"])

        turnover_tests = [0.12, 0.16, 0.20, 0.24]
        rows = []
        for turn in turnover_tests:
            test_params = ModelParams(**{**params.__dict__, "annual_turnover": float(turn)})
            test_result = simulate_policy(test_params, policy, current_fte)
            rows.append({
                "Turnover": float(turn),
                "SWB/Visit": float(test_result["annual_swb_per_visit"]),
                "EBITDA (3yr)": float(test_result["ebitda_proxy_annual"]),
                "Δ SWB": float(test_result["annual_swb_per_visit"]) - base_swb,
                "Δ EBITDA": float(test_result["ebitda_proxy_annual"]) - base_ebitda,
            })
        df_turn = pd.DataFrame(rows)
        st.dataframe(df_turn, hide_index=True, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# CHARTS
# ============================================================
st.markdown("## Financial Projections")
st.markdown("**Three-year staffing and performance outlook**")

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
fig1.update_layout(height=450, hovermode="x unified", paper_bgcolor="white", plot_bgcolor="rgba(250, 248, 243, 0.25)")
st.plotly_chart(fig1, use_container_width=True)

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(x=dates, y=util * 100, mode="lines+markers", name="Utilization %"), secondary_y=False)
fig2.add_trace(go.Scatter(x=dates, y=load_post, mode="lines+markers", name="Load PPPD"), secondary_y=True)
fig2.update_layout(height=450, hovermode="x unified", paper_bgcolor="white", plot_bgcolor="rgba(250, 248, 243, 0.25)")
st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# TABLES
# ============================================================
st.markdown("## Detailed Results")
annual = R["annual_summary"]
tab1, tab2 = st.tabs(["Annual Summary", "Monthly Ledger"])
with tab1:
    st.dataframe(annual, hide_index=True, use_container_width=True)
with tab2:
    st.dataframe(ledger, hide_index=True, use_container_width=True, height=420)

# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div style='text-align:center; padding: 1.75rem 0; color: {GOLD_MUTED};'>
  <div style='font-size: 0.9rem; font-style: italic;'>
    predict. perform. prosper.
  </div>
  <div style='font-size: 0.75rem; margin-top: 0.4rem; color:#999;'>
    Bramhall Co. | Predictive Staffing Model • {MODEL_VERSION}
  </div>
</div>
""",
    unsafe_allow_html=True,
)
