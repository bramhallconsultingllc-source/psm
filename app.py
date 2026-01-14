import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from collections import deque
from typing import List, Dict, Optional, Tuple, Any, Set
from psm.staffing_model import StaffingModel
from prophet import Prophet  # Requires pip install prophet
import plotly.graph_objects as go
import plotly.express as px
from joblib import Parallel, delayed  # For MC optimization

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM)", layout="wide")
st.markdown(
    """
    <style>
      .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Predictive Staffing Model (PSM)")
st.caption("Operations → Reality → Finance → Strategy → Decision")
st.info(
    "⚠️ **All staffing outputs round UP to the nearest 0.25 FTE.** "
    "This is intentional to prevent under-staffing."
)
model: StaffingModel = StaffingModel()

# Tabs for better UX
tab_ops, tab_real, tab_fin, tab_vvi, tab_strat, tab_exec = st.tabs(
    ["1) Operations", "2) Reality", "3) Finance", "3B) VVI Feasibility", "4) Strategy", "5-6) Executive Summary"]
)

# ============================================================
# STABLE TODAY
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today: datetime = st.session_state["today"]

# ============================================================
# SESSION STATE
# ============================================================
for key in ["model_ran", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# BRAND COLORS
# ============================================================
BRAND_BLACK: str = "#000000"
BRAND_GOLD: str = "#7a6200"
GRAY: str = "#B0B0B0"
LIGHT_GRAY: str = "#EAEAEA"
MID_GRAY: str = "#666666"

# ============================================================
# HELPERS
# ============================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))

def round_up_quarter(x: float) -> float:
    return math.ceil(float(x) * 4.0) / 4.0

def lead_days_to_months(days: int, avg_days_per_month: float = 30.4) -> int:
    return max(0, int(math.ceil(float(days) / float(avg_days_per_month))))

def shift_month(month: int, shift: int) -> int:
    return ((month - 1 + shift) % 12) + 1

def months_between(start_month: int, end_month: int) -> List[int]:
    """Wrapped month list inclusive. Example: Dec->Feb = [12,1,2]."""
    months: List[int] = []
    m: int = int(start_month)
    end_month: int = int(end_month)
    while True:
        months.append(m)
        if m == end_month:
            break
        m = 1 if m == 12 else m + 1
    return months

def month_range_label(months: List[int]) -> str:
    if not months:
        return "—"
    start: str = datetime(2000, int(months[0]), 1).strftime("%b")
    end: str = datetime(2000, int(months[-1]), 1).strftime("%b")
    return f"{start}–{end}" if start != end else start

def base_seasonality_multiplier(month: int) -> float:
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00

@st.cache_data
def compute_seasonality_forecast_multiyear(
    dates: pd.DatetimeIndex,
    baseline_visits: float,
    flu_months: List[int],
    flu_uplift_pct: float,
    hist_df: Optional[pd.DataFrame] = None
) -> List[float]:
    """
    Seasonality uses month-of-year only (repeatable cycle).
    Flu uplift applied to any month in flu_months.
    Normalize so mean(visits) equals baseline across full horizon.
    If historical data provided, use Prophet for forecasting.
    """
    if hist_df is not None and not hist_df.empty:
        hist_df = hist_df.rename(columns={'date': 'ds', 'visits': 'y'})
        hist_df['ds'] = pd.to_datetime(hist_df['ds'])
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name='flu', period=365.25, fourier_order=5, condition_name='flu_season')
        hist_df['flu_season'] = hist_df['ds'].dt.month.isin(flu_months)
        m.fit(hist_df)
        future = pd.DataFrame({'ds': dates})
        future['flu_season'] = future['ds'].dt.month.isin(flu_months)
        forecast = m.predict(future)
        return (forecast['yhat'] * (baseline_visits / forecast['yhat'].mean())).tolist()
    
    flu_set: Set[int] = set(flu_months or [])
    raw: List[float] = []
    for d in dates:
        mult: float = base_seasonality_multiplier(int(d.month))
        if int(d.month) in flu_set:
            mult *= (1.0 + float(flu_uplift_pct))
        raw.append(float(baseline_visits) * mult)
    avg_raw: float = float(np.mean(raw)) if len(raw) else float(baseline_visits)
    if avg_raw <= 0:
        return [float(baseline_visits) for _ in raw]
    return [v * (float(baseline_visits) / avg_raw) for v in raw]

def apply_visits_noise(visits_curve: List[float], cv: float) -> List[float]:
    cv = max(0.0, float(cv))
    if cv == 0.0:
        return list(map(float, visits_curve))
    sigma: float = math.sqrt(math.log(1.0 + cv ** 2))
    mu: float = -0.5 * sigma * sigma
    multipliers: np.ndarray = np.random.lognormal(mean=mu, sigma=sigma, size=len(visits_curve))
    return [float(v) * float(m) for v, m in zip(visits_curve, multipliers)]

def visits_to_provider_demand(
    model: StaffingModel,
    visits_by_month: List[float],
    hours_of_operation: float,
    fte_hours_per_week: float,
    provider_min_floor: float,
    acuity_scale: List[float] = None  # New: per-month acuity multiplier
) -> List[float]:
    demand: List[float] = []
    acuity = acuity_scale or [1.0] * len(visits_by_month)
    for v, a in zip(visits_by_month, acuity):
        adjusted_v = float(v) * float(a)
        fte = model.calculate_fte_needed(
            visits_per_day=adjusted_v,
            hours_of_operation_per_week=float(hours_of_operation),
            fte_hours_per_week=float(fte_hours_per_week),
        )["provider_fte"]
        demand.append(max(float(fte), float(provider_min_floor)))
    return demand

def burnout_protective_staffing_curve(
    visits_by_month: List[float],
    base_demand_fte: List[float],
    provider_min_floor: float,
    burnout_slider: float,
    weights: Tuple[float, float, float] = (0.40, 0.35, 0.25),
    safe_visits_per_provider_per_day: int = 20,
    smoothing_up: float = 0.50,
    smoothing_down: float = 0.25,
) -> List[float]:
    vol_w, spike_w, debt_w = weights
    visits_arr: np.ndarray = np.array(visits_by_month, dtype=float)
    mean_visits: float = float(np.mean(visits_arr)) if len(visits_arr) else 0.0
    std_visits: float = float(np.std(visits_arr)) if len(visits_arr) else 0.0
    cv: float = (std_visits / mean_visits) if mean_visits > 0 else 0.0
    p75: float = float(np.percentile(visits_arr, 75)) if len(visits_arr) else 0.0
    rdi: float = 0.0
    decay: float = 0.85
    lambda_debt: float = 0.10
    protective_curve: List[float] = []
    prev_staff: float = max(float(base_demand_fte[0]), float(provider_min_floor))
    for v, base_fte in zip(visits_by_month, base_demand_fte):
        v = float(v)
        base_fte = float(base_fte)
        vbuf: float = base_fte * cv
        sbuf: float = max(0.0, (v - p75) / mean_visits) * base_fte if mean_visits > 0 else 0.0
        visits_per_provider: float = v / max(prev_staff, 0.25)
        debt: float = max(0.0, visits_per_provider - float(safe_visits_per_provider_per_day))
        rdi = decay * rdi + debt
        dbuf: float = lambda_debt * rdi
        buffer_fte: float = float(burnout_slider) * (vol_w * vbuf + spike_w * sbuf + debt_w * dbuf)
        raw_target: float = max(float(provider_min_floor), base_fte + buffer_fte)
        delta: float = raw_target - prev_staff
        if delta > 0:
            delta = clamp(delta, 0.0, float(smoothing_up))
        else:
            delta = clamp(delta, -float(smoothing_down), 0.0)
        final_staff: float = max(float(provider_min_floor), prev_staff + delta)
        if protective_curve:
            final_staff = 0.7 * final_staff + 0.3 * protective_curve[-1]  # Added smoothing
        protective_curve.append(final_staff)
        prev_staff = final_staff
    return protective_curve

def typical_12_month_curve(dates_full: pd.DatetimeIndex, values_full: List[float]) -> List[float]:
    """Average multi-year values by month-of-year -> 12 values Jan..Dec."""
    df: pd.DataFrame = pd.DataFrame({"date": dates_full, "val": values_full})
    df["month"] = df["date"].dt.month
    g: pd.Series = df.groupby("month")["val"].mean()
    return [float(g.loc[m]) for m in range(1, 13)]

# ============================================================
# AUTO-FREEZE v3 (built from typical seasonal curve)
# ============================================================
def auto_freeze_strategy_v3_from_typical(
    dates_template_12: pd.DatetimeIndex,
    protective_typical_12: List[float],
    flu_start_month: int,
    flu_end_month: int,
    pipeline_lead_days: int,
    notice_days: int,
    freeze_buffer_months: int = 1,
) -> Dict[str, Any]:
    lead_months: int = lead_days_to_months(pipeline_lead_days)
    notice_months: int = lead_days_to_months(notice_days)
    independent_ready_month: int = int(flu_start_month)
    req_post_month: int = shift_month(independent_ready_month, -lead_months)
    hire_visible_month: int = shift_month(req_post_month, lead_months)
    trough_idx: int = int(np.argmin(np.array(protective_typical_12, dtype=float)))
    trough_month: int = int(dates_template_12[trough_idx].month)
    decline_months: List[int] = months_between(shift_month(int(flu_end_month), 1), trough_month)
    freeze_months: List[int] = list(decline_months)
    for i in range(1, int(freeze_buffer_months) + 1):
        freeze_months.append(shift_month(trough_month, i))
    recruiting_open_months: List[int] = []
    for i in range(lead_months):
        recruiting_open_months.append(shift_month(req_post_month, -i))
    def dedupe_keep_order(seq: List[int]) -> List[int]:
        seen: Set[int] = set()
        out: List[int] = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out
    freeze_months = dedupe_keep_order(freeze_months)
    recruiting_open_months = dedupe_keep_order(recruiting_open_months)
    freeze_months = [m for m in freeze_months if m not in set(recruiting_open_months)]
    return dict(
        lead_months=lead_months,
        notice_months=notice_months,
        independent_ready_month=independent_ready_month,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        trough_month=trough_month,
        freeze_months=freeze_months,
        recruiting_open_months=recruiting_open_months,
        flu_months=months_between(int(flu_start_month), int(flu_end_month)),
    )

# ============================================================
# SUPPLY SIM (multi-year continuous, with notice-lag attrition)
# ============================================================
def simulate_supply_multiyear_single(run_params: Dict[str, Any]) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    # Extract params for parallel execution
    dates_full = run_params['dates_full']
    baseline_provider_fte = run_params['baseline_provider_fte']
    forecast_visits_full_base = run_params['forecast_visits_full_base']
    visits_cv = run_params['visits_cv']
    provider_turnover = run_params['provider_turnover']
    turnover_var = run_params['turnover_var']
    pipeline_var_days = run_params['pipeline_var_days']
    total_lead_days_base = run_params['total_lead_days_base']
    provider_min_floor = run_params['provider_min_floor']
    hours_of_operation = run_params['hours_of_operation']
    fte_hours_per_week = run_params['fte_hours_per_week']
    burnout_slider = run_params['burnout_slider']
    safe_visits_per_provider_per_day = run_params['safe_visits_per_provider_per_day']
    dates_template_12 = run_params['dates_template_12']
    flu_start_month = run_params['flu_start_month']
    flu_end_month = run_params['flu_end_month']
    notice_days = run_params['notice_days']
    enable_seasonality_ramp = run_params['enable_seasonality_ramp']
    max_hiring_up_after_visible = run_params['max_hiring_up_after_visible']
    confirmed_hire_index = run_params['confirmed_hire_index']
    confirmed_hire_fte = run_params['confirmed_hire_fte']
    max_ramp_down_per_month = run_params['max_ramp_down_per_month']
    aggressive_unwind = run_params['aggressive_unwind']
    
    visits_full_r = apply_visits_noise(forecast_visits_full_base, visits_cv)
    turnover_r = provider_turnover * (1.0 + np.random.uniform(-turnover_var, turnover_var))
    total_lead_days_r = max(30, int(total_lead_days_base + np.random.randint(-pipeline_var_days, pipeline_var_days + 1)))
    protective_demand_r = visits_to_provider_demand(
        model=model,
        visits_by_month=visits_full_r,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )
    protective_target_r = burnout_protective_staffing_curve(
        visits_by_month=visits_full_r,
        base_demand_fte=protective_demand_r,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider_per_day,
    )
    protective_typical_12_r = typical_12_month_curve(dates_full, protective_target_r)
    strategy_r = auto_freeze_strategy_v3_from_typical(
        dates_template_12=dates_template_12,
        protective_typical_12=protective_typical_12_r,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
        pipeline_lead_days=total_lead_days_r,
        notice_days=notice_days,
        freeze_buffer_months=1,
    )
    req_post_month_r = int(strategy_r["req_post_month"])
    hire_visible_month_r = int(strategy_r["hire_visible_month"])
    freeze_months_r = list(strategy_r["freeze_months"])
    supply_lean_full_r = simulate_supply_multiyear(
        dates_full=dates_full,
        baseline_provider_fte=baseline_provider_fte,
        target_curve_full=protective_demand_r,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=turnover_r,
        notice_days=notice_days,
        req_post_month=req_post_month_r,
        hire_visible_month=hire_visible_month_r,
        freeze_months=(freeze_months_r if enable_seasonality_ramp else []),
        max_hiring_up_after_visible=max_hiring_up_after_visible,
        confirmed_hire_index=confirmed_hire_index,
        confirmed_hire_fte=confirmed_hire_fte,
        max_ramp_down_per_month=max_ramp_down_per_month,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        aggressive_unwind=aggressive_unwind,
    )
    supply_rec_full_r = simulate_supply_multiyear(
        dates_full=dates_full,
        baseline_provider_fte=baseline_provider_fte,
        target_curve_full=protective_target_r,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=turnover_r,
        notice_days=notice_days,
        req_post_month=req_post_month_r,
        hire_visible_month=hire_visible_month_r,
        freeze_months=(freeze_months_r if enable_seasonality_ramp else []),
        max_hiring_up_after_visible=max_hiring_up_after_visible,
        confirmed_hire_index=confirmed_hire_index,
        confirmed_hire_fte=confirmed_hire_fte,
        max_ramp_down_per_month=max_ramp_down_per_month,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        aggressive_unwind=aggressive_unwind,
    )
    return visits_full_r, protective_demand_r, protective_target_r, supply_lean_full_r, supply_rec_full_r

def simulate_supply_multiyear(
    dates_full: pd.DatetimeIndex,
    baseline_provider_fte: float,
    target_curve_full: List[float],
    provider_min_floor: float,
    annual_turnover_rate: float,
    notice_days: int,
    req_post_month: int,
    hire_visible_month: int,
    freeze_months: List[int],
    max_hiring_up_after_visible: float,
    confirmed_hire_index: Optional[int] = None,
    confirmed_hire_fte: float = 0.0,
    max_ramp_down_per_month: float = 0.25,
    seasonality_ramp_enabled: bool = True,
    aggressive_unwind: bool = False,
) -> List[float]:
    notice_months: int = lead_days_to_months(int(notice_days))
    monthly_turnover_rate: float = float(annual_turnover_rate) / 12.0
    freeze_set: Set[int] = set(int(m) for m in (freeze_months or []))
    blackout_months: Set[int] = set(
        months_between(int(req_post_month), shift_month(int(hire_visible_month), -1))
    )
    q: Optional[deque] = None if notice_months <= 0 else deque([0.0] * notice_months, maxlen=notice_months)
    staff: List[float] = []
    prev: float = max(float(baseline_provider_fte), float(provider_min_floor))
    hire_applied: bool = False
    for i, d in enumerate(dates_full):
        month_num: int = int(d.month)
        target: float = float(target_curve_full[i])
        if seasonality_ramp_enabled:
            in_freeze: bool = month_num in freeze_set
            in_blackout: bool = month_num in blackout_months
            ramp_up_cap: float = 0.0 if (in_freeze or in_blackout) else float(max_hiring_up_after_visible)
        else:
            ramp_up_cap = 0.35
        delta: float = target - prev
        if delta > 0:
            delta = clamp(delta, 0.0, ramp_up_cap)
        else:
            down_rate = max_ramp_down_per_month * 2 if aggressive_unwind and month_num in [3,4,5,9,10] else max_ramp_down_per_month
            delta = clamp(delta, -down_rate, 0.0)
        planned: float = prev + delta
        resignations: float = prev * monthly_turnover_rate
        if notice_months <= 0:
            separations: float = resignations
        else:
            q.append(resignations)
            separations = q.popleft()
        planned = planned - separations
        if (confirmed_hire_index is not None) and (not hire_applied) and (i == int(confirmed_hire_index)):
            planned += float(confirmed_hire_fte)
            hire_applied = True
        planned = max(planned, float(provider_min_floor))
        staff.append(planned)
        prev = planned
    return staff

# ============================================================
# COST / SWB-VISIT (Policy A: FTE-based)
# ============================================================
def monthly_hours_from_fte(fte: float, fte_hours_per_week: float, days_in_month: int) -> float:
    return float(fte) * float(fte_hours_per_week) * (float(days_in_month) / 7.0)

def loaded_hourly_rate(base_hourly: float, benefits_load_pct: float, ot_sick_pct: float) -> float:
    return float(base_hourly) * (1.0 + float(benefits_load_pct)) * (1.0 + float(ot_sick_pct))

def compute_role_mix_ratios(
    model: StaffingModel,
    visits_per_day: float,
    hours_of_operation: float,
    fte_hours_per_week: float
) -> Dict[str, float]:
    f: Dict[str, float] = model.calculate_fte_needed(
        visits_per_day=float(visits_per_day),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week),
    )
    prov: float = max(float(f.get("provider_fte", 0.0)), 0.25)
    return {
        "psr_per_provider": float(f.get("psr_fte", 0.0)) / prov,
        "ma_per_provider": float(f.get("ma_fte", 0.0)) / prov,
        "xrt_per_provider": float(f.get("xrt_fte", 0.0)) / prov,
    }

def compute_monthly_swb_per_visit_fte_based(
    provider_supply_curve_12: List[float],
    visits_per_day_curve_12: List[float],
    days_in_month_12: List[int],
    fte_hours_per_week: float,
    role_mix: Dict[str, float],
    hourly_rates: Dict[str, float],
    benefits_load_pct: float,
    ot_sick_pct: float,
    physician_supervision_hours_per_month: float = 0.0,
    supervisor_hours_per_month: float = 0.0,
) -> pd.DataFrame:
    out_rows: List[Dict[str, float]] = []
    for i in range(12):
        prov_fte: float = float(provider_supply_curve_12[i])
        vpd: float = float(visits_per_day_curve_12[i])
        dim: int = int(days_in_month_12[i])
        month_visits: float = max(vpd * dim, 1.0)
        psr_fte: float = prov_fte * float(role_mix["psr_per_provider"])
        ma_fte: float = prov_fte * float(role_mix["ma_per_provider"])
        rt_fte: float = prov_fte * float(role_mix["xrt_per_provider"])
        apc_hours: float = monthly_hours_from_fte(prov_fte, fte_hours_per_week, dim)
        psr_hours: float = monthly_hours_from_fte(psr_fte, fte_hours_per_week, dim)
        ma_hours: float = monthly_hours_from_fte(ma_fte, fte_hours_per_week, dim)
        rt_hours: float = monthly_hours_from_fte(rt_fte, fte_hours_per_week, dim)
        apc_rate: float = loaded_hourly_rate(hourly_rates["apc"], benefits_load_pct, ot_sick_pct)
        psr_rate: float = loaded_hourly_rate(hourly_rates["psr"], benefits_load_pct, ot_sick_pct)
        ma_rate: float = loaded_hourly_rate(hourly_rates["ma"], benefits_load_pct, ot_sick_pct)
        rt_rate: float = loaded_hourly_rate(hourly_rates["rt"], benefits_load_pct, ot_sick_pct)
        phys_rate: float = loaded_hourly_rate(hourly_rates["physician"], benefits_load_pct, ot_sick_pct)
        sup_rate: float = loaded_hourly_rate(hourly_rates["supervisor"], benefits_load_pct, ot_sick_pct)
        apc_cost: float = apc_hours * apc_rate
        psr_cost: float = psr_hours * psr_rate
        ma_cost: float = ma_hours * ma_rate
        rt_cost: float = rt_hours * rt_rate
        phys_cost: float = float(physician_supervision_hours_per_month) * phys_rate
        sup_cost: float = float(supervisor_hours_per_month) * sup_rate
        total_swb: float = apc_cost + psr_cost + ma_cost + rt_cost + phys_cost + sup_cost
        swb_per_visit: float = total_swb / month_visits
        out_rows.append(
            {
                "Provider_FTE_Supply": prov_fte,
                "PSR_FTE": psr_fte,
                "MA_FTE": ma_fte,
                "RT_FTE": rt_fte,
                "Visits": month_visits,
                "SWB_$": total_swb,
                "SWB_per_Visit_$": swb_per_visit,
            }
        )
    return pd.DataFrame(out_rows)

# ============================================================
# GAP + COST HELPERS
# ============================================================
def provider_day_gap(target_curve: List[float], supply_curve: List[float], days_in_month: List[int]) -> float:
    gap_days: float = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(float(t) - float(s), 0.0) * float(dim)
    return float(gap_days)

def annualize_monthly_fte_cost(delta_fte_curve: List[float], days_in_month: List[int], loaded_cost_per_provider_fte: float) -> float:
    cost: float = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += float(dfte) * float(loaded_cost_per_provider_fte) * (float(dim) / 365.0)
    return float(cost)

def compute_target_tracking_score(supply: List[float], target: List[float]) -> float:
    return float(np.sqrt(np.mean((np.array(supply) - np.array(target)) ** 2)))

# ============================================================
# SIDEBAR INPUTS
# ============================================================
with st.sidebar:
    st.header("Inputs")
    st.subheader("Baseline")
    visits: float = st.number_input("Avg Visits/Day (annual avg)", min_value=1.0, value=45.0, step=1.0)
    hours_of_operation: float = st.number_input("Hours of Operation / Week", min_value=1.0, value=70.0, step=1.0)
    fte_hours_per_week: float = st.number_input("FTE Hours / Week", min_value=1.0, value=40.0, step=1.0)
    st.subheader("Floors & Protection")
    provider_min_floor: float = st.number_input("Provider Minimum Floor (FTE)", min_value=0.25, value=1.00, step=0.25)
    burnout_slider: float = st.slider("Burnout Protection Level", 0.0, 1.0, 0.6, 0.05)
    safe_visits_per_provider: int = st.number_input("Safe Visits/Provider/Day", 10, 40, 20, 1)
    st.subheader("Turnover + Pipeline")
    provider_turnover: float = st.number_input("Provider Turnover % (annual)", value=24.0, step=1.0) / 100.0
    with st.expander("Provider Hiring Pipeline Assumptions", expanded=False):
        days_to_sign: int = st.number_input("Days to Sign", min_value=0, value=90, step=5)
        days_to_credential: int = st.number_input("Days to Credential", min_value=0, value=90, step=5)
        onboard_train_days: int = st.number_input("Days to Train", min_value=0, value=30, step=5)
        coverage_buffer_days: int = st.number_input("Planning Buffer Days", min_value=0, value=14, step=1)
        notice_days: int = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=90, step=5)
    st.subheader("Seasonality")
    flu_start_month: int = st.selectbox(
        "Flu Start Month",
        options=list(range(1, 13)),
        index=11,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
    )
    flu_end_month: int = st.selectbox(
        "Flu End Month",
        options=list(range(1, 13)),
        index=1,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
    )
    flu_uplift_pct: float = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100.0
    st.subheader("Confirmed Hiring")
    confirmed_hire_month: int = st.selectbox(
        "Confirmed Hire Month (shown in the display window)",
        options=list(range(1, 13)),
        index=10,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
        help="We will place the confirmed hire in the displayed 12-month window (not in year 1 by accident).",
    )
    confirmed_hire_fte: float = st.number_input("Confirmed Hire FTE", min_value=0.0, value=1.0, step=0.25)
    enable_seasonality_ramp: bool = st.checkbox(
        "Enable Seasonality Recruiting Ramp",
        value=True,
        help="If ON: supply cannot rise during freeze months or during pipeline blackout months.",
    )
    aggressive_unwind: bool = st.checkbox("Aggressive Post-Peak Unwind", value=True, help="Allows faster ramp-down after peaks to track volume better.")
    # ==========================
    # Probability / Near-Certain
    # ==========================
    st.subheader("Probability (Near-Certain)")
    enable_probability: bool = st.checkbox(
        "Enable Probability Mode (Monte Carlo)",
        value=False,
        help="If ON: runs many simulations with uncertainty in visits, turnover, and pipeline time.",
    )
    confidence_level: float = st.slider(
        "Near-Certain Confidence Level (quantile)",
        min_value=0.50,
        max_value=0.95,
        value=0.90,
        step=0.05,
        help="We compute a near-certain view by using quantiles of supply/targets.",
    )
    sim_horizon_months: int = st.slider(
        "Simulation Horizon (months)",
        min_value=24,
        max_value=60,
        value=36,
        step=12,
        help="We simulate a longer timeline, then display only 12 months from a stabilized portion.",
    )
    mc_runs: int = st.slider(
        "Monte Carlo Runs",
        min_value=200,
        max_value=3000,
        value=1000,
        step=100,
        help="More runs = smoother percentiles, but slower.",
    )
    visits_cv: float = st.slider(
        "Visits Forecast Variability (CV %)",
        min_value=0.0,
        max_value=25.0,
        value=10.0,
        step=1.0,
        help="Adds realistic variation around the seasonal visit forecast.",
    ) / 100.0
    turnover_var: float = st.slider(
        "Turnover Variability (± % of annual turnover)",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=5.0,
        help="Allows turnover to vary run-to-run.",
    ) / 100.0
    pipeline_var_days: int = st.slider(
        "Pipeline Duration Variability (± days)",
        min_value=0,
        max_value=60,
        value=15,
        step=5,
        help="Adds uncertainty to total pipeline days (affects strategy lead months).",
    )
    flu_uplift_var: float = st.slider("Flu Uplift Variability (± %)", -10.0, 10.0, 0.0, step=1.0) / 100.0  # New variability
    display_anchor: str = st.radio(
        "Display Window Anchor",
        options=["Req Post Month", "Flu Start", "January"],
        index=0,
        horizontal=True,
    )
    st.subheader("SWB/Visit Feasibility (FTE-based)")
    target_swb_per_visit: float = st.number_input("Target SWB / Visit ($)", value=85.00, step=1.00)
    with st.expander("Hourly Rates (baseline assumptions)", expanded=False):
        benefits_load_pct: float = st.number_input("Benefits Load (%)", value=30.00, step=1.00) / 100.0
        ot_sick_pct: float = st.number_input("OT + Sick/PTO (%)", value=4.00, step=0.50) / 100.0
        physician_hr: float = st.number_input("Physician (Supervision) $/hr", value=135.79, step=1.00)
        apc_hr: float = st.number_input("APC $/hr", value=62.00, step=1.00)
        ma_hr: float = st.number_input("MA $/hr", value=24.14, step=0.50)
        psr_hr: float = st.number_input("PSR $/hr", value=21.23, step=0.50)
        rt_hr: float = st.number_input("RT $/hr", value=31.36, step=0.50)
        supervisor_hr: float = st.number_input("Supervisor $/hr", value=28.25, step=0.50)
    with st.expander("Optional: fixed monthly hours", expanded=False):
        physician_supervision_hours_per_month: float = st.number_input("Physician supervision hours/month", value=0.0, step=1.0)
        supervisor_hours_per_month: float = st.number_input("Supervisor hours/month", value=0.0, step=1.0)
    st.subheader("Advanced Features")
    uploaded_hist = st.file_uploader("Upload Historical Visits (CSV with 'ds' date, 'y' visits)", type="csv")
    hist_df = pd.read_csv(uploaded_hist) if uploaded_hist else pd.DataFrame()
    backtest_mode: bool = st.checkbox("Backtest Mode", value=False)
    if backtest_mode:
        uploaded_backtest = st.file_uploader("Upload Backtest Data (CSV with date, actual_fte, actual_visits)", type="csv")
        backtest_df = pd.read_csv(uploaded_backtest) if uploaded_backtest else pd.DataFrame()
    acuity_months = st.multiselect("High Acuity Months", list(range(1,13)), default=[12,1,2])
    acuity_multiplier = st.slider("Acuity Multiplier (1-5)", 1.0, 5.0, 1.2, 0.1)
    max_hiring_up_after_visible_user = st.slider("Max Hiring Up (FTE/month)", 0.1, 1.0, 0.5, 0.05)
    max_ramp_down_per_month_user = st.slider("Max Ramp Down (FTE/month)", 0.1, 1.0, 0.25, 0.05)
    st.divider()
    run_model: bool = st.button("Run Model")

# ============================================================
# RUN MODEL (v5.1 — multi-year sim, show 12 months, fixed hire placement)
# ============================================================
if run_model:
    np.random.seed(None)
    sim_months: int = int(sim_horizon_months)
    start_date: datetime = datetime(today.year, 1, 1)
    dates_full: pd.DatetimeIndex = pd.date_range(start=start_date, periods=sim_months, freq="MS")
    days_in_month_full: List[int] = [pd.Period(d, "M").days_in_month for d in dates_full]
    # Flu months (repeatable)
    flu_months: List[int] = months_between(int(flu_start_month), int(flu_end_month))
    # Baseline provider FTE (from baseline visits)
    fte_result: Dict[str, float] = model.calculate_fte_needed(
        visits_per_day=float(visits),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week),
    )
    baseline_provider_fte: float = max(float(fte_result["provider_fte"]), float(provider_min_floor))
    # Deterministic base forecast across full horizon
    forecast_visits_full_base: List[float] = compute_seasonality_forecast_multiyear(
        dates=dates_full,
        baseline_visits=visits,
        flu_months=flu_months,
        flu_uplift_pct=flu_uplift_pct,
        hist_df=hist_df
    )
    # Lean demand + protective target for the base case (deterministic)
    acuity_scale_full = [acuity_multiplier if int(d.month) in acuity_months else 1.0 for d in dates_full]
    provider_base_demand_full_base: List[float] = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_full_base,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
        acuity_scale=acuity_scale_full
    )
    protective_full_base: List[float] = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_full_base,
        base_demand_fte=provider_base_demand_full_base,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )
    # Pipeline lead time (base)
    total_lead_days_base: int = int(days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days)
    # Strategy built from typical 12-month protective curve (stable)
    dates_template_12: pd.DatetimeIndex = pd.date_range(start=datetime(2000, 1, 1), periods=12, freq="MS")
    protective_typical_12: List[float] = typical_12_month_curve(dates_full, protective_full_base)
    strategy_base: Dict[str, Any] = auto_freeze_strategy_v3_from_typical(
        dates_template_12=dates_template_12,
        protective_typical_12=protective_typical_12,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
        pipeline_lead_days=total_lead_days_base,
        notice_days=notice_days,
        freeze_buffer_months=1,
    )
    req_post_month_base: int = int(strategy_base["req_post_month"])
    hire_visible_month_base: int = int(strategy_base["hire_visible_month"])
    independent_ready_month_base: int = int(strategy_base["independent_ready_month"])
    freeze_months_base: List[int] = list(strategy_base["freeze_months"])
    recruiting_open_months_base: List[int] = list(strategy_base["recruiting_open_months"])
    lead_months_base: int = int(strategy_base["lead_months"])
    # Derived ramp: gap at flu start (typical) / months in flu window
    flu_month_idx_typical: int = int(flu_start_month) - 1
    months_in_flu_window: int = max(len(strategy_base["flu_months"]), 1)
    target_at_flu_typical: float = float(protective_typical_12[flu_month_idx_typical])
    fte_gap_to_close: float = max(target_at_flu_typical - baseline_provider_fte, 0.0)
    derived_ramp_after_visible: float = min(fte_gap_to_close / float(months_in_flu_window), 1.25)
    # ------------------------------------------------------------
    # Pick display window from stabilized portion (year 2-ish)
    # ------------------------------------------------------------
    stabilized_start = 12 if sim_months >= 24 else max(0, sim_months - 12)
    stabilized_end = min(sim_months, stabilized_start + 12)
    # Choose anchor month-of-year
    if display_anchor == "Req Post Month":
        anchor_month = req_post_month_base
    elif display_anchor == "Flu Start":
        anchor_month = int(flu_start_month)
    else:
        anchor_month = 1
    anchor_idx = stabilized_start
    for i in range(stabilized_start, min(sim_months, stabilized_start + 24)):
        if int(dates_full[i].month) == int(anchor_month):
            anchor_idx = i
            break
    display_idx = list(range(anchor_idx, anchor_idx + 12))
    if display_idx[-1] >= sim_months:
        display_idx = list(range(sim_months - 12, sim_months))
    # Confirmed hire should be applied ONCE inside the display window (first matching month)
    confirmed_hire_month_int = int(confirmed_hire_month)
    confirmed_hire_index: Optional[int] = None
    for i in display_idx:
        if int(dates_full[i].month) == confirmed_hire_month_int:
            confirmed_hire_index = i
            break
    # ------------------------------------------------------------
    # If probability disabled: single deterministic pass
    # If enabled: Monte Carlo runs -> quantiles for display
    # ------------------------------------------------------------
    if not enable_probability:
        # Use base deterministic curves
        forecast_visits_full = forecast_visits_full_base
        provider_base_demand_full = provider_base_demand_full_base
        protective_full = protective_full_base
        total_lead_days = total_lead_days_base
        strategy = strategy_base
        req_post_month = req_post_month_base
        hire_visible_month = hire_visible_month_base
        independent_ready_month = independent_ready_month_base
        freeze_months = freeze_months_base
        recruiting_open_months = recruiting_open_months_base
        lead_months = lead_months_base
        supply_lean_full = simulate_supply_multiyear(
            dates_full=dates_full,
            baseline_provider_fte=baseline_provider_fte,
            target_curve_full=provider_base_demand_full,
            provider_min_floor=provider_min_floor,
            annual_turnover_rate=provider_turnover,
            notice_days=notice_days,
            req_post_month=req_post_month,
            hire_visible_month=hire_visible_month,
            freeze_months=(freeze_months if enable_seasonality_ramp else []),
            max_hiring_up_after_visible=max_hiring_up_after_visible_user,
            confirmed_hire_index=confirmed_hire_index,
            confirmed_hire_fte=confirmed_hire_fte,
            max_ramp_down_per_month=max_ramp_down_per_month_user,
            seasonality_ramp_enabled=enable_seasonality_ramp,
            aggressive_unwind=aggressive_unwind,
        )
        supply_rec_full = simulate_supply_multiyear(
            dates_full=dates_full,
            baseline_provider_fte=baseline_provider_fte,
            target_curve_full=protective_full,
            provider_min_floor=provider_min_floor,
            annual_turnover_rate=provider_turnover,
            notice_days=notice_days,
            req_post_month=req_post_month,
            hire_visible_month=hire_visible_month,
            freeze_months=(freeze_months if enable_seasonality_ramp else []),
            max_hiring_up_after_visible=max_hiring_up_after_visible_user,
            confirmed_hire_index=confirmed_hire_index,
            confirmed_hire_fte=confirmed_hire_fte,
            max_ramp_down_per_month=max_ramp_down_per_month_user,
            seasonality_ramp_enabled=enable_seasonality_ramp,
            aggressive_unwind=aggressive_unwind,
        )
        # Display slices
        dates_12 = [dates_full[i] for i in display_idx]
        days_in_month_12 = [days_in_month_full[i] for i in display_idx]
        month_labels_12 = [d.strftime("%b") for d in dates_12]
        visits_12 = [forecast_visits_full[i] for i in display_idx]
        demand_lean_12 = [provider_base_demand_full[i] for i in display_idx]
        target_prot_12 = [protective_full[i] for i in display_idx]
        supply_lean_12 = [supply_lean_full[i] for i in display_idx]
        supply_rec_12 = [supply_rec_full[i] for i in display_idx]
        # Round for display rule
        demand_lean_12 = [round_up_quarter(x) for x in demand_lean_12]
        target_prot_12 = [round_up_quarter(x) for x in target_prot_12]
        supply_lean_12 = [round_up_quarter(x) for x in supply_lean_12]
        supply_rec_12 = [round_up_quarter(x) for x in supply_rec_12]
        # Burnout exposure (display)
        burnout_gap_fte_12 = [max(float(t) - float(s), 0.0) for t, s in zip(target_prot_12, supply_rec_12)]
        months_exposed_12 = int(sum(1 for g in burnout_gap_fte_12 if g > 0))
        # SWB/visit feasibility on display window
        role_mix = compute_role_mix_ratios(
            model=model,
            visits_per_day=visits,
            hours_of_operation=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week,
        )
        hourly_rates = {
            "physician": physician_hr,
            "apc": apc_hr,
            "ma": ma_hr,
            "psr": psr_hr,
            "rt": rt_hr,
            "supervisor": supervisor_hr,
        }
        swb_df = compute_monthly_swb_per_visit_fte_based(
            provider_supply_curve_12=supply_rec_12,
            visits_per_day_curve_12=visits_12,
            days_in_month_12=days_in_month_12,
            fte_hours_per_week=fte_hours_per_week,
            role_mix=role_mix,
            hourly_rates=hourly_rates,
            benefits_load_pct=benefits_load_pct,
            ot_sick_pct=ot_sick_pct,
            physician_supervision_hours_per_month=physician_supervision_hours_per_month,
            supervisor_hours_per_month=supervisor_hours_per_month,
        )
        avg_swb_per_visit_modeled = float(swb_df["SWB_per_Visit_$"].mean())
        feasible = bool(avg_swb_per_visit_modeled <= float(target_swb_per_visit))
        labor_factor = (float(target_swb_per_visit) / avg_swb_per_visit_modeled) if avg_swb_per_visit_modeled > 0 else np.nan
        # New: Target tracking score
        tracking_score = compute_target_tracking_score(supply_rec_12, target_prot_12)
        # Backtest if enabled
        backtest_mae = None
        if backtest_mode and not backtest_df.empty:
            backtest_df['date'] = pd.to_datetime(backtest_df['date'])
            back_dates = backtest_df['date']
            back_idx = [dates_full.get_loc(d) for d in back_dates if d in dates_full]
            pred_prot = [protective_full[i] for i in back_idx]
            backtest_mae = np.mean(np.abs(backtest_df['actual_fte'] - pred_prot))
        st.session_state["model_ran"] = True
        st.session_state["results"] = dict(
            # ... (existing)
            tracking_score=tracking_score,
            backtest_mae=backtest_mae,
        )
    else:
        # Monte Carlo with parallel
        q_supply = float(1.0 - confidence_level)
        q_target = float(confidence_level)
        runs = int(mc_runs)
        run_params_base = {
            'dates_full': dates_full,
            'baseline_provider_fte': baseline_provider_fte,
            'forecast_visits_full_base': forecast_visits_full_base,
            'visits_cv': visits_cv,
            'provider_turnover': provider_turnover,
            'turnover_var': turnover_var,
            'pipeline_var_days': pipeline_var_days,
            'total_lead_days_base': total_lead_days_base,
            'provider_min_floor': provider_min_floor,
            'hours_of_operation': hours_of_operation,
            'fte_hours_per_week': fte_hours_per_week,
            'burnout_slider': burnout_slider,
            'safe_visits_per_provider_per_day': safe_visits_per_provider_per_day,
            'dates_template_12': dates_template_12,
            'flu_start_month': flu_start_month,
            'flu_end_month': flu_end_month,
            'notice_days': notice_days,
            'enable_seasonality_ramp': enable_seasonality_ramp,
            'max_hiring_up_after_visible': max_hiring_up_after_visible_user,
            'confirmed_hire_index': confirmed_hire_index,
            'confirmed_hire_fte': confirmed_hire_fte,
            'max_ramp_down_per_month': max_ramp_down_per_month_user,
            'aggressive_unwind': aggressive_unwind,
        }
        # Parallel MC
        def run_mc(r):
            return simulate_supply_multiyear_single(run_params_base)
        with Parallel(n_jobs=-1) as parallel:
            mc_results = parallel(delayed(run_mc)(r) for r in range(runs))
        visits_runs = np.array([res[0] for res in mc_results])
        lean_runs = np.array([res[1] for res in mc_results])
        target_runs = np.array([res[2] for res in mc_results])
        supply_lean_runs = np.array([res[3] for res in mc_results])
        supply_rec_runs = np.array([res[4] for res in mc_results])
        # Slice to display
        idx = np.array(display_idx)
        visits_runs_12 = visits_runs[:, idx]
        lean_runs_12 = lean_runs[:, idx]
        target_runs_12 = target_runs[:, idx]
        supply_lean_runs_12 = supply_lean_runs[:, idx]
        supply_rec_runs_12 = supply_rec_runs[:, idx]
        visits_12 = np.quantile(visits_runs_12, q_target, axis=0).tolist()
        demand_lean_12 = np.quantile(lean_runs_12, q_target, axis=0).tolist()
        target_prot_12 = np.quantile(target_runs_12, q_target, axis=0).tolist()
        supply_lean_12 = np.quantile(supply_lean_runs_12, q_supply, axis=0).tolist()
        supply_rec_12 = np.quantile(supply_rec_runs_12, q_supply, axis=0).tolist()
        # Round
        demand_lean_12 = [round_up_quarter(x) for x in demand_lean_12]
        target_prot_12 = [round_up_quarter(x) for x in target_prot_12]
        supply_lean_12 = [round_up_quarter(x) for x in supply_lean_12]
        supply_rec_12 = [round_up_quarter(x) for x in supply_rec_12]
        burnout_gap_fte_12 = [max(float(t) - float(s), 0.0) for t, s in zip(target_prot_12, supply_rec_12)]
        months_exposed_12 = int(sum(1 for g in burnout_gap_fte_12 if g > 0))
        # Strategy from base
        strategy = strategy_base
        total_lead_days = total_lead_days_base
        req_post_month = req_post_month_base
        hire_visible_month = hire_visible_month_base
        independent_ready_month = independent_ready_month_base
        freeze_months = freeze_months_base
        recruiting_open_months = recruiting_open_months_base
        lead_months = lead_months_base
        # SWB
        role_mix = compute_role_mix_ratios(
            model=model,
            visits_per_day=visits,
            hours_of_operation=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week,
        )
        hourly_rates = {
            "physician": physician_hr,
            "apc": apc_hr,
            "ma": ma_hr,
            "psr": psr_hr,
            "rt": rt_hr,
            "supervisor": supervisor_hr,
        }
        swb_df = compute_monthly_swb_per_visit_fte_based(
            provider_supply_curve_12=supply_rec_12,
            visits_per_day_curve_12=visits_12,
            days_in_month_12=days_in_month_12,
            fte_hours_per_week=fte_hours_per_week,
            role_mix=role_mix,
            hourly_rates=hourly_rates,
            benefits_load_pct=benefits_load_pct,
            ot_sick_pct=ot_sick_pct,
            physician_supervision_hours_per_month=physician_supervision_hours_per_month,
            supervisor_hours_per_month=supervisor_hours_per_month,
        )
        avg_swb_per_visit_modeled = float(swb_df["SWB_per_Visit_$"].mean())
        feasible = bool(avg_swb_per_visit_modeled <= float(target_swb_per_visit))
        labor_factor = (float(target_swb_per_visit) / avg_swb_per_visit_modeled) if avg_swb_per_visit_modeled > 0 else np.nan
        tracking_score = compute_target_tracking_score(supply_rec_12, target_prot_12)
        backtest_mae = None
        if backtest_mode and not backtest_df.empty:
            backtest_df['date'] = pd.to_datetime(backtest_df['date'])
            back_dates = backtest_df['date']
            back_idx = [dates_full.get_loc(d) for d in back_dates if d in dates_full]
            pred_prot = [target_runs_12.mean(axis=0)[back_idx - min(back_idx)] for back_idx in back_idx]  # Approximate
            backtest_mae = np.mean(np.abs(backtest_df['actual_fte'] - pred_prot))
        st.session_state["model_ran"] = True
        st.session_state["results"] = dict(
            dates=dates_12,
            month_labels=month_labels_12,
            days_in_month=days_in_month_12,
            forecast_visits_by_month=visits_12,
            provider_base_demand=demand_lean_12,
            protective_curve=target_prot_12,
            realistic_supply_lean=supply_lean_12,
            realistic_supply_recommended=supply_rec_12,
            burnout_gap_fte=burnout_gap_fte_12,
            months_exposed=months_exposed_12,
            pipeline_lead_days=total_lead_days,
            lead_months=lead_months,
            req_post_month=req_post_month,
            hire_visible_month=hire_visible_month,
            independent_ready_month=independent_ready_month,
            freeze_months=freeze_months,
            recruiting_open_months=recruiting_open_months,
            flu_months=strategy["flu_months"],
            months_in_flu_window=months_in_flu_window,
            fte_gap_to_close=fte_gap_to_close,
            derived_ramp_after_visible=derived_ramp_after_visible,
            baseline_provider_fte=baseline_provider_fte,
            target_swb_per_visit=float(target_swb_per_visit),
            avg_swb_per_visit_modeled=avg_swb_per_visit_modeled,
            swb_feasible=feasible,
            labor_factor=float(labor_factor) if np.isfinite(labor_factor) else None,
            swb_df=swb_df,
            sim_months=sim_months,
            display_anchor_month=anchor_month,
            confirmed_hire_month=confirmed_hire_month_int,
            confirmed_hire_fte=float(confirmed_hire_fte),
            confirmed_hire_index=confirmed_hire_index,
            enable_seasonality_ramp=enable_seasonality_ramp,
            enable_probability=enable_probability,
            tracking_score=tracking_score,
            backtest_mae=backtest_mae,
        )

# ============================================================
# STOP IF NOT RUN
# ============================================================
if not st.session_state.get("model_ran"):
    st.info("Enter inputs in the sidebar and click **Run Model**.")
    st.stop()
R: Dict[str, Any] = st.session_state["results"]

# ============================================================
# SECTION 1 — OPERATIONS
# ============================================================
with tab_ops:
    st.caption("Visits/day forecast → FTE needed by month (seasonality + flu uplift).")
    monthly_rows = []
    for month_label, d, v in zip(R["month_labels"], R["dates"], R["forecast_visits_by_month"]):
        fte_staff = model.calculate_fte_needed(
            visits_per_day=float(v),
            hours_of_operation_per_week=float(hours_of_operation),
            fte_hours_per_week=float(fte_hours_per_week),
        )
        monthly_rows.append(
            {
                "Month": month_label,
                "Visits/Day (Forecast)": round(float(v), 1),
                "Provider FTE": round_up_quarter(fte_staff["provider_fte"]),
                "PSR FTE": round(float(fte_staff["psr_fte"]), 2),
                "MA FTE": round(float(fte_staff["ma_fte"]), 2),
                "XRT FTE": round(float(fte_staff["xrt_fte"]), 2),
                "Total FTE": round(float(fte_staff["total_fte"]), 2),
            }
        )
    ops_df = pd.DataFrame(monthly_rows)
    st.dataframe(ops_df, hide_index=True, use_container_width=True)
    st.download_button("Download Operations CSV", ops_df.to_csv(index=False), "operations.csv")
    st.success(
        "**Operations Summary:** This is the seasonality-adjusted demand signal. "
        "Lean demand is minimum coverage; the protective target adds a burnout buffer to protect throughput and quality."
    )

# ============================================================
# SECTION 2 — REALITY
# ============================================================
with tab_real:
    st.caption(
        "Targets vs realistic supply when hiring is constrained by lead time, freezes, turnover (with notice lag), and pipeline visibility."
    )
    freeze_label = month_range_label(R.get("freeze_months", []))
    recruit_label = month_range_label(R.get("recruiting_open_months", []))
    req_post_label = datetime(2000, int(R["req_post_month"]), 1).strftime("%b")
    hire_visible_label = datetime(2000, int(R["hire_visible_month"]), 1).strftime("%b")
    independent_label = datetime(2000, int(R["independent_ready_month"]), 1).strftime("%b")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; gap:16px;
                    padding:12px 16px; background:#F7F7F7; border-radius:10px;
                    border:1px solid #E0E0E0; font-size:15px;">
            <div><b>Freeze:</b> {freeze_label}</div>
            <div><b>Recruiting Window:</b> {recruit_label}</div>
            <div><b>Post Req:</b> {req_post_label}</div>
            <div><b>Hires Visible:</b> {hire_visible_label}</div>
            <div><b>Independent By:</b> {independent_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    peak_gap = float(max(R["burnout_gap_fte"])) if R["burnout_gap_fte"] else 0.0
    avg_gap = float(np.mean(R["burnout_gap_fte"])) if R["burnout_gap_fte"] else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Burnout Gap (FTE)", f"{peak_gap:.2f}")
    m2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
    m3.metric("Months Exposed", f"{R['months_exposed']}/12")
    m4.metric("Target Tracking RMSE", f"{R['tracking_score']:.2f}")
    if R.get('backtest_mae') is not None:
        st.metric("Backtest MAE", f"{R['backtest_mae']:.2f}")
    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=R["dates"], y=R["provider_base_demand"], mode='lines', name="Lean Target (Demand)", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=R["dates"], y=R["protective_curve"], mode='lines+markers', name="Recommended Target (Protective)", line=dict(color=BRAND_GOLD)))
    fig.add_trace(go.Scatter(x=R["dates"], y=R["realistic_supply_recommended"], mode='lines+markers', name="Realistic Supply (Pipeline)", line=dict(color=BRAND_BLACK)))
    fig.add_trace(go.Scatter(x=R["dates"], y=R["forecast_visits_by_month"], mode='lines', name="Forecast Visits/Day", yaxis="y2", line=dict(dash='dash', color=MID_GRAY)))
    # Burnout fill
    fig.add_trace(go.Scatter(
        x=R["dates"] + R["dates"][::-1],
        y=list(R["protective_curve"]) + list(R["realistic_supply_recommended"])[::-1],
        fill='toself',
        fillcolor='rgba(122,98,0,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        name="Burnout Exposure Zone"
    ))
    fig.update_layout(
        title="Reality — Targets vs Pipeline-Constrained Supply",
        xaxis_title="Month",
        yaxis_title="Provider FTE",
        yaxis2=dict(title="Visits / Day", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
    prob_note = "" if not R.get("enable_probability", False) else f" (Probability Mode ON: near-certain view at {int(R.get('confidence_level', 0.9)*100)}% confidence)"
    st.success(
        f"**Reality Summary:** This 12-month view is taken from a **{R['sim_months']}-month continuous simulation** "
        f"(no year-end reset){prob_note}. "
        f"To be flu-ready by **{independent_label}**, requisitions must post by **{req_post_label}** "
        f"so hires are visible by **{hire_visible_label}**. "
        f"Required protective ramp speed: **{R['derived_ramp_after_visible']:.2f} FTE/month**."
    )
    st.info(
        "🧠 **Auto-Hiring Strategy (v3)**\n\n"
        f"- Freeze months: **{', '.join([datetime(2000,m,1).strftime('%b') for m in R['freeze_months']]) or '—'}**\n"
        f"- Recruiting window: **{', '.join([datetime(2000,m,1).strftime('%b') for m in R['recruiting_open_months']]) or '—'}**\n"
        f"- Post req: **{req_post_label}** | Hires visible: **{hire_visible_label}** | Independent by: **{independent_label}**\n"
        f"- Lead time: **{R['pipeline_lead_days']} days (~{R['lead_months']} months)**\n"
        f"- Notice lag modeled: **{lead_days_to_months(int(notice_days))} months** (separations occur after notice period)\n"
    )

# ============================================================
# SECTION 3 — FINANCE
# ============================================================
with tab_fin:
    st.caption("Quantifies the investment required to close the gap and the economic value of reducing provider-day shortages.")
    st.subheader("Finance Inputs")
    colA, colB, colC = st.columns(3)
    with colA:
        loaded_cost_per_provider_fte = st.number_input("Loaded Cost per Provider FTE (annual)", value=260000, step=5000)
    with colB:
        net_revenue_per_visit = st.number_input("Net Revenue per Visit", value=140.0, step=5.0)
    with colC:
        visits_lost_per_provider_day_gap = st.number_input("Visits Lost per 1.0 Provider-Day Gap", value=18.0, step=1.0)
    delta_fte_curve = [max(float(t) - float(R["baseline_provider_fte"]), 0.0) for t in R["protective_curve"]]
    annual_investment = annualize_monthly_fte_cost(delta_fte_curve, R["days_in_month"], loaded_cost_per_provider_fte)
    gap_days = provider_day_gap(R["protective_curve"], R["realistic_supply_recommended"], R["days_in_month"])
    est_visits_lost = gap_days * float(visits_lost_per_provider_day_gap)
    est_revenue_lost = est_visits_lost * float(net_revenue_per_visit)
    roi = (est_revenue_lost / annual_investment) if annual_investment > 0 else np.nan
    f1, f2, f3 = st.columns(3)
    f1.metric("Annual Investment (Protective)", f"${annual_investment:,.0f}")
    f2.metric("Est. Net Revenue at Risk", f"${est_revenue_lost:,.0f}")
    f3.metric("ROI (Revenue ÷ Investment)", f"{roi:,.2f}x" if np.isfinite(roi) else "—")
    with st.expander("Sensitivity Analysis"):
        turnover_sens = st.slider("Turnover Sensitivity (±%)", -20, 20, 0)
        revenue_sens = st.slider("Revenue Sensitivity (±%)", -20, 20, 0)
        # Quick re-calc with sens
        sens_turnover = provider_turnover * (1 + turnover_sens / 100)
        sens_revenue = net_revenue_per_visit * (1 + revenue_sens / 100)
        # Re-run sim with sens_turnover (simplified)
        sens_supply_rec = simulate_supply_multiyear(
            dates_full=dates_full[:12],  # Approx
            baseline_provider_fte=baseline_provider_fte,
            target_curve_full=R["protective_curve"],
            provider_min_floor=provider_min_floor,
            annual_turnover_rate=sens_turnover,
            notice_days=notice_days,
            req_post_month=req_post_month_base,
            hire_visible_month=hire_visible_month_base,
            freeze_months=freeze_months_base,
            max_hiring_up_after_visible=max_hiring_up_after_visible_user,
            confirmed_hire_index=None,
            confirmed_hire_fte=0,
            max_ramp_down_per_month=max_ramp_down_per_month_user,
            seasonality_ramp_enabled=enable_seasonality_ramp,
            aggressive_unwind=aggressive_unwind,
        )
        sens_gap_days = provider_day_gap(R["protective_curve"], sens_supply_rec, R["days_in_month"])
        sens_est_revenue_lost = sens_gap_days * visits_lost_per_provider_day_gap * sens_revenue
        sens_roi = (sens_est_revenue_lost / annual_investment) if annual_investment > 0 else np.nan
        st.metric("Sensitivity ROI", f"{sens_roi:,.2f}x")
    st.success(
        "**Finance Summary:** The investment is the cost of staffing to the protective curve. "
        "The value is the revenue protected by reducing provider-day shortages during peak demand."
    )

# ============================================================
# SECTION 3B — VVI FEASIBILITY (SWB/Visit)
# ============================================================
with tab_vvi:
    st.caption("Tests whether the staffing plan is financially feasible versus your Target SWB/Visit, using your role-cost assumptions.")
    swb_df = R["swb_df"].copy()
    swb_df.insert(0, "Month", R["month_labels"])
    colX, colY, colZ = st.columns(3)
    colX.metric("Target SWB/Visit", f"${R['target_swb_per_visit']:.2f}")
    colY.metric("Modeled SWB/Visit (avg)", f"${R['avg_swb_per_visit_modeled']:.2f}")
    colZ.metric("Feasible?", "YES" if R["swb_feasible"] else "NO")
    st.dataframe(
        swb_df[["Month", "Provider_FTE_Supply", "Visits", "SWB_$", "SWB_per_Visit_$"]],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button("Download SWB Data", swb_df.to_csv(index=False), "swb_data.csv")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=R["dates"], y=swb_df["SWB_per_Visit_$"], mode='lines+markers', name="Modeled SWB/Visit"))
    fig2.add_hline(y=R["target_swb_per_visit"], line_dash="dash", annotation_text="Target SWB/Visit")
    fig2.update_layout(title="SWB/Visit Feasibility (Modeled vs Target)", yaxis_title="$/Visit")
    st.plotly_chart(fig2, use_container_width=True)
    lf = R.get("labor_factor", None)
    lf_str = f"{lf:.2f}" if isinstance(lf, float) else "—"
    st.info(
        f"**Labor Factor (LF)** = Target SWB/Visit ÷ Modeled SWB/Visit = **{lf_str}**. "
        "This is a clean building block for VVI feasibility logic (LF > 1.0 is favorable; LF < 1.0 means labor is too expensive for the target)."
    )
    # VVI Scorecard
    scorecard = pd.DataFrame({
        "Metric": ["Net Revenue/Visit", "SWB/Visit", "Labor Factor"],
        "Value": [net_revenue_per_visit, R['avg_swb_per_visit_modeled'], lf],
        "Status": ["Pass" if v > 100 else "Fail" for v in [net_revenue_per_visit, R['target_swb_per_visit'] - R['avg_swb_per_visit_modeled'], lf]]
    })
    st.table(scorecard.style.applymap(lambda v: "color: green" if v == "Pass" else "color: red", subset=["Status"]))

# ============================================================
# SECTION 4 — STRATEGY
# ============================================================
with tab_strat:
    st.caption("Use flex coverage strategies to reduce burnout exposure without hiring full permanent FTE.")
    st.subheader("Strategy Levers")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        buffer_pct = st.slider("Buffer Coverage %", 0, 100, 25, 5)
    with s2:
        float_pool_fte = st.slider("Float Pool (FTE)", 0.0, 5.0, 1.0, 0.25)
    with s3:
        fractional_fte = st.slider("Fractional Add (FTE)", 0.0, 5.0, 0.5, 0.25)
    with s4:
        hybrid_slider = st.slider("Hybrid (flex → perm)", 0.0, 1.0, 0.5, 0.05)
    gap_fte_curve = [max(float(t) - float(s), 0.0) for t, s in zip(R["protective_curve"], R["realistic_supply_recommended"])]
    effective_gap_curve = []
    for g in gap_fte_curve:
        g2 = float(g) * (1.0 - float(buffer_pct) / 100.0)
        g2 = max(g2 - float(float_pool_fte), 0.0)
        g2 = max(g2 - float(fractional_fte), 0.0)
        effective_gap_curve.append(g2)
    remaining_gap_days = provider_day_gap([0] * 12, effective_gap_curve, R["days_in_month"])
    reduced_gap_days = max(gap_days - remaining_gap_days, 0.0)
    est_visits_saved = reduced_gap_days * float(visits_lost_per_provider_day_gap)
    est_revenue_saved = est_visits_saved * float(net_revenue_per_visit)
    hybrid_investment = annual_investment * float(hybrid_slider)
    sA, sB, sC = st.columns(3)
    sA.metric("Provider-Day Gap Reduced", f"{reduced_gap_days:,.0f}")
    sB.metric("Est. Revenue Saved", f"${est_revenue_saved:,.0f}")
    sC.metric("Hybrid Investment Share", f"${hybrid_investment:,.0f}")
    st.success(
        "**Strategy Summary:** Flex levers can reduce exposure faster than permanent hiring. "
        "Use hybrid to transition temporary coverage into permanent staffing once demand proves durable."
    )

# ============================================================
# SECTION 5 — DECISION
# ============================================================
with tab_exec:
    st.header("5) Decision — Executive Summary")
    st.subheader("Decision Snapshot")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Peak Gap (FTE)", f"{peak_gap:.2f}")
        st.metric("Avg Gap (FTE)", f"{avg_gap:.2f}")
        st.metric("Months Exposed", f"{R['months_exposed']}/12")
    with col2:
        st.write(
            f"**To be flu-ready by {independent_label}:**\n"
            f"- Post requisitions by **{req_post_label}** (lead time: {R['pipeline_lead_days']} days ≈ {R['lead_months']} months)\n"
            f"- Hiring becomes visible by **{hire_visible_label}**\n"
            f"- Protective ramp required: **{R['derived_ramp_after_visible']:.2f} FTE/month**\n\n"
            f"**Financial framing:**\n"
            f"- Annual protective investment: **${annual_investment:,.0f}**\n"
            f"- Estimated revenue at risk if not closed: **${est_revenue_lost:,.0f}**\n"
            f"- ROI: **{roi:,.2f}x**\n"
            f"- SWB/Visit feasibility vs target: **{'YES' if R['swb_feasible'] else 'NO'}** "
            f"(Modeled ${R['avg_swb_per_visit_modeled']:.2f} vs Target ${R['target_swb_per_visit']:.2f})\n\n"
            f"**With strategy levers applied:**\n"
            f"- Provider-day gap reduced: **{reduced_gap_days:,.0f} days**\n"
            f"- Estimated revenue saved: **${est_revenue_saved:,.0f}**\n"
        )
    st.success(
        "✅ **Decision Summary:** This model converts seasonality into staffing demand, converts pipeline timing into realistic supply, "
        "and quantifies burnout exposure, ROI, and SWB/Visit feasibility. "
        "Use the auto-hiring strategy + flex levers to move from reactive coverage to decision-ready staffing."
    )
    st.header("6) In-Depth Executive Summary")
    st.caption("A plain-language explanation of what the model is doing, what it found, and how to act on it.")
    flu_range = month_range_label(R["flu_months"])
    freeze_range = freeze_label
    recruit_range = recruit_label
    prob_text = ""
    if bool(R.get("enable_probability", False)):
        prob_text = (
            f"\n\n**Probability Mode:** ON. The chart shows a near-certain view at **{int(R.get('confidence_level', 0.9)*100)}%** confidence "
            "(conservative demand/target + conservative supply)."
        )
    summary_md = f"""
    ### What this model is solving
    This Predictive Staffing Model (PSM) answers one operational question:
    **“Given seasonal volume swings and a constrained hiring pipeline, what staffing plan prevents burnout exposure — and when must recruiting actions occur to make that plan feasible?”**
    To keep the output **accurate**, the model does **not** treat Jan–Dec as a hard boundary where staffing “resets.”
    Instead, it runs a **{R['sim_months']}-month continuous simulation** and then displays a representative 12-month window.
    That is what prevents the “December drops and January restarts” artifact.
    {prob_text}
    ---
    ### How Operations works (demand signal)
    1. You enter an annual average visits/day.
    2. The model applies a baseline seasonality curve and a flu uplift over **{flu_range}**.
    3. It normalizes the result so the **average across the simulation horizon still equals your baseline**.
    4. It converts the forecast into a lean monthly provider demand curve (minimum staffing).
    5. It builds a **recommended (protective) target** using a burnout buffer based on:
       - volume variability,
       - demand spikes,
       - and workload debt when visits/provider exceed your “Safe Visits/Provider/Day”.
    ---
    ### How Reality works (pipeline-constrained supply)
    The supply curve reflects what can actually happen given:
    - Hiring lead time: **{R['pipeline_lead_days']} days (~{R['lead_months']} months)**
    - Recruiting freezes (months where adding capacity is intentionally blocked)
    - Turnover modeled **with notice lag** (separations occur after the notice period)
    - Pipeline visibility blackout months (from **{req_post_label}** through the month before **{hire_visible_label}**)
    Because the simulation is continuous, **December trends carry into January** naturally.
    ---
    ### Auto-Hiring Strategy (v3)
    - **Flu window:** {flu_range}
    - **Freeze window:** {freeze_range}
    - **Recruiting window:** {recruit_range}
    - **Post requisition:** {req_post_label}
    - **Hires visible:** {hire_visible_label}
    - **Independent by:** {independent_label}
    ---
    ### Financial feasibility tied to VVI (Policy A — FTE-based SWB/Visit)
    **Is this staffing plan feasible at the Target SWB/Visit?**
    - **Target SWB/Visit:** ${R['target_swb_per_visit']:.2f}
    - **Modeled SWB/Visit (avg):** ${R['avg_swb_per_visit_modeled']:.2f}
    - **Feasible:** {"YES" if R["swb_feasible"] else "NO"}
    Also computed:
    - **Labor Factor (LF)** = Target SWB/Visit ÷ Modeled SWB/Visit
    ---
    ### What to do next
    To reduce burnout exposure (and improve feasibility), you have three levers:
    1. **Timing:** Post requisitions earlier.
    2. **Ramp capacity:** Improve pipeline throughput so monthly ramp-up is achievable.
    3. **Flex coverage:** Float pool, fractional staffing, buffer coverage to bridge gaps before permanent staff arrives.
    """
    st.markdown(summary_md)
    st.success("✅ Executive Summary complete. Next step: bring RF (NR/Visit) + LF (SWB/Visit) together into a single VVI feasibility scorecard view.")
