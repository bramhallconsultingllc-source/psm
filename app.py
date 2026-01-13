import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from collections import deque

from psm.staffing_model import StaffingModel


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM)", layout="centered")

st.markdown(
    """
    <style>
      .block-container {
        max-width: 1200px;
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

model = StaffingModel()

# ============================================================
# STABLE TODAY
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]

# ============================================================
# SESSION STATE
# ============================================================
for key in ["model_ran", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# BRAND COLORS
# ============================================================
BRAND_BLACK = "#000000"
BRAND_GOLD = "#7a6200"
GRAY = "#B0B0B0"
LIGHT_GRAY = "#EAEAEA"
MID_GRAY = "#666666"

# ============================================================
# HELPERS
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(x, hi))

def round_up_quarter(x: float) -> float:
    return math.ceil(float(x) * 4.0) / 4.0

def lead_days_to_months(days: int, avg_days_per_month: float = 30.4) -> int:
    return max(0, int(math.ceil(float(days) / float(avg_days_per_month))))

def shift_month(month: int, shift: int) -> int:
    return ((month - 1 + shift) % 12) + 1

def months_between(start_month: int, end_month: int):
    """Wrapped month list inclusive. Example: Dec->Feb = [12,1,2]."""
    months = []
    m = start_month
    while True:
        months.append(m)
        if m == end_month:
            break
        m = 1 if m == 12 else m + 1
    return months

def month_range_label(months):
    if not months:
        return "—"
    start = datetime(2000, int(months[0]), 1).strftime("%b")
    end = datetime(2000, int(months[-1]), 1).strftime("%b")
    return f"{start}–{end}" if start != end else start

def base_seasonality_multiplier(month: int):
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00

def compute_seasonality_forecast_multiyear(dates, baseline_visits, flu_months, flu_uplift_pct):
    """
    Seasonality uses month-of-year only (repeatable cycle).
    Flu uplift applied to any month in flu_months.
    Normalize so mean(visits) equals baseline across full horizon.
    """
    raw = []
    for d in dates:
        mult = base_seasonality_multiplier(d.month)
        if d.month in set(flu_months):
            mult *= (1.0 + float(flu_uplift_pct))
        raw.append(float(baseline_visits) * mult)

    avg_raw = float(np.mean(raw)) if len(raw) else float(baseline_visits)
    if avg_raw <= 0:
        return [float(baseline_visits) for _ in raw]
    return [v * (float(baseline_visits) / avg_raw) for v in raw]

def visits_to_provider_demand(model, visits_by_month, hours_of_operation, fte_hours_per_week, provider_min_floor):
    demand = []
    for v in visits_by_month:
        fte = model.calculate_fte_needed(
            visits_per_day=float(v),
            hours_of_operation_per_week=float(hours_of_operation),
            fte_hours_per_week=float(fte_hours_per_week)
        )["provider_fte"]
        demand.append(max(float(fte), float(provider_min_floor)))
    return demand

def burnout_protective_staffing_curve(
    visits_by_month,
    base_demand_fte,
    provider_min_floor,
    burnout_slider,
    weights=(0.40, 0.35, 0.25),
    safe_visits_per_provider_per_day=20,
    smoothing_up=0.50,
    smoothing_down=0.25,
):
    vol_w, spike_w, debt_w = weights

    visits_arr = np.array(visits_by_month, dtype=float)
    mean_visits = float(np.mean(visits_arr)) if len(visits_arr) else 0.0
    std_visits = float(np.std(visits_arr)) if len(visits_arr) else 0.0
    cv = (std_visits / mean_visits) if mean_visits > 0 else 0.0
    p75 = float(np.percentile(visits_arr, 75)) if len(visits_arr) else 0.0

    rdi = 0.0
    decay = 0.85
    lambda_debt = 0.10

    protective_curve = []
    prev_staff = max(float(base_demand_fte[0]), float(provider_min_floor))

    for v, base_fte in zip(visits_by_month, base_demand_fte):
        v = float(v)
        base_fte = float(base_fte)

        vbuf = base_fte * cv
        sbuf = max(0.0, (v - p75) / mean_visits) * base_fte if mean_visits > 0 else 0.0

        visits_per_provider = v / max(prev_staff, 0.25)
        debt = max(0.0, visits_per_provider - float(safe_visits_per_provider_per_day))

        rdi = decay * rdi + debt
        dbuf = lambda_debt * rdi

        buffer_fte = float(burnout_slider) * (vol_w * vbuf + spike_w * sbuf + debt_w * dbuf)
        raw_target = max(float(provider_min_floor), base_fte + buffer_fte)

        delta = raw_target - prev_staff
        if delta > 0:
            delta = clamp(delta, 0.0, float(smoothing_up))
        else:
            delta = clamp(delta, -float(smoothing_down), 0.0)

        final_staff = max(float(provider_min_floor), prev_staff + delta)
        protective_curve.append(final_staff)
        prev_staff = final_staff

    return protective_curve

def typical_12_month_curve(dates_full, values_full):
    """Average multi-year values by month-of-year -> 12 values Jan..Dec."""
    df = pd.DataFrame({"date": dates_full, "val": values_full})
    df["month"] = df["date"].dt.month
    g = df.groupby("month")["val"].mean()
    return [float(g.loc[m]) for m in range(1, 13)]

# ============================================================
# AUTO-FREEZE v3 (built from typical seasonal curve)
# ============================================================
def auto_freeze_strategy_v3_from_typical(
    dates_template_12,            # Jan..Dec template dates
    protective_typical_12,        # 12 values
    flu_start_month,
    flu_end_month,
    pipeline_lead_days,
    notice_days,
    freeze_buffer_months=1,
):
    lead_months = lead_days_to_months(pipeline_lead_days)
    notice_months = lead_days_to_months(notice_days)

    independent_ready_month = int(flu_start_month)
    req_post_month = shift_month(independent_ready_month, -lead_months)
    hire_visible_month = shift_month(req_post_month, lead_months)

    trough_idx = int(np.argmin(np.array(protective_typical_12, dtype=float)))
    trough_month = int(dates_template_12[trough_idx].month)

    decline_months = months_between(shift_month(int(flu_end_month), 1), trough_month)

    freeze_months = list(decline_months)
    for i in range(1, int(freeze_buffer_months) + 1):
        freeze_months.append(shift_month(trough_month, i))

    recruiting_open_months = []
    for i in range(lead_months + 1):
        recruiting_open_months.append(shift_month(req_post_month, -i))

    def dedupe_keep_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    freeze_months = dedupe_keep_order(freeze_months)
    recruiting_open_months = dedupe_keep_order(recruiting_open_months)

    # Recruiting-open wins
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
def simulate_supply_multiyear(
    dates_full,
    baseline_provider_fte,
    target_curve_full,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    req_post_month,
    hire_visible_month,
    freeze_months,
    max_hiring_up_after_visible,
    confirmed_hire_month=None,
    confirmed_hire_fte=0.0,
    max_ramp_down_per_month=0.25,
    seasonality_ramp_enabled=True,
):
    """
    Continuous multi-year supply simulation.

    Core behaviors:
    - No reset at year-end (dates are continuous).
    - Attrition uses notice-lag: resignations scheduled now, separations occur notice_months later.
    - Ramp-up is blocked during:
        * freeze_months, and
        * the "pipeline blackout" months (req_post_month..month_before(hire_visible_month)).
      This ensures you can't increase supply until hires are visible, but carry-over from Dec->Jan stays intact.
    """
    notice_months = lead_days_to_months(int(notice_days))
    monthly_turnover_rate = float(annual_turnover_rate) / 12.0

    freeze_set = set(int(m) for m in (freeze_months or []))

    # Months where hiring is "blackout" (no visible hires yet)
    blackout_months = set(months_between(int(req_post_month), shift_month(int(hire_visible_month), -1)))

    # Attrition lag queue
    q = deque([0.0] * max(notice_months, 0), maxlen=max(notice_months, 0) if notice_months > 0 else None)

    staff = []
    prev = max(float(baseline_provider_fte), float(provider_min_floor))
    hire_applied = False

    for i, d in enumerate(dates_full):
        month_num = int(d.month)
        target = float(target_curve_full[i])

        # Can we ramp up this month?
        if seasonality_ramp_enabled:
            in_freeze = month_num in freeze_set
            in_blackout = month_num in blackout_months
            ramp_up_cap = 0.0 if (in_freeze or in_blackout) else float(max_hiring_up_after_visible)
        else:
            ramp_up_cap = 0.35

        # Move toward target
        delta = target - prev
        if delta > 0:
            delta = clamp(delta, 0.0, ramp_up_cap)
        else:
            delta = clamp(delta, -float(max_ramp_down_per_month), 0.0)

        planned = prev + delta

        # Notice-lag attrition:
        # resignations scheduled now based on current supply; separations occur later.
        resignations = prev * monthly_turnover_rate
        separations = 0.0
        if notice_months <= 0:
            separations = resignations
        else:
            q.append(resignations)
            separations = q.popleft()

        planned = planned - separations

        # Confirmed hire (one-time at first occurrence of month)
        if (not hire_applied) and (confirmed_hire_month is not None) and (int(month_num) == int(confirmed_hire_month)):
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

def compute_role_mix_ratios(model: StaffingModel, visits_per_day: float, hours_of_operation: float, fte_hours_per_week: float):
    f = model.calculate_fte_needed(
        visits_per_day=float(visits_per_day),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week)
    )
    prov = max(float(f.get("provider_fte", 0.0)), 0.25)
    return {
        "psr_per_provider": float(f.get("psr_fte", 0.0)) / prov,
        "ma_per_provider": float(f.get("ma_fte", 0.0)) / prov,
        "xrt_per_provider": float(f.get("xrt_fte", 0.0)) / prov,
    }

def compute_monthly_swb_per_visit_fte_based(
    provider_supply_curve_12,
    visits_per_day_curve_12,
    days_in_month_12,
    fte_hours_per_week,
    role_mix,
    hourly_rates,              # dict: {"apc":, "ma":, "psr":, "rt":, "supervisor":, "physician":}
    benefits_load_pct,
    ot_sick_pct,
    physician_supervision_hours_per_month=0.0,
    supervisor_hours_per_month=0.0,
):
    out_rows = []

    for i in range(12):
        prov_fte = float(provider_supply_curve_12[i])
        vpd = float(visits_per_day_curve_12[i])
        dim = int(days_in_month_12[i])
        month_visits = max(vpd * dim, 1.0)

        psr_fte = prov_fte * float(role_mix["psr_per_provider"])
        ma_fte  = prov_fte * float(role_mix["ma_per_provider"])
        rt_fte  = prov_fte * float(role_mix["xrt_per_provider"])

        apc_hours = monthly_hours_from_fte(prov_fte, fte_hours_per_week, dim)
        psr_hours = monthly_hours_from_fte(psr_fte, fte_hours_per_week, dim)
        ma_hours  = monthly_hours_from_fte(ma_fte,  fte_hours_per_week, dim)
        rt_hours  = monthly_hours_from_fte(rt_fte,  fte_hours_per_week, dim)

        apc_rate = loaded_hourly_rate(hourly_rates["apc"], benefits_load_pct, ot_sick_pct)
        psr_rate = loaded_hourly_rate(hourly_rates["psr"], benefits_load_pct, ot_sick_pct)
        ma_rate  = loaded_hourly_rate(hourly_rates["ma"],  benefits_load_pct, ot_sick_pct)
        rt_rate  = loaded_hourly_rate(hourly_rates["rt"],  benefits_load_pct, ot_sick_pct)

        phys_rate = loaded_hourly_rate(hourly_rates["physician"], benefits_load_pct, ot_sick_pct)
        sup_rate  = loaded_hourly_rate(hourly_rates["supervisor"], benefits_load_pct, ot_sick_pct)

        apc_cost = apc_hours * apc_rate
        psr_cost = psr_hours * psr_rate
        ma_cost  = ma_hours  * ma_rate
        rt_cost  = rt_hours  * rt_rate

        phys_cost = float(physician_supervision_hours_per_month) * phys_rate
        sup_cost  = float(supervisor_hours_per_month) * sup_rate

        total_swb = apc_cost + psr_cost + ma_cost + rt_cost + phys_cost + sup_cost
        swb_per_visit = total_swb / month_visits

        out_rows.append({
            "Provider_FTE_Supply": prov_fte,
            "PSR_FTE": psr_fte,
            "MA_FTE": ma_fte,
            "RT_FTE": rt_fte,
            "Visits": month_visits,
            "SWB_$": total_swb,
            "SWB_per_Visit_$": swb_per_visit,
        })

    return pd.DataFrame(out_rows)

# ============================================================
# GAP + COST HELPERS
# ============================================================
def provider_day_gap(target_curve, supply_curve, days_in_month):
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(float(t) - float(s), 0.0) * float(dim)
    return float(gap_days)

def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    cost = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += float(dfte) * float(loaded_cost_per_provider_fte) * (float(dim) / 365.0)
    return float(cost)

# ============================================================
# SIDEBAR INPUTS
# ============================================================
with st.sidebar:
    st.header("Inputs")

    st.subheader("Baseline")
    visits = st.number_input("Avg Visits/Day (annual avg)", min_value=1.0, value=45.0, step=1.0)
    hours_of_operation = st.number_input("Hours of Operation / Week", min_value=1.0, value=70.0, step=1.0)
    fte_hours_per_week = st.number_input("FTE Hours / Week", min_value=1.0, value=40.0, step=1.0)

    st.subheader("Floors & Protection")
    provider_min_floor = st.number_input("Provider Minimum Floor (FTE)", min_value=0.25, value=1.00, step=0.25)
    burnout_slider = st.slider("Burnout Protection Level", 0.0, 1.0, 0.6, 0.05)
    safe_visits_per_provider = st.number_input("Safe Visits/Provider/Day", 10, 40, 20, 1)

    st.subheader("Turnover + Pipeline")
    provider_turnover = st.number_input("Provider Turnover % (annual)", value=24.0, step=1.0) / 100.0

    with st.expander("Provider Hiring Pipeline Assumptions", expanded=False):
        days_to_sign = st.number_input("Days to Sign", min_value=0, value=90, step=5)
        days_to_credential = st.number_input("Days to Credential", min_value=0, value=90, step=5)
        onboard_train_days = st.number_input("Days to Train", min_value=0, value=30, step=5)
        coverage_buffer_days = st.number_input("Planning Buffer Days", min_value=0, value=14, step=1)
        notice_days = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=90, step=5)

    st.subheader("Seasonality")
    flu_start_month = st.selectbox(
        "Flu Start Month",
        options=list(range(1, 13)),
        index=11,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    flu_end_month = st.selectbox(
        "Flu End Month",
        options=list(range(1, 13)),
        index=1,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100.0

    st.subheader("Confirmed Hiring (Month-Based)")
    confirmed_hire_month = st.selectbox(
        "Confirmed Hire Independent Month",
        options=list(range(1, 13)),
        index=10,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    confirmed_hire_fte = st.number_input("Confirmed Hire FTE", min_value=0.0, value=1.0, step=0.25)

    enable_seasonality_ramp = st.checkbox(
        "Enable Seasonality Recruiting Ramp",
        value=True,
        help="If ON: supply cannot rise during recruiting freeze months or during pipeline blackout months."
    )

# ==========================
# Probability / Near-Certain
# ==========================
    st.subheader("Probability (Near-Certain)")
    
    enable_probability = st.checkbox(
        "Enable Probability Mode (Monte Carlo)",
        value=False,
        help="If ON: runs many simulations with uncertainty in visits, turnover, and pipeline time."
    )
    
    confidence_level = st.slider(
        "Near-Certain Confidence Level (quantile)",
        min_value=0.50, max_value=0.95, value=0.90, step=0.05,
        help="Example: 0.90 shows the staffing needed to be sufficient ~90% of the time."
    )
    
    sim_horizon_months = st.slider(
        "Simulation Horizon (months)",
        min_value=24, max_value=60, value=36, step=12,
        help="We simulate a longer timeline, then display only 12 months."
    )
    
    mc_runs = st.slider(
        "Monte Carlo Runs",
        min_value=200, max_value=3000, value=1000, step=100,
        help="More runs = smoother percentiles, but slower."
    )
    
    visits_cv = st.slider(
        "Visits Forecast Variability (CV %)",
        min_value=0.0, max_value=25.0, value=10.0, step=1.0,
        help="Adds realistic variation around the seasonal visit forecast."
    ) / 100.0
    
    turnover_var = st.slider(
        "Turnover Variability (± % of annual turnover)",
        min_value=0.0, max_value=50.0, value=20.0, step=5.0,
        help="Allows month-to-month turnover to vary."
    ) / 100.0
    
    pipeline_var_days = st.slider(
        "Pipeline Duration Variability (± days)",
        min_value=0, max_value=60, value=15, step=5,
        help="Adds uncertainty to total pipeline days."
    )
    
    display_anchor = st.radio(
        "Display Window Anchor",
        options=["Req Post Month", "Flu Start", "January"],
        index=0,
        horizontal=True,
    )
    
    st.subheader("SWB/Visit Feasibility (FTE-based)")
    target_swb_per_visit = st.number_input("Target SWB / Visit ($)", value=85.00, step=1.00)

    with st.expander("Hourly Rates (baseline assumptions)", expanded=False):
        benefits_load_pct = st.number_input("Benefits Load (%)", value=30.00, step=1.00) / 100.0
        ot_sick_pct = st.number_input("OT + Sick/PTO (%)", value=4.00, step=0.50) / 100.0

        physician_hr = st.number_input("Physician (Supervision) $/hr", value=135.79, step=1.00)
        apc_hr = st.number_input("APC $/hr", value=62.00, step=1.00)
        ma_hr = st.number_input("MA $/hr", value=24.14, step=0.50)
        psr_hr = st.number_input("PSR $/hr", value=21.23, step=0.50)
        rt_hr = st.number_input("RT $/hr", value=31.36, step=0.50)
        supervisor_hr = st.number_input("Supervisor $/hr", value=28.25, step=0.50)

    with st.expander("Optional: fixed monthly hours", expanded=False):
        physician_supervision_hours_per_month = st.number_input("Physician supervision hours/month", value=0.0, step=1.0)
        supervisor_hours_per_month = st.number_input("Supervisor hours/month", value=0.0, step=1.0)

    st.divider()
    run_model = st.button("Run Model")

# ============================================================
# RUN MODEL (v5 — multi-year sim, show 12 months)
# ============================================================
if run_model:
    # --- Simulation horizon: 36 months (near-certain trend stabilization) ---
    sim_months = 36
    start_date = datetime(today.year, 1, 1)
    dates_full = pd.date_range(start=start_date, periods=sim_months, freq="MS")

    # --- Base calendar info ---
    days_in_month_full = [pd.Period(d, "M").days_in_month for d in dates_full]

    # --- Flu months (repeatable) ---
    flu_months = months_between(int(flu_start_month), int(flu_end_month))

    # --- Forecast visits across full horizon ---
    forecast_visits_full = compute_seasonality_forecast_multiyear(
        dates=dates_full,
        baseline_visits=visits,
        flu_months=flu_months,
        flu_uplift_pct=flu_uplift_pct
    )

    # --- Baseline provider FTE (from baseline visits) ---
    fte_result = model.calculate_fte_needed(
        visits_per_day=float(visits),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week),
    )
    baseline_provider_fte = max(float(fte_result["provider_fte"]), float(provider_min_floor))

    # --- Lean demand full curve ---
    provider_base_demand_full = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_full,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    # --- Protective target full curve ---
    protective_full = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_full,
        base_demand_fte=provider_base_demand_full,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # --- Pipeline lead time ---
    total_lead_days = int(days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days)

    # --- Strategy built from typical 12-month protective curve (most stable) ---
    dates_template_12 = pd.date_range(start=datetime(2000, 1, 1), periods=12, freq="MS")
    protective_typical_12 = typical_12_month_curve(dates_full, protective_full)

    strategy = auto_freeze_strategy_v3_from_typical(
        dates_template_12=dates_template_12,
        protective_typical_12=protective_typical_12,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
        pipeline_lead_days=total_lead_days,
        notice_days=notice_days,
        freeze_buffer_months=1,
    )

    req_post_month = strategy["req_post_month"]
    hire_visible_month = strategy["hire_visible_month"]
    independent_ready_month = strategy["independent_ready_month"]
    freeze_months = strategy["freeze_months"]
    recruiting_open_months = strategy["recruiting_open_months"]
    lead_months = strategy["lead_months"]

    # --- Derived ramp: gap at flu start (typical) / months in flu window ---
    flu_month_idx_typical = int(flu_start_month) - 1
    months_in_flu_window = max(len(strategy["flu_months"]), 1)

    target_at_flu_typical = float(protective_typical_12[flu_month_idx_typical])
    fte_gap_to_close = max(target_at_flu_typical - baseline_provider_fte, 0.0)
    derived_ramp_after_visible = min(fte_gap_to_close / float(months_in_flu_window), 1.25)

    # --- Supply simulation (continuous, no reset; notice-lag attrition) ---
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
        max_hiring_up_after_visible=derived_ramp_after_visible,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        seasonality_ramp_enabled=enable_seasonality_ramp,
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
        max_hiring_up_after_visible=derived_ramp_after_visible,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        seasonality_ramp_enabled=enable_seasonality_ramp,
    )

    # --- Choose a 12-month display window from stabilized portion (months 13–24) ---
    # Anchor A: start display at Req Post month within year 2.
    # This guarantees carry-over continuity while showing a clean “actions → outcomes” year.
    stabilized_start = 12  # month 13 (0-indexed)
    stabilized_end = 24    # up to month 24 (exclusive for searching anchor)
    anchor_idx = stabilized_start
    for i in range(stabilized_start, stabilized_end):
        if int(dates_full[i].month) == int(req_post_month):
            anchor_idx = i
            break

    display_idx = list(range(anchor_idx, anchor_idx + 12))
    # Safety: if we run past horizon, back up to last 12 months
    if display_idx[-1] >= len(dates_full):
        display_idx = list(range(len(dates_full) - 12, len(dates_full)))

    dates_12 = [dates_full[i] for i in display_idx]
    days_in_month_12 = [days_in_month_full[i] for i in display_idx]
    month_labels_12 = [d.strftime("%b") for d in dates_12]

    visits_12 = [forecast_visits_full[i] for i in display_idx]
    demand_lean_12 = [provider_base_demand_full[i] for i in display_idx]
    target_prot_12 = [protective_full[i] for i in display_idx]
    supply_lean_12 = [supply_lean_full[i] for i in display_idx]
    supply_rec_12 = [supply_rec_full[i] for i in display_idx]

    full_hire_points = [(d.strftime("%Y-%m"), float(s)) for d, s in zip(dates_full, supply_rec_full) if d.month == hire_month]
    st.caption(f"DEBUG: supply points in hire month (FULL horizon) = {full_hire_points[:6]} ...")

    # ==========================
    # TRUTH TEST (debug)
    # ==========================
    hire_month = int(confirmed_hire_month)
    hire_fte = float(confirmed_hire_fte)
    
    # 1) Does the DISPLAY window include the hire month?
    display_months = [d.month for d in dates_12]
    st.caption(f"DEBUG: display months = {[datetime(2000,m,1).strftime('%b') for m in display_months]}")
    
    # 2) What is supply in the hire month inside the display window?
    hire_points = [(d.strftime("%Y-%m"), float(s)) for d, s in zip(dates_12, supply_rec_12) if d.month == hire_month]
    st.caption(f"DEBUG: hire month = {datetime(2000,hire_month,1).strftime('%b')} | hire fte = {hire_fte:.2f}")
    st.caption(f"DEBUG: supply points in hire month (display window) = {hire_points}")


    # Round outputs up to quarter FTE (display + business rule)
    demand_lean_12 = [round_up_quarter(x) for x in demand_lean_12]
    target_prot_12 = [round_up_quarter(x) for x in target_prot_12]
    supply_rec_12 = [round_up_quarter(x) for x in supply_rec_12]
    supply_lean_12 = [round_up_quarter(x) for x in supply_lean_12]

    # Burnout exposure (display window)
    burnout_gap_fte_12 = [max(float(t) - float(s), 0.0) for t, s in zip(target_prot_12, supply_rec_12)]
    months_exposed_12 = int(sum(1 for g in burnout_gap_fte_12 if g > 0))

    # --- SWB/Visit Feasibility (Policy A) on display window ---
    role_mix = compute_role_mix_ratios(
        model=model,
        visits_per_day=visits,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week
    )

    hourly_rates = {
        "physician": physician_hr,
        "apc": apc_hr,
        "ma": ma_hr,
        "psr": psr_hr,
        "rt": rt_hr,
        "supervisor": supervisor_hr
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

    # Store results
    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        # Display window
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

        # Strategy
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

        # Baseline
        baseline_provider_fte=baseline_provider_fte,

        # SWB feasibility
        target_swb_per_visit=float(target_swb_per_visit),
        avg_swb_per_visit_modeled=avg_swb_per_visit_modeled,
        swb_feasible=feasible,
        labor_factor=float(labor_factor) if np.isfinite(labor_factor) else None,
        swb_df=swb_df,

        # For narrative / transparency
        sim_months=sim_months,
        display_anchor_month=req_post_month,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        enable_seasonality_ramp=enable_seasonality_ramp,
    )

# ============================================================
# STOP IF NOT RUN
# ============================================================
if not st.session_state.get("model_ran"):
    st.info("Enter inputs in the sidebar and click **Run Model**.")
    st.stop()

R = st.session_state["results"]

# ============================================================
# SECTION 1 — OPERATIONS
# ============================================================
st.markdown("---")
st.header("1) Operations — Seasonality Staffing Requirements")
st.caption("Visits/day forecast → FTE needed by month (seasonality + flu uplift).")

monthly_rows = []
for month_label, d, v in zip(R["month_labels"], R["dates"], R["forecast_visits_by_month"]):
    fte_staff = model.calculate_fte_needed(
        visits_per_day=float(v),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week)
    )
    monthly_rows.append({
        "Month": month_label,
        "Visits/Day (Forecast)": round(float(v), 1),
        "Provider FTE": round_up_quarter(fte_staff["provider_fte"]),
        "PSR FTE": round(float(fte_staff["psr_fte"]), 2),
        "MA FTE": round(float(fte_staff["ma_fte"]), 2),
        "XRT FTE": round(float(fte_staff["xrt_fte"]), 2),
        "Total FTE": round(float(fte_staff["total_fte"]), 2),
    })

ops_df = pd.DataFrame(monthly_rows)
st.dataframe(ops_df, hide_index=True, use_container_width=True)

st.success(
    "**Operations Summary:** This is the seasonality-adjusted demand signal. "
    "Lean demand is minimum coverage; the protective target adds a burnout buffer to protect throughput and quality."
)

# ============================================================
# SECTION 2 — REALITY
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Targets vs realistic supply when hiring is constrained by lead time, freezes, turnover (with notice lag), and pipeline visibility.")

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

m1, m2, m3 = st.columns(3)
m1.metric("Peak Burnout Gap (FTE)", f"{peak_gap:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
m3.metric("Months Exposed", f"{R['months_exposed']}/12")

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

freeze_set = set(R.get("freeze_months", []))
for d in R["dates"]:
    if int(d.month) in freeze_set:
        ax1.axvspan(d, d + timedelta(days=27), alpha=0.12, color=BRAND_GOLD, linewidth=0)

ax1.plot(R["dates"], R["provider_base_demand"], linestyle=":", linewidth=1.2, color=GRAY, label="Lean Target (Demand)")
ax1.plot(R["dates"], R["protective_curve"], linewidth=2.0, color=BRAND_GOLD, marker="o", markersize=4,
         label="Recommended Target (Protective)")
ax1.plot(R["dates"], R["realistic_supply_recommended"], linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=4,
         label="Realistic Supply (Pipeline)")

ax1.fill_between(
    R["dates"],
    np.array(R["realistic_supply_recommended"], dtype=float),
    np.array(R["protective_curve"], dtype=float),
    where=np.array(R["protective_curve"], dtype=float) > np.array(R["realistic_supply_recommended"], dtype=float),
    color=BRAND_GOLD,
    alpha=0.12,
    label="Burnout Exposure Zone"
)

ax1.set_title("Reality — Targets vs Pipeline-Constrained Supply", fontsize=16, fontweight="bold", pad=16, color=BRAND_BLACK)
ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax1.set_xticks(R["dates"])
ax1.set_xticklabels(R["month_labels"], fontsize=11, color=BRAND_BLACK)
ax1.tick_params(axis="y", labelsize=11, colors=BRAND_BLACK)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

ax2 = ax1.twinx()
ax2.plot(R["dates"], R["forecast_visits_by_month"], linestyle="-.", linewidth=1.4, color=MID_GRAY, label="Forecast Visits/Day")
ax2.set_ylabel("Visits / Day", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax2.tick_params(axis="y", labelsize=11, colors=BRAND_BLACK)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False,
    fontsize=11
)

plt.tight_layout()
st.pyplot(fig)

st.success(
    f"**Reality Summary:** This 12-month view is taken from a **{R['sim_months']}-month continuous simulation** "
    f"(no year-end reset). To be flu-ready by **{independent_label}**, requisitions must post by **{req_post_label}** "
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

# ------------------------------------------------------------
# Confirmed hire marker (thin vertical line)
# ------------------------------------------------------------
confirmed_month = int(R.get("confirmed_hire_month", 0) or 0)
confirmed_fte = float(R.get("confirmed_hire_fte", 0.0) or 0.0)

# Only draw if meaningful
if confirmed_month in range(1, 13) and confirmed_fte > 0:
    # Find the confirmed hire date inside the DISPLAY window
    confirmed_date = None
    for d in R["dates"]:
        if int(d.month) == confirmed_month:
            confirmed_date = d
            break

    if confirmed_date is not None:
        ax1.axvline(
            confirmed_date,
            color=BRAND_BLACK,
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            zorder=1
        )

        # Optional small label near the top of plot area
        y_top = ax1.get_ylim()[1]
        ax1.text(
            confirmed_date,
            y_top,
            f" Confirmed Hire (+{confirmed_fte:.2f} FTE)",
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            color=BRAND_BLACK,
            alpha=0.8
        )

# ============================================================
# SECTION 3 — FINANCE
# ============================================================
st.markdown("---")
st.header("3) Finance — ROI Investment Case")
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

st.success(
    "**Finance Summary:** The investment is the cost of staffing to the protective curve. "
    "The value is the revenue protected by reducing provider-day shortages during peak demand."
)

# ============================================================
# SECTION 3B — VVI FEASIBILITY (SWB/Visit)
# ============================================================
st.markdown("---")
st.header("3B) VVI Feasibility — SWB/Visit (FTE-based)")
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
    use_container_width=True
)

fig2, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(R["dates"], swb_df["SWB_per_Visit_$"].values, linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=3, label="Modeled SWB/Visit")
ax.axhline(R["target_swb_per_visit"], linewidth=2.0, linestyle="--", color=BRAND_GOLD, label="Target SWB/Visit")
ax.set_title("SWB/Visit Feasibility (Modeled vs Target)", fontsize=14, fontweight="bold")
ax.set_ylabel("$/Visit")
ax.set_xticks(R["dates"])
ax.set_xticklabels(R["month_labels"])
ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
st.pyplot(fig2)

lf = R.get("labor_factor", None)
lf_str = f"{lf:.2f}" if isinstance(lf, float) else "—"
st.info(
    f"**Labor Factor (LF)** = Target SWB/Visit ÷ Modeled SWB/Visit = **{lf_str}**. "
    "This is a clean building block for VVI feasibility logic (LF > 1.0 is favorable; LF < 1.0 means labor is too expensive for the target)."
)

# ============================================================
# SECTION 4 — STRATEGY
# ============================================================
st.markdown("---")
st.header("4) Strategy — Closing the Gap with Flexible Coverage")
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

remaining_gap_days = provider_day_gap([0]*12, effective_gap_curve, R["days_in_month"])
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
st.markdown("---")
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
        f"- SWB/Visit feasibility vs target: **{'YES' if R['swb_feasible'] else 'NO'}** (Modeled ${R['avg_swb_per_visit_modeled']:.2f} vs Target ${R['target_swb_per_visit']:.2f})\n\n"
        f"**With strategy levers applied:**\n"
        f"- Provider-day gap reduced: **{reduced_gap_days:,.0f} days**\n"
        f"- Estimated revenue saved: **${est_revenue_saved:,.0f}**\n"
    )

st.success(
    "✅ **Decision Summary:** This model converts seasonality into staffing demand, converts pipeline timing into realistic supply, "
    "and quantifies burnout exposure, ROI, and SWB/Visit feasibility. "
    "Use the auto-hiring strategy + flex levers to move from reactive coverage to decision-ready staffing."
)

# ============================================================
# SECTION 6 — IN-DEPTH EXECUTIVE SUMMARY (Narrative)
# ============================================================
st.markdown("---")
st.header("6) In-Depth Executive Summary")
st.caption("A plain-language explanation of what the model is doing, what it found, and how to act on it.")

flu_range = month_range_label(R["flu_months"])
freeze_range = freeze_label
recruit_range = recruit_label

summary_md = f"""
### What this model is solving
This Predictive Staffing Model (PSM) answers one operational question:

**“Given seasonal volume swings and a constrained hiring pipeline, what staffing plan prevents burnout exposure — and when must recruiting actions occur to make that plan feasible?”**

To keep the output **accurate**, the model does **not** treat Jan–Dec as a hard boundary where staffing “resets.”
Instead, it runs a **{R['sim_months']}-month continuous simulation** and then displays a representative 12-month window.

That is exactly what prevents the “December drops and January restarts” artifact.

---

### How Operations works (demand signal)
1. You enter an annual average visits/day.
2. The model applies a baseline seasonality curve and a flu uplift over **{flu_range}**.
3. It normalizes the result so the **average across the simulation horizon still equals your baseline**.
4. It converts the forecast into a lean monthly provider demand curve (minimum staffing).

Then it builds a **recommended (protective) target** that adds a burnout buffer based on:
- volume variability,
- demand spikes,
- and cumulative workload debt when visits per provider exceed your “Safe Visits/Provider/Day”.

---

### How Reality works (pipeline-constrained supply)
The supply curve reflects what can actually happen given:
- Hiring lead time: **{R['pipeline_lead_days']} days (~{R['lead_months']} months)**
- Recruiting freezes (months where adding capacity is intentionally blocked)
- Turnover modeled **with notice lag** (separations occur after the notice period, not instantly)
- Pipeline visibility blackout months (from **{req_post_label}** through the month before **{hire_visible_label}**)

Because the simulation is continuous, **December trends carry into January** naturally.

---

### Auto-Hiring Strategy (v3)
The model recommends:
- **Flu window:** {flu_range}
- **Freeze window:** {freeze_range}
- **Recruiting window:** {recruit_range}
- **Post requisition:** {req_post_label}
- **Hires visible:** {hire_visible_label}
- **Independent by:** {independent_label}

This matches your intent: freeze ahead of demand decreases (knowing separations lag), and post requisitions early enough to hit flu season with independent capacity.

---

### Financial feasibility tied to VVI (Policy A — FTE-based SWB/Visit)
The model uses your hourly assumptions to test:

**Is this staffing plan feasible at the Target SWB/Visit?**

- **Target SWB/Visit:** ${R['target_swb_per_visit']:.2f}  
- **Modeled SWB/Visit (avg):** ${R['avg_swb_per_visit_modeled']:.2f}  
- **Feasible:** {"YES" if R["swb_feasible"] else "NO"}

It also computes a VVI-ready building block:
- **Labor Factor (LF)** = Target SWB/Visit ÷ Modeled SWB/Visit

---

### What it found (burnout exposure)
- **Peak burnout gap:** {peak_gap:.2f} FTE  
- **Average burnout gap:** {avg_gap:.2f} FTE  
- **Months exposed:** {R['months_exposed']}/12

---

### What to do next
To reduce burnout exposure (and improve feasibility), you have three levers:
1. **Timing:** Post requisitions earlier (shifts hires visible earlier).
2. **Ramp capacity:** Improve pipeline throughput so monthly ramp-up is achievable.
3. **Flex coverage:** Float pool, fractional staffing, buffer coverage to bridge gaps before permanent staff arrives.

"""
st.markdown(summary_md)

st.success("✅ Executive Summary complete. Next step: bring RF (NR/Visit) + LF (SWB/Visit) together into a single VVI feasibility scorecard view.")
