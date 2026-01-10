import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ============================================================
# ✅ PAGE CONFIG (OPTION A — Centered)
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM)", layout="centered")

# ============================================================
# ✅ "WIDER BUT NOT WIDE" CONTAINER
# ------------------------------------------------------------
# Streamlit supports only "centered" or "wide" layouts.
# This CSS widens the centered content area without going full wide.
# ============================================================
st.markdown(
    """
    <style>
      /* Widen the centered content area */
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
    "⚠️ **All staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)

model = StaffingModel()


# ============================================================
# ✅ STABLE TODAY (prevents moving windows on reruns)
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()

today = st.session_state["today"]


# ============================================================
# ✅ SESSION STATE
# ============================================================
for key in ["model_ran", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ============================================================
# ✅ HELPERS
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def monthly_index(d: datetime, anchor: datetime):
    return (d.year - anchor.year) * 12 + (d.month - anchor.month)


def base_seasonality_multiplier(month: int):
    """Baseline seasonality curve outside flu uplift."""
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00


def build_flu_window(current_year: int, flu_start_month: int, flu_end_month: int):
    """Builds flu season window between start month and end month (can cross year boundary)."""
    flu_start_date = datetime(current_year, flu_start_month, 1)

    if flu_end_month < flu_start_month:
        flu_end_date = datetime(current_year + 1, flu_end_month, 1)
    else:
        flu_end_date = datetime(current_year, flu_end_month, 1)

    # last day of flu_end month
    flu_end_date = flu_end_date + timedelta(days=32)
    flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)

    return flu_start_date, flu_end_date


def in_window(d: datetime, start: datetime, end: datetime):
    return start <= d <= end


def compute_seasonality_forecast(dates, baseline_visits, flu_start, flu_end, flu_uplift_pct):
    """Applies monthly seasonality multipliers and flu uplift, then normalizes to annual baseline."""
    raw = []
    for d in dates:
        mult = base_seasonality_multiplier(d.month)
        if in_window(d.to_pydatetime(), flu_start, flu_end):
            mult *= (1 + flu_uplift_pct)
        raw.append(baseline_visits * mult)

    avg_raw = np.mean(raw)
    normalized = [v * (baseline_visits / avg_raw) for v in raw]
    return normalized


def visits_to_provider_demand(model, visits_by_month, hours_of_operation, fte_hours_per_week, provider_min_floor):
    """Converts visits/day forecast to provider FTE demand by month."""
    demand = []
    for v in visits_by_month:
        fte = model.calculate_fte_needed(
            visits_per_day=v,
            hours_of_operation_per_week=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week
        )["provider_fte"]
        demand.append(max(fte, provider_min_floor))
    return demand


# ============================================================
# ✅ BURNOUT-PROTECTIVE CURVE (Recommended Target)
# ============================================================

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
    """
    Builds a recommended (burnout-protective) target curve.

    - Buffer is driven by: volume variability, demand spikes, and cumulative workload debt.
    - Smoothing caps month-to-month changes.
    """
    vol_w, spike_w, debt_w = weights

    visits_arr = np.array(visits_by_month)
    mean_visits = np.mean(visits_arr)
    std_visits = np.std(visits_arr)
    cv = (std_visits / mean_visits) if mean_visits > 0 else 0.0
    p75 = np.percentile(visits_arr, 75)

    rdi = 0.0
    decay = 0.85
    lambda_debt = 0.10

    protective_curve = []
    prev_staff = max(base_demand_fte[0], provider_min_floor)

    for v, base_fte in zip(visits_by_month, base_demand_fte):

        vbuf = base_fte * cv
        sbuf = max(0.0, (v - p75) / mean_visits) * base_fte if mean_visits > 0 else 0

        visits_per_provider = v / max(prev_staff, 0.25)
        debt = max(0.0, visits_per_provider - safe_visits_per_provider_per_day)

        rdi = decay * rdi + debt
        dbuf = lambda_debt * rdi

        buffer_fte = burnout_slider * (vol_w * vbuf + spike_w * sbuf + debt_w * dbuf)
        raw_target = max(provider_min_floor, base_fte + buffer_fte)

        delta = raw_target - prev_staff
        if delta > 0:
            delta = clamp(delta, 0.0, smoothing_up)
        else:
            delta = clamp(delta, -smoothing_down, 0.0)

        final_staff = max(provider_min_floor, prev_staff + delta)
        protective_curve.append(final_staff)
        prev_staff = final_staff

    return protective_curve


def in_any_freeze_window(d, freeze_windows):
    """
    Returns True if date `d` falls inside ANY freeze window.
    Supports windows that may cross year boundaries.
    """
    if not freeze_windows:
        return False

    for start, end in freeze_windows:
        if start is None or end is None:
            continue

        # Normal window
        if start <= end and start <= d <= end:
            return True

        # Year-crossing window (e.g., Nov → Mar)
        if end < start and (d >= start or d <= end):
            return True

    return False


def pipeline_supply_curve_v2(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    req_post_date,
    pipeline_lead_days,
    max_hiring_up_after_pipeline,
    confirmed_hire_month=None,       # ✅ month input: 1–12 (preferred)
    confirmed_hire_fte=0.0,
    confirmed_hire_date=None,        # optional exact date input
    max_ramp_down_per_month=0.25,
    seasonality_ramp_enabled=True,
    freeze_windows=None,
):
    """
    Pipeline-aware realistic supply curve (v2).

    ✅ FIXES:
    - Prevents Dec→Jan resets (supply carries forward)
    - Confirmed hire reliably adds FTE in correct month (month-loop safe)
    - Freeze windows supported (list of datetime start/end)
    - Hiring ramp-up blocked during freeze + before hires visible
    - Attrition begins ONLY after notice period

    Inputs:
    - confirmed_hire_month: int (1–12), month-based start (recommended)
    - confirmed_hire_date: date object (optional override)
    """

    if freeze_windows is None:
        freeze_windows = []

    # Attrition per month
    monthly_attrition_fte = baseline_fte * (annual_turnover_rate / 12)

    # Attrition begins AFTER notice lag
    effective_attrition_start = today + timedelta(days=int(notice_days))

    # Hiring becomes visible only after pipeline completes
    hire_visible_date = req_post_date + timedelta(days=int(pipeline_lead_days))

    # ✅ Confirmed hire datetime logic
    confirmed_hire_dt = None
    if confirmed_hire_date:
        confirmed_hire_dt = datetime.combine(confirmed_hire_date, datetime.min.time())
    elif confirmed_hire_month:
        # Find matching month in the dates list
        for d in dates:
            if d.month == confirmed_hire_month:
                confirmed_hire_dt = d.to_pydatetime()
                break

    hire_applied = False
    staff = []
    prev = max(baseline_fte, provider_min_floor)

    for d, target in zip(dates, target_curve):
        d_py = d.to_pydatetime()

        # ✅ Freeze logic (supports multiple windows)
        in_freeze = in_any_freeze_window(d_py, freeze_windows)

        # ✅ Ramp cap logic
        if seasonality_ramp_enabled:
            if in_freeze or (d_py < hire_visible_date):
                ramp_up_cap = 0.0
            else:
                ramp_up_cap = max_hiring_up_after_pipeline
        else:
            ramp_up_cap = 0.35

        # ✅ Move supply toward target
        delta = target - prev
        if delta > 0:
            delta = clamp(delta, 0.0, ramp_up_cap)
        else:
            delta = clamp(delta, -max_ramp_down_per_month, 0.0)

        planned = prev + delta

        # ✅ Attrition applies ONLY after notice lag
        if d_py >= effective_attrition_start:
            planned -= monthly_attrition_fte

        # ✅ Confirmed hire applies once when month reached
        if (not hire_applied) and confirmed_hire_dt and (d_py >= confirmed_hire_dt):
            planned += confirmed_hire_fte
            hire_applied = True

        planned = max(planned, provider_min_floor)

        staff.append(planned)
        prev = planned  # ✅ carry forward ALWAYS

    return staff

# ============================================================
# ✅ COST HELPERS
# ============================================================

def provider_day_gap(target_curve, supply_curve, days_in_month):
    """Total provider-days of under-staffing (area between curves)."""
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(t - s, 0) * dim
    return gap_days


def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    """Annualized cost of added FTE (pro-rated by days in month)."""
    cost = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += dfte * loaded_cost_per_provider_fte * (dim / 365)
    return cost


# ============================================================
# ✅ SIDEBAR (ONLY INPUTS + RUN + TOGGLE)
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
    provider_turnover = st.number_input("Provider Turnover % (annual)", value=24.0, step=1.0) / 100

    with st.expander("Provider Hiring Pipeline Assumptions", expanded=False):
        days_to_sign = st.number_input("Days to Sign", min_value=0, value=90, step=5)
        days_to_credential = st.number_input("Days to Credential", min_value=0, value=90, step=5)
        onboard_train_days = st.number_input("Days to Train", min_value=0, value=30, step=5)
        coverage_buffer_days = st.number_input("Planning Buffer Days", min_value=0, value=14, step=1)
        notice_days = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=90, step=5)

    # ✅ Seasonality (outside the expander)
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
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100

    # ✅ Hiring Freeze Windows
    st.subheader("Hiring Freeze Windows")

    freeze1_start_date = st.date_input(
        "Freeze Window 1 Start",
        value=datetime(today.year, 11, 1).date(),
        help="Example: Nov 1"
    )
    freeze1_end_date = st.date_input(
        "Freeze Window 1 End",
        value=datetime(today.year, 12, 31).date(),
        help="Example: Dec 31"
    )

    freeze2_start_date = st.date_input(
        "Freeze Window 2 Start",
        value=datetime(today.year + 1, 1, 1).date(),
        help="Example: Jan 1"
    )
    freeze2_end_date = st.date_input(
        "Freeze Window 2 End",
        value=datetime(today.year + 1, 4, 30).date(),
        help="Example: Apr 30"
    )

    # ✅ Safety checks
    if freeze1_end_date <= freeze1_start_date:
        st.error("❌ Freeze Window 1 End must be after Start.")
        st.stop()

    if freeze2_end_date <= freeze2_start_date:
        st.error("❌ Freeze Window 2 End must be after Start.")
        st.stop()

    # ✅ Confirmed Hiring
    st.subheader("Confirmed Hiring (Month-Based)")
    
    confirmed_hire_month = st.selectbox(
        "Confirmed Hire Start Month (Independent)",
        options=list(range(1, 13)),
        index=10,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    
    confirmed_hire_fte = st.number_input(
        "Confirmed Hire FTE",
        min_value=0.0,
        value=1.0,
        step=0.25,
        help="FTE capacity of the confirmed hire (e.g. 1.0 or 0.75)."
    )

    enable_seasonality_ramp = st.checkbox(
        "Enable Seasonality Recruiting Ramp",
        value=True,
        help="If ON: supply cannot rise until hires become visible (pipeline completion)."
    )

    st.divider()
    run_model = st.button("Run Model")

def months_between(start_month, end_month):
    """
    Returns list of month numbers in a wrapped window.
    Example: Dec(12) → Feb(2) returns [12,1,2]
    """
    months = []
    m = start_month
    while True:
        months.append(m)
        if m == end_month:
            break
        m = 1 if m == 12 else m + 1
    return months


def shift_month(month, shift):
    """Shift month integer forward/backward with wraparound."""
    return ((month - 1 + shift) % 12) + 1


def month_to_date(dates, month_num):
    """Return datetime from dates matching month_num."""
    for d in dates:
        if d.month == month_num:
            return d.to_pydatetime()
    return None


def auto_hiring_strategy_v2(
    dates,
    flu_start_month,
    flu_end_month,
    pipeline_lead_days,
    notice_days,
    freeze_buffer_months=1,
):
    """
    SMART AUTO-FREEZE V2:
    - Freeze hiring DURING flu window (+ optional buffer months after)
    - Post req far enough in advance so provider is independent by flu start
    - Unfreeze in the months prior to req posting
    """

    # ✅ Flu months (wrap safe)
    flu_months = months_between(flu_start_month, flu_end_month)

    # ✅ Independent month = flu_start_month (core business assumption)
    independent_month = flu_start_month

    # ✅ Lead months from pipeline days
    lead_months = lead_days_to_months(pipeline_lead_days)

    # ✅ Req posting month = independent month minus lead months
    req_post_month = shift_month(independent_month, -lead_months)

    # ✅ Hiring freeze window:
    # Freeze from flu_start → flu_end (+ buffer months after)
    freeze_months = list(flu_months)
    for i in range(1, freeze_buffer_months + 1):
        freeze_months.append(shift_month(flu_end_month, i))

    freeze_months = sorted(set(freeze_months))

    # ✅ Hiring unfreeze months:
    # Unfreeze for all months leading into req_post_month (lead_months months prior)
    unfreeze_months = []
    for i in range(lead_months + 1):
        unfreeze_months.append(shift_month(req_post_month, -i))

    unfreeze_months = sorted(set(unfreeze_months))

    # ✅ Convert months → datetime freeze windows for plotting + model
    freeze_windows = []
    for m in freeze_months:
        start = month_to_date(dates, m)
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(1)).to_pydatetime()
        freeze_windows.append((start, end))

    return dict(
        independent_month=independent_month,
        req_post_month=req_post_month,
        independent_date=month_to_date(dates, independent_month),
        req_post_date=month_to_date(dates, req_post_month),
        freeze_windows=freeze_windows,
        freeze_months=freeze_months,
        unfreeze_months=unfreeze_months,
        lead_months=lead_months,
    )

# ============================================================
# ✅ RUN MODEL
# ============================================================
if run_model:

    # ✅ Month-loop dates anchored to dummy year
    dates = build_month_loop_dates(dummy_year=2000)
    month_labels = [d.strftime("%b") for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    # ============================================================
    # ✅ Baseline provider FTE
    # ============================================================
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

    # ============================================================
    # ✅ Forecast visits w/ seasonality (month-loop)
    # ============================================================
    forecast_visits_by_month = compute_seasonality_forecast(
        dates=dates,
        baseline_visits=visits,
        flu_start=datetime(2000, flu_start_month, 1),
        flu_end=datetime(2000, flu_end_month, 28),  # month-loop safe
        flu_uplift_pct=flu_uplift_pct,
    )

    # ============================================================
    # ✅ Lean demand curve
    # ============================================================
    provider_base_demand = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_by_month,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    # ============================================================
    # ✅ Protective (recommended) curve
    # ============================================================
    protective_curve = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # ============================================================
    # ✅ Pipeline Lead Time
    # ============================================================
    total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

    # ============================================================
    # ✅ AUTO FREEZE + AUTO REQ + AUTO INDEPENDENT DATE
    # ============================================================
    strategy = auto_hiring_strategy_v2(
        dates=dates,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
        pipeline_lead_days=total_lead_days,
        notice_days=notice_days,
        freeze_buffer_months=1,
    )

    req_post_date = strategy["req_post_date"]
    independent_date = strategy["independent_date"]
    freeze_windows = strategy["freeze_windows"]
    months_in_pipeline = strategy["lead_months"]


    # ============================================================
    # ✅ Derived ramp after independent month
    # ============================================================
    peak_idx = strategy["peak_idx"]
    target_at_peak = protective_curve[peak_idx]
    fte_gap_to_close = max(target_at_peak - baseline_provider_fte, 0)

    derived_ramp_after_independent = min(fte_gap_to_close / max(months_in_pipeline, 1), 1.25)

    # ============================================================
    # ✅ Supply curves (lean + recommended)
    # ============================================================
    realistic_supply_lean = pipeline_supply_curve(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        pipeline_lead_days=total_lead_days,
        max_hiring_up_after_pipeline=derived_ramp_after_independent,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        seasonality_ramp_enabled=True,
        freeze_windows=freeze_windows,
    )

    realistic_supply_recommended = pipeline_supply_curve(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=protective_curve,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        pipeline_lead_days=total_lead_days,
        max_hiring_up_after_pipeline=derived_ramp_after_independent,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        seasonality_ramp_enabled=True,
        freeze_windows=freeze_windows,
    )

    # ============================================================
    # ✅ Burnout gap + exposure
    # ============================================================
    burnout_gap_fte = [
        max(t - s, 0)
        for t, s in zip(protective_curve, realistic_supply_recommended)
    ]
    months_exposed = sum(1 for g in burnout_gap_fte if g > 0)

    # ============================================================
    # ✅ Store results
    # ============================================================
    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        dates=dates,
        month_labels=month_labels,
        days_in_month=days_in_month,
        baseline_provider_fte=baseline_provider_fte,

        forecast_visits_by_month=forecast_visits_by_month,

        provider_base_demand=provider_base_demand,
        protective_curve=protective_curve,

        realistic_supply_lean=realistic_supply_lean,
        realistic_supply_recommended=realistic_supply_recommended,

        burnout_gap_fte=burnout_gap_fte,
        months_exposed=months_exposed,

        req_post_date=req_post_date,
        independent_date=independent_date,

        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,

        derived_ramp_after_independent=derived_ramp_after_independent,

        pipeline_lead_days=total_lead_days,
        months_in_pipeline=months_in_pipeline,

        freeze_windows=freeze_windows,
    )

    # --------------------------------------------------------
    # ✅ Burnout gap + exposure
    # --------------------------------------------------------
    burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    months_exposed = sum([1 for g in burnout_gap_fte if g > 0])

    # --------------------------------------------------------
    # ✅ Store results
    # --------------------------------------------------------
    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        dates=dates,
        month_labels=month_labels,
        days_in_month=days_in_month,
        baseline_provider_fte=baseline_provider_fte,

        flu_start_date=flu_start_date,
        flu_end_date=flu_end_date,
        forecast_visits_by_month=forecast_visits_by_month,

        provider_base_demand=provider_base_demand,
        protective_curve=protective_curve,

        realistic_supply_lean=realistic_supply_lean,
        realistic_supply_recommended=realistic_supply_recommended,

        burnout_gap_fte=burnout_gap_fte,
        months_exposed=months_exposed,

        req_post_date=req_post_date,
        independent_ready_date=independent_ready_date,

        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,

        enable_seasonality_ramp=enable_seasonality_ramp,
        derived_ramp_after_independent=derived_ramp_after_independent,

        months_in_flu_window=months_in_flu_window,
        fte_gap_to_close=fte_gap_to_close,
        pipeline_lead_days=total_lead_days,

        freeze_windows=freeze_windows,
    )
# ============================================================
# ✅ RUN MODEL
# ============================================================
if run_model:

    # --------------------------------------------------------
    # ✅ Month Loop — Anchor all months to a dummy year (2000)
    # --------------------------------------------------------
    anchor_year = 2000
    dates = pd.date_range(start=datetime(anchor_year, 1, 1), periods=12, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    # --------------------------------------------------------
    # ✅ Flu Window (month-based)
    # --------------------------------------------------------
    flu_start_date, flu_end_date = build_flu_window(anchor_year, flu_start_month, flu_end_month)

    # --------------------------------------------------------
    # ✅ Freeze windows (still supported for now, can remove later)
    # --------------------------------------------------------
    freeze_windows = [
        (datetime.combine(freeze1_start_date.replace(year=anchor_year), datetime.min.time()),
         datetime.combine(freeze1_end_date.replace(year=anchor_year), datetime.min.time())),
        (datetime.combine(freeze2_start_date.replace(year=anchor_year), datetime.min.time()),
         datetime.combine(freeze2_end_date.replace(year=anchor_year), datetime.min.time())),
    ]

    # --------------------------------------------------------
    # ✅ Baseline provider FTE
    # --------------------------------------------------------
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

    # --------------------------------------------------------
    # ✅ Forecast visits w/ seasonality
    # --------------------------------------------------------
    forecast_visits_by_month = compute_seasonality_forecast(
        dates=dates,
        baseline_visits=visits,
        flu_start=flu_start_date,
        flu_end=flu_end_date,
        flu_uplift_pct=flu_uplift_pct,
    )

    # --------------------------------------------------------
    # ✅ Lean demand curve
    # --------------------------------------------------------
    provider_base_demand = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_by_month,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    # --------------------------------------------------------
    # ✅ Protective curve (recommended)
    # --------------------------------------------------------
    protective_curve = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # --------------------------------------------------------
    # ✅ AUTO-CALCULATED RECRUITING TIMELINE
    # --------------------------------------------------------
    staffing_needed_by = flu_start_date
    total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

    req_post_date = staffing_needed_by - timedelta(days=total_lead_days)
    independent_ready_date = staffing_needed_by

    # --------------------------------------------------------
    # ✅ Flu month index
    # --------------------------------------------------------
    flu_month_idx = next(i for i, d in enumerate(dates) if d.month == flu_start_date.month)

    flu_end_idx = flu_month_idx
    for i, d in enumerate(dates):
        if d.to_pydatetime() <= flu_end_date:
            flu_end_idx = i

    months_in_flu_window = max(flu_end_idx - flu_month_idx + 1, 1)

    target_at_flu = protective_curve[flu_month_idx]
    supply_at_independent = baseline_provider_fte
    fte_gap_to_close = max(target_at_flu - supply_at_independent, 0)

    derived_ramp_after_independent = fte_gap_to_close / months_in_flu_window
    derived_ramp_after_independent = min(derived_ramp_after_independent, 1.25)

    # --------------------------------------------------------
    # ✅ Confirmed Hire Month (apply based on month only)
    # --------------------------------------------------------
    confirmed_hire_month_dt = datetime(anchor_year, confirmed_hire_month, 1)

    # --------------------------------------------------------
    # ✅ SUPPLY CURVES (LEAN + RECOMMENDED)
    # --------------------------------------------------------
    realistic_supply_lean = pipeline_supply_curve(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        pipeline_lead_days=total_lead_days,
        max_hiring_up_after_pipeline=derived_ramp_after_independent,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        freeze_windows=freeze_windows,
    )

    realistic_supply_recommended = pipeline_supply_curve(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=protective_curve,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        pipeline_lead_days=total_lead_days,
        max_hiring_up_after_pipeline=derived_ramp_after_independent,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        freeze_windows=freeze_windows,
    )

    # --------------------------------------------------------
    # ✅ Burnout gap + exposure
    # --------------------------------------------------------
    burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    months_exposed = sum([1 for g in burnout_gap_fte if g > 0])

    # --------------------------------------------------------
    # ✅ Store results
    # --------------------------------------------------------
    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        dates=dates,
        month_labels=month_labels,
        days_in_month=days_in_month,
        baseline_provider_fte=baseline_provider_fte,

        flu_start_date=flu_start_date,
        flu_end_date=flu_end_date,
        forecast_visits_by_month=forecast_visits_by_month,

        provider_base_demand=provider_base_demand,
        protective_curve=protective_curve,

        realistic_supply_lean=realistic_supply_lean,
        realistic_supply_recommended=realistic_supply_recommended,

        burnout_gap_fte=burnout_gap_fte,
        months_exposed=months_exposed,

        req_post_date=req_post_date,
        independent_ready_date=independent_ready_date,

        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,

        enable_seasonality_ramp=enable_seasonality_ramp,
        derived_ramp_after_independent=derived_ramp_after_independent,

        months_in_flu_window=months_in_flu_window,
        fte_gap_to_close=fte_gap_to_close,
        pipeline_lead_days=total_lead_days,

        freeze_windows=freeze_windows,
    )

# ============================================================
# ✅ STOP IF NOT RUN
# ============================================================
if not st.session_state.get("model_ran"):
    st.info("Enter inputs in the sidebar and click **Run Model**.")
    st.stop()

R = st.session_state["results"]


# ============================================================
# ✅ SECTION 1 — OPERATIONS
# ============================================================
st.markdown("---")
st.header("1) Operations — Seasonality Staffing Requirements")
st.caption("Visits/day forecast → staff/day → FTE needed by month (all based on seasonality).")

monthly_rows = []
for month_label, d, v in zip(R["month_labels"], R["dates"], R["forecast_visits_by_month"]):
    fte_staff = model.calculate_fte_needed(
        visits_per_day=v,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week
    )

    provider_day = (fte_staff["provider_fte"] * fte_hours_per_week) / max(hours_of_operation, 1)
    psr_day      = (fte_staff["psr_fte"]      * fte_hours_per_week) / max(hours_of_operation, 1)
    ma_day       = (fte_staff["ma_fte"]       * fte_hours_per_week) / max(hours_of_operation, 1)
    xrt_day      = (fte_staff["xrt_fte"]      * fte_hours_per_week) / max(hours_of_operation, 1)

    monthly_rows.append({
        "Month": month_label,
        "Visits/Day (Forecast)": round(v, 1),

        "Providers Needed/Day": round(provider_day, 2),
        "PSR Needed/Day": round(psr_day, 2),
        "MA Needed/Day": round(ma_day, 2),
        "XRT Needed/Day": round(xrt_day, 2),

        "Provider FTE": round(fte_staff["provider_fte"], 2),
        "PSR FTE": round(fte_staff["psr_fte"], 2),
        "MA FTE": round(fte_staff["ma_fte"], 2),
        "XRT FTE": round(fte_staff["xrt_fte"], 2),
        "Total FTE": round(fte_staff["total_fte"], 2),
    })

ops_df = pd.DataFrame(monthly_rows)
st.dataframe(ops_df, hide_index=True, use_container_width=True)

st.success(
    "**Operations Summary:** This is the seasonality-adjusted demand signal that drives the model. "
    "Lean demand represents minimum coverage; protective demand adds a burnout buffer to protect throughput and quality."
)


# ============================================================
# ✅ SECTION 2 — REALITY (Presentation Ready)
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Compares lean vs recommended targets against realistic supply given hiring lead time + attrition.")

R = st.session_state["results"]

# ------------------------------------------------------------
# ✅ Brand Colors
# ------------------------------------------------------------
BLACK = "#000000"
GOLD  = "#7a6200"
GRAY  = "#B0B0B0"
LIGHT_GRAY = "#EAEAEA"

# ------------------------------------------------------------
# ✅ Helper: month range label (ex: Dec–May)
# ------------------------------------------------------------
def month_range_label(months):
    if not months:
        return "—"
    m_sorted = months[:]  # already looped in order
    start = datetime(2000, m_sorted[0], 1).strftime("%b")
    end   = datetime(2000, m_sorted[-1], 1).strftime("%b")
    return f"{start}–{end}" if start != end else start

freeze_label = month_range_label(R["freeze_months"])
recruit_label = month_range_label(R["recruiting_open_months"])
req_post_label = datetime(2000, R["req_post_month"], 1).strftime("%b")
hire_visible_label = datetime(2000, R["hire_visible_month"], 1).strftime("%b")
independent_label = datetime(2000, R["independent_ready_month"], 1).strftime("%b")

# ------------------------------------------------------------
# ✅ Executive timeline row
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# ✅ Metrics row (clean & centered)
# ------------------------------------------------------------
peak_gap = max(R["burnout_gap_fte"])
avg_gap = float(np.mean(R["burnout_gap_fte"]))

m1, m2, m3 = st.columns(3)
m1.metric("Peak Burnout Gap (FTE)", f"{peak_gap:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
m3.metric("Months Exposed", f"{R['months_exposed']}/12")

# ------------------------------------------------------------
# ✅ Plot Setup (taller + sharper)
# ------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# ------------------------------------------------------------
# ✅ Shade Hiring Freeze Months
# ------------------------------------------------------------
freeze_months = R.get("freeze_months", [])
for i, d in enumerate(R["dates"]):
    month_num = d.month
    if month_num in freeze_months:
        ax1.axvspan(
            d,
            d + timedelta(days=27),
            alpha=0.12,
            color=GOLD,
            linewidth=0
        )

# ------------------------------------------------------------
# ✅ Lines (thin + crisp)
# ------------------------------------------------------------
ax1.plot(
    R["dates"], R["provider_base_demand"],
    linestyle=":", linewidth=1.2, color=GRAY,
    label="Lean Target (Demand)"
)

ax1.plot(
    R["dates"], R["protective_curve"],
    linewidth=2.0, color=GOLD,
    marker="o", markersize=4,
    label="Recommended Target (Protective)"
)

ax1.plot(
    R["dates"], R["realistic_supply_recommended"],
    linewidth=2.0, color=BLACK,
    marker="o", markersize=4,
    label="Realistic Supply (Pipeline)"
)

# ------------------------------------------------------------
# ✅ Burnout Exposure Zone Fill
# ------------------------------------------------------------
ax1.fill_between(
    R["dates"],
    R["realistic_supply_recommended"],
    R["protective_curve"],
    where=np.array(R["protective_curve"]) > np.array(R["realistic_supply_recommended"]),
    color=GOLD,
    alpha=0.12,
    label="Burnout Exposure Zone"
)

# ------------------------------------------------------------
# ✅ Axis Styling
# ------------------------------------------------------------
ax1.set_title("Reality — Targets vs Pipeline-Constrained Supply",
              fontsize=16, fontweight="bold", pad=16, color=BLACK)

ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold", color=BLACK)
ax1.set_xticks(R["dates"])
ax1.set_xticklabels(R["month_labels"], fontsize=11, color=BLACK)

ax1.tick_params(axis='y', labelsize=11, colors=BLACK)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

# ------------------------------------------------------------
# ✅ Visits Secondary Axis
# ------------------------------------------------------------
ax2 = ax1.twinx()
ax2.plot(
    R["dates"], R["forecast_visits_by_month"],
    linestyle="-.", linewidth=1.4,
    color="#666666",
    label="Forecast Visits/Day"
)
ax2.set_ylabel("Visits / Day", fontsize=12, fontweight="bold", color=BLACK)
ax2.tick_params(axis='y', labelsize=11, colors=BLACK)

# ------------------------------------------------------------
# ✅ Clean Legend (bottom)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# ✅ Summary statement (updated language)
# ------------------------------------------------------------
st.success(
    f"**Reality Summary:** To be flu-ready by **{independent_label}**, post requisitions by **{req_post_label}** "
    f"so new hires are visible by **{hire_visible_label}**. "
    f"Protective ramp speed required: **{R['derived_ramp_after_independent_ready']:.2f} FTE/month**."
)

st.info(
    f"🧠 **Auto Hiring Strategy**\n\n"
    f"- Independent by: **{independent_date.strftime('%b %d')}**\n"
    f"- Post req by: **{req_post_date.strftime('%b %d')}**\n"
    f"- Freeze months: **{', '.join([datetime(2000,m,1).strftime('%b') for m in strategy['freeze_months']])}**\n"
    f"- Unfreeze months: **{', '.join([datetime(2000,m,1).strftime('%b') for m in strategy['unfreeze_months']])}**\n"
)

# ============================================================
# ✅ EXECUTIVE TAKEAWAY BOX
# ============================================================

req_post_by = R["req_post_date"].strftime("%b %d")
solo_by = R["solo_ready_date"].strftime("%b %d")
pipeline_days = R["pipeline_lead_days"]

st.markdown(
    f"""
    <div style="
        border-left: 6px solid {BRAND_GOLD};
        background-color: #fafafa;
        padding: 14px 18px;
        border-radius: 10px;
        margin-top: 8px;
        font-size: 15px;
        line-height: 1.45;
    ">
        <b style="color:{BRAND_BLACK}; font-size:16px;">Executive Takeaway</b><br>
        The clinic faces <b>{months_exposed}/12 months</b> of burnout exposure, peaking at <b>{peak_gap:.2f} FTE</b>.
        To be flu-ready by <b>{solo_by}</b>, requisitions must post by <b>{req_post_by}</b>
        (<b>{pipeline_days} pipeline days</b>).
        Without earlier requisition timing or flex coverage, staffing shortfalls will persist through the peak season.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ✅ SECTION 3 — FINANCE (ROI INVESTMENT CASE)
# ============================================================
st.markdown("---")
st.header("3) Finance — ROI Investment Case")
st.caption("Quantifies the investment required to close the gap and the economic value of reducing provider-day shortages.")

# Finance inputs (in main page, not sidebar — sidebar remains inputs only)
st.subheader("Finance Inputs")
colA, colB, colC = st.columns(3)
with colA:
    loaded_cost_per_provider_fte = st.number_input("Loaded Cost per Provider FTE (annual)", value=260000, step=5000)
with colB:
    net_revenue_per_visit = st.number_input("Net Revenue per Visit", value=140.0, step=5.0)
with colC:
    visits_lost_per_provider_day_gap = st.number_input("Visits Lost per 1.0 Provider-Day Gap", value=18.0, step=1.0)

# Calculate annualized cost to meet protective target (added FTE above baseline)
delta_fte_curve = [max(t - R["baseline_provider_fte"], 0) for t in R["protective_curve"]]
annual_investment = annualize_monthly_fte_cost(delta_fte_curve, R["days_in_month"], loaded_cost_per_provider_fte)

# Value of closing burnout gap (provider-day shortages)
gap_days = provider_day_gap(R["protective_curve"], R["realistic_supply_recommended"], R["days_in_month"])
est_visits_lost = gap_days * visits_lost_per_provider_day_gap
est_revenue_lost = est_visits_lost * net_revenue_per_visit

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
# ✅ SECTION 4 — STRATEGY (BUFFER + FLOAT POOL + FRACTIONAL + HYBRID)
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

# Strategy impact model
# - Buffer coverage reduces gap proportionally
# - Float pool and fractional reduce gap directly
# - Hybrid shifts some flex into permanent investment

gap_fte_curve = [max(t - s, 0) for t, s in zip(R["protective_curve"], R["realistic_supply_recommended"])]

effective_gap_curve = []
for g in gap_fte_curve:
    g2 = g * (1 - buffer_pct / 100)
    g2 = max(g2 - float_pool_fte, 0)
    g2 = max(g2 - fractional_fte, 0)
    effective_gap_curve.append(g2)

reduced_gap_days = provider_day_gap([0]*12, effective_gap_curve, R["days_in_month"])  # treat as "remaining" gap
reduced_gap_days = max(gap_days - reduced_gap_days, 0)

est_visits_saved = reduced_gap_days * visits_lost_per_provider_day_gap
est_revenue_saved = est_visits_saved * net_revenue_per_visit

# Hybrid: portion of strategy treated as permanent investment
hybrid_investment = annual_investment * hybrid_slider

sA, sB, sC = st.columns(3)
sA.metric("Provider-Day Gap Reduced", f"{reduced_gap_days:,.0f}")
sB.metric("Est. Revenue Saved", f"${est_revenue_saved:,.0f}")
sC.metric("Hybrid Investment Share", f"${hybrid_investment:,.0f}")

st.success(
    "**Strategy Summary:** Flex levers can reduce exposure faster than permanent hiring. "
    "Use hybrid to transition temporary coverage into permanent staffing once demand proves durable."
)


# ============================================================
# ✅ SECTION 5 — DECISION (EXECUTIVE SUMMARY)
# ============================================================
st.markdown("---")
st.header("5) Decision — Executive Summary")

st.subheader("Decision Snapshot")

peak_gap = max(R["burnout_gap_fte"])
avg_gap = np.mean(R["burnout_gap_fte"])

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Peak Gap (FTE)", f"{peak_gap:.2f}")
    st.metric("Avg Gap (FTE)", f"{avg_gap:.2f}")
    st.metric("Months Exposed", f"{R['months_exposed']}/12")
with col2:
    st.write(
        f"**To be flu-ready by {R['solo_ready_date'].strftime('%b %d')}:**\n"
        f"- Post requisitions by **{R['req_post_date'].strftime('%b %d')}** (lead time: {R['pipeline_lead_days']} days)\n"
        f"- Ramp supply at **{R['derived_ramp_after_solo']:.2f} FTE/month** during flu window\n"
        f"- Annual protective investment: **${annual_investment:,.0f}**\n"
        f"- Estimated revenue at risk if not closed: **${est_revenue_lost:,.0f}**\n"
        f"- ROI: **{roi:,.2f}x**\n\n"
        f"**With strategy levers applied:**\n"
        f"- Provider-day gap reduced: **{reduced_gap_days:,.0f} days**\n"
        f"- Estimated revenue saved: **${est_revenue_saved:,.0f}**\n"
    )

st.success(
    "✅ **Decision Summary:** This model translates seasonality into demand, translates pipeline timing into reality, "
    "and quantifies the ROI of closing the staffing gap. Use the recruiting ramp + strategy levers to move from "
    "reactive coverage to decision-ready staffing."
)
