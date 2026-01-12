# app.py — Predictive Staffing Model (PSM) v4.7
# Major updates (per latest alignment):
# 1) Recruiting Window is a TRUE window: Post Req (latest) → Freeze By (latest allowed)
#    - NOT "all months leading up to Post Req"
# 2) Freeze months NEVER overlap recruiting window (recruiting wins)
# 3) Freeze-by logic is turnover + notice-lag aware:
#    - Stop permanent hiring early enough that expected attrition (after notice) can shed incremental flu staffing (X)
#      by the baseline/trough month.
# 4) Reality Supply is a STOCK with a PIPELINE QUEUE:
#    - Decision variable is "hire starts" in recruiting months
#    - Those hires only convert to usable supply after lead_months (with optional ramp to independence)
#    - Supply does NOT snap up instantly in late months
#    - Supply does NOT ramp down just because demand dips (declines via attrition / no replacement)
# 5) Display remains Jan–Dec; carryover statement = Dec ending supply (starting supply next Jan)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

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
    "⚠️ **All staffing outputs round UP to the nearest 0.25 FTE/day.** "
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


def base_seasonality_multiplier(month: int) -> float:
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

    flu_end_date = flu_end_date + timedelta(days=32)
    flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)
    return flu_start_date, flu_end_date


def in_window(d: datetime, start: datetime, end: datetime) -> bool:
    return start <= d <= end


def compute_seasonality_forecast(dates, baseline_visits, flu_start, flu_end, flu_uplift_pct):
    """Seasonality + flu uplift, normalized back to baseline annual average."""
    raw = []
    for d in dates:
        mult = base_seasonality_multiplier(d.month)
        if in_window(d.to_pydatetime(), flu_start, flu_end):
            mult *= (1 + flu_uplift_pct)
        raw.append(baseline_visits * mult)

    avg_raw = float(np.mean(raw)) if len(raw) else baseline_visits
    return [v * (baseline_visits / avg_raw) for v in raw]


def visits_to_provider_demand(model, visits_by_month, hours_of_operation, fte_hours_per_week, provider_min_floor):
    """Visits/day -> provider FTE demand."""
    demand = []
    for v in visits_by_month:
        fte = model.calculate_fte_needed(
            visits_per_day=v,
            hours_of_operation_per_week=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week
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
    """Protective target curve with buffer + smoothing."""
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
        vbuf = base_fte * cv
        sbuf = max(0.0, (v - p75) / mean_visits) * base_fte if mean_visits > 0 else 0.0

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


def months_between(start_month, end_month):
    """Wrapped month window, inclusive. Example: Nov->Feb = [11,12,1,2]."""
    months = []
    m = start_month
    while True:
        months.append(m)
        if m == end_month:
            break
        m = 1 if m == 12 else m + 1
    return months


def shift_month(month: int, shift: int) -> int:
    """Shift month int with wrap."""
    return ((month - 1 + shift) % 12) + 1


def lead_days_to_months(days: int, avg_days_per_month: float = 30.4) -> int:
    return max(0, int(math.ceil(days / avg_days_per_month)))


def month_index_map(dates):
    # Safe because within a 12-month run each month appears once
    return {d.month: i for i, d in enumerate(dates)}


def month_range_label(months):
    if not months:
        return "—"
    start = datetime(2000, months[0], 1).strftime("%b")
    end = datetime(2000, months[-1], 1).strftime("%b")
    return f"{start}–{end}" if start != end else start


def dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ============================================================
# DISCRETE ATTRITION (notice-lagged; no wrap)
# ============================================================
def build_attrition_schedule_discrete_one_year(
    n_months: int,
    expected_avg_fte: float,
    annual_turnover_rate: float,
    notice_months: int,
    fte_granularity: float = 0.25,
):
    """
    Discrete attrition events distributed across the year.
    Notice lag shifts the capacity drop into later months.
    IMPORTANT: drops that would occur after December are NOT wrapped into January.
    """
    annual_expected_separations_fte = max(0.0, expected_avg_fte * annual_turnover_rate)
    if annual_expected_separations_fte <= 0:
        return [0.0] * n_months

    # Quantize into 0.25 events + spread any remainder softly (prevents cliffs at low-FTE clinics)
    full_events = int(math.floor(annual_expected_separations_fte / fte_granularity))
    remainder = annual_expected_separations_fte - full_events * fte_granularity

    idxs = []
    if full_events > 0:
        for k in range(full_events):
            idxs.append(int(math.floor(k * n_months / full_events)))
        idxs = [clamp(i, 0, n_months - 1) for i in idxs]

    drop = [0.0] * n_months

    # Apply discrete events
    for i in idxs:
        j = i + int(notice_months)
        if 0 <= j < n_months:
            drop[j] += fte_granularity

    # Spread remainder (still notice-lagged)
    if remainder > 1e-9:
        for m in range(n_months):
            j = m + int(notice_months)
            if 0 <= j < n_months:
                drop[j] += remainder / n_months

    return drop


# ============================================================
# CONFIRMED HIRE RAMP (month-based independence; no wrap)
# ============================================================
def build_confirmed_hire_ramp_one_year(
    dates,
    independent_month: int | None,
    hire_fte: float,
    ramp_months: int,
):
    """
    Additive FTE contribution:
    - ramps up in the months leading to independence
    - full from independence through end of horizon
    - no wrap beyond the 12 months
    """
    n = len(dates)
    add = [0.0] * n
    if independent_month is None or hire_fte <= 0:
        return add

    m2i = month_index_map(dates)
    independent_idx = m2i.get(int(independent_month), None)
    if independent_idx is None:
        return add

    ramp_months = max(0, int(ramp_months))

    if ramp_months == 0:
        for i in range(independent_idx, n):
            add[i] += float(hire_fte)
        return add

    start_idx = max(0, independent_idx - ramp_months)
    steps = independent_idx - start_idx
    if steps <= 0:
        for i in range(independent_idx, n):
            add[i] += float(hire_fte)
        return add

    # ramp segment
    for s, i in enumerate(range(start_idx, independent_idx)):
        frac = (s + 1) / steps
        add[i] += float(hire_fte) * frac

    # full segment
    for i in range(independent_idx, n):
        add[i] += float(hire_fte)

    return add


# ============================================================
# AUTO-STRATEGY v5 (Recruiting Window = Post Req → Freeze By)
# ============================================================
def auto_strategy_v5(
    dates,
    protective_curve,
    baseline_provider_fte,
    annual_turnover_rate,
    flu_start_month,
    pipeline_lead_days,
    notice_days,
    freeze_buffer_months=1,
):
    """
    Produces:
    - Post Req month (latest acceptable posting month for readiness)
    - Hires Visible month (Post Req + lead months)
    - Freeze By month (latest posting month allowed so turnover can shed incremental flu staffing by trough/baseline)
    - Recruiting window months (Post Req → Freeze By)
    - Freeze months (after Freeze By through trough/baseline, excluding recruiting months)

    Key properties:
    - Recruiting and Freeze NEVER overlap (recruiting wins).
    - Freeze-by incorporates notice lag (attrition hits capacity after notice).
    """
    n = len(dates)
    m2i = month_index_map(dates)

    lead_months = lead_days_to_months(int(pipeline_lead_days))
    notice_months = lead_days_to_months(int(notice_days))

    # "Independent by" month = flu start month (readiness target)
    independent_ready_month = int(flu_start_month)

    # Latest acceptable month to POST to be independent by readiness (best-effort month math)
    req_post_month = shift_month(independent_ready_month, -lead_months)

    # Hires become "visible" (i.e., independent capacity appears) after lead months
    hire_visible_month = shift_month(req_post_month, lead_months)

    # Baseline / trough month = min protective need month
    trough_idx = int(np.argmin(np.array(protective_curve)))
    trough_month = int(dates[trough_idx].month)

    # Incremental flu staffing to "shed" (X):
    # Use protective target at readiness minus baseline demand at trough (this is the "extra seasonal headcount").
    ready_idx = m2i.get(independent_ready_month, 0)
    target_at_ready = float(protective_curve[ready_idx])
    baseline_at_trough = float(protective_curve[trough_idx])
    X = max(target_at_ready - baseline_at_trough, 0.0)

    # Baseline B for attrition expectation (use max of baseline staffing need and baseline supply)
    B = max(float(baseline_provider_fte), float(baseline_at_trough), 0.25)
    T = max(float(annual_turnover_rate), 0.0)

    # Daily expected attrition in FTE/day
    attr_per_day = (B * T) / 365.0 if (B > 0 and T > 0) else 0.0

    # Days needed (in expectation) to shed X
    if attr_per_day > 0:
        delta_days = X / attr_per_day
    else:
        # If turnover is 0, you cannot "shed" via turnover; freeze-by becomes effectively "never"
        delta_days = float("inf")

    delta_months = 0 if not np.isfinite(delta_days) else int(math.ceil(delta_days / 30.4))

    # Total months required before trough for staffing to shed:
    # notice lag (when attrition hits capacity) + time to shed X
    total_shed_months = int(notice_months + delta_months)

    # Freeze By = latest month you can keep posting permanent reqs
    # and still have expected attrition reduce X by trough month.
    if np.isfinite(delta_days):
        freeze_by_month = shift_month(trough_month, -total_shed_months)
    else:
        # If no turnover, freeze-by is effectively trough month (or earlier) to avoid overstaffing;
        # we clamp to just before trough buffer.
        freeze_by_month = shift_month(trough_month, -notice_months)

    # Recruiting window = Post Req → Freeze By (inclusive)
    # If freeze_by is "before" post_req in calendar sense, months_between will wrap; that's acceptable as a plan window.
    recruiting_open_months = months_between(req_post_month, freeze_by_month)

    # Freeze months: after Freeze By through trough (plus buffer), excluding recruiting months
    freeze_start = shift_month(freeze_by_month, 1)
    freeze_end = trough_month
    freeze_months = months_between(freeze_start, freeze_end)

    # Add buffer after trough
    for i in range(1, int(freeze_buffer_months) + 1):
        freeze_months.append(shift_month(trough_month, i))
    freeze_months = dedupe_keep_order(freeze_months)

    # Remove recruiting overlap (hard rule)
    recruiting_set = set(recruiting_open_months)
    freeze_months = [m for m in freeze_months if m not in recruiting_set]

    return dict(
        lead_months=lead_months,
        notice_months=notice_months,
        independent_ready_month=independent_ready_month,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        trough_month=trough_month,
        recruiting_open_months=recruiting_open_months,
        freeze_months=freeze_months,
        X_incremental_flu_fte=X,
        baseline_at_trough=baseline_at_trough,
    )


# ============================================================
# PIPELINE SUPPLY (hire-starts → delayed graduation; no instant supply)
# ============================================================
def add_pipeline_contribution(
    add_curve,
    start_idx: int,
    start_fte: float,
    lead_months: int,
    ramp_months: int,
):
    """
    Convert a hire START at start_idx into supply additions over time.

    - Graduation/independence occurs at grad_idx = start_idx + lead_months
    - Optional ramp occurs in the ramp_months immediately preceding grad_idx
      (linear ramp). Full FTE from grad_idx onward.
    - No wrap beyond 12-month horizon.
    """
    n = len(add_curve)
    if start_fte <= 0:
        return

    grad_idx = start_idx + int(lead_months)
    if grad_idx >= n:
        return  # becomes independent next year; outside this 12-month horizon

    ramp_months = max(0, int(ramp_months))

    if ramp_months == 0:
        for i in range(grad_idx, n):
            add_curve[i] += float(start_fte)
        return

    ramp_start = max(0, grad_idx - ramp_months)
    steps = grad_idx - ramp_start
    if steps <= 0:
        for i in range(grad_idx, n):
            add_curve[i] += float(start_fte)
        return

    # ramp segment (partial contribution)
    for s, i in enumerate(range(ramp_start, grad_idx)):
        frac = (s + 1) / steps
        add_curve[i] += float(start_fte) * frac

    # full segment
    for i in range(grad_idx, n):
        add_curve[i] += float(start_fte)


def simulate_supply_with_pipeline_decisions(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_months,
    recruiting_open_months,
    freeze_months,
    lead_months,
    hire_start_cap_per_month,
    confirmed_independent_month=None,
    confirmed_hire_fte=0.0,
    confirmed_ramp_months=1,
    pipeline_ramp_months=1,
):
    """
    Reality supply simulation with:
    - Attrition (notice-lag)
    - Confirmed hire ramp
    - Discretionary hire STARTS only during recruiting window (not during freeze)
      which become supply after lead_months (with optional ramp)
    - Supply does NOT ramp down with seasonality; only declines via attrition / no replacement.

    "Best case" policy:
    - In each recruiting-allowed month, start enough hires (capped) so that projected supply
      at readiness month meets the target as closely as possible.
    """
    n = len(dates)
    m2i = month_index_map(dates)

    # Attrition schedule
    attr_drop = build_attrition_schedule_discrete_one_year(
        n_months=n,
        expected_avg_fte=max(float(baseline_fte), float(provider_min_floor)),
        annual_turnover_rate=float(annual_turnover_rate),
        notice_months=int(notice_months),
        fte_granularity=0.25,
    )

    # Confirmed hire schedule
    confirmed_add = build_confirmed_hire_ramp_one_year(
        dates=dates,
        independent_month=int(confirmed_independent_month) if confirmed_independent_month else None,
        hire_fte=float(confirmed_hire_fte),
        ramp_months=int(confirmed_ramp_months),
    )

    recruiting_set = set(recruiting_open_months or [])
    freeze_set = set(freeze_months or [])

    # We will build discretionary pipeline additions over time as we decide hire starts
    pipeline_add = [0.0] * n
    hire_starts = [0.0] * n

    # Choose a readiness month = flu start month if present in target curve logic:
    # We infer it as the month where target is max in winter (or simply the month of max target).
    # BUT: better is to use the month with the maximum target (protective) as readiness anchor.
    ready_idx = int(np.argmax(np.array(target_curve)))
    ready_month = int(dates[ready_idx].month)

    prev = max(float(baseline_fte), float(provider_min_floor))
    supply = []

    for i, d in enumerate(dates):
        month_num = int(d.month)

        # Apply stock evolution for this month
        planned = prev

        # Attrition hits after notice lag (already baked into attr_drop)
        planned -= float(attr_drop[i])

        # Confirmed hire adds
        planned += float(confirmed_add[i])

        # Pipeline additions already scheduled from prior hire starts
        planned += float(pipeline_add[i])

        # Decide hire starts (only during recruiting window, and not during freeze)
        can_start = (month_num in recruiting_set) and (month_num not in freeze_set)

        if can_start and i <= ready_idx:
            # Project supply at readiness if we make NO additional starts after this month
            # (fast forward using known schedules + current state)
            proj_prev = planned
            proj_supply = proj_prev

            for j in range(i + 1, ready_idx + 1):
                proj_supply = proj_supply
                proj_supply -= float(attr_drop[j])
                proj_supply += float(confirmed_add[j])
                proj_supply += float(pipeline_add[j])
                # no additional discretionary starts assumed
                proj_supply = max(proj_supply, float(provider_min_floor))

            gap_at_ready = max(float(target_curve[ready_idx]) - proj_supply, 0.0)

            # Count remaining recruiting-allowed months from i..ready_idx
            remaining_idxs = []
            for k in range(i, ready_idx + 1):
                mn = int(dates[k].month)
                if (mn in recruiting_set) and (mn not in freeze_set):
                    remaining_idxs.append(k)

            remaining_months = max(len(remaining_idxs), 1)

            # Best-case: spread remaining readiness gap across remaining recruiting months
            start_fte = gap_at_ready / remaining_months

            # Cap hire starts per month (operational ramp constraint)
            start_fte = clamp(start_fte, 0.0, float(hire_start_cap_per_month))

            # Quantize to 0.25 for realism
            start_fte = math.ceil(start_fte / 0.25) * 0.25 if start_fte > 0 else 0.0

            if start_fte > 0:
                hire_starts[i] += start_fte
                add_pipeline_contribution(
                    add_curve=pipeline_add,
                    start_idx=i,
                    start_fte=start_fte,
                    lead_months=int(lead_months),
                    ramp_months=int(pipeline_ramp_months),
                )

        planned = max(planned, float(provider_min_floor))
        supply.append(planned)
        prev = planned

    return supply, hire_starts, pipeline_add, ready_month


# ============================================================
# COST HELPERS
# ============================================================
def provider_day_gap(target_curve, supply_curve, days_in_month):
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(float(t) - float(s), 0.0) * dim
    return float(gap_days)


def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    cost = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += float(dfte) * float(loaded_cost_per_provider_fte) * (dim / 365.0)
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
        index=11,  # Dec
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    flu_end_month = st.selectbox(
        "Flu End Month",
        options=list(range(1, 13)),
        index=1,   # Feb
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100.0

    st.subheader("Confirmed Hiring (Month-Based)")
    confirmed_hire_month = st.selectbox(
        "Confirmed Hire Independent Month",
        options=list(range(1, 13)),
        index=10,  # Nov default
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    confirmed_hire_fte = st.number_input("Confirmed Hire FTE", min_value=0.0, value=1.0, step=0.25)

    enable_seasonality_ramp = st.checkbox(
        "Enable Seasonality Recruiting Ramp",
        value=True,
        help="If ON: hiring starts occur only during Recruiting Window; Freeze months block starts; pipeline delay applies before supply increases."
    )

    st.caption("App Version: v4.7 (Recruiting Window = Post→Freeze; Pipeline-delayed supply)")

    st.divider()
    run_model = st.button("Run Model")


# ============================================================
# RUN MODEL (v4.7) — Jan–Dec
# ============================================================
if run_model:
    current_year = today.year
    dates = pd.date_range(start=datetime(current_year, 1, 1), periods=12, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    flu_start_date, flu_end_date = build_flu_window(current_year, flu_start_month, flu_end_month)

    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    baseline_provider_fte = max(float(fte_result["provider_fte"]), float(provider_min_floor))

    forecast_visits_by_month = compute_seasonality_forecast(
        dates=dates,
        baseline_visits=visits,
        flu_start=flu_start_date,
        flu_end=flu_end_date,
        flu_uplift_pct=flu_uplift_pct,
    )

    provider_base_demand = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_by_month,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    protective_curve = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    total_lead_days = int(days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days)
    lead_months = lead_days_to_months(total_lead_days)
    notice_months = lead_days_to_months(int(notice_days))
    confirmed_ramp_months = lead_days_to_months(int(onboard_train_days))

    # Strategy v5 (new recruiting window + freeze-by math)
    strategy = auto_strategy_v5(
        dates=dates,
        protective_curve=protective_curve,
        baseline_provider_fte=baseline_provider_fte,
        annual_turnover_rate=provider_turnover,
        flu_start_month=flu_start_month,
        pipeline_lead_days=total_lead_days,
        notice_days=notice_days,
        freeze_buffer_months=1,
    )

    req_post_month = int(strategy["req_post_month"])
    hire_visible_month = int(strategy["hire_visible_month"])
    independent_ready_month = int(strategy["independent_ready_month"])
    freeze_months = list(strategy["freeze_months"])
    recruiting_open_months = list(strategy["recruiting_open_months"])
    trough_month = int(strategy["trough_month"])
    X_incremental_flu_fte = float(strategy["X_incremental_flu_fte"])

    # Ramp cap for hire STARTS per month (best-case throughput constraint)
    # Use the incremental gap-to-close at readiness divided by months available in recruiting window (clamped).
    m2i = month_index_map(dates)
    ready_idx = m2i.get(independent_ready_month, 0)
    baseline_at_trough = float(strategy["baseline_at_trough"])
    target_at_ready = float(protective_curve[ready_idx])
    fte_gap_to_close = max(target_at_ready - max(baseline_provider_fte, baseline_at_trough), 0.0)

    # Months in recruiting window up to readiness (for a sane cap)
    rw_idxs = []
    recruiting_set = set(recruiting_open_months)
    freeze_set = set(freeze_months)
    for i, d in enumerate(dates):
        if i > ready_idx:
            break
        mn = int(d.month)
        if (mn in recruiting_set) and (mn not in freeze_set):
            rw_idxs.append(i)
    months_available = max(len(rw_idxs), 1)

    # Hire start cap: "how much net FTE you need to start per month" (clamped)
    derived_hire_start_cap = min(fte_gap_to_close / months_available, 1.25)

    # Simulate supply against lean and protective targets using pipeline decisions
    if enable_seasonality_ramp:
        supply_lean, hire_starts_lean, pipeline_add_lean, ready_month_lean = simulate_supply_with_pipeline_decisions(
            dates=dates,
            baseline_fte=baseline_provider_fte,
            target_curve=provider_base_demand,
            provider_min_floor=provider_min_floor,
            annual_turnover_rate=provider_turnover,
            notice_months=notice_months,
            recruiting_open_months=recruiting_open_months,
            freeze_months=freeze_months,
            lead_months=lead_months,
            hire_start_cap_per_month=derived_hire_start_cap,
            confirmed_independent_month=int(confirmed_hire_month),
            confirmed_hire_fte=float(confirmed_hire_fte),
            confirmed_ramp_months=confirmed_ramp_months,
            pipeline_ramp_months=confirmed_ramp_months,
        )

        supply_rec, hire_starts_rec, pipeline_add_rec, ready_month_rec = simulate_supply_with_pipeline_decisions(
            dates=dates,
            baseline_fte=baseline_provider_fte,
            target_curve=protective_curve,
            provider_min_floor=provider_min_floor,
            annual_turnover_rate=provider_turnover,
            notice_months=notice_months,
            recruiting_open_months=recruiting_open_months,
            freeze_months=freeze_months,
            lead_months=lead_months,
            hire_start_cap_per_month=derived_hire_start_cap,
            confirmed_independent_month=int(confirmed_hire_month),
            confirmed_hire_fte=float(confirmed_hire_fte),
            confirmed_ramp_months=confirmed_ramp_months,
            pipeline_ramp_months=confirmed_ramp_months,
        )
    else:
        # If ramp disabled, treat all months as recruiting-open and no freeze
        all_months = [d.month for d in dates]
        supply_rec, hire_starts_rec, pipeline_add_rec, ready_month_rec = simulate_supply_with_pipeline_decisions(
            dates=dates,
            baseline_fte=baseline_provider_fte,
            target_curve=protective_curve,
            provider_min_floor=provider_min_floor,
            annual_turnover_rate=provider_turnover,
            notice_months=notice_months,
            recruiting_open_months=all_months,
            freeze_months=[],
            lead_months=lead_months,
            hire_start_cap_per_month=derived_hire_start_cap,
            confirmed_independent_month=int(confirmed_hire_month),
            confirmed_hire_fte=float(confirmed_hire_fte),
            confirmed_ramp_months=confirmed_ramp_months,
            pipeline_ramp_months=confirmed_ramp_months,
        )
        supply_lean = supply_rec
        hire_starts_lean = hire_starts_rec
        pipeline_add_lean = pipeline_add_rec

    burnout_gap_fte = [max(float(t) - float(s), 0.0) for t, s in zip(protective_curve, supply_rec)]
    months_exposed = int(sum(1 for g in burnout_gap_fte if g > 0))

    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        dates=dates,
        month_labels=month_labels,
        days_in_month=days_in_month,

        baseline_provider_fte=baseline_provider_fte,

        flu_start_date=flu_start_date,
        flu_end_date=flu_end_date,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,

        forecast_visits_by_month=forecast_visits_by_month,
        provider_base_demand=provider_base_demand,
        protective_curve=protective_curve,

        realistic_supply_lean=supply_lean,
        realistic_supply_recommended=supply_rec,

        hire_starts_recommended=hire_starts_rec,
        hire_starts_lean=hire_starts_lean,

        burnout_gap_fte=burnout_gap_fte,
        months_exposed=months_exposed,

        pipeline_lead_days=total_lead_days,
        lead_months=lead_months,
        notice_months=notice_months,

        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        independent_ready_month=independent_ready_month,
        trough_month=trough_month,

        freeze_months=freeze_months,
        recruiting_open_months=recruiting_open_months,

        derived_hire_start_cap=derived_hire_start_cap,
        fte_gap_to_close=fte_gap_to_close,
        X_incremental_flu_fte=X_incremental_flu_fte,
        baseline_at_trough=baseline_at_trough,

        confirmed_hire_month=int(confirmed_hire_month),
        confirmed_hire_fte=float(confirmed_hire_fte),
        confirmed_ramp_months=confirmed_ramp_months,

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
        visits_per_day=v,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week
    )
    monthly_rows.append({
        "Month": month_label,
        "Visits/Day (Forecast)": round(float(v), 1),
        "Provider FTE": round(float(fte_staff["provider_fte"]), 2),
        "PSR FTE": round(float(fte_staff["psr_fte"]), 2),
        "MA FTE": round(float(fte_staff["ma_fte"]), 2),
        "XRT FTE": round(float(fte_staff["xrt_fte"]), 2),
        "Total FTE": round(float(fte_staff["total_fte"]), 2),
    })

ops_df = pd.DataFrame(monthly_rows)
st.dataframe(ops_df, hide_index=True, use_container_width=True)

st.success(
    "**Operations Summary:** This is the seasonality-adjusted demand signal. "
    "Lean demand is minimum coverage; the protective target adds a burnout buffer."
)


# ============================================================
# SECTION 2 — REALITY (Jan–Dec display)
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Targets vs realistic supply when hiring is constrained by lead time, recruiting window, freeze discipline, and notice-lag attrition.")

freeze_months = R.get("freeze_months", []) or []
recruiting_open_months = R.get("recruiting_open_months", []) or []
req_post_month = int(R.get("req_post_month", 1))
hire_visible_month = int(R.get("hire_visible_month", 1))
independent_ready_month = int(R.get("independent_ready_month", 1))
trough_month = int(R.get("trough_month", 1))

freeze_label = month_range_label(freeze_months)
recruit_label = month_range_label(recruiting_open_months)
req_post_label = datetime(2000, req_post_month, 1).strftime("%b")
hire_visible_label = datetime(2000, hire_visible_month, 1).strftime("%b")
independent_label = datetime(2000, independent_ready_month, 1).strftime("%b")
trough_label = datetime(2000, trough_month, 1).strftime("%b")

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
m3.metric("Months Exposed", f"{int(R['months_exposed'])}/12")

# Carryover statement (no 24-month simulation)
carryover_next_jan = float(R["realistic_supply_recommended"][-1]) if R["realistic_supply_recommended"] else float("nan")
st.info(
    f"**Carryover (continuity):** Projected starting supply next January = **{carryover_next_jan:.2f} FTE** "
    f"(equal to this year’s December ending supply)."
)

# Display hiring plan summary (optional but useful guidance)
total_hire_starts = float(np.sum(np.array(R.get("hire_starts_recommended", [0.0] * 12))))
st.success(
    f"**Plan Output:** Recommended total permanent hire starts within recruiting window = **{total_hire_starts:.2f} FTE** "
    f"(constrained by pipeline + freeze + ramp)."
)

freeze_set = set(R.get("freeze_months", []))
recruit_set = set(R.get("recruiting_open_months", []))

# categorical x positions for stable plotting
x = np.arange(12)
labels = [d.strftime("%b") for d in R["dates"]]

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Shade freeze months (gold) and recruiting months (light gray-blue) for clarity
for i, d in enumerate(R["dates"]):
    mn = int(d.month)
    if mn in recruit_set:
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.06, color="#6aa7ff", linewidth=0)
    if mn in freeze_set:
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.12, color=BRAND_GOLD, linewidth=0)

ax1.plot(x, R["provider_base_demand"], linestyle=":", linewidth=1.2, color=GRAY, label="Lean Target (Demand)")
ax1.plot(x, R["protective_curve"], linewidth=2.0, color=BRAND_GOLD, marker="o", markersize=4, label="Recommended Target (Protective)")
ax1.plot(x, R["realistic_supply_recommended"], linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=4, label="Realistic Supply (Pipeline)")

ax1.fill_between(
    x,
    R["realistic_supply_recommended"],
    R["protective_curve"],
    where=np.array(R["protective_curve"]) > np.array(R["realistic_supply_recommended"]),
    color=BRAND_GOLD,
    alpha=0.12,
    label="Burnout Exposure Zone"
)

ax1.set_title("Reality — Targets vs Pipeline-Constrained Supply (Jan–Dec)", fontsize=16, fontweight="bold", pad=16, color=BRAND_BLACK)
ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11, color=BRAND_BLACK)
ax1.tick_params(axis="y", labelsize=11, colors=BRAND_BLACK)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

ax2 = ax1.twinx()
ax2.plot(x, R["forecast_visits_by_month"], linestyle="-.", linewidth=1.4, color=MID_GRAY, label="Forecast Visits/Day")
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
    f"**Reality Summary:** To be independent by **{independent_label}**, post requisitions by **{req_post_label}** "
    f"(lead time: ~{int(R['lead_months'])} months). "
    f"Freeze discipline begins after the recruiting window so turnover (after notice) can normalize staffing by **{trough_label}**."
)

st.info(
    "🧠 **Auto-Strategy (v5: window-based recruiting + turnover/notice-aware freeze-by)**\n\n"
    f"- Recruiting window (post reqs / start hires): **{', '.join([datetime(2000,m,1).strftime('%b') for m in R['recruiting_open_months']]) or '—'}**\n"
    f"- Freeze months (no new permanent starts): **{', '.join([datetime(2000,m,1).strftime('%b') for m in R['freeze_months']]) or '—'}**\n"
    f"- Post req: **{req_post_label}** | Hires visible: **{hire_visible_label}** | Independent by: **{independent_label}**\n"
    f"- Lead time: **{int(R['pipeline_lead_days'])} days (~{int(R['lead_months'])} months)**\n"
    f"- Notice lag: **~{int(R['notice_months'])} months**\n"
    f"- Incremental flu staffing to shed (X): **{float(R['X_incremental_flu_fte']):.2f} FTE** by trough month **{trough_label}**\n"
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
annual_investment = annualize_monthly_fte_cost(delta_fte_curve, R["days_in_month"], float(loaded_cost_per_provider_fte))

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
    g2 = g * (1 - buffer_pct / 100.0)
    g2 = max(g2 - float_pool_fte, 0.0)
    g2 = max(g2 - fractional_fte, 0.0)
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
st.markdown("---")
st.header("5) Decision — Executive Summary")

peak_gap = float(max(R["burnout_gap_fte"])) if R["burnout_gap_fte"] else 0.0
avg_gap = float(np.mean(R["burnout_gap_fte"])) if R["burnout_gap_fte"] else 0.0

st.subheader("Decision Snapshot")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Peak Gap (FTE)", f"{peak_gap:.2f}")
    st.metric("Avg Gap (FTE)", f"{avg_gap:.2f}")
    st.metric("Months Exposed", f"{int(R['months_exposed'])}/12")
with col2:
    st.write(
        f"**Recruiting actions (best-case, constrained):**\n"
        f"- Recruiting window: **{recruit_label}**\n"
        f"- Total permanent hire starts (recommended): **{total_hire_starts:.2f} FTE**\n"
        f"- Post req by: **{req_post_label}** to be independent by **{independent_label}**\n\n"
        f"**Freeze discipline:**\n"
        f"- Freeze months: **{freeze_label}** (no new permanent starts; allow notice-lag turnover to reduce excess)\n"
        f"- Incremental flu staffing to shed by trough (**{trough_label}**): **{float(R['X_incremental_flu_fte']):.2f} FTE**\n\n"
        f"**Financial framing:**\n"
        f"- Annual protective investment: **${annual_investment:,.0f}**\n"
        f"- Estimated revenue at risk if not closed: **${est_revenue_lost:,.0f}**\n"
        f"- ROI: **{roi:,.2f}x**\n\n"
        f"**Carryover:**\n"
        f"- Projected starting supply next Jan: **{carryover_next_jan:.2f} FTE**\n"
    )

st.success(
    "✅ **Decision Summary:** This model converts seasonality into staffing demand, converts pipeline timing into realistic supply, "
    "and operationalizes recruiting windows + freeze discipline using turnover and notice lag. "
    "It produces a best-case staffing plan based on your inputs, without forcing unrealistic supply spikes or resets."
)
