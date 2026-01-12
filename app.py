# app.py — Predictive Staffing Model (PSM) v4.4
# Accuracy fixes:
# - REMOVE cyclic steady-state forcing (was invalid with one-time confirmed hires)
# - Attrition: discrete events + notice-lag, BUT no wrap into next year
# - Confirmed hire: ramp-to-independence, BUT no wrap into next year
# - Auto-Freeze v3 retained (month-loop safe)
# - Reality chart rotated to Req Post month (story-first display)

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
    """Wrapped month window, e.g., Dec->Feb = [12,1,2]."""
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
    return {d.month: i for i, d in enumerate(dates)}


def rotate_by_month(dates, series_list, anchor_month: int):
    """Rotate for display only."""
    m2i = month_index_map(dates)
    k = m2i.get(anchor_month, 0)

    rot_dates = list(dates[k:]) + list(dates[:k])
    rot_labels = [d.strftime("%b") for d in rot_dates]
    rot_series = []
    for s in series_list:
        s = list(s)
        rot_series.append(s[k:] + s[:k])

    return rot_dates, rot_labels, rot_series, k


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
# AUTO-FREEZE v3 (month-loop safe)
# ============================================================
def auto_freeze_strategy_v3(
    dates,
    protective_curve,
    flu_start_month,
    flu_end_month,
    pipeline_lead_days,
    notice_days,
    freeze_buffer_months=1,
):
    lead_months = lead_days_to_months(pipeline_lead_days)
    notice_months = lead_days_to_months(notice_days)

    independent_ready_month = flu_start_month
    req_post_month = shift_month(independent_ready_month, -lead_months)
    hire_visible_month = shift_month(req_post_month, lead_months)

    trough_idx = int(np.argmin(np.array(protective_curve)))
    trough_month = dates[trough_idx].month

    decline_months = months_between(shift_month(flu_end_month, 1), trough_month)

    freeze_months = list(decline_months)
    for i in range(1, freeze_buffer_months + 1):
        freeze_months.append(shift_month(trough_month, i))
    freeze_months = dedupe_keep_order(freeze_months)

    recruiting_open_months = []
    for i in range(lead_months + 1):
        recruiting_open_months.append(shift_month(req_post_month, -i))
    recruiting_open_months = dedupe_keep_order(recruiting_open_months)

    # recruiting-open wins
    freeze_months = [m for m in freeze_months if m not in recruiting_open_months]

    return dict(
        lead_months=lead_months,
        notice_months=notice_months,
        independent_ready_month=independent_ready_month,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        trough_month=trough_month,
        freeze_months=freeze_months,
        recruiting_open_months=recruiting_open_months,
        flu_months=months_between(flu_start_month, flu_end_month),
    )


# ============================================================
# ACCURATE ONE-YEAR SUPPLY (NO WRAP OF ONE-TIME EVENTS)
# ============================================================
def build_attrition_schedule_discrete_one_year(
    n_months: int,
    expected_avg_fte: float,
    annual_turnover_rate: float,
    notice_months: int,
    fte_granularity: float = 0.25,
):
    annual_expected_separations_fte = max(0.0, expected_avg_fte * annual_turnover_rate)
    if annual_expected_separations_fte <= 0:
        return [0.0] * n_months

    # number of full 0.25-FTE events + remainder
    full_events = int(math.floor(annual_expected_separations_fte / fte_granularity))
    remainder = annual_expected_separations_fte - full_events * fte_granularity

    idxs = []
    if full_events > 0:
        for k in range(full_events):
            idxs.append(int(math.floor(k * n_months / full_events)))
        idxs = [clamp(i, 0, n_months - 1) for i in idxs]

    drop = [0.0] * n_months

    # apply full discrete events
    for i in idxs:
        j = i + int(notice_months)
        if 0 <= j < n_months:
            drop[j] += fte_granularity

    # apply remainder as a small expected-value drop (also notice-lagged)
    if remainder > 1e-9:
        # spread remainder across year to avoid one big weird step
        for m in range(n_months):
            j = m + int(notice_months)
            if 0 <= j < n_months:
                drop[j] += remainder / n_months

    return drop

def build_confirmed_hire_ramp_one_year(
    n_months: int,
    independent_idx: int | None,
    hire_fte: float,
    ramp_months: int,
):
    """
    Additive FTE contribution:
    - ramps up in the months leading to independence
    - full from independence through December
    - no wrap into next January
    """
    add = [0.0] * n_months
    if independent_idx is None or hire_fte <= 0:
        return add

    ramp_months = max(0, int(ramp_months))

    if ramp_months == 0:
        for i in range(independent_idx, n_months):
            add[i] += hire_fte
        return add

    start_idx = max(0, independent_idx - ramp_months)

    steps = independent_idx - start_idx
    if steps <= 0:
        for i in range(independent_idx, n_months):
            add[i] += hire_fte
        return add

    # ramp segment
    for s, i in enumerate(range(start_idx, independent_idx)):
        frac = (s + 1) / steps
        add[i] += hire_fte * frac

    # full segment
    for i in range(independent_idx, n_months):
        add[i] += hire_fte

    return add


def pipeline_supply_curve_one_year_v44(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_months,
    hire_visible_month,
    freeze_months,
    max_hiring_up_after_visible,
    confirmed_independent_month=None,
    confirmed_hire_fte=0.0,
    confirmed_ramp_months=1,
    seasonality_ramp_enabled=True,
):
    """
    One-year forward simulation (most accurate for planning-year actions).

    Ramp-up rules:
    - If seasonality_ramp_enabled: no ramp-up before hires visible; freeze blocks net-add starts
      (we cap supply at freeze ceiling when freeze begins)
    - Otherwise ramp-up fixed at 0.35

    Attrition:
    - discrete drops with notice lag (no wrap into next year)

    Confirmed hire:
    - ramps into independence (no wrap into next year)
    """
    n = len(dates)
    m2i = month_index_map(dates)
    hv_idx = m2i.get(hire_visible_month, 0)
    freeze_set = set(freeze_months or [])

    # Attrition schedule: approximate expected avg FTE as baseline for distribution
    attr_drop = build_attrition_schedule_discrete_one_year(
        n_months=n,
        expected_avg_fte=max(float(baseline_fte), float(provider_min_floor)),
        annual_turnover_rate=float(annual_turnover_rate),
        notice_months=int(notice_months),
        fte_granularity=0.25,
    )

    # Confirmed hire ramp schedule
    independent_idx = m2i.get(int(confirmed_independent_month), None) if confirmed_independent_month else None
    hire_add = build_confirmed_hire_ramp_one_year(
        n_months=n,
        independent_idx=independent_idx,
        hire_fte=float(confirmed_hire_fte),
        ramp_months=int(confirmed_ramp_months),
    )

    staff = []
    prev = max(float(baseline_fte), float(provider_min_floor))

    freeze_ceiling = None
    prev_in_freeze = False

    for i, (d, target) in enumerate(zip(dates, target_curve)):
        month_num = d.month
        in_freeze = month_num in freeze_set

        # Freeze ceiling captures supply when freeze begins
        if seasonality_ramp_enabled and in_freeze and (not prev_in_freeze):
            freeze_ceiling = prev
        if not in_freeze:
            freeze_ceiling = None

        # Ramp-up cap
        if seasonality_ramp_enabled:
            if i < hv_idx:
                ramp_up_cap = 0.0
            else:
                ramp_up_cap = float(max_hiring_up_after_visible)
        else:
            ramp_up_cap = 0.35

        # Move toward target (Reality supply should not auto-ramp DOWN with seasonality)
        gap = float(target) - prev
        
        # only hire upward toward target (subject to ramp caps)
        if gap > 0:
            hire_step = clamp(gap, 0.0, ramp_up_cap)
        else:
            hire_step = 0.0  # no forced ramp-down
        
        planned = prev + hire_step

        # Attrition drop (notice-lagged)
        planned -= float(attr_drop[i])

        # Confirmed hire contribution
        planned += float(hire_add[i])

        # Freeze policy: no net-add starts during freeze (cap at ceiling)
        if seasonality_ramp_enabled and in_freeze and freeze_ceiling is not None:
            planned = min(planned, freeze_ceiling)

        planned = max(planned, float(provider_min_floor))
        staff.append(planned)
        prev = planned
        prev_in_freeze = in_freeze

    return staff


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
        help="If ON: supply cannot ramp up before hires are visible, and freeze blocks net-add starts."
    )

    st.caption("App Version: v4.4 (accuracy corrected)")

    st.divider()
    run_model = st.button("Run Model")


# ============================================================
# RUN MODEL (v4.4)
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

    strategy = auto_freeze_strategy_v3(
        dates=dates,
        protective_curve=protective_curve,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
        pipeline_lead_days=total_lead_days,
        notice_days=notice_days,
        freeze_buffer_months=1,
    )

    req_post_month = int(strategy["req_post_month"])
    hire_visible_month = int(strategy["hire_visible_month"])
    independent_ready_month = int(strategy["independent_ready_month"])
    freeze_months = list(strategy["freeze_months"])
    recruiting_open_months = list(strategy["recruiting_open_months"])
    flu_months = list(strategy["flu_months"])

    # derived ramp (same method you’ve been using)
    flu_month_idx = month_index_map(dates).get(flu_start_month, 0)
    months_in_flu_window = max(len(flu_months), 1)

    target_at_flu = float(protective_curve[flu_month_idx])
    fte_gap_to_close = max(target_at_flu - baseline_provider_fte, 0.0)
    derived_ramp_after_visible = min(fte_gap_to_close / months_in_flu_window, 1.25)

    # ONE-YEAR supply sim (accurate for one-time actions)
    realistic_supply_lean = pipeline_supply_curve_one_year_v44(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_months=notice_months,
        hire_visible_month=hire_visible_month,
        freeze_months=freeze_months if enable_seasonality_ramp else [],
        max_hiring_up_after_visible=derived_ramp_after_visible,
        confirmed_independent_month=int(confirmed_hire_month),
        confirmed_hire_fte=float(confirmed_hire_fte),
        confirmed_ramp_months=confirmed_ramp_months,
        seasonality_ramp_enabled=enable_seasonality_ramp,
    )

    realistic_supply_recommended = pipeline_supply_curve_one_year_v44(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=protective_curve,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_months=notice_months,
        hire_visible_month=hire_visible_month,
        freeze_months=freeze_months if enable_seasonality_ramp else [],
        max_hiring_up_after_visible=derived_ramp_after_visible,
        confirmed_independent_month=int(confirmed_hire_month),
        confirmed_hire_fte=float(confirmed_hire_fte),
        confirmed_ramp_months=confirmed_ramp_months,
        seasonality_ramp_enabled=enable_seasonality_ramp,
    )

    burnout_gap_fte = [max(float(t) - float(s), 0.0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    months_exposed = int(sum(1 for g in burnout_gap_fte if g > 0))

    rotation_anchor_month = req_post_month  # Req Post anchor

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
        flu_months=flu_months,

        forecast_visits_by_month=forecast_visits_by_month,
        provider_base_demand=provider_base_demand,
        protective_curve=protective_curve,

        realistic_supply_lean=realistic_supply_lean,
        realistic_supply_recommended=realistic_supply_recommended,

        burnout_gap_fte=burnout_gap_fte,
        months_exposed=months_exposed,

        pipeline_lead_days=total_lead_days,
        lead_months=lead_months,
        notice_months=notice_months,

        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        independent_ready_month=independent_ready_month,

        freeze_months=freeze_months,
        recruiting_open_months=recruiting_open_months,

        derived_ramp_after_visible=derived_ramp_after_visible,
        fte_gap_to_close=fte_gap_to_close,
        months_in_flu_window=months_in_flu_window,

        confirmed_hire_month=int(confirmed_hire_month),
        confirmed_hire_fte=float(confirmed_hire_fte),
        confirmed_ramp_months=confirmed_ramp_months,

        enable_seasonality_ramp=enable_seasonality_ramp,
        rotation_anchor_month=rotation_anchor_month,
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
# SECTION 2 — REALITY (ROTATED DISPLAY)
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Targets vs realistic supply when hiring is constrained by lead time, freezes, and turnover (discrete attrition + notice lag).")

freeze_months = R.get("freeze_months", []) or []
recruiting_open_months = R.get("recruiting_open_months", []) or []
req_post_month = int(R.get("req_post_month", 1))
hire_visible_month = int(R.get("hire_visible_month", 1))
independent_ready_month = int(R.get("independent_ready_month", 1))

freeze_label = month_range_label(freeze_months)
recruit_label = month_range_label(recruiting_open_months)
req_post_label = datetime(2000, req_post_month, 1).strftime("%b")
hire_visible_label = datetime(2000, hire_visible_month, 1).strftime("%b")
independent_label = datetime(2000, independent_ready_month, 1).strftime("%b")

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

# --- CATEGORICAL X AXIS FIX (prevents date-sorting loops) ---

anchor_month = int(R.get("rotation_anchor_month", req_post_month))
rot_dates, rot_labels, (rot_lean, rot_prot, rot_supply, rot_visits), _ = rotate_by_month(
    R["dates"],
    [R["provider_base_demand"], R["protective_curve"], R["realistic_supply_recommended"], R["forecast_visits_by_month"]],
    anchor_month=anchor_month
)

freeze_set = set(R.get("freeze_months", []))

# categorical x positions
x = np.arange(len(rot_labels))

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# shade freeze months using categorical spans
for i, d in enumerate(rot_dates):
    if d.month in freeze_set:
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.12, color=BRAND_GOLD, linewidth=0)

ax1.plot(x, rot_lean, linestyle=":", linewidth=1.2, color=GRAY, label="Lean Target (Demand)")
ax1.plot(x, rot_prot, linewidth=2.0, color=BRAND_GOLD, marker="o", markersize=4, label="Recommended Target (Protective)")
ax1.plot(x, rot_supply, linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=4, label="Realistic Supply (Pipeline)")

ax1.fill_between(
    x,
    rot_supply,
    rot_prot,
    where=np.array(rot_prot) > np.array(rot_supply),
    color=BRAND_GOLD,
    alpha=0.12,
    label="Burnout Exposure Zone"
)

ax1.set_title("Reality — Targets vs Pipeline-Constrained Supply", fontsize=16, fontweight="bold", pad=16, color=BRAND_BLACK)
ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold", color=BRAND_BLACK)

ax1.set_xticks(x)
ax1.set_xticklabels(rot_labels, fontsize=11, color=BRAND_BLACK)
ax1.tick_params(axis="y", labelsize=11, colors=BRAND_BLACK)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

ax2 = ax1.twinx()
ax2.plot(x, rot_visits, linestyle="-.", linewidth=1.4, color=MID_GRAY, label="Forecast Visits/Day")
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
    f"**Reality Summary:** To be flu-ready by **{independent_label}**, requisitions must post by **{req_post_label}** "
    f"so hires are visible by **{hire_visible_label}**. "
    f"Required protective ramp speed: **{float(R['derived_ramp_after_visible']):.2f} FTE/month**."
)

st.info(
    "🧠 **Auto-Hiring Strategy (v3)**\n\n"
    f"- Freeze months: **{', '.join([datetime(2000,m,1).strftime('%b') for m in R['freeze_months']]) or '—'}**\n"
    f"- Recruiting window: **{', '.join([datetime(2000,m,1).strftime('%b') for m in R['recruiting_open_months']]) or '—'}**\n"
    f"- Post req: **{req_post_label}** | Hires visible: **{hire_visible_label}** | Independent by: **{independent_label}**\n"
    f"- Lead time: **{int(R['pipeline_lead_days'])} days (~{int(R['lead_months'])} months)**\n"
    f"- Notice lag: **~{int(R['notice_months'])} months** (attrition hits capacity after notice)\n"
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
        f"**To be flu-ready by {independent_label}:**\n"
        f"- Post requisitions by **{req_post_label}** (lead time: {int(R['pipeline_lead_days'])} days ≈ {int(R['lead_months'])} months)\n"
        f"- Hiring becomes visible by **{hire_visible_label}**\n"
        f"- Protective ramp required: **{float(R['derived_ramp_after_visible']):.2f} FTE/month**\n\n"
        f"**Financial framing:**\n"
        f"- Annual protective investment: **${annual_investment:,.0f}**\n"
        f"- Estimated revenue at risk if not closed: **${est_revenue_lost:,.0f}**\n"
        f"- ROI: **{roi:,.2f}x**\n\n"
        f"**With strategy levers applied:**\n"
        f"- Provider-day gap reduced: **{reduced_gap_days:,.0f} days**\n"
        f"- Estimated revenue saved: **${est_revenue_saved:,.0f}**\n"
    )

st.success(
    "✅ **Decision Summary:** This model converts seasonality into staffing demand, converts pipeline timing into realistic supply, "
    "and quantifies burnout exposure and the ROI of closing gaps. "
    "Use the auto-hiring strategy + flex levers to move from reactive coverage to decision-ready staffing."
)

# ============================================================
# SECTION 6 — IN-DEPTH EXECUTIVE SUMMARY
# ============================================================
st.markdown("---")
st.header("6) In-Depth Executive Summary")
st.caption("A plain-language explanation of what the model is doing, what it found, and how to act on it.")

flu_range = month_range_label(R["flu_months"])
freeze_range = month_range_label(R["freeze_months"])
recruit_range = month_range_label(R["recruiting_open_months"])

summary_md = f"""
### What this model is solving
This Predictive Staffing Model (PSM) answers:

**“Given seasonal demand swings and a constrained hiring pipeline, what staffing plan prevents burnout exposure — and when must recruiting actions occur to make it feasible?”**

This is a **12-month planning-year simulation**.  
One-time actions (confirmed hire, notice-lag attrition) **do not wrap** into next year because that would be inaccurate.

---

### How Operations works (demand signal)
- Applies seasonality + flu uplift over **{flu_range}**
- Normalizes back to the baseline annual average
- Converts forecast visits/day into monthly staffing demand
- Builds a protective target with a burnout buffer + smoothing

---

### How Reality works (supply constraints)
Supply is constrained by:
- Pipeline lead time: **{int(R['pipeline_lead_days'])} days (~{int(R['lead_months'])} months)**
- Hiring visibility: supply cannot rise before **{hire_visible_label}**
- Freeze policy: blocks net-add starts during **{freeze_range}**
- Turnover: modeled as **discrete events** hitting capacity after **~{int(R['notice_months'])} months** notice lag

---

### The Auto-Hiring Strategy (v3)
- Flu window: {flu_range}
- Freeze window: {freeze_range}
- Recruiting window: {recruit_range}
- Post req: {req_post_label}
- Hires visible: {hire_visible_label}
- Independent by: {independent_label}

---

### How to read the Reality chart
The chart is rotated to begin at **Post Req ({req_post_label})** so you see:
**recruiting actions → pipeline visibility → flu readiness**, without implying the year boundary is a staffing reset.
"""
st.markdown(summary_md)

st.success("✅ Executive Summary complete. Next: connect feasibility to VVI (RF/LF) so financial performance gates recommended staffing moves.")
