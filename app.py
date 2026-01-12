# app.py — Predictive Staffing Model (PSM) v4.9
# Internal engine: 24-month continuous simulation (real-life Dec→Jan continuity)
# Display layer: Jan–Dec only (Year 1)
#
# Key updates:
# - Recruiting Window is a TRUE window: Post-By → Freeze-By (indices, not "all months back")
# - Freeze never overlaps recruiting (recruiting wins)
# - Freeze-by is turnover + notice-lag aware, anchored to Baseline Date (BD; default Apr)
# - Supply is a STOCK with PIPELINE: hire starts → delayed graduation (+ optional ramp)
# - Attrition is discrete-ish + notice-lag capacity drop (applies across Dec→Jan within 24m)
# - Confirmed hires can become independent in Year 2 and WILL count (real life)
#
# NOTE: We DO NOT "plan 24 months." We SIMULATE 24 months so Year 1 is mathematically correct.

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
    """Seasonality + flu uplift, normalized back to baseline annual average (within the horizon)."""
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


def lead_days_to_months(days: int, avg_days_per_month: float = 30.4) -> int:
    return max(0, int(math.ceil(days / avg_days_per_month)))


def label_month(d: pd.Timestamp, base_year: int) -> str:
    """Month label with subtle Y2 marker for strategy items that land in Year 2."""
    if int(d.year) == int(base_year):
        return d.strftime("%b")
    return d.strftime("%b") + " (Y2)"


def month_range_label_from_indices(dates, idxs, base_year: int) -> str:
    if not idxs:
        return "—"
    start = label_month(dates[idxs[0]], base_year)
    end = label_month(dates[idxs[-1]], base_year)
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
# ATTRITION SCHEDULE (24-month horizon; notice-lagged; no wrap beyond horizon)
# ============================================================
def build_attrition_schedule_discrete_horizon(
    n_months: int,
    expected_avg_fte: float,
    annual_turnover_rate: float,
    notice_months: int,
    fte_granularity: float = 0.25,
):
    """
    Discrete-ish attrition events spread across the horizon.
    Annual turnover rate applies per 12 months (so 24m horizon doubles expected separations).
    Notice lag shifts the capacity drop into later months.
    """
    years = n_months / 12.0
    expected_separations_fte = max(0.0, expected_avg_fte * annual_turnover_rate * years)
    if expected_separations_fte <= 0:
        return [0.0] * n_months

    full_events = int(math.floor(expected_separations_fte / fte_granularity))
    remainder = expected_separations_fte - full_events * fte_granularity

    idxs = []
    if full_events > 0:
        for k in range(full_events):
            idxs.append(int(math.floor(k * n_months / full_events)))
        idxs = [clamp(i, 0, n_months - 1) for i in idxs]

    drop = [0.0] * n_months

    for i in idxs:
        j = i + int(notice_months)
        if 0 <= j < n_months:
            drop[j] += fte_granularity

    if remainder > 1e-9:
        for m in range(n_months):
            j = m + int(notice_months)
            if 0 <= j < n_months:
                drop[j] += remainder / n_months

    return drop


# ============================================================
# CONFIRMED HIRE RAMP (can land in Year 2; real-life)
# ============================================================
def build_confirmed_hire_ramp_horizon(
    dates,
    independent_month: int | None,
    independent_year_offset: int,
    hire_fte: float,
    ramp_months: int,
):
    """
    independent_year_offset:
      0 = independent month occurs in Year 1
      1 = independent month occurs in Year 2
    """
    n = len(dates)
    add = [0.0] * n
    if independent_month is None or hire_fte <= 0:
        return add

    base_year = int(dates[0].year)
    target_year = base_year + int(independent_year_offset)

    independent_idx = None
    for i, d in enumerate(dates):
        if int(d.year) == target_year and int(d.month) == int(independent_month):
            independent_idx = i
            break
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

    for s, i in enumerate(range(start_idx, independent_idx)):
        frac = (s + 1) / steps
        add[i] += float(hire_fte) * frac

    for i in range(independent_idx, n):
        add[i] += float(hire_fte)

    return add


# ============================================================
# PIPELINE: hire starts → delayed contribution (with ramp)
# ============================================================
def add_pipeline_contribution(
    add_curve,
    start_idx: int,
    start_fte: float,
    lead_months: int,
    ramp_months: int,
):
    n = len(add_curve)
    if start_fte <= 0:
        return

    grad_idx = start_idx + int(lead_months)
    if grad_idx >= n:
        return

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

    for s, i in enumerate(range(ramp_start, grad_idx)):
        frac = (s + 1) / steps
        add_curve[i] += float(start_fte) * frac

    for i in range(grad_idx, n):
        add_curve[i] += float(start_fte)


# ============================================================
# STRATEGY: Post-By, Freeze-By, Recruiting Window, Freeze Months (INDEX-BASED)
# ============================================================
def compute_strategy_indices(
    dates_24,
    protective_curve_24,
    baseline_provider_fte,
    annual_turnover_rate,
    flu_start_month,
    baseline_month_bd,
    lead_months,
    notice_months,
    freeze_buffer_months=1,
):
    """
    Outputs indices (0..23) so windows can cross Dec→Jan naturally.
    - RD_idx: readiness month in Year 1 (flu_start_month in base_year)
    - BD_idx: baseline month AFTER RD (default Apr; can be in Year 2)
    - PostBy_idx: latest start month for postings to be ready by RD (RD - lead)
    - FreezeBy_idx: latest month to keep starting permanents so attrition (after notice) can shed X by BD
    - Recruiting window indices: PostBy → FreezeBy (inclusive) if feasible
    - Freeze indices: (FreezeBy+1) → BD (+ buffer), excluding recruiting indices
    """
    base_year = int(dates_24[0].year)

    # RD = flu start month in Year 1
    RD_idx = None
    for i, d in enumerate(dates_24):
        if int(d.year) == base_year and int(d.month) == int(flu_start_month):
            RD_idx = i
            break
    if RD_idx is None:
        RD_idx = 11  # fallback to Dec of Year 1

    # BD = baseline month AFTER RD (so it can be in Year 2 if BD month <= RD month)
    BD_idx = None
    for i in range(RD_idx + 1, len(dates_24)):
        d = dates_24[i]
        if int(d.month) == int(baseline_month_bd):
            BD_idx = i
            break
    if BD_idx is None:
        BD_idx = min(RD_idx + 4, len(dates_24) - 1)  # safe fallback

    # Post-By
    PostBy_idx = max(0, RD_idx - int(lead_months))

    # Incremental seasonal staffing X to shed by BD:
    # use protective at RD minus protective at BD as "seasonal extra"
    target_at_RD = float(protective_curve_24[RD_idx])
    baseline_at_BD = float(protective_curve_24[BD_idx])
    X = max(target_at_RD - baseline_at_BD, 0.0)

    # Attrition expectation
    B = max(float(baseline_provider_fte), float(baseline_at_BD), 0.25)
    T = max(float(annual_turnover_rate), 0.0)
    attr_per_day = (B * T) / 365.0 if (B > 0 and T > 0) else 0.0

    if attr_per_day > 0:
        delta_days = X / attr_per_day
        delta_months = int(math.ceil(delta_days / 30.4))
    else:
        # No turnover means you can't shed X via attrition; freeze-by becomes "as early as possible"
        delta_months = 999

    total_shed_months = int(notice_months + delta_months)
    FreezeBy_idx = BD_idx - total_shed_months
    FreezeBy_idx = clamp(FreezeBy_idx, 0, len(dates_24) - 1)

    # Recruiting window feasibility
    feasible = FreezeBy_idx >= PostBy_idx

    recruiting_idxs = list(range(PostBy_idx, FreezeBy_idx + 1)) if feasible else []

    # Freeze indices (after FreezeBy through BD + buffer), exclude recruiting overlap
    freeze_end = min(BD_idx + int(freeze_buffer_months), len(dates_24) - 1)
    freeze_idxs = list(range(FreezeBy_idx + 1, freeze_end + 1))
    recruiting_set = set(recruiting_idxs)
    freeze_idxs = [i for i in freeze_idxs if i not in recruiting_set]

    return dict(
        base_year=base_year,
        RD_idx=int(RD_idx),
        BD_idx=int(BD_idx),
        PostBy_idx=int(PostBy_idx),
        FreezeBy_idx=int(FreezeBy_idx),
        recruiting_idxs=recruiting_idxs,
        freeze_idxs=freeze_idxs,
        X_incremental_flu_fte=float(X),
        baseline_at_BD=float(baseline_at_BD),
        feasible=bool(feasible),
    )


# ============================================================
# SUPPLY SIMULATION (24m): stock + pipeline + notice-lag attrition
# ============================================================
def simulate_supply_24m_best_case(
    dates_24,
    target_curve_24,
    baseline_fte,
    provider_min_floor,
    annual_turnover_rate,
    notice_months,
    lead_months,
    recruiting_idxs,
    freeze_idxs,
    hire_start_cap_per_month,
    confirmed_add_24,
    pipeline_ramp_months,
):
    """
    Best-case: start hires in recruiting months to hit RD, but don't create post-peak bloat by BD.
    We enforce:
      - starts only in recruiting_idxs
      - no starts in freeze_idxs (also redundant since recruiting/freeze non-overlap)
      - starts are capped per month and quantized to 0.25
      - supply changes only via attrition hits + pipeline + confirmed
    """
    n = len(dates_24)
    recruiting_set = set(recruiting_idxs or [])
    freeze_set = set(freeze_idxs or [])

    # Attrition schedule across horizon (notice-lag already included)
    attr_drop = build_attrition_schedule_discrete_horizon(
        n_months=n,
        expected_avg_fte=max(float(baseline_fte), float(provider_min_floor)),
        annual_turnover_rate=float(annual_turnover_rate),
        notice_months=int(notice_months),
        fte_granularity=0.25,
    )

    pipeline_add = [0.0] * n
    hire_starts = [0.0] * n
    supply = []

    prev = max(float(baseline_fte), float(provider_min_floor))

    # Determine RD and BD indices for the target curve by looking at the max-winter / min-baseline? —
    # We'll expect caller to use strategy indices for decisions, but simulation just needs the curve.
    # We'll infer "readiness" as max target in months 0..11 (Year 1) to avoid targeting Year 2.
    RD_idx = int(np.argmax(np.array(target_curve_24[:12])))

    # For bloat guard, use minimum target in months 12..23 as baseline guardrail
    # (this aligns with "normalize by BD" in spirit even if BD is explicit elsewhere)
    BD_guard_idx = 12 + int(np.argmin(np.array(target_curve_24[12:24])))

    for i in range(n):
        planned = prev

        # Attrition hits capacity
        planned -= float(attr_drop[i])

        # Confirmed hire additions (already ramped)
        planned += float(confirmed_add_24[i])

        # Pipeline graduations from prior starts
        planned += float(pipeline_add[i])

        # Decide hire starts (only within recruiting set, not in freeze)
        can_start = (i in recruiting_set) and (i not in freeze_set) and (i <= RD_idx)

        if can_start:
            # Project supply at RD if we do NOTHING else starting after this month
            proj = planned
            for j in range(i + 1, RD_idx + 1):
                proj -= float(attr_drop[j])
                proj += float(confirmed_add_24[j])
                proj += float(pipeline_add[j])
                proj = max(proj, float(provider_min_floor))

            gap_at_RD = max(float(target_curve_24[RD_idx]) - proj, 0.0)

            # Remaining recruiting months to RD (including i)
            remaining = [k for k in recruiting_idxs if k >= i and k <= RD_idx]
            remaining_months = max(len(remaining), 1)

            start_fte = gap_at_RD / remaining_months
            start_fte = clamp(start_fte, 0.0, float(hire_start_cap_per_month))

            # Bloat guard: don't push Year 2 baseline too high.
            # Conservative: if the start would be independent by BD_guard, it contributes ~full FTE there.
            grad_idx = i + int(lead_months)
            if grad_idx <= BD_guard_idx:
                # Project baseline supply at BD_guard without this start
                proj_bd = planned
                for j in range(i + 1, BD_guard_idx + 1):
                    proj_bd -= float(attr_drop[j])
                    proj_bd += float(confirmed_add_24[j])
                    proj_bd += float(pipeline_add[j])
                    proj_bd = max(proj_bd, float(provider_min_floor))

                bd_room = max(float(target_curve_24[BD_guard_idx]) - proj_bd, 0.0)
                start_fte = min(start_fte, bd_room)

            # Quantize to 0.25
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

    return supply, hire_starts, pipeline_add, attr_drop


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
        "Flu Start Month (Readiness Month)",
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

    st.subheader("Baseline Date (BD)")
    baseline_month_bd = st.selectbox(
        "Return-to-Baseline Month (default Apr)",
        options=list(range(1, 13)),
        index=3,  # Apr
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )

    st.subheader("Confirmed Hiring (Month-Based Independence)")
    confirmed_hire_month = st.selectbox(
        "Confirmed Hire Independent Month",
        options=list(range(1, 13)),
        index=10,  # Nov
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    confirmed_hire_year = st.selectbox(
        "Confirmed Hire Independence Year",
        options=[0, 1],
        index=0,
        format_func=lambda x: "Year 1" if x == 0 else "Year 2"
    )
    confirmed_hire_fte = st.number_input("Confirmed Hire FTE", min_value=0.0, value=1.0, step=0.25)

    enable_pipeline_constraints = st.checkbox(
        "Enable Recruiting Window + Freeze Discipline",
        value=True,
        help="If ON: starts only in recruiting window; freeze blocks starts; pipeline delay applies before supply increases."
    )

    st.caption("App Version: v4.9 (24m engine; 12m display; window-based recruiting; pipeline supply)")
    st.divider()
    run_model = st.button("Run Model")


# ============================================================
# RUN MODEL — 24-month engine, 12-month display
# ============================================================
if run_model:
    current_year = today.year
    base_year = current_year

    # 24-month continuous horizon
    dates_24 = pd.date_range(start=datetime(current_year, 1, 1), periods=24, freq="MS")

    # Display horizon (Year 1 only)
    dates_12 = dates_24[:12]
    month_labels_12 = [d.strftime("%b") for d in dates_12]
    days_in_month_12 = [pd.Period(d, "M").days_in_month for d in dates_12]

    # Flu window spans years as needed
    flu_start_date, flu_end_date = build_flu_window(current_year, flu_start_month, flu_end_month)

    # Baseline FTE from operations staffing calc
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    baseline_provider_fte = max(float(fte_result["provider_fte"]), float(provider_min_floor))

    # Demand signals across full 24 months (so Year 1 stays physically correct)
    forecast_visits_24 = compute_seasonality_forecast(
        dates=dates_24,
        baseline_visits=visits,
        flu_start=flu_start_date,
        flu_end=flu_end_date,
        flu_uplift_pct=flu_uplift_pct,
    )

    provider_base_demand_24 = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_24,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    protective_curve_24 = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_24,
        base_demand_fte=provider_base_demand_24,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # Pipeline params
    total_lead_days = int(days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days)
    lead_months = lead_days_to_months(total_lead_days)
    notice_months = lead_days_to_months(int(notice_days))
    ramp_months = lead_days_to_months(int(onboard_train_days))

    # Confirmed hire additions (can land in Year 2; counts)
    confirmed_add_24 = build_confirmed_hire_ramp_horizon(
        dates=dates_24,
        independent_month=int(confirmed_hire_month),
        independent_year_offset=int(confirmed_hire_year),
        hire_fte=float(confirmed_hire_fte),
        ramp_months=int(ramp_months),
    )

    # Strategy indices (Post-By, Freeze-By, windows) — based on protective curve
    strategy = compute_strategy_indices(
        dates_24=dates_24,
        protective_curve_24=protective_curve_24,
        baseline_provider_fte=baseline_provider_fte,
        annual_turnover_rate=provider_turnover,
        flu_start_month=flu_start_month,
        baseline_month_bd=baseline_month_bd,
        lead_months=lead_months,
        notice_months=notice_months,
        freeze_buffer_months=1,
    )

    RD_idx = strategy["RD_idx"]
    BD_idx = strategy["BD_idx"]
    PostBy_idx = strategy["PostBy_idx"]
    FreezeBy_idx = strategy["FreezeBy_idx"]
    recruiting_idxs = strategy["recruiting_idxs"]
    freeze_idxs = strategy["freeze_idxs"]
    X_incremental_flu_fte = float(strategy["X_incremental_flu_fte"])

    # Hire start throughput cap (best-case operational constraint)
    # Use gap at RD divided by available recruiting months (clamped).
    target_at_RD = float(protective_curve_24[RD_idx])
    baseline_at_BD = float(strategy["baseline_at_BD"])
    fte_gap_to_close = max(target_at_RD - max(baseline_provider_fte, baseline_at_BD), 0.0)

    if enable_pipeline_constraints and recruiting_idxs:
        months_available = len([i for i in recruiting_idxs if i <= RD_idx])
        months_available = max(months_available, 1)
    else:
        months_available = max(RD_idx + 1, 1)

    hire_start_cap = min(fte_gap_to_close / months_available, 1.25)

    # If constraints off: treat all months as recruiting, no freeze
    if not enable_pipeline_constraints:
        recruiting_idxs = list(range(0, 24))
        freeze_idxs = []

    # Simulate realistic supply vs protective target (24m), then slice to 12m display
    supply_24, hire_starts_24, pipeline_add_24, attr_drop_24 = simulate_supply_24m_best_case(
        dates_24=dates_24,
        target_curve_24=protective_curve_24,
        baseline_fte=baseline_provider_fte,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_months=notice_months,
        lead_months=lead_months,
        recruiting_idxs=recruiting_idxs,
        freeze_idxs=freeze_idxs,
        hire_start_cap_per_month=hire_start_cap,
        confirmed_add_24=confirmed_add_24,
        pipeline_ramp_months=ramp_months,
    )

    # Slice for display
    demand_lean_12 = provider_base_demand_24[:12]
    demand_prot_12 = protective_curve_24[:12]
    supply_12 = supply_24[:12]
    visits_12 = forecast_visits_24[:12]

    burnout_gap_12 = [max(float(t) - float(s), 0.0) for t, s in zip(demand_prot_12, supply_12)]
    months_exposed_12 = int(sum(1 for g in burnout_gap_12 if g > 0))

    # Carryover = Month 13 supply (Jan Year 2)
    carryover_next_jan = float(supply_24[12]) if len(supply_24) > 12 else float("nan")

    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        base_year=base_year,
        dates_24=dates_24,
        dates_12=dates_12,
        month_labels_12=month_labels_12,
        days_in_month_12=days_in_month_12,

        baseline_provider_fte=baseline_provider_fte,

        flu_start_date=flu_start_date,
        flu_end_date=flu_end_date,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
        baseline_month_bd=baseline_month_bd,

        forecast_visits_24=forecast_visits_24,
        provider_base_demand_24=provider_base_demand_24,
        protective_curve_24=protective_curve_24,

        realistic_supply_24=supply_24,
        hire_starts_24=hire_starts_24,
        pipeline_add_24=pipeline_add_24,
        attr_drop_24=attr_drop_24,
        confirmed_add_24=confirmed_add_24,

        demand_lean_12=demand_lean_12,
        demand_prot_12=demand_prot_12,
        supply_12=supply_12,
        visits_12=visits_12,

        burnout_gap_12=burnout_gap_12,
        months_exposed_12=months_exposed_12,

        pipeline_lead_days=total_lead_days,
        lead_months=lead_months,
        notice_months=notice_months,
        ramp_months=ramp_months,

        RD_idx=RD_idx,
        BD_idx=BD_idx,
        PostBy_idx=PostBy_idx,
        FreezeBy_idx=FreezeBy_idx,
        recruiting_idxs=recruiting_idxs,
        freeze_idxs=freeze_idxs,

        X_incremental_flu_fte=X_incremental_flu_fte,
        fte_gap_to_close=fte_gap_to_close,
        hire_start_cap=hire_start_cap,

        carryover_next_jan=carryover_next_jan,
        strategy_feasible=strategy["feasible"],
        enable_pipeline_constraints=enable_pipeline_constraints,
    )


# ============================================================
# STOP IF NOT RUN
# ============================================================
if not st.session_state.get("model_ran"):
    st.info("Enter inputs in the sidebar and click **Run Model**.")
    st.stop()

R = st.session_state["results"]
base_year = int(R["base_year"])


# ============================================================
# SECTION 1 — OPERATIONS (Jan–Dec display)
# ============================================================
st.markdown("---")
st.header("1) Operations — Seasonality Staffing Requirements (Jan–Dec)")
st.caption("Visits/day forecast → FTE needed by month (seasonality + flu uplift).")

monthly_rows = []
for month_label, d, v in zip(R["month_labels_12"], R["dates_12"], R["visits_12"]):
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
# SECTION 2 — REALITY (Jan–Dec chart; 24m engine behind it)
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure (Jan–Dec)")
st.caption("Chart shows Jan–Dec. Calculations run 24 months to preserve real Dec→Jan continuity.")

# Strategy labels from indices (may include Year 2 in windows)
RD_label = label_month(R["dates_24"][R["RD_idx"]], base_year)
BD_label = label_month(R["dates_24"][R["BD_idx"]], base_year)
post_by_label = label_month(R["dates_24"][R["PostBy_idx"]], base_year)
freeze_by_label = label_month(R["dates_24"][R["FreezeBy_idx"]], base_year)

recruit_label = month_range_label_from_indices(R["dates_24"], R["recruiting_idxs"], base_year)
freeze_label = month_range_label_from_indices(R["dates_24"], R["freeze_idxs"], base_year)

st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; gap:16px;
                padding:12px 16px; background:#F7F7F7; border-radius:10px;
                border:1px solid #E0E0E0; font-size:15px;">
        <div><b>Recruiting Window:</b> {recruit_label}</div>
        <div><b>Post-By:</b> {post_by_label}</div>
        <div><b>Freeze-By:</b> {freeze_by_label}</div>
        <div><b>Ready-By (RD):</b> {RD_label}</div>
        <div><b>Baseline-By (BD):</b> {BD_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

peak_gap = float(max(R["burnout_gap_12"])) if R["burnout_gap_12"] else 0.0
avg_gap = float(np.mean(R["burnout_gap_12"])) if R["burnout_gap_12"] else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Peak Burnout Gap (FTE)", f"{peak_gap:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
m3.metric("Months Exposed", f"{int(R['months_exposed_12'])}/12")

st.info(
    f"**Carryover (real continuity):** Projected starting supply next January = "
    f"**{float(R['carryover_next_jan']):.2f} FTE** (from month 13 of the 24-month simulation)."
)

if not R.get("strategy_feasible", True):
    st.warning(
        "⚠️ **Feasibility warning:** Given the current pipeline + turnover + notice assumptions, "
        "the model could not construct a feasible Post-By → Freeze-By recruiting window that fully meets readiness. "
        "The supply curve still reflects best-case starts, but you may need earlier action or shorter cycle time."
    )

# Shade months (only on Jan–Dec chart):
# - Recruiting months that fall within Year 1
# - Freeze months that fall within Year 1
recruit_set_y1 = set([i for i in R["recruiting_idxs"] if 0 <= i <= 11])
freeze_set_y1 = set([i for i in R["freeze_idxs"] if 0 <= i <= 11])

x = np.arange(12)
labels = R["month_labels_12"]

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

for i in range(12):
    if i in recruit_set_y1:
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.06, color="#6aa7ff", linewidth=0)
    if i in freeze_set_y1:
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.12, color=BRAND_GOLD, linewidth=0)

ax1.plot(x, R["demand_lean_12"], linestyle=":", linewidth=1.2, color=GRAY, label="Lean Target (Demand)")
ax1.plot(x, R["demand_prot_12"], linewidth=2.0, color=BRAND_GOLD, marker="o", markersize=4, label="Recommended Target (Protective)")
ax1.plot(x, R["supply_12"], linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=4, label="Realistic Supply (Pipeline)")

ax1.fill_between(
    x,
    R["supply_12"],
    R["demand_prot_12"],
    where=np.array(R["demand_prot_12"]) > np.array(R["supply_12"]),
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
ax2.plot(x, R["visits_12"], linestyle="-.", linewidth=1.4, color=MID_GRAY, label="Forecast Visits/Day")
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

total_starts_y1 = float(np.sum(np.array(R["hire_starts_24"][:12])))
total_starts_y2 = float(np.sum(np.array(R["hire_starts_24"][12:24])))

st.success(
    f"**Reality Summary:** This is a best-case plan constrained by your pipeline inputs. "
    f"Recruiting window: **{recruit_label}**. Freeze discipline begins after **{freeze_by_label}** "
    f"to allow notice-lag turnover to normalize by **{BD_label}**. "
    f"Year-1 permanent hire starts (within Jan–Dec): **{total_starts_y1:.2f} FTE**."
)

st.info(
    "🧠 **Strategy Notes**\n\n"
    f"- Lead time: **{int(R['pipeline_lead_days'])} days (~{int(R['lead_months'])} months)**\n"
    f"- Notice lag: **~{int(R['notice_months'])} months**\n"
    f"- Incremental seasonal staffing to shed (X): **{float(R['X_incremental_flu_fte']):.2f} FTE**\n"
    f"- Start-cap (throughput): **{float(R['hire_start_cap']):.2f} FTE / month**\n"
    f"- Confirmed hire: **{float(np.max(np.array(R['confirmed_add_24']))):.2f} FTE max contribution (ramped)**\n"
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

delta_fte_curve = [max(float(t) - float(R["baseline_provider_fte"]), 0.0) for t in R["demand_prot_12"]]
annual_investment = annualize_monthly_fte_cost(delta_fte_curve, R["days_in_month_12"], float(loaded_cost_per_provider_fte))

gap_days = provider_day_gap(R["demand_prot_12"], R["supply_12"], R["days_in_month_12"])
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

gap_fte_curve = [max(float(t) - float(s), 0.0) for t, s in zip(R["demand_prot_12"], R["supply_12"])]

effective_gap_curve = []
for g in gap_fte_curve:
    g2 = g * (1 - buffer_pct / 100.0)
    g2 = max(g2 - float_pool_fte, 0.0)
    g2 = max(g2 - fractional_fte, 0.0)
    effective_gap_curve.append(g2)

remaining_gap_days = provider_day_gap([0] * 12, effective_gap_curve, R["days_in_month_12"])
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
    st.metric("Months Exposed", f"{int(R['months_exposed_12'])}/12")
with col2:
    st.write(
        f"**Action plan (based on your inputs):**\n"
        f"- **Post-By:** {post_by_label} (to be ready by {RD_label})\n"
        f"- **Recruiting Window:** {recruit_label}\n"
        f"- **Freeze-By:** {freeze_by_label} (normalize by {BD_label} via notice-lag turnover)\n\n"
        f"**Continuity:**\n"
        f"- Starting supply next Jan (Y2) projected at **{float(R['carryover_next_jan']):.2f} FTE**\n\n"
        f"**Financial framing:**\n"
        f"- Annual protective investment: **${annual_investment:,.0f}**\n"
        f"- Estimated revenue at risk: **${est_revenue_lost:,.0f}**\n"
        f"- ROI: **{roi:,.2f}x**\n"
    )

st.success(
    "✅ **Decision Summary:** The model simulates real staffing physics across Dec→Jan (24-month engine) "
    "while keeping a clean Jan–Dec executive view. Recruiting and freeze guidance reflect pipeline lead time, "
    "notice-lag attrition, and burnout protection."
)
