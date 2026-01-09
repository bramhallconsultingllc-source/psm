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


def pipeline_supply_curve(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    req_post_date,
    pipeline_lead_days,
    max_hiring_up_after_pipeline,
    confirmed_hire_date=None,
    confirmed_hire_fte=0.0,
    max_ramp_down_per_month=0.25,
    seasonality_ramp_enabled=True,
    freeze_windows=None,
):
    """
    Pipeline-aware realistic supply curve.

    - Attrition begins AFTER notice period.
    - Hiring ramp UP is blocked until hires become visible (req_post_date + pipeline days).
    - Ramp UP is blocked during ANY freeze window.
    - Confirmed hire creates a one-time supply jump on hire date.
    """

    if freeze_windows is None:
        freeze_windows = []

    monthly_attrition_fte = baseline_fte * (annual_turnover_rate / 12)

    # Attrition begins after notice lag
    effective_attrition_start = today + timedelta(days=int(notice_days))

    # Hiring becomes visible only after pipeline completes
    hire_visible_date = req_post_date + timedelta(days=int(pipeline_lead_days))

    # ✅ Convert confirmed hire date into datetime (applied once)
    confirmed_hire_dt = None
    if confirmed_hire_date:
        confirmed_hire_dt = datetime.combine(confirmed_hire_date, datetime.min.time())

    hire_applied = False

    staff = []
    prev = max(baseline_fte, provider_min_floor)

    for d, target in zip(dates, target_curve):
        d_py = d.to_pydatetime()

        # -------------------------------
        # ✅ Hiring freeze logic (supports multiple windows)
        # -------------------------------
        in_freeze = in_any_freeze_window(d_py, freeze_windows)

        # -------------------------------
        # ✅ Ramp cap logic
        # -------------------------------
        if seasonality_ramp_enabled:
            # No ramp-up during freeze OR before hires are visible
            if in_freeze or (d_py < hire_visible_date):
                ramp_up_cap = 0.0
            else:
                ramp_up_cap = max_hiring_up_after_pipeline
        else:
            ramp_up_cap = 0.35

        # -------------------------------
        # ✅ Move supply toward target
        # -------------------------------
        delta = target - prev
        if delta > 0:
            delta = clamp(delta, 0.0, ramp_up_cap)
        else:
            delta = clamp(delta, -max_ramp_down_per_month, 0.0)

        planned = prev + delta

        # -------------------------------
        # ✅ Attrition loss after notice lag
        # -------------------------------
        if d_py >= effective_attrition_start:
            planned -= monthly_attrition_fte

        # -------------------------------
        # ✅ Confirmed Hire (hard jump once)
        # -------------------------------
        if (not hire_applied) and confirmed_hire_dt and (d_py >= confirmed_hire_dt):
            planned += confirmed_hire_fte
            hire_applied = True

        planned = max(planned, provider_min_floor)

        staff.append(planned)
        prev = planned

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
    flu_uplift_pct = st.number_input(
        "Flu Uplift (%)",
        min_value=0.0,
        value=20.0,
        step=5.0
    ) / 100

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

    st.subheader("Confirmed Hiring")

    confirmed_hire_date = st.date_input(
        "Confirmed Hire Start Date",
        value=datetime(today.year, 11, 1).date(),
        help="Date the confirmed provider begins seeing patients independently."
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

# ============================================================
# ✅ RUN MODEL
# ============================================================
if run_model:

    current_year = today.year
    dates = pd.date_range(start=datetime(current_year, 1, 1), periods=12, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    flu_start_date, flu_end_date = build_flu_window(current_year, flu_start_month, flu_end_month)
    
    # ✅ Convert freeze dates into datetimes for model comparisons
    freeze_start = datetime.combine(freeze_start_date, datetime.min.time())
    freeze_end = datetime.combine(freeze_end_date, datetime.min.time())


    # Baseline provider FTE
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

    # Forecast visits
    forecast_visits_by_month = compute_seasonality_forecast(
        dates=dates,
        baseline_visits=visits,
        flu_start=flu_start_date,
        flu_end=flu_end_date,
        flu_uplift_pct=flu_uplift_pct,
    )

    # Lean demand curve
    provider_base_demand = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_by_month,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    # Protective (recommended) target curve
    protective_curve = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # ============================================================
    # ✅ AUTO-CALCULATED RECRUITING RAMP
    # ------------------------------------------------------------
    # staffing_needed_by = flu_start_date
    # req_post_date = staffing_needed_by - (days_to_sign + credential + training + buffer)
    # solo_ready_date = staffing_needed_by
    # ramp after solo is derived from FTE gap to close during flu window
    # ============================================================

    staffing_needed_by = flu_start_date
    total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

    req_post_date = staffing_needed_by - timedelta(days=total_lead_days)
    solo_ready_date = staffing_needed_by

    flu_month_idx = next(i for i, d in enumerate(dates) if d.to_pydatetime().month == flu_start_date.month)

    flu_end_idx = flu_month_idx
    for i, d in enumerate(dates):
        if d.to_pydatetime() <= flu_end_date:
            flu_end_idx = i

    months_in_flu_window = max(flu_end_idx - flu_month_idx + 1, 1)

    target_at_flu = protective_curve[flu_month_idx]
    supply_at_solo = baseline_provider_fte
    fte_gap_to_close = max(target_at_flu - supply_at_solo, 0)

    derived_ramp_after_solo = fte_gap_to_close / months_in_flu_window
    derived_ramp_after_solo = min(derived_ramp_after_solo, 1.25)
    
       
    # ============================================================
    # ✅ SUPPLY CURVES (LEAN + RECOMMENDED)
    # ============================================================

    # ✅ Convert sidebar freeze dates into datetimes for model comparisons
    freeze_start = datetime.combine(freeze_start_date, datetime.min.time())
    freeze_end = datetime.combine(freeze_end_date, datetime.min.time())

    realistic_supply_lean = pipeline_supply_curve(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        pipeline_lead_days=total_lead_days,
        max_hiring_up_after_pipeline=derived_ramp_after_solo,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        hiring_freeze_start=freeze_start,
        hiring_freeze_end=freeze_end,
        confirmed_hire_date=confirmed_hire_date,
        confirmed_hire_fte=confirmed_hire_fte,
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
        max_hiring_up_after_pipeline=derived_ramp_after_solo,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        hiring_freeze_start=freeze_start,
        hiring_freeze_end=freeze_end,
        confirmed_hire_date=confirmed_hire_date,
        confirmed_hire_fte=confirmed_hire_fte,
    )

    burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    months_exposed = sum([1 for g in burnout_gap_fte if g > 0])

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
        solo_ready_date=solo_ready_date,
        freeze_start=freeze_start,
        freeze_end=freeze_end,
        confirmed_hire_date=confirmed_hire_date,
        confirmed_hire_fte=confirmed_hire_fte,
        enable_seasonality_ramp=enable_seasonality_ramp,
        derived_ramp_after_solo=derived_ramp_after_solo,
        months_in_flu_window=months_in_flu_window,
        fte_gap_to_close=fte_gap_to_close,
        pipeline_lead_days=total_lead_days,
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
# ✅ SECTION 2 — REALITY
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Compares lean vs recommended targets against realistic supply given hiring lead time + attrition.")

fig, ax1 = plt.subplots(figsize=(10, 4))

# ------------------------------------------------------------
# ✅ Shade ALL Hiring Freeze Windows
# ------------------------------------------------------------
freeze_windows = R.get("freeze_windows", [])

chart_start = R["dates"][0].to_pydatetime()
chart_end = R["dates"][-1].to_pydatetime() + timedelta(days=27)

for start, end in freeze_windows:
    if start is None or end is None:
        continue

    # Clamp to chart display range
    shade_start = max(start, chart_start)
    shade_end = min(end, chart_end)

    if shade_start < shade_end:
        ax1.axvspan(shade_start, shade_end, alpha=0.10)

if freeze_windows:
    ax1.text(
        R["dates"][0],
        ax1.get_ylim()[1] * 0.98,
        "Hiring Freeze",
        fontsize=9,
        alpha=0.7,
        va="top",
    )

# ------------------------------------------------------------
# ✅ Visual cue: Shade hiring freeze window
# ------------------------------------------------------------
freeze_start = R.get("freeze_start")
freeze_end = R.get("freeze_end")
if freeze_start and freeze_end:
    # Clamp to chart range
    chart_start = R["dates"][0].to_pydatetime()
    chart_end = R["dates"][-1].to_pydatetime() + timedelta(days=27)

    # Handle freeze window crossing year boundary
    if freeze_end < freeze_start:
        # Shade from freeze_start → chart_end
        left_start = max(freeze_start, chart_start)
        left_end = chart_end
        if left_start < left_end:
            ax1.axvspan(left_start, left_end, alpha=0.10)

        # Shade from chart_start → freeze_end
        right_start = chart_start
        right_end = min(freeze_end, chart_end)
        if right_start < right_end:
            ax1.axvspan(right_start, right_end, alpha=0.10)
    else:
        shade_start = max(freeze_start, chart_start)
        shade_end = min(freeze_end, chart_end)
        if shade_start < shade_end:
            ax1.axvspan(shade_start, shade_end, alpha=0.10)

    ax1.text(
        R["dates"][0],
        ax1.get_ylim()[1] * 0.98,
        "Hiring Freeze",
        fontsize=9,
        alpha=0.7,
        va="top",
    )

ax1.plot(R["dates"], R["provider_base_demand"], linestyle=":", linewidth=2, label="Lean Target (Demand)")
ax1.plot(R["dates"], R["protective_curve"], linewidth=3, marker="o", label="Recommended Target (Protective)")
ax1.plot(R["dates"], R["realistic_supply_recommended"], linewidth=3, marker="o", label="Realistic Supply (Pipeline)")

ax1.fill_between(
    R["dates"],
    R["realistic_supply_recommended"],
    R["protective_curve"],
    where=np.array(R["protective_curve"]) > np.array(R["realistic_supply_recommended"]),
    alpha=0.25,
    label="Burnout Exposure Zone"
)

ax1.set_title("Reality — Volume, Targets, Supply & Burnout Exposure")
ax1.set_ylabel("Provider FTE")
ax1.set_xticks(R["dates"])
ax1.set_xticklabels(R["month_labels"])
ax1.grid(axis="y", linestyle=":", alpha=0.35)

ax2 = ax1.twinx()
ax2.plot(R["dates"], R["forecast_visits_by_month"], linestyle="-.", linewidth=2.5, label="Forecast Visits/Day")
ax2.set_ylabel("Visits / Day")

# Markers
for marker_date, label in [
    (R["req_post_date"], "Req Post By"),
    (R.get("hire_visible_date"), "Hires Visible"),
    (R["solo_ready_date"], "Solo By"),
]:
    if marker_date is None:
        continue
    if R["dates"][0].to_pydatetime() <= marker_date <= R["dates"][-1].to_pydatetime():
        ax1.axvline(marker_date, linestyle="--", linewidth=1.5, alpha=0.6)
        ax1.annotate(label, xy=(marker_date, ax1.get_ylim()[1]), xytext=(marker_date, ax1.get_ylim()[1] + 0.2),
                     ha="center", fontsize=9, rotation=90)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, -0.25), ncol=2)

plt.tight_layout()
st.pyplot(fig)

k1, k2, k3 = st.columns(3)
k1.metric("Peak Burnout Gap (FTE)", f"{max(R['burnout_gap_fte']):.2f}")
k2.metric("Avg Burnout Gap (FTE)", f"{np.mean(R['burnout_gap_fte']):.2f}")
k3.metric("Months Exposed", f"{R['months_exposed']}/12")

if R.get("enable_seasonality_ramp"):
    st.success(
        f"**Reality Summary:** To meet flu demand, requisitions must post by **{R['req_post_date'].strftime('%b %d')}** "
        f"({R['pipeline_lead_days']} pipeline days), so providers go solo by **{R['solo_ready_date'].strftime('%b %d')}**. "
        f"Derived ramp speed is **{R['derived_ramp_after_solo']:.2f} FTE/month** during flu season."
    )
else:
    st.warning("Reality Summary: Seasonality Recruiting Ramp is OFF — supply uses a generic ramp only.")


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
