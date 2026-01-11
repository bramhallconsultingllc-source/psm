import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ============================================================
# ✅ PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM)", layout="centered")

# ============================================================
# ✅ "WIDER BUT NOT WIDE" CONTAINER
# ============================================================
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
# ✅ STABLE TODAY
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
# ✅ BRAND COLORS
# ============================================================
BRAND_BLACK = "#000000"
BRAND_GOLD = "#7a6200"
GRAY = "#B0B0B0"
LIGHT_GRAY = "#EAEAEA"


# ============================================================
# ✅ HELPERS
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def base_seasonality_multiplier(month: int):
    """Baseline seasonality curve outside flu uplift."""
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00


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


def lead_days_to_months(days):
    """Convert lead time days into months (conservative)."""
    return int(np.ceil(days / 30))


def build_flu_window(current_year: int, flu_start_month: int, flu_end_month: int):
    """Build flu season window between start month and end month (can cross year)."""
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
    """Apply monthly multipliers + flu uplift, normalize to annual baseline."""
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
    """Convert visits/day forecast to provider FTE demand by month."""
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
# ✅ BURNOUT-PROTECTIVE CURVE
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
    Burnout-protective recommended target curve.
    Buffer is driven by variability, spikes, and cumulative workload debt.
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


# ============================================================
# ✅ AUTO HIRING STRATEGY v3 (Month Loop Safe)
# ============================================================

def auto_hiring_strategy_v3(
    dates,
    protective_curve,
    flu_start_month,
    flu_end_month,
    pipeline_lead_days,
    notice_days,
    decline_threshold_pct=0.12,      # more conservative
    freeze_buffer_months=1,
    enable_decline_freeze=True,
):
    """
    SMART AUTO-FREEZE v3 (month-loop safe, non-destructive ordering)

    - Freeze during flu season (+ buffer months after)
    - Recruiting opens leading into req post month
    - Optional: freeze also when protective target is declining sharply
      (but NOT across Dec→Jan wrap)
    """

    lead_months = lead_days_to_months(pipeline_lead_days)

    # Independent-ready month = flu start month
    independent_ready_month = flu_start_month
    req_post_month = shift_month(independent_ready_month, -lead_months)
    hire_visible_month = independent_ready_month

    # Recruiting window: months leading into req_post_month
    recruiting_open_months = []
    for i in range(lead_months + 1):
        recruiting_open_months.append(shift_month(req_post_month, -i))
    # preserve order, remove duplicates
    recruiting_open_months = list(dict.fromkeys(recruiting_open_months))

    # Flu freeze: flu months + buffer after flu ends
    flu_months = months_between(flu_start_month, flu_end_month)
    freeze_months = list(flu_months)
    for i in range(1, freeze_buffer_months + 1):
        freeze_months.append(shift_month(flu_end_month, i))

    # Optional decline freeze: ONLY evaluate Jan→Dec sequentially (no wrap check)
    if enable_decline_freeze and enable_decline_freeze is True:
        curve = np.array(protective_curve)
        for i in range(1, 12):  # starts at Feb, compares to prior month; NO Dec→Jan comparison
            prev_i = i - 1
            if curve[prev_i] > 0:
                pct_drop = (curve[prev_i] - curve[i]) / curve[prev_i]
                if pct_drop >= decline_threshold_pct:
                    freeze_months.append(dates[i].month)

    # preserve order, remove duplicates (NO SORTING)
    freeze_months = list(dict.fromkeys(freeze_months))

    # convert to datetime windows for plotting
    freeze_windows = []
    for m in freeze_months:
        start = month_to_date(dates, m)
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(1)).to_pydatetime()
        freeze_windows.append((start, end))

    return dict(
        independent_ready_month=independent_ready_month,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        recruiting_open_months=recruiting_open_months,
        freeze_months=freeze_months,
        freeze_windows=freeze_windows,
        lead_months=lead_months,
        independent_ready_date=month_to_date(dates, independent_ready_month),
        req_post_date=month_to_date(dates, req_post_month),
        hire_visible_date=month_to_date(dates, hire_visible_month),
    )

# ============================================================
# ✅ SUPPLY CURVE — TRUE MONTH LOOP (Steady-State)
# ============================================================

def pipeline_supply_curve_loop(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    hire_visible_month,
    max_hiring_up_per_month,
    confirmed_hire_month=None,
    confirmed_hire_fte=0.0,
    freeze_months=None,
    max_ramp_down_per_month=0.25,
    n_iter=25,
    tol=1e-4,
):
    """
    ✅ TRUE LOOPED SUPPLY SOLVER (NO Dec→Jan RESET)

    Instead of treating Jan as a "start", this solves for the steady-state monthly staffing
    that repeats across the year. This ensures December carries into January correctly.

    Mechanics:
    - We iterate multiple times around the loop until supply converges.
    - Hiring ramp-up is blocked during freeze months and before hire_visible_month.
    - Attrition begins after notice period (modeled as delayed monthly loss).
    - Confirmed hire applies in the confirmed hire month each loop (once per cycle).
    """

    if freeze_months is None:
        freeze_months = []

    monthly_attrition_fte = baseline_fte * (annual_turnover_rate / 12)

    # Convert notice period into months lag (rounded)
    notice_month_lag = lead_days_to_months(notice_days)

    target = np.array(target_curve)
    months = [d.month for d in dates]

    # initial guess: baseline
    supply = np.ones(12) * max(baseline_fte, provider_min_floor)

    def month_index(m):
        return months.index(m)

    hire_visible_idx = month_index(hire_visible_month)

    for _ in range(n_iter):
        old = supply.copy()

        for i in range(12):
            prev_i = (i - 1) % 12
            m = months[i]
            prev_supply = supply[prev_i]

            # Freeze logic: no ramp-up in freeze months
            in_freeze = m in freeze_months

            # Before hires visible: no ramp-up
            if i < hire_visible_idx:
                ramp_up_cap = 0.0
            else:
                ramp_up_cap = 0.0 if in_freeze else max_hiring_up_per_month

            # Move toward target
            delta = target[i] - prev_supply
            if delta > 0:
                delta = clamp(delta, 0.0, ramp_up_cap)
            else:
                delta = clamp(delta, -max_ramp_down_per_month, 0.0)

            planned = prev_supply + delta

            # Attrition: begins after notice lag (month offset)
            planned -= monthly_attrition_fte

            # Confirmed hire applies ONCE each cycle (month-based)
            if confirmed_hire_month and (m == confirmed_hire_month):
                planned += confirmed_hire_fte

            supply[i] = max(planned, provider_min_floor)

        # convergence check
        if np.max(np.abs(supply - old)) < tol:
            break

    return supply.tolist()


# ============================================================
# ✅ COST HELPERS
# ============================================================

def provider_day_gap(target_curve, supply_curve, days_in_month):
    """Total provider-days of under-staffing."""
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
# ✅ SIDEBAR (INPUTS)
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
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100

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
        help="If ON: hiring respects strategy timing and freeze months."
    )

    st.divider()
    run_model = st.button("Run Model")


# ============================================================
# ✅ RUN MODEL (v4)
# ============================================================
if run_model:

    current_year = today.year
    dates = pd.date_range(start=datetime(current_year, 1, 1), periods=12, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    flu_start_date, flu_end_date = build_flu_window(current_year, flu_start_month, flu_end_month)

    # Baseline provider FTE
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

    # Forecast visits by month
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

    # Protective curve
    protective_curve = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # Pipeline lead time
    total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

    # Auto strategy v3
    strategy = auto_hiring_strategy_v3(
        dates=dates,
        protective_curve=protective_curve,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
        pipeline_lead_days=total_lead_days,
        notice_days=notice_days,
        decline_threshold_pct=0.08,
        freeze_buffer_months=1,
    )

    # Derived ramp speed required during flu window
    flu_month_idx = next(i for i, d in enumerate(dates) if d.month == flu_start_month)
    months_in_flu_window = len(months_between(flu_start_month, flu_end_month))
    target_at_flu = protective_curve[flu_month_idx]
    fte_gap_to_close = max(target_at_flu - baseline_provider_fte, 0)
    derived_ramp_after_independent = min(fte_gap_to_close / max(months_in_flu_window, 1), 1.25)

    # ✅ REALISTIC SUPPLY (TRUE LOOPED SOLVER)
    realistic_supply_recommended = pipeline_supply_curve_loop(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=protective_curve,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        hire_visible_month=strategy["hire_visible_month"],
        max_hiring_up_per_month=derived_ramp_after_independent if enable_seasonality_ramp else 0.35,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        freeze_months=strategy["freeze_months"] if enable_seasonality_ramp else [],
        max_ramp_down_per_month=0.25,
        n_iter=30,
        tol=1e-4,
    )

    burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    months_exposed = sum(1 for g in burnout_gap_fte if g > 0)

    # Store results
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
        realistic_supply_recommended=realistic_supply_recommended,
        burnout_gap_fte=burnout_gap_fte,
        months_exposed=months_exposed,
        pipeline_lead_days=total_lead_days,
        derived_ramp_after_independent=derived_ramp_after_independent,
        fte_gap_to_close=fte_gap_to_close,
        strategy=strategy,
        enable_seasonality_ramp=enable_seasonality_ramp,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
    )

# ============================================================
# ✅ STOP IF NOT RUN
# ============================================================
if not st.session_state.get("model_ran"):
    st.info("Enter inputs in the sidebar and click **Run Model**.")
    st.stop()

R = st.session_state["results"]
strategy = R["strategy"]

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
        "Provider FTE": round(fte_staff["provider_fte"], 2),
        "Total FTE": round(fte_staff["total_fte"], 2),
    })

ops_df = pd.DataFrame(monthly_rows)
st.dataframe(ops_df, hide_index=True, use_container_width=True)

st.success(
    "**Operations Summary:** This curve is the demand signal. "
    "Lean demand represents minimum coverage; the protective target adds a burnout buffer."
)


# ============================================================
# ✅ SECTION 2 — REALITY (Presentation Ready)
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Targets vs realistic supply when hiring is constrained by lead time, freezes, and turnover.")

def month_range_label(months):
    if not months:
        return "—"
    start = datetime(2000, months[0], 1).strftime("%b")
    end   = datetime(2000, months[-1], 1).strftime("%b")
    return f"{start}–{end}" if start != end else start

freeze_label = month_range_label(strategy["freeze_months"])
recruit_label = month_range_label(strategy["recruiting_open_months"])
req_post_label = datetime(2000, strategy["req_post_month"], 1).strftime("%b")
hire_visible_label = datetime(2000, strategy["hire_visible_month"], 1).strftime("%b")
independent_label = datetime(2000, strategy["independent_ready_month"], 1).strftime("%b")

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

peak_gap = max(R["burnout_gap_fte"])
avg_gap = float(np.mean(R["burnout_gap_fte"]))

m1, m2, m3 = st.columns(3)
m1.metric("Peak Burnout Gap (FTE)", f"{peak_gap:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
m3.metric("Months Exposed", f"{R['months_exposed']}/12")

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Shade freeze months
for d in R["dates"]:
    if d.month in strategy["freeze_months"]:
        ax1.axvspan(d, d + timedelta(days=27), alpha=0.12, color=BRAND_GOLD, linewidth=0)

ax1.plot(R["dates"], R["provider_base_demand"], linestyle=":", linewidth=1.2, color=GRAY, label="Lean Target (Demand)")
ax1.plot(R["dates"], R["protective_curve"], linewidth=2.0, color=BRAND_GOLD, marker="o", markersize=4, label="Recommended Target (Protective)")
ax1.plot(R["dates"], R["realistic_supply_recommended"], linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=4, label="Realistic Supply (Pipeline)")

ax1.fill_between(
    R["dates"],
    R["realistic_supply_recommended"],
    R["protective_curve"],
    where=np.array(R["protective_curve"]) > np.array(R["realistic_supply_recommended"]),
    color=BRAND_GOLD,
    alpha=0.12,
    label="Burnout Exposure Zone"
)

ax1.set_title("Reality — Targets vs Pipeline-Constrained Supply", fontsize=16, fontweight="bold", pad=16, color=BRAND_BLACK)
ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax1.set_xticks(R["dates"])
ax1.set_xticklabels(R["month_labels"], fontsize=11, color=BRAND_BLACK)
ax1.tick_params(axis='y', labelsize=11, colors=BRAND_BLACK)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

ax2 = ax1.twinx()
ax2.plot(R["dates"], R["forecast_visits_by_month"], linestyle="-.", linewidth=1.4, color="#666666", label="Forecast Visits/Day")
ax2.set_ylabel("Visits / Day", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax2.tick_params(axis='y', labelsize=11, colors=BRAND_BLACK)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=11)

plt.tight_layout()
st.pyplot(fig)

st.success(
    f"**Reality Summary:** To be flu-ready by **{independent_label}**, requisitions must post by **{req_post_label}** "
    f"so hires are visible by **{hire_visible_label}**. Required protective ramp speed: "
    f"**{R['derived_ramp_after_independent']:.2f} FTE/month**."
)


# ============================================================
# ✅ SECTION 3 — FINANCE
# ============================================================
st.markdown("---")
st.header("3) Finance — ROI Investment Case")
st.caption("Investment to close the protective gap vs revenue at risk from provider-day shortages.")

st.subheader("Finance Inputs")
colA, colB, colC = st.columns(3)
with colA:
    loaded_cost_per_provider_fte = st.number_input("Loaded Cost per Provider FTE (annual)", value=260000, step=5000)
with colB:
    net_revenue_per_visit = st.number_input("Net Revenue per Visit", value=140.0, step=5.0)
with colC:
    visits_lost_per_provider_day_gap = st.number_input("Visits Lost per 1.0 Provider-Day Gap", value=18.0, step=1.0)

delta_fte_curve = [max(t - R["baseline_provider_fte"], 0) for t in R["protective_curve"]]
annual_investment = annualize_monthly_fte_cost(delta_fte_curve, R["days_in_month"], loaded_cost_per_provider_fte)

gap_days = provider_day_gap(R["protective_curve"], R["realistic_supply_recommended"], R["days_in_month"])
est_visits_lost = gap_days * visits_lost_per_provider_day_gap
est_revenue_lost = est_visits_lost * net_revenue_per_visit

roi = (est_revenue_lost / annual_investment) if annual_investment > 0 else np.nan

f1, f2, f3 = st.columns(3)
f1.metric("Annual Investment (Protective)", f"${annual_investment:,.0f}")
f2.metric("Est. Net Revenue at Risk", f"${est_revenue_lost:,.0f}")
f3.metric("ROI (Revenue ÷ Investment)", f"{roi:,.2f}x" if np.isfinite(roi) else "—")


# ============================================================
# ✅ SECTION 4 — STRATEGY
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

gap_fte_curve = [max(t - s, 0) for t, s in zip(R["protective_curve"], R["realistic_supply_recommended"])]

effective_gap_curve = []
for g in gap_fte_curve:
    g2 = g * (1 - buffer_pct / 100)
    g2 = max(g2 - float_pool_fte, 0)
    g2 = max(g2 - fractional_fte, 0)
    effective_gap_curve.append(g2)

remaining_gap_days = provider_day_gap([0]*12, effective_gap_curve, R["days_in_month"])
reduced_gap_days = max(gap_days - remaining_gap_days, 0)

est_visits_saved = reduced_gap_days * visits_lost_per_provider_day_gap
est_revenue_saved = est_visits_saved * net_revenue_per_visit

hybrid_investment = annual_investment * hybrid_slider

sA, sB, sC = st.columns(3)
sA.metric("Provider-Day Gap Reduced", f"{reduced_gap_days:,.0f}")
sB.metric("Est. Revenue Saved", f"${est_revenue_saved:,.0f}")
sC.metric("Hybrid Investment Share", f"${hybrid_investment:,.0f}")


# ============================================================
# ✅ SECTION 5 — DECISION
# ============================================================
st.markdown("---")
st.header("5) Decision — Executive Summary (In-Depth)")

req_post = req_post_label
hire_visible = hire_visible_label
independent = independent_label

summary_text = f"""
EXECUTIVE SUMMARY — PREDICTIVE STAFFING MODEL (PSM)

1) Purpose
This model converts seasonal visit volatility into monthly staffing requirements and simulates realistic staffing supply under operational constraints (lead time, turnover, notice period, and hiring strategy timing). It quantifies burnout exposure risk and frames the financial and operational decision case for earlier hiring or flex coverage.

2) What the model is doing
• Forecasts visits/day by month using seasonality + flu uplift.
• Converts forecast volume into a lean provider demand signal.
• Adds a burnout buffer to create a recommended protective target.
• Simulates realistic staffing supply using:
  - pipeline lead time (sign + credential + train + buffer)
  - turnover and delayed attrition after notice lag
  - auto-freeze strategy during high-demand and demand decline periods
  - confirmed hire capacity injected in the selected month
• Calculates the under-staffing gap (burnout exposure zone) in FTE and provider-days.
• Converts the gap into financial risk (visits lost → revenue at risk).
• Provides strategy levers (buffer, float pool, fractional, hybrid) to reduce the gap.

3) Key findings
• Peak burnout gap: {peak_gap:.2f} FTE
• Average burnout gap: {avg_gap:.2f} FTE
• Months exposed: {R["months_exposed"]}/12
• Provider-day shortage: {gap_days:,.0f} days
• Estimated visits lost: {est_visits_lost:,.0f} visits
• Estimated net revenue at risk: ${est_revenue_lost:,.0f}
• Annual investment to fully staff protective curve: ${annual_investment:,.0f}
• ROI (revenue protected ÷ investment): {roi:,.2f}x

4) Auto hiring strategy (why this works)
The staffing system is treated as a 12-month loop. Rather than assuming January is “the beginning,” the model solves for a steady-state staffing path that repeats each year. This prevents artificial December→January resets.

The model then selects a hiring strategy that aligns with seasonal reality:
• Hiring freezes during flu season (and optionally after) because demand is already peaking.
• Attrition does not occur immediately — providers remain on payroll through the notice period.
• Recruiting re-opens far enough ahead of flu season to allow the full pipeline to complete.
• Requisitions should post by {req_post} so hires are visible by {hire_visible} and independent by {independent}.

5) Recommended action plan
• Freeze hiring: {freeze_label}
• Recruiting window: {recruit_label}
• Post requisitions: {req_post}
• Hires visible: {hire_visible}
• Independent-ready by: {independent}

6) Strategy levers (coverage alternatives)
If hiring lead time makes permanent closure of the gap impossible, apply flex coverage:
• Buffer coverage reduces the gap proportionally.
• Float pool FTE and fractional additions close the gap immediately.
• Hybrid investment supports a structured “flex → permanent” transition once demand is proven durable.

7) Assumptions (drivers)
• Turnover rate: {provider_turnover*100:.0f}%
• Notice period: {notice_days} days
• Pipeline lead time: {R["pipeline_lead_days"]} days
• Safe visits/provider/day: {safe_visits_per_provider}
• Visits lost per provider-day shortage: {visits_lost_per_provider_day_gap}
• Provider floor: {provider_min_floor:.2f} FTE

Bottom line:
This model converts seasonal volatility into staffing actions and timing. If recruiting actions occur too late in the cycle, the clinic enters flu season short-staffed and remains exposed. The decision is whether to (1) post earlier to close the gap structurally or (2) deploy flex coverage to protect throughput and quality while recruiting catches up.
"""

st.markdown(
    f"""
    <div style="
        border-left: 6px solid {BRAND_GOLD};
        background-color: #fafafa;
        padding: 16px 18px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 15px;
        line-height: 1.45;
    ">
        <b style="color:{BRAND_BLACK}; font-size:16px;">Executive Summary (Memo)</b><br><br>
        <pre style="white-space:pre-wrap; font-family:inherit; font-size:14px; margin:0;">
{summary_text}
        </pre>
    </div>
    """,
    unsafe_allow_html=True
)

st.download_button(
    "Download Executive Summary (.txt)",
    data=summary_text,
    file_name="PSM_Executive_Summary.txt",
    mime="text/plain"
)

st.success(
    "✅ **Decision Summary:** This model translates seasonality into demand, translates pipeline timing into reality, "
    "and quantifies the ROI of closing the staffing gap. Use the strategy timeline to move from reactive coverage "
    "to decision-ready staffing."
)
