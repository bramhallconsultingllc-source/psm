import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ============================================================
# ✅ Stable "today" for consistent chart windows across reruns
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]


# ============================================================
# ✅ Session State Init (Crash-Proof)
# ============================================================
for key in ["daily_result", "fte_result", "fte_df", "calculated"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ============================================================
# ✅ Helper Functions (Single Source of Truth)
# ============================================================
def base_seasonality_multiplier(month: int):
    """Simple seasonal curve (winter high, summer low)."""
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00


def build_flu_window(current_year: int, flu_start_month: int, flu_end_month: int):
    """Return flu_start_date, flu_end_date; supports year wrap (e.g. Dec → Feb)."""
    flu_start_date = datetime(current_year, flu_start_month, 1)

    if flu_end_month < flu_start_month:
        flu_end_date = datetime(current_year + 1, flu_end_month, 1)
    else:
        flu_end_date = datetime(current_year, flu_end_month, 1)

    flu_end_date = flu_end_date + timedelta(days=32)
    flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)

    return flu_start_date, flu_end_date


def in_window(d: datetime, start: datetime, end: datetime):
    return start <= d <= end


def monthly_index(d: datetime, anchor: datetime):
    """Months elapsed between d and anchor (0 if same month)."""
    return (d.year - anchor.year) * 12 + (d.month - anchor.month)


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def compute_seasonality_forecast(dates, baseline_visits, flu_start, flu_end, flu_uplift_pct):
    """
    Returns normalized monthly visit/day forecast.
    Baseline visits is treated as annual average. Seasonal redistribution keeps avg stable.
    """
    raw = []
    for d in dates:
        mult = base_seasonality_multiplier(d.month)
        if in_window(d, flu_start, flu_end):
            mult *= (1 + flu_uplift_pct)
        raw.append(baseline_visits * mult)

    avg_raw = np.mean(raw)
    normalized = [v * (baseline_visits / avg_raw) for v in raw]
    return normalized


def visits_to_provider_demand(model, visits_by_month, hours_of_operation, fte_hours_per_week, provider_min_floor):
    """
    Convert visits forecast into provider FTE demand curve.
    Demand is never allowed below provider_min_floor.
    """
    demand = []
    for v in visits_by_month:
        fte = model.calculate_fte_needed(
            visits_per_day=v,
            hours_of_operation_per_week=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week,
        )["provider_fte"]
        demand.append(max(fte, provider_min_floor))
    return demand


# ============================================================
# ✅ Burnout-Protective Staffing Formula
# ============================================================
def burnout_protective_staffing_curve(
    dates,
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
    Builds a burnout-protective staffing curve using:
      - Volatility buffer (CV)
      - Spike buffer (above P75)
      - Recovery debt buffer (fatigue accumulation)
      - Anti-whiplash smoothing (rate-limited staffing changes)
      - Provider minimum floor

    burnout_slider ∈ [0,1] controls total protection level.
    weights = (vol_w, spike_w, debt_w)
    """

    vol_w, spike_w, debt_w = weights

    visits_arr = np.array(visits_by_month)
    mean_visits = np.mean(visits_arr)
    std_visits = np.std(visits_arr)
    cv = (std_visits / mean_visits) if mean_visits > 0 else 0.0

    p75 = np.percentile(visits_arr, 75)

    # ---- Buffers computed per month ----
    volatility_buffer = []
    spike_buffer = []
    recovery_buffer = []

    # Recovery Debt Index (RDI)
    rdi = 0.0
    decay = 0.85
    lambda_debt = 0.10  # translates debt into FTE (tunable but stable)

    protective_curve = []
    prev_staff = max(base_demand_fte[0], provider_min_floor)

    for i, (d, v, base_fte) in enumerate(zip(dates, visits_by_month, base_demand_fte)):

        # (1) volatility buffer: same for all months, scales with base_fte
        vbuf = base_fte * cv

        # (2) spike buffer: only above P75
        sbuf = max(0.0, (v - p75) / mean_visits) * base_fte

        # (3) recovery debt buffer: models sustained overload
        # assume staffing ≈ prev_staff for workload calc (conservative)
        visits_per_provider = v / max(prev_staff, 0.25)
        debt = max(0.0, visits_per_provider - safe_visits_per_provider_per_day)
        rdi = decay * rdi + debt
        dbuf = lambda_debt * rdi

        # total weighted buffer scaled by slider
        buffer_fte = burnout_slider * (
            vol_w * vbuf +
            spike_w * sbuf +
            debt_w * dbuf
        )

        raw_target = max(provider_min_floor, base_fte + buffer_fte)

        # ---- anti-whiplash smoothing (rate-limited) ----
        delta = raw_target - prev_staff
        if delta > 0:
            delta = clamp(delta, 0.0, smoothing_up)
        else:
            delta = clamp(delta, -smoothing_down, 0.0)

        final_staff = max(provider_min_floor, prev_staff + delta)

        # store
        volatility_buffer.append(vbuf)
        spike_buffer.append(sbuf)
        recovery_buffer.append(dbuf)
        protective_curve.append(final_staff)

        prev_staff = final_staff

    buffers = {
        "volatility_buffer": volatility_buffer,
        "spike_buffer": spike_buffer,
        "recovery_debt_buffer": recovery_buffer,
        "cv": cv,
        "p75": p75
    }

    return protective_curve, buffers


# ============================================================
# ✅ Conservative Attrition Projection
# ============================================================
def conservative_attrition_curve(
    dates,
    peak_staffing,
    provider_min_floor,
    annual_turnover_rate,
    freeze_start_date,
    notice_days,
):
    """
    Conservative attrition:
    - Attrition begins after freeze_start_date + notice_days
    - Cannot reduce staffing below provider_min_floor
    - Only burns down the amount above the floor
    """
    monthly_attrition = peak_staffing * (annual_turnover_rate / 12)
    effective_start = freeze_start_date + timedelta(days=int(notice_days))

    curve = []
    for d in dates:
        if d < effective_start:
            curve.append(peak_staffing)
        else:
            months = monthly_index(d, effective_start)
            loss = months * monthly_attrition
            max_burn = max(peak_staffing - provider_min_floor, 0)
            loss = min(loss, max_burn)
            curve.append(max(peak_staffing - loss, provider_min_floor))

    return curve, effective_start


def conservative_freeze_forecast(target_curve, attrition_curve):
    """
    Forecasted actual staffing under freeze:
    - before attrition, meets target
    - after attrition begins, declines as attrition constrains staffing
    """
    return [min(t, a) for t, a in zip(target_curve, attrition_curve)]


# ============================================================
# ✅ Page Setup
# ============================================================
st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")

st.title("Predictive Staffing Model (PSM)")
st.caption("A staffing calculator using linear interpolation + conservative rounding rules.")

st.info(
    "⚠️ **All daily staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)

model = StaffingModel()


# ============================================================
# ✅ Baseline Inputs
# ============================================================
st.markdown("## Baseline Inputs")

visits = st.number_input(
    "Average Visits per Day (Annual Average)",
    min_value=1.0,
    value=45.0,
    step=1.0
)

st.markdown("### Weekly Inputs (for FTE conversion)")

hours_of_operation = st.number_input(
    "Hours of Operation per Week",
    min_value=1.0,
    value=70.0,
    step=1.0
)

fte_hours_per_week = st.number_input(
    "FTE Hours per Week (default 40)",
    min_value=1.0,
    value=40.0,
    step=1.0
)

provider_min_floor = st.number_input(
    "Provider Minimum Floor (FTE)",
    min_value=0.25,
    value=1.00,
    step=0.25,
    help="Prevents unrealistic staffing projections (e.g., demand curves dropping to 0)."
)


# ============================================================
# ✅ Burnout Protection Control (B)
# ============================================================
st.markdown("## Burnout Protection Controls")

burnout_slider = st.slider(
    "Burnout Protection Level",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="""
0.0 = Lean (volume-only staffing)
0.3 = Balanced
0.6 = Protective (recommended)
1.0 = Max protection (strong buffers + smoothing)
"""
)

safe_visits_per_provider = st.number_input(
    "Safe Visits per Provider per Day Threshold",
    min_value=10,
    max_value=40,
    value=20,
    step=1,
    help="Used for Recovery Debt modeling. Higher = more aggressive staffing (less protective)."
)


# ============================================================
# ✅ Turnover Assumptions
# ============================================================
st.markdown("## Role-Specific Turnover Assumptions")

planning_months = st.number_input("Planning Horizon (months)", min_value=1, value=12, step=1)

t1, t2 = st.columns(2)

with t1:
    provider_turnover = st.number_input("Provider Turnover %", value=24.0, step=1.0) / 100
    psr_turnover = st.number_input("PSR Turnover %", value=30.0, step=1.0) / 100

with t2:
    ma_turnover = st.number_input("MA Turnover %", value=40.0, step=1.0) / 100
    xrt_turnover = st.number_input("XRT Turnover %", value=20.0, step=1.0) / 100


# ============================================================
# ✅ Flu Season Settings
# ============================================================
st.markdown("## Flu Season Settings")

flu_c1, flu_c2, flu_c3 = st.columns(3)

with flu_c1:
    flu_start_month = st.selectbox(
        "Flu Start Month",
        options=list(range(1, 13)),
        index=11,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
    )

with flu_c2:
    flu_end_month = st.selectbox(
        "Flu End Month",
        options=list(range(1, 13)),
        index=1,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
    )

with flu_c3:
    flu_uplift_pct = st.number_input(
        "Flu Uplift (%)",
        min_value=0.0,
        value=20.0,
        step=5.0,
        help="Applies ONLY during flu months. Model re-normalizes so annual average stays constant."
    ) / 100


# ============================================================
# ✅ Provider Pipeline Inputs
# ============================================================
st.markdown("## Provider Hiring Glidepath Inputs")

with st.expander("Provider Pipeline Assumptions", expanded=False):

    days_to_sign = st.number_input("Days to Sign (Req → Signed Offer)", min_value=1, value=90, step=5)
    days_to_credential = st.number_input("Days to Credential (Signed → Credentialed)", min_value=1, value=90, step=5)
    onboard_train_days = st.number_input("Onboard/Train Days (Credentialed → Solo)", min_value=0, value=30, step=5)

    coverage_buffer_days = st.number_input("Buffer Days (Planning Margin)", min_value=0, value=14, step=1)

    utilization_factor = st.number_input(
        "Hiring Effectiveness Factor",
        min_value=0.10,
        max_value=1.00,
        value=0.90,
        step=0.05,
        help="Included for planning realism (not yet applied to headcount math)."
    )

    notice_days = st.number_input(
        "Provider Resignation Notice Period (days)",
        min_value=0,
        max_value=180,
        value=75,
        step=5,
        help="Attrition begins only after providers leave following notice period."
    )


# ============================================================
# ✅ CALCULATE BUTTON
# ============================================================
if st.button("Calculate Staffing"):

    st.session_state["calculated"] = True

    daily_result = model.calculate(visits)
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )

    st.session_state["daily_result"] = daily_result
    st.session_state["fte_result"] = fte_result

    fte_df = pd.DataFrame(
        {
            "Role": ["Provider", "PSR", "MA", "XRT", "TOTAL"],
            "FTE Needed": [
                round(fte_result["provider_fte"], 2),
                round(fte_result["psr_fte"], 2),
                round(fte_result["ma_fte"], 2),
                round(fte_result["xrt_fte"], 2),
                round(fte_result["total_fte"], 2),
            ],
        }
    )

    st.session_state["fte_df"] = fte_df


# ============================================================
# ✅ MAIN OUTPUT
# ============================================================
if st.session_state.get("fte_df") is None:
    st.info("Enter inputs above and click **Calculate Staffing** to generate outputs.")
    st.stop()

fte_result = st.session_state["fte_result"]
fte_df = st.session_state["fte_df"]

baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)


# ============================================================
# ✅ Baseline Output Table
# ============================================================
st.markdown("---")
st.subheader("Baseline Full-Time Employees (FTEs) Needed")
st.dataframe(fte_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ Calendar-Year Timeline (Jan → Dec)
# ============================================================
current_year = today.year
chart_start = datetime(current_year, 1, 1)
dates = pd.date_range(start=chart_start, periods=12, freq="MS")
month_labels = [d.strftime("%b") for d in dates]

flu_start_date, flu_end_date = build_flu_window(current_year, flu_start_month, flu_end_month)


# ============================================================
# ✅ STEP A2: Seasonality Forecast (Visits/Day by Month)
# ============================================================
st.markdown("---")
st.subheader("Seasonality Forecast (Month-by-Month Projection)")
st.caption("Baseline visits/day is treated as your annual average. Seasonality redistributes volume across the year.")

forecast_visits_by_month = compute_seasonality_forecast(
    dates=dates,
    baseline_visits=visits,
    flu_start=flu_start_date,
    flu_end=flu_end_date,
    flu_uplift_pct=flu_uplift_pct,
)

forecast_df = pd.DataFrame({
    "Month": month_labels,
    "Forecast Visits/Day": np.round(forecast_visits_by_month, 1),
})
st.dataframe(forecast_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ Base Staffing Target Curve (Demand)
# ============================================================
provider_base_demand = visits_to_provider_demand(
    model=model,
    visits_by_month=forecast_visits_by_month,
    hours_of_operation=hours_of_operation,
    fte_hours_per_week=fte_hours_per_week,
    provider_min_floor=provider_min_floor,
)

# ============================================================
# ✅ Burnout-Protective Staffing Curve (Guaranteed Defined)
# ============================================================
protective_curve = None
buffers = {}

try:
    protective_curve, buffers = burnout_protective_staffing_curve(
        dates=dates,
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )
except Exception as e:
    st.warning(f"Burnout protective curve failed. Defaulting to base demand curve. Error: {e}")
    protective_curve = provider_base_demand
    buffers = {
        "volatility_buffer": [0]*len(dates),
        "spike_buffer": [0]*len(dates),
        "recovery_debt_buffer": [0]*len(dates),
        "cv": 0,
        "p75": 0
    }

# ============================================================
# ✅ A6: Executive + Analyst Views (Cleaner + Realistic)
# ============================================================
st.markdown("---")
st.subheader("Provider Seasonality + Staffing Plan (Executive View)")
st.caption(
    "Executive summary: forecasted volumes, recommended staffing target, burnout risk exposure, "
    "and a best-case realistic staffing path accounting for hiring limits + attrition."
)

# ------------------------------------------------------------
# ✅ Realistic Staffing Supply Curve
# ------------------------------------------------------------
def realistic_staffing_supply_curve(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    max_hiring_up_per_month=0.50,
    max_ramp_down_per_month=0.25,
):
    """
    Builds a 'best-case realistic staffing' curve:
      - Starts at baseline
      - Can move toward target only within ramp limits
      - Attrition begins after notice period (applied gradually)
      - Never drops below provider_min_floor
    """

    # Convert annual turnover into monthly attrition based on baseline (conservative)
    monthly_attrition_fte = baseline_fte * (annual_turnover_rate / 12)

    # ✅ Attrition starts notice_days after TODAY (realistic)
    effective_attrition_start = today + timedelta(days=int(notice_days))

    staff = []
    prev = max(baseline_fte, provider_min_floor)

    for d, target in zip(dates, target_curve):

        # ---- Step 1: Attempt to move toward target (realistic hiring / ramp constraints)
        delta = target - prev
        if delta > 0:
            delta = clamp(delta, 0.0, max_hiring_up_per_month)
        else:
            delta = clamp(delta, -max_ramp_down_per_month, 0.0)

        planned = prev + delta

        # ---- Step 2: Apply attrition after notice lag
        if d >= effective_attrition_start:
            months_elapsed = monthly_index(d, effective_attrition_start)
            attrition_loss = months_elapsed * monthly_attrition_fte
            planned = max(planned - attrition_loss, provider_min_floor)

        # ---- Step 3: Bound by floor
        planned = max(planned, provider_min_floor)

        staff.append(planned)
        prev = planned

    return staff


# Build best-case realistic staffing curve
realistic_actual_staffing = realistic_staffing_supply_curve(
    dates=dates,
    baseline_fte=baseline_provider_fte,
    target_curve=protective_curve,  # "what we want"
    provider_min_floor=provider_min_floor,
    annual_turnover_rate=provider_turnover,
    notice_days=notice_days,
    max_hiring_up_per_month=0.50,
    max_ramp_down_per_month=0.25,
)

# Burnout gap = (target - actual) if actual below target
burnout_gap = [max(t - a, 0) for t, a in zip(protective_curve, realistic_actual_staffing)]


# ------------------------------------------------------------
# ✅ Hiring Pipeline Milestones (for Executive Markers)
# Anchor to flu_start_date as the moment staffing is needed by
# ------------------------------------------------------------
staffing_needed_by = flu_start_date
total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

req_post_date = staffing_needed_by - timedelta(days=total_lead_days)
signed_date = req_post_date + timedelta(days=days_to_sign)
credentialed_date = signed_date + timedelta(days=days_to_credential)
solo_ready_date = credentialed_date + timedelta(days=onboard_train_days)

# Chart boundaries for safety
chart_min = dates[0].to_pydatetime()
chart_max = dates[-1].to_pydatetime()


# ------------------------------------------------------------
# ✅ EXECUTIVE PLOT (3 Lines + Burnout Risk Shading)
# ------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 4))

# Left axis: staffing
ax1.plot(dates, protective_curve, linewidth=3.5, marker="o",
         label="Recommended Target FTE (Burnout-Protective)")
ax1.plot(dates, realistic_actual_staffing, linewidth=3, marker="o",
         label="Best-Case Realistic Staffing (Supply)")
ax1.plot(dates, provider_base_demand, linewidth=2, linestyle=":",
         label="Base Demand Target (Volume Only)")

# Burnout risk shading: where actual < protective target
ax1.fill_between(
    dates,
    realistic_actual_staffing,
    protective_curve,
    where=np.array(protective_curve) > np.array(realistic_actual_staffing),
    alpha=0.25,
    label="Burnout Exposure Zone"
)

ax1.set_title("Provider Seasonality + Staffing Plan (Executive Summary)")
ax1.set_ylabel("Provider FTE")
ax1.set_ylim(0, max(protective_curve) + 1.5)
ax1.set_xticks(dates)
ax1.set_xticklabels(month_labels)
ax1.grid(axis="y", linestyle=":", alpha=0.35)

# Right axis: volume
ax2 = ax1.twinx()
ax2.plot(dates, forecast_visits_by_month, linestyle="-.", linewidth=2.5,
         label="Forecasted Volume (Visits/Day)")
ax2.set_ylabel("Visits / Day")

# ------------------------------------------------------------
# ✅ Minimal Hiring Markers (only show if inside chart window)
# ------------------------------------------------------------
ymax = ax1.get_ylim()[1]
for marker_date, label in [
    (req_post_date, "Req Post By"),
    (signed_date, "Sign By"),
    (solo_ready_date, "Solo By")
]:
    if chart_min <= marker_date <= chart_max:
        ax1.axvline(marker_date, linestyle="--", linewidth=1.5, alpha=0.6)
        ax1.annotate(
            label,
            xy=(marker_date, ymax),
            xytext=(marker_date, ymax + 0.25),
            ha="center",
            fontsize=9,
            rotation=90
        )

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=2
)

ax1.set_xlim(dates[0], dates[-1])
plt.tight_layout()
st.pyplot(fig)


# ------------------------------------------------------------
# ✅ Executive KPIs (Decision-focused)
# ------------------------------------------------------------
max_gap = max(burnout_gap)
avg_gap = np.mean(burnout_gap)
months_exposed = sum([1 for g in burnout_gap if g > 0])

st.markdown("### Executive Readout")

k1, k2, k3 = st.columns(3)
k1.metric("Peak Burnout Gap (FTE)", f"{max_gap:.2f}")
k2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
k3.metric("Months Exposed", f"{months_exposed}/12")

st.info(
    f"""
✅ **Executive Interpretation**
- The **Recommended Target FTE** includes burnout protection and should be considered the safe staffing goal.
- The **Best-Case Realistic Staffing** line reflects a constrained supply path: hiring ramp limits + attrition after notice.
- The shaded area shows **burnout exposure** when staffing supply is below the burnout-protective target.
- Forecasted volume is shown on the right axis.
- Provider staffing never drops below the **minimum floor of {provider_min_floor:.2f} FTE**.
"""
)


# ============================================================
# ✅ Analyst View (Expandable)
# ============================================================
with st.expander("Analyst View (Details + Assumptions)", expanded=False):

    st.caption("This section explains buffers and mechanics behind the executive view.")

    analyst_df = pd.DataFrame({
        "Month": month_labels,
        "Visits/Day": np.round(forecast_visits_by_month, 1),
        "Base Demand Target FTE": np.round(provider_base_demand, 2),
        "Burnout-Protective Target FTE": np.round(protective_curve, 2),
        "Best-Case Realistic Supply FTE": np.round(realistic_actual_staffing, 2),
        "Burnout Gap (FTE)": np.round(burnout_gap, 2),
    })

    st.dataframe(analyst_df, hide_index=True, use_container_width=True)

    st.markdown("#### Modeling Assumptions")
    st.write(f"""
    - Burnout protection level: **{burnout_slider:.2f}**
    - Safe visits/provider/day threshold: **{safe_visits_per_provider}**
    - Attrition modeled at **{provider_turnover*100:.1f}% annually**, beginning **{notice_days} days after today**
    - Hiring ramp limit: **+0.50 FTE/month**
    - Ramp-down limit: **-0.25 FTE/month**
    - Provider floor: **{provider_min_floor:.2f} FTE**
    """)
