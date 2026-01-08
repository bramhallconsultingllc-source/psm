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
# ✅ Burnout-Protective Staffing Curve (NEW)
# ============================================================
protective_curve, buffers = burnout_protective_staffing_curve(
    dates=dates,
    visits_by_month=forecast_visits_by_month,
    base_demand_fte=provider_base_demand,
    provider_min_floor=provider_min_floor,
    burnout_slider=burnout_slider,
    safe_visits_per_provider_per_day=safe_visits_per_provider,
)

peak_staffing = max(protective_curve)


# ============================================================
# ✅ STEP A6: Provider Seasonality + Hiring Glidepath (Executive View)
# ============================================================
st.markdown("---")
st.subheader("Provider Seasonality + Hiring Glidepath (Executive View)")
st.caption(
    "Shows predicted volume (right axis), base demand staffing, burnout-protective staffing, "
    "attrition risk under freeze, and forecasted actual staffing."
)

# ------------------------------------------------------------
# ✅ Colors
# ------------------------------------------------------------
COLOR_SIGNING = "#7a6200"         # Sunshine Gold
COLOR_CREDENTIALING = "#3b78c2"   # deep blue
COLOR_TRAINING = "#2e9b6a"        # green/teal
COLOR_FLU_SEASON = "#f4c542"      # warm flu highlight
COLOR_FREEZE = "#9c9c9c"          # freeze gray


# ------------------------------------------------------------
# ✅ Provider pipeline timeline (anchor to flu_start_date)
# ------------------------------------------------------------
staffing_needed_by = flu_start_date
total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

req_post_date = staffing_needed_by - timedelta(days=total_lead_days)
signed_date = req_post_date + timedelta(days=days_to_sign)
credentialed_date = signed_date + timedelta(days=days_to_credential)
solo_ready_date = credentialed_date + timedelta(days=onboard_train_days)


# ------------------------------------------------------------
# ✅ Conservative Freeze Logic (based on protective curve)
# ------------------------------------------------------------
annual_turnover_rate = provider_turnover
monthly_attrition_fte = peak_staffing * (annual_turnover_rate / 12)

max_burnable = max(peak_staffing - provider_min_floor, 0)
months_to_burn_off = max_burnable / max(monthly_attrition_fte, 0.01)

freeze_start_date = flu_end_date - timedelta(days=int(months_to_burn_off * 30.4))
freeze_start_date = max(freeze_start_date, chart_start)
freeze_end_date = flu_end_date


# ------------------------------------------------------------
# ✅ Attrition projection + forecasted actual staffing
# ------------------------------------------------------------
attrition_line, effective_attrition_start = conservative_attrition_curve(
    dates=dates,
    peak_staffing=peak_staffing,
    provider_min_floor=provider_min_floor,
    annual_turnover_rate=annual_turnover_rate,
    freeze_start_date=freeze_start_date,
    notice_days=notice_days,
)

forecast_actual_staffing = conservative_freeze_forecast(
    target_curve=protective_curve,
    attrition_curve=attrition_line,
)


# ============================================================
# ✅ Plot (Dual Axis)
# ============================================================
fig, ax1 = plt.subplots(figsize=(12, 4))

# Left axis: Provider FTE curves
ax1.plot(dates, provider_base_demand, linewidth=2.5, marker="o",
         label="Base Staffing Target (Demand Curve)")
ax1.plot(dates, protective_curve, linewidth=3.5, marker="o",
         label=f"Burnout-Protective Staffing (Level {burnout_slider:.2f})")
ax1.plot(dates, attrition_line, linestyle="--", linewidth=2,
         label="Attrition Projection (Freeze, No Backfill)")
ax1.plot(dates, forecast_actual_staffing, linewidth=3,
         label="Forecasted Actual Staffing (Freeze Plan)")

ax1.set_title("Provider Seasonality + Hiring Glidepath (Executive Summary)")
ax1.set_ylabel("Provider FTE Needed")
ax1.set_ylim(0, max(protective_curve) + 1.5)

ax1.set_xticks(dates)
ax1.set_xticklabels(month_labels)
ax1.grid(axis="y", linestyle=":", alpha=0.35)

# Right axis: Visits/day forecast
ax2 = ax1.twinx()
ax2.plot(dates, forecast_visits_by_month, linestyle="-.", linewidth=2.5,
         label="Predicted Volume (Visits/Day)")
ax2.set_ylabel("Visits / Day")

# Shaded timeline blocks
ax1.axvspan(req_post_date, signed_date, color=COLOR_SIGNING, alpha=0.22)
ax1.axvspan(signed_date, credentialed_date, color=COLOR_CREDENTIALING, alpha=0.18)
ax1.axvspan(credentialed_date, solo_ready_date, color=COLOR_TRAINING, alpha=0.18)
ax1.axvspan(flu_start_date, flu_end_date, color=COLOR_FLU_SEASON, alpha=0.16)
ax1.axvspan(freeze_start_date, freeze_end_date, color=COLOR_FREEZE, alpha=0.15)

# Freeze markers
ymax = ax1.get_ylim()[1]
ax1.axvline(freeze_start_date, linestyle=":", linewidth=2, alpha=0.9)
ax1.axvline(effective_attrition_start, linestyle="--", linewidth=2, alpha=0.9)

ax1.annotate(
    "Freeze Starts",
    xy=(freeze_start_date, ymax),
    xytext=(freeze_start_date, ymax + 0.2),
    ha="center",
    fontsize=10,
    arrowprops=dict(arrowstyle="-|>", lw=1),
)
ax1.annotate(
    "Attrition Begins",
    xy=(effective_attrition_start, ymax),
    xytext=(effective_attrition_start, ymax + 0.2),
    ha="center",
    fontsize=10,
    arrowprops=dict(arrowstyle="-|>", lw=1),
)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2
)

ax1.set_xlim(dates[0], dates[-1])

plt.tight_layout()
st.pyplot(fig)


# ============================================================
# ✅ Transparency Table (Explains the Burnout Buffer)
# ============================================================
st.markdown("---")
st.subheader("Burnout Protection Transparency Table")
st.caption("Shows the components that drive burnout-protective staffing (volatility, spike, recovery debt).")

transparency_df = pd.DataFrame({
    "Month": month_labels,
    "Visits/Day": np.round(forecast_visits_by_month, 1),
    "Base Demand FTE": np.round(provider_base_demand, 2),
    "Burnout-Protective FTE": np.round(protective_curve, 2),
    "Volatility Buffer (raw)": np.round(buffers["volatility_buffer"], 2),
    "Spike Buffer (raw)": np.round(buffers["spike_buffer"], 2),
    "Recovery Debt Buffer (raw)": np.round(buffers["recovery_debt_buffer"], 2),
})

st.dataframe(transparency_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ Executive Interpretation
# ============================================================
st.info(
    f"""
✅ **Executive Interpretation**
- **Base staffing target** follows predicted demand based on seasonality.
- **Burnout-protective staffing** adds buffers for:
  - **volatility** (demand chaos),
  - **spikes** (peak overload),
  - **recovery debt** (fatigue accumulation over time).
- Protection level is **user-controlled** via slider: **{burnout_slider:.2f}**
- Anti-whiplash smoothing prevents unstable month-to-month staffing swings.
- Provider staffing never drops below the **minimum floor of {provider_min_floor:.2f} FTE**.
"""
)
