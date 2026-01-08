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

if "runs" not in st.session_state:
    st.session_state["runs"] = []


# ============================================================
# ✅ Helper Functions (GLOBAL)
# ============================================================
def base_seasonality_multiplier(month: int):
    """
    Winter (Dec-Feb): 1.20
    Summer (Jun-Aug): 0.80
    Spring/Fall: 1.00
    """
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00


def is_in_flu_window(date_obj, flu_start_date, flu_end_date):
    return flu_start_date <= date_obj <= flu_end_date


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


# ============================================================
# ✅ GLOBAL INPUTS
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

model = StaffingModel()


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
# ✅ Flu Season Settings (GLOBAL)
# ============================================================
st.markdown("## Flu Season Settings")

flu_c1, flu_c2, flu_c3 = st.columns(3)

with flu_c1:
    flu_start_month = st.selectbox(
        "Flu Start Month",
        options=list(range(1, 13)),
        index=11,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
        key="flu_start_month_global"
    )

with flu_c2:
    flu_end_month = st.selectbox(
        "Flu End Month",
        options=list(range(1, 13)),
        index=1,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
        key="flu_end_month_global"
    )

with flu_c3:
    flu_uplift_pct = st.number_input(
        "Flu Uplift (%)",
        min_value=0.0,
        value=20.0,
        step=5.0,
        help="Applies ONLY during flu months. The model re-normalizes so annual avg stays equal."
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
        step=0.05
    )

    notice_days = st.number_input(
        "Provider Resignation Notice Period (days)",
        min_value=0,
        max_value=180,
        value=75,
        step=5,
        help="Attrition starts AFTER providers leave following notice period."
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
                fte_result["provider_fte"],
                fte_result["psr_fte"],
                fte_result["ma_fte"],
                fte_result["xrt_fte"],
                fte_result["total_fte"],
            ],
        }
    )

    fte_df["FTE Needed"] = fte_df["FTE Needed"].round(2)
    st.session_state["fte_df"] = fte_df


# ============================================================
# ✅ MAIN OUTPUT
# ============================================================
if st.session_state.get("fte_df") is None:
    st.info("Enter inputs above and click **Calculate Staffing** to generate outputs.")
    st.stop()

daily_result = st.session_state["daily_result"]
fte_result = st.session_state["fte_result"]
fte_df = st.session_state["fte_df"]

baseline_provider_fte = fte_result["provider_fte"]


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
chart_end = datetime(current_year, 12, 31)

dates = pd.date_range(start=chart_start, end=chart_end, freq="MS")
month_labels = [d.strftime("%b") for d in dates]


# ============================================================
# ✅ Flu Season Dates (handle wrap across year)
# ============================================================
flu_start_date = datetime(current_year, flu_start_month, 1)

if flu_end_month < flu_start_month:
    flu_end_date = datetime(current_year + 1, flu_end_month, 1)
else:
    flu_end_date = datetime(current_year, flu_end_month, 1)

flu_end_date = flu_end_date + timedelta(days=32)
flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)


# ============================================================
# ✅ STEP A2: Seasonality Forecast (Visits/Day by Month)
# ============================================================
st.markdown("---")
st.subheader("Seasonality Forecast (Month-by-Month Projection)")
st.caption("Baseline visits/day is treated as your annual average. Seasonality redistributes volume across the year.")

raw_forecast_visits = []

for d in dates:
    mult = base_seasonality_multiplier(d.month)

    if is_in_flu_window(d, flu_start_date, flu_end_date):
        mult = mult * (1 + flu_uplift_pct)

    raw_forecast_visits.append(visits * mult)

# Normalize so average stays equal to baseline visits input
avg_forecast = np.mean(raw_forecast_visits)
forecast_visits_by_month = [v * (visits / avg_forecast) for v in raw_forecast_visits]

forecast_df = pd.DataFrame({
    "Month": month_labels,
    "Forecast Visits/Day": np.round(forecast_visits_by_month, 1),
})

st.dataframe(forecast_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ Translate Forecast Visits → Provider FTE Demand
# ============================================================
provider_fte_demand = []

for v in forecast_visits_by_month:
    fte_month = model.calculate_fte_needed(
        visits_per_day=v,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    provider_fte_demand.append(fte_month["provider_fte"])


# ============================================================
# ✅ STEP A6: Provider Seasonality + Hiring Glidepath (Executive View)
# ============================================================
st.markdown("---")
st.subheader("Provider Seasonality + Hiring Glidepath (Executive View)")

st.caption(
    "This chart shows (1) predicted volume (right axis), (2) staffing target based on demand, "
    "(3) attrition risk if you freeze hiring, and (4) forecasted actual staffing under the freeze plan."
)

# ------------------------------------------------------------
# ✅ Controls (executive-facing)
# ------------------------------------------------------------
notice_days = st.number_input(
    "Provider Resignation Notice Period (days)",
    min_value=0,
    max_value=180,
    value=75,
    step=5,
    help="Attrition begins only after providers exit following notice period.",
)

# ------------------------------------------------------------
# ✅ Colors (distinct + Sunshine Gold)
# ------------------------------------------------------------
COLOR_SIGNING = "#7a6200"         # Sunshine Gold
COLOR_CREDENTIALING = "#3b78c2"   # deep blue
COLOR_TRAINING = "#2e9b6a"        # green/teal
COLOR_FLU_SEASON = "#f4c542"      # warm flu highlight
COLOR_FREEZE = "#9c9c9c"          # freeze gray

# ------------------------------------------------------------
# ✅ Chart window: Calendar year Jan → Dec
# ------------------------------------------------------------
current_year = today.year
chart_start = datetime(current_year, 1, 1)
chart_end = datetime(current_year, 12, 31)

dates = pd.date_range(start=chart_start, end=chart_end, freq="MS")
month_labels = [d.strftime("%b") for d in dates]

# ------------------------------------------------------------
# ✅ Compute flu season dates (handles wrap across year)
# ------------------------------------------------------------
flu_start_date = datetime(current_year, flu_start_month, 1)

if flu_end_month < flu_start_month:
    flu_end_date = datetime(current_year + 1, flu_end_month, 1)
else:
    flu_end_date = datetime(current_year, flu_end_month, 1)

flu_end_date = flu_end_date + timedelta(days=32)
flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)

# ------------------------------------------------------------
# ✅ Seasonality multiplier (base curve)
# ------------------------------------------------------------
def base_seasonality_multiplier(month: int):
    if month in [12, 1, 2]:  # Winter
        return 1.20
    if month in [6, 7, 8]:  # Summer
        return 0.80
    return 1.00             # Spring/Fall

# ------------------------------------------------------------
# ✅ Predicted Visits/Day line (seasonality + manual flu uplift)
# ------------------------------------------------------------
forecast_visits_by_month = []
for d in dates:
    base = visits * base_seasonality_multiplier(d.month)

    # apply manual uplift only inside flu window
    if flu_start_date <= d <= flu_end_date:
        base = base * (1 + flu_uplift_pct)

    forecast_visits_by_month.append(base)

# normalize so annual average equals baseline visits/day
avg_forecast = np.mean(forecast_visits_by_month)
forecast_visits_by_month = [v * (visits / avg_forecast) for v in forecast_visits_by_month]

# ------------------------------------------------------------
# ✅ Convert predicted volume into Provider FTE demand
# ------------------------------------------------------------
provider_fte_demand = []
for v in forecast_visits_by_month:
    fte_month = model.calculate_fte_needed(
        visits_per_day=v,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    provider_fte_demand.append(fte_month["provider_fte"])

staffing_target = provider_fte_demand
peak_staffing = max(staffing_target)

# ------------------------------------------------------------
# ✅ Attrition parameters
# ------------------------------------------------------------
baseline_provider_fte = fte_result["provider_fte"]
provider_turnover_rate = provider_turnover  # annual
monthly_attrition_fte = baseline_provider_fte * (provider_turnover_rate / 12)

# ------------------------------------------------------------
# ✅ Provider pipeline timeline (anchor to flu_start_date)
# ------------------------------------------------------------
staffing_needed_by = flu_start_date
total_provider_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

req_post_date = staffing_needed_by - timedelta(days=total_provider_lead_days)
signed_date = req_post_date + timedelta(days=days_to_sign)
credentialed_date = signed_date + timedelta(days=days_to_credential)
solo_ready_date = credentialed_date + timedelta(days=onboard_train_days)

# ------------------------------------------------------------
# ✅ Auto freeze logic
# Freeze begins early enough so attrition can burn off peak by flu end
# ------------------------------------------------------------
overhang_fte = max(peak_staffing - baseline_provider_fte, 0)
months_to_burn_off = overhang_fte / max(monthly_attrition_fte, 0.01)

freeze_start_date = flu_end_date - timedelta(days=int(months_to_burn_off * 30.4))
freeze_start_date = max(freeze_start_date, chart_start)
freeze_end_date = flu_end_date

# ------------------------------------------------------------
# ✅ FIXED Attrition Projection (No Backfill Risk)
# - Starts at peak staffing
# - Stays flat until freeze + notice lag
# - Declines only AFTER providers exit
# ------------------------------------------------------------
effective_attrition_start = freeze_start_date + timedelta(days=int(notice_days))

attrition_line = []
for d in dates:
    if d < effective_attrition_start:
        attrition_line.append(peak_staffing)
    else:
        months_elapsed = (d.year - effective_attrition_start.year) * 12 + (d.month - effective_attrition_start.month)
        loss = months_elapsed * monthly_attrition_fte
        attrition_line.append(max(peak_staffing - loss, 0))

# ------------------------------------------------------------
# ✅ Forecasted Actual Staffing (Freeze Plan)
# - Before freeze: meets staffing target
# - After freeze + notice lag: declines gradually
# ------------------------------------------------------------
forecast_actual_staffing = []
for d, target in zip(dates, staffing_target):
    if d < effective_attrition_start:
        forecast_actual_staffing.append(target)
    else:
        months_elapsed = (d.year - effective_attrition_start.year) * 12 + (d.month - effective_attrition_start.month)
        loss = months_elapsed * monthly_attrition_fte
        forecast_actual_staffing.append(max(target - loss, 0))

# ============================================================
# ✅ Plot (Two Axes)
# ============================================================
fig, ax1 = plt.subplots(figsize=(12, 4))

# ---- Left axis: Provider FTE ----
ax1.plot(dates, staffing_target, linewidth=3, marker="o",
         label="Staffing Target (Demand Curve)")
ax1.plot(dates, attrition_line, linestyle="--", linewidth=2,
         label="Attrition Projection (No Backfill Risk)")
ax1.plot(dates, forecast_actual_staffing, linewidth=3,
         label="Forecasted Actual Staffing (Freeze Plan)")

ax1.set_title("Provider Seasonality + Hiring Glidepath (Executive Summary)")
ax1.set_ylabel("Provider FTE Needed")
ax1.set_ylim(0, max(staffing_target) + 1)

ax1.set_xticks(dates)
ax1.set_xticklabels(month_labels)
ax1.grid(axis="y", linestyle=":", alpha=0.35)

# ---- Right axis: Predicted Volume ----
ax2 = ax1.twinx()
ax2.plot(dates, forecast_visits_by_month, linestyle="-.", linewidth=2.5,
         label="Predicted Volume (Visits/Day)")
ax2.set_ylabel("Visits / Day")

# ------------------------------------------------------------
# ✅ Shaded Windows
# ------------------------------------------------------------
ax1.axvspan(req_post_date, signed_date, color=COLOR_SIGNING, alpha=0.22)
ax1.axvspan(signed_date, credentialed_date, color=COLOR_CREDENTIALING, alpha=0.18)
ax1.axvspan(credentialed_date, solo_ready_date, color=COLOR_TRAINING, alpha=0.18)

ax1.axvspan(flu_start_date, flu_end_date, color=COLOR_FLU_SEASON, alpha=0.16)
ax1.axvspan(freeze_start_date, freeze_end_date, color=COLOR_FREEZE, alpha=0.15)

# ------------------------------------------------------------
# ✅ Vertical Markers (Freeze Starts + Attrition Begins)
# ------------------------------------------------------------
ax1.axvline(freeze_start_date, linestyle=":", linewidth=2, alpha=0.9)
ax1.axvline(effective_attrition_start, linestyle="--", linewidth=2, alpha=0.9)

# ------------------------------------------------------------
# ✅ Marker Labels (top of chart)
# ------------------------------------------------------------
ymax = ax1.get_ylim()[1]

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

# ------------------------------------------------------------
# ✅ Legend UNDER the chart (lines only)
# ------------------------------------------------------------
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

plt.tight_layout()
st.pyplot(fig)

# ============================================================
# ✅ Executive Interpretation Box
# ============================================================
st.info(
    f"""
✅ **Executive Interpretation**
- **Predicted volume** fluctuates seasonally, with flu season uplift applied only during flu months.
- **Staffing target** rises and falls to match demand (demand-based staffing curve).
- If you implement a **hiring freeze**, you do *not* lose providers immediately.
  - Providers typically give **{notice_days} days’ notice**, protecting flu season coverage.
  - Staffing erosion begins only **after the notice period**, causing staffing to drift down into spring/summer.
- The **attrition projection line** shows the risk if you freeze hiring and do not backfill.
- The **forecasted actual staffing line** shows what staffing will look like under that freeze plan in reality.
"""
)
