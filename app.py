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
# ✅ Provider baseline
# ------------------------------------------------------------
baseline_provider_fte = fte_result["provider_fte"]
provider_turnover_rate = provider_turnover  # annual %

# ------------------------------------------------------------
# ✅ Resignation notice lag
# ------------------------------------------------------------
notice_days = st.number_input(
    "Provider Resignation Notice Period (days)",
    min_value=0,
    max_value=180,
    value=75,
    step=5,
    help="Attrition only impacts staffing after this many days post-freeze.",
)

# ------------------------------------------------------------
# ✅ Colors (distinct + Sunshine Gold)
# ------------------------------------------------------------
COLOR_SIGNING = "#7a6200"         # Sunshine Gold
COLOR_CREDENTIALING = "#3b78c2"   # deep blue
COLOR_TRAINING = "#2e9b6a"        # teal green
COLOR_FLU_SEASON = "#f4c542"      # warm flu highlight
COLOR_FREEZE = "#9c9c9c"          # freeze gray

# ------------------------------------------------------------
# ✅ Calendar year view: Jan → Dec
# ------------------------------------------------------------
current_year = today.year
chart_start = datetime(current_year, 1, 1)
chart_end = datetime(current_year, 12, 31)

dates = pd.date_range(start=chart_start, end=chart_end, freq="MS")
month_labels = [d.strftime("%b") for d in dates]

# ------------------------------------------------------------
# ✅ Flu season dates (handles wrap into next year)
# ------------------------------------------------------------
flu_start_date = datetime(current_year, flu_start_month, 1)

if flu_end_month < flu_start_month:
    flu_end_date = datetime(current_year + 1, flu_end_month, 1)
else:
    flu_end_date = datetime(current_year, flu_end_month, 1)

flu_end_date = flu_end_date + timedelta(days=32)
flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)

# ------------------------------------------------------------
# ✅ Seasonality multiplier
# ------------------------------------------------------------
def base_seasonality_multiplier(month: int):
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00

# ------------------------------------------------------------
# ✅ Predicted Volume Line (Seasonality + Flu Uplift inside flu window only)
# ------------------------------------------------------------
forecast_visits_by_month = []

for d in dates:
    m = base_seasonality_multiplier(d.month)

    # apply flu uplift only during flu season window
    if flu_start_date <= d <= flu_end_date:
        m = 1 + flu_uplift_pct

    forecast_visits_by_month.append(visits * m)

# normalize so annual avg stays equal to baseline visits/day
avg_forecast = np.mean(forecast_visits_by_month)
forecast_visits_by_month = [v * (visits / avg_forecast) for v in forecast_visits_by_month]

# ------------------------------------------------------------
# ✅ Convert forecast visits → Provider FTE demand (staffing target)
# ------------------------------------------------------------
staffing_target = []

for v in forecast_visits_by_month:
    fte_month = model.calculate_fte_needed(
        visits_per_day=v,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    staffing_target.append(fte_month["provider_fte"])

peak_staffing = max(staffing_target)

# ------------------------------------------------------------
# ✅ Provider pipeline timeline (anchor to flu_start)
# ------------------------------------------------------------
staffing_needed_by = flu_start_date
total_provider_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

req_post_date = staffing_needed_by - timedelta(days=total_provider_lead_days)
signed_date = req_post_date + timedelta(days=days_to_sign)
credentialed_date = signed_date + timedelta(days=days_to_credential)
solo_ready_date = credentialed_date + timedelta(days=onboard_train_days)

# ------------------------------------------------------------
# ✅ Auto Freeze Logic
# Freeze begins early enough that turnover drifts staffing back down by flu_end
# ------------------------------------------------------------
monthly_attrition_from_peak = peak_staffing * (provider_turnover_rate / 12)

overhang_fte = max(peak_staffing - baseline_provider_fte, 0)
months_to_burn_off = overhang_fte / max(monthly_attrition_from_peak, 0.01)

freeze_start_date = flu_end_date - timedelta(days=int(months_to_burn_off * 30.4))
freeze_start_date = max(freeze_start_date, chart_start)
freeze_end_date = flu_end_date

# ------------------------------------------------------------
# ✅ Effective Freeze Start (includes notice lag)
# ------------------------------------------------------------
effective_freeze_start = freeze_start_date + timedelta(days=int(notice_days))

# ------------------------------------------------------------
# ✅ Attrition Risk Line (Option B)
# Flat at peak until freeze+notice, then slopes down
# ------------------------------------------------------------
attrition_line = []

for d in dates:
    if d < effective_freeze_start:
        attrition_line.append(peak_staffing)
    else:
        months_elapsed = (d.year - effective_freeze_start.year) * 12 + (d.month - effective_freeze_start.month)
        loss = months_elapsed * monthly_attrition_from_peak
        attrition_line.append(max(peak_staffing - loss, 0))

# ------------------------------------------------------------
# ✅ Forecast Actual Staffing Line (Freeze + Notice + Attrition)
# Matches target until effective_freeze_start, then erodes
# ------------------------------------------------------------
forecast_actual_staffing = []

for d, target in zip(dates, staffing_target):
    if d < effective_freeze_start:
        forecast_actual_staffing.append(target)
    else:
        months_elapsed = (d.year - effective_freeze_start.year) * 12 + (d.month - effective_freeze_start.month)
        loss = months_elapsed * monthly_attrition_from_peak
        forecast_actual_staffing.append(max(target - loss, 0))

# ============================================================
# ✅ Plot (Two Axes)
# ============================================================
fig, ax1 = plt.subplots(figsize=(12, 4))

# ---- Left axis: Provider staffing ----
l1, = ax1.plot(dates, staffing_target, linewidth=3, marker="o",
               label="Staffing Target (Demand Curve)")
l2, = ax1.plot(dates, attrition_line, linestyle="--", linewidth=2,
               label="Attrition Projection (No Backfill Risk)")
l3, = ax1.plot(dates, forecast_actual_staffing, linewidth=3,
               label="Forecasted Actual Staffing (Freeze Plan)")

ax1.set_ylabel("Provider FTE Needed")
ax1.set_ylim(0, max(staffing_target) + 1)
ax1.set_xticks(dates)
ax1.set_xticklabels(month_labels)
ax1.grid(axis="y", linestyle=":", alpha=0.35)

# ---- Right axis: Volume ----
ax2 = ax1.twinx()
l4, = ax2.plot(dates, forecast_visits_by_month, linestyle="-.", linewidth=2.5,
               label="Predicted Volume (Visits/Day)")
ax2.set_ylabel("Visits / Day")

# ------------------------------------------------------------
# ✅ Shading blocks (no legend entries)
# ------------------------------------------------------------
ax1.axvspan(req_post_date, signed_date, color=COLOR_SIGNING, alpha=0.25)
ax1.axvspan(signed_date, credentialed_date, color=COLOR_CREDENTIALING, alpha=0.20)
ax1.axvspan(credentialed_date, solo_ready_date, color=COLOR_TRAINING, alpha=0.20)

ax1.axvspan(flu_start_date, flu_end_date, color=COLOR_FLU_SEASON, alpha=0.18)
ax1.axvspan(freeze_start_date, freeze_end_date, color=COLOR_FREEZE, alpha=0.18)

# ------------------------------------------------------------
# ✅ Legend UNDER chart (lines only)
# ------------------------------------------------------------
lines = [l1, l2, l3, l4]
labels = [line.get_label() for line in lines]

ax1.legend(
    lines, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=2,
    frameon=False
)

plt.tight_layout()
st.pyplot(fig)

# ============================================================
# ✅ Block Key
# ============================================================
st.markdown("### Block Key (Shaded Windows)")
st.markdown(
    f"""
    <div style="font-size: 14px; line-height: 1.8;">
        <span style="background-color:{COLOR_SIGNING}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
        &nbsp; Req Posted → Signed Offer (Signing Window)<br>

        <span style="background-color:{COLOR_CREDENTIALING}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
        &nbsp; Signed → Credentialed (Credentialing Window)<br>

        <span style="background-color:{COLOR_TRAINING}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
        &nbsp; Credentialed → Solo Ready (Training / Onboarding Window)<br>

        <span style="background-color:{COLOR_FLU_SEASON}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
        &nbsp; Flu Season (Peak Demand Window)<br>

        <span style="background-color:{COLOR_FREEZE}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
        &nbsp; Hiring Freeze (Attrition Allowed)
    </div>
    """,
    unsafe_allow_html=True,
)
