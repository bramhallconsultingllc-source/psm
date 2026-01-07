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
# ✅ Role-Specific Hiring Assumptions
# ============================================================
st.markdown("## Role-Specific Hiring Assumptions")

c1, c2 = st.columns(2)

with c1:
    provider_tth = st.number_input("Provider — Avg Time to Hire (days)", value=120, step=5)
    psr_tth = st.number_input("PSR — Avg Time to Hire (days)", value=45, step=5)
    ma_tth = st.number_input("MA — Avg Time to Hire (days)", value=60, step=5)
    xrt_tth = st.number_input("XRT — Avg Time to Hire (days)", value=60, step=5)

with c2:
    provider_ramp = st.number_input("Provider — Training/Ramp Days", value=14, step=1)
    psr_ramp = st.number_input("PSR — Training/Ramp Days", value=7, step=1)
    ma_ramp = st.number_input("MA — Training/Ramp Days", value=10, step=1)
    xrt_ramp = st.number_input("XRT — Training/Ramp Days", value=10, step=1)


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
        help="Manually adjust the anticipated flu season volume increase (applies only inside flu season window)."
    ) / 100


# ============================================================
# ✅ Provider Pipeline Inputs
# ============================================================
st.markdown("## Provider Hiring Glidepath Inputs")

with st.expander("Provider Pipeline Assumptions", expanded=False):

    days_to_sign = st.number_input("Days to Sign (Req → Signed Offer)", min_value=1, value=90, step=5)
    days_to_credential = st.number_input("Days to Credential (Signed → Credentialed)", min_value=1, value=90, step=5)
    onboard_train_days = st.number_input("Onboard/Train Days (Credentialed → Solo)", min_value=0, value=30, step=5)

    coverage_buffer_days = st.number_input(
        "Buffer Days (Planning Margin)",
        min_value=0,
        value=14,
        step=1
    )

    utilization_factor = st.number_input(
        "Hiring Effectiveness Factor",
        min_value=0.10,
        max_value=1.00,
        value=0.90,
        step=0.05
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
# ✅ STEP A2: Seasonality Forecast (Visits/Day by Month)
# ============================================================
st.markdown("---")
st.subheader("Seasonality Forecast (Month-by-Month Projection)")
st.caption("Baseline visits/day is treated as your annual average. Seasonality redistributes volume across the year.")

# -------------------------
# Define flu season dates
# -------------------------
current_year = today.year
flu_start_date = datetime(current_year, flu_start_month, 1)

if flu_end_month < flu_start_month:
    flu_end_date = datetime(current_year + 1, flu_end_month, 1)
else:
    flu_end_date = datetime(current_year, flu_end_month, 1)

flu_end_date = flu_end_date + timedelta(days=32)
flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)

# -------------------------
# Build 12-month projection timeline
# -------------------------
chart_start = today
chart_end = today + timedelta(days=365)

dates = pd.date_range(start=chart_start, end=chart_end, freq="MS")
month_labels = [d.strftime("%b") for d in dates]


# -------------------------
# Seasonality base multipliers
# -------------------------
def base_seasonality_multiplier(month: int):
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00


# -------------------------
# Apply flu uplift only inside flu season window
# -------------------------
multipliers = []
for d in dates:
    m = base_seasonality_multiplier(d.month)

    if flu_start_date <= d <= flu_end_date:
        m = 1 + flu_uplift_pct  # apply manual uplift only inside flu window

    multipliers.append(m)

avg_multiplier = np.mean(multipliers)
normalized_multipliers = [m / avg_multiplier for m in multipliers]

forecast_visits_by_month = [visits * m for m in normalized_multipliers]

forecast_df = pd.DataFrame({
    "Month": month_labels,
    "Seasonality Multiplier": np.round(normalized_multipliers, 2),
    "Forecast Visits/Day": np.round(forecast_visits_by_month, 1)
})

st.dataframe(forecast_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ STEP A3: Translate Forecast Visits → Provider FTE Needed
# ============================================================
provider_fte_by_month = []
for v in forecast_visits_by_month:
    fte_month = model.calculate_fte_needed(
        visits_per_day=v,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    provider_fte_by_month.append(fte_month["provider_fte"])


# ============================================================
# ✅ STEP A6: Provider Seasonality + Hiring Glidepath (Executive View)
# ============================================================
st.markdown("---")
st.subheader("Provider Seasonality + Hiring Glidepath (Executive View)")

st.caption(
    "This chart shows: "
    "(1) provider staffing target based on demand seasonality, "
    "(2) attrition risk if you freeze hiring, and "
    "(3) forecasted actual staffing if you freeze and allow attrition to erode staffing."
)

# -------------------------
# Colors
# -------------------------
COLOR_SIGNING = "#7a6200"         # Sunshine Gold
COLOR_CREDENTIALING = "#3b78c2"
COLOR_TRAINING = "#2e9b6a"
COLOR_FLU_SEASON = "#f4c542"
COLOR_FREEZE = "#9c9c9c"


# -------------------------
# Provider pipeline timeline (anchor to flu_start)
# -------------------------
staffing_needed_by = flu_start_date
total_provider_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

req_post_date = staffing_needed_by - timedelta(days=total_provider_lead_days)
signed_date = req_post_date + timedelta(days=days_to_sign)
credentialed_date = signed_date + timedelta(days=days_to_credential)
solo_ready_date = credentialed_date + timedelta(days=onboard_train_days)

# -------------------------
# Staffing Target = provider FTE need by month
# -------------------------
staffing_target = provider_fte_by_month

# -------------------------
# Attrition Projection (No Backfill Risk)
# -------------------------
monthly_attrition_fte = baseline_provider_fte * (provider_turnover / 12)

attrition_line = []
for d in dates:
    months_elapsed = (d.year - chart_start.year) * 12 + (d.month - chart_start.month)
    attrition_loss = months_elapsed * monthly_attrition_fte
    attrition_line.append(max(baseline_provider_fte - attrition_loss, 0))

# -------------------------
# ✅ Auto Freeze Logic (Provider Only)
# Freeze starts early enough so attrition naturally returns staffing to baseline by flu_end
# -------------------------

peak_provider_fte = max(staffing_target)

# Overhang above baseline at peak
overhang_fte = max(peak_provider_fte - baseline_provider_fte, 0)

# Monthly attrition burn rate (FTE/month)
monthly_attrition_fte = baseline_provider_fte * (provider_turnover / 12)

# Months required to drift back to baseline
months_to_burn_off = overhang_fte / max(monthly_attrition_fte, 0.01)

# Freeze starts this many months before flu end
freeze_start_date = flu_end_date - timedelta(days=int(months_to_burn_off * 30.4))

# Clip freeze start so it can't start before today
freeze_start_date = max(freeze_start_date, today)

freeze_end_date = flu_end_date

# -------------------------
# ✅ Forecasted Actual Staffing (with Auto Freeze)
# -------------------------

forecast_actual = []
current_staffing = staffing_target[0]

# Monthly attrition burn rate (FTE/month)
monthly_attrition_fte = baseline_provider_fte * (provider_turnover / 12)

for i, d in enumerate(dates):

    target = staffing_target[i]

    # Before freeze: staffing tracks the target
    if d < freeze_start_date:
        current_staffing = target

    # During freeze: staffing erodes due to attrition (no backfill)
    else:
        current_staffing = max(current_staffing - monthly_attrition_fte, 0)

    forecast_actual.append(current_staffing)

# ============================================================
# ✅ Plot
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(dates, staffing_target, linewidth=3, marker="o",
        label="Staffing Target (Seasonality Curve)")
ax.plot(dates, attrition_line, linestyle="--", linewidth=2,
        label="Attrition Projection (No Backfill Risk)")
ax.plot(dates, forecast_actual, linewidth=3,
        label="Forecasted Actual Staffing (Freeze Plan)")

# Shaded windows
ax.axvspan(req_post_date, signed_date, color=COLOR_SIGNING, alpha=0.22)
ax.axvspan(signed_date, credentialed_date, color=COLOR_CREDENTIALING, alpha=0.18)
ax.axvspan(credentialed_date, solo_ready_date, color=COLOR_TRAINING, alpha=0.18)

ax.axvspan(flu_start_date, flu_end_date, color=COLOR_FLU_SEASON, alpha=0.15)
ax.axvspan(freeze_start_date, freeze_end_date, color=COLOR_FREEZE, alpha=0.15)

# Formatting
ax.set_title("Provider Seasonality Curve + Hiring Glidepath (Executive Summary)")
ax.set_ylabel("Provider FTE Need")

ax.set_ylim(0, max(staffing_target) + 1)
ax.set_xlim(chart_start, chart_end)

ax.set_xticks(dates)
ax.set_xticklabels(month_labels)

ax.grid(axis="y", linestyle=":", alpha=0.35)

# Lines-only legend outside
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))

plt.tight_layout()
st.pyplot(fig)


# ============================================================
# ✅ Block Key (Shaded Windows)
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
        &nbsp; Flu Season (Peak Coverage Window)<br>

        <span style="background-color:{COLOR_FREEZE}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
        &nbsp; Hiring Freeze (No Backfill)
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ✅ Timeline Summary
# ============================================================
st.markdown("---")
st.subheader("Provider Timeline Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Req Posted By", req_post_date.strftime("%b %d, %Y"))
    st.metric("Signed By", signed_date.strftime("%b %d, %Y"))

with c2:
    st.metric("Credentialed By", credentialed_date.strftime("%b %d, %Y"))
    st.metric("Solo Ready By", solo_ready_date.strftime("%b %d, %Y"))

with c3:
    st.metric("Flu Starts", flu_start_date.strftime("%b %d, %Y"))
    st.metric("Freeze Starts", freeze_start_date.strftime("%b %d, %Y"))
    st.metric("Flu Ends", flu_end_date.strftime("%b %d, %Y"))


st.info(
    """
✅ **Executive Interpretation**
- Baseline staffing represents the annual average (100%).
- Staffing rises above baseline in winter due to flu season demand.
- Staffing can fall below baseline in summer because:
  1) census is lower,
  2) vacations increase,
  3) hiring pauses and attrition reduces staffing naturally.
- The staffing target shows what demand requires.
- Forecasted actual staffing shows what happens if you freeze hiring and allow attrition to erode staffing.
"""
)
