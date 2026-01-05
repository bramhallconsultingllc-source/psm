import streamlit as st
import pandas as pd
from psm.staffing_model import StaffingModel

st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")

st.title("Predictive Staffing Model (PSM)")
st.caption("A simple staffing calculator using linear interpolation + conservative rounding rules.")

st.info(
    "⚠️ **All daily staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)

# -------------------------
# Inputs
# -------------------------

visits = st.number_input(
    "Average Visits per Day",
    min_value=1.0,
    value=45.0,
    step=1.0,
)

st.markdown("### Weekly Inputs (for FTE conversion)")

hours_of_operation = st.number_input(
    "Hours of Operation per Week",
    min_value=1.0,
    value=70.0,
    step=1.0,
)

fte_hours_per_week = st.number_input(
    "FTE Hours per Week (default 40)",
    min_value=1.0,
    value=40.0,
    step=1.0,
)

model = StaffingModel()

# -------------------------
# Calculate
# -------------------------

if st.button("Calculate Staffing"):

    # Daily staffing (rounded up)
    daily_result = model.calculate(visits)

    st.subheader("Staffing Output (FTE / Day) — Rounded Up")

    daily_df = pd.DataFrame(
        {
            "Role": ["Provider", "PSR", "MA", "XRT", "TOTAL"],
            "FTE/Day": [
                daily_result["provider_day"],
                daily_result["psr_day"],
                daily_result["ma_day"],
                daily_result["xrt_day"],
                daily_result["total_day"],
            ],
        }
    )

    st.dataframe(daily_df, hide_index=True, use_container_width=True)

    # ✅ Weekly FTE conversion (exact)
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )

    st.subheader("Full-Time Employees (FTEs) Needed")

    st.caption(
        "✅ FTE is calculated exactly (not rounded). "
        "Daily staffing is still conservatively rounded UP."
    )

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

    # Optional: round display to 2 decimals (ONLY for display)
    fte_df["FTE Needed"] = fte_df["FTE Needed"].round(2)

    st.dataframe(fte_df, hide_index=True, use_container_width=True)
