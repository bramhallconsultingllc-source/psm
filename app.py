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

    # ============================================================
    # ✅ STEP 4: Staffing Summary + Productivity + Interpretation
    # ============================================================

    st.markdown("---")
    st.subheader("Staffing Summary + Interpretation")

    provider_day = result["provider_day"]
    psr_day = result["psr_day"]
    ma_day = result["ma_day"]
    xrt_day = result["xrt_day"]
    total_day = result["total_day"]

    # --- Productivity Snapshot ---
    patients_per_provider = visits / provider_day if provider_day > 0 else 0
    patients_per_ma = visits / ma_day if ma_day > 0 else 0
    visits_per_total_staff = visits / total_day if total_day > 0 else 0

    # --- Interpretation Rules (simple + practical) ---
    interpretation = []

    # Staffing conservatism note
    interpretation.append(
        "Staffing outputs are intentionally conservative (rounded UP) to reduce under-coverage risk."
    )

    # Provider workload interpretation
    if patients_per_provider >= 30:
        interpretation.append(
            "Provider workload is relatively high. Monitor wait times, documentation lag, and end-of-day spillover."
        )
    elif 22 <= patients_per_provider < 30:
        interpretation.append(
            "Provider workload is within a typical efficient range. Maintain good flow standards to protect throughput."
        )
    else:
        interpretation.append(
            "Provider workload appears low. Confirm demand is real and avoid overstaffing during slow sessions."
        )

    # MA workload interpretation
    if patients_per_ma >= 22:
        interpretation.append(
            "MA workload is relatively high. If flow slows, MA coverage is usually the first constraint."
        )
    elif 16 <= patients_per_ma < 22:
        interpretation.append(
            "MA workload is balanced. Keep role clarity tight to prevent drift into inefficiency."
        )
    else:
        interpretation.append(
            "MA workload appears low. If labor costs are rising, this is a likely area to optimize."
        )

    # Total staff efficiency
    if visits_per_total_staff >= 10:
        interpretation.append(
            "Total staffing is lean. Protect reliability with strong shift handoffs and clear standards."
        )
    elif 7 <= visits_per_total_staff < 10:
        interpretation.append(
            "Total staffing is balanced. This is a stable operating posture for most clinics."
        )
    else:
        interpretation.append(
            "Total staffing is heavier than typical. Confirm visit complexity or workflow friction before accepting this as the norm."
        )

    # --- Layout in 3 columns ---
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Daily Staffing Target")
        st.metric("Provider / Day", provider_day)
        st.metric("PSR / Day", psr_day)
        st.metric("MA / Day", ma_day)
        st.metric("XRT / Day", xrt_day)
        st.metric("TOTAL / Day", total_day)

    with c2:
        st.markdown("### Productivity Snapshot")
        st.metric("Patients per Provider", f"{patients_per_provider:.1f}")
        st.metric("Patients per MA", f"{patients_per_ma:.1f}")
        st.metric("Visits per Total Staff", f"{visits_per_total_staff:.1f}")

    with c3:
        st.markdown("### Interpretation")
        for line in interpretation:
            st.markdown(f"- {line}")

