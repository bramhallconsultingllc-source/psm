import streamlit as st
import pandas as pd
from psm.staffing_model import StaffingModel

st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")

st.title("Predictive Staffing Model (PSM)")
st.caption("A simple staffing calculator using linear interpolation + conservative rounding rules.")

st.info("⚠️ **All staffing outputs round UP to the nearest 0.25 FTE/day.** This is intentional to prevent under-staffing.")

visits = st.number_input("Average Visits per Day", min_value=1.0, value=45.0, step=1.0)

model = StaffingModel()

if st.button("Calculate Staffing"):
    result = model.calculate(visits)

    st.subheader("Staffing Output (FTE / Day)")

    df = pd.DataFrame(
        {
            "Role": ["Provider", "PSR", "MA", "XRT", "TOTAL"],
            "FTE/Day": [
                result["provider_day"],
                result["psr_day"],
                result["ma_day"],
                result["xrt_day"],
                result["total_day"],
            ],
        }
    )

    st.dataframe(df, hide_index=True, use_container_width=True)

