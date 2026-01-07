import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ✅ Stable "today" for consistent chart windows across reruns
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]


# -------------------------
# Session State Init
# -------------------------
if "runs" not in st.session_state:
    st.session_state["runs"] = []


# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")

st.title("Predictive Staffing Model (PSM)")
st.caption("A simple staffing calculator using linear interpolation + conservative rounding rules.")

st.info(
    "⚠️ **All daily staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)


# -------------------------
# Inputs ✅ NOT INDENTED
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
# Role-Specific Hiring Assumptions
# -------------------------

st.markdown("### Role-Specific Hiring Assumptions")

st.caption(
    "Different roles require different recruiting lead time and training ramp. "
    "These values drive the role-specific glidepath timeline."
)

c1, c2 = st.columns(2)

with c1:
    provider_tth = st.number_input("Provider — Average Time to Hire (days)", value=120, step=5)
    psr_tth = st.number_input("PSR — Average Time to Hire (days)", value=45, step=5)
    ma_tth = st.number_input("MA — Average Time to Hire (days)", value=60, step=5)
    xrt_tth = st.number_input("XRT — Average Time to Hire (days)", value=60, step=5)

with c2:
    provider_ramp = st.number_input("Provider — Training/Ramp Days", value=14, step=1)
    psr_ramp = st.number_input("PSR — Training/Ramp Days", value=7, step=1)
    ma_ramp = st.number_input("MA — Training/Ramp Days", value=10, step=1)
    xrt_ramp = st.number_input("XRT — Training/Ramp Days", value=10, step=1)

st.markdown("### Flu Season Settings")

flu_c1, flu_c2 = st.columns(2)

with flu_c1:
    flu_start_month = st.selectbox(
        "Flu Season Start Month",
        options=list(range(1, 13)),
        index=10,  # Default Nov
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )

with flu_c2:
    flu_end_month = st.selectbox(
        "Flu Season End Month",
        options=list(range(1, 13)),
        index=1,  # Default Feb
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )

# -------------------------
# Turnover Assumptions
# -------------------------

st.markdown("### Role-Specific Turnover Assumptions")

st.caption(
    "Turnover % represents expected annual attrition for each role. "
    "We use this to build a planning buffer in your hiring target."
)

planning_months = st.number_input(
    "Planning Horizon (months)",
    min_value=1,
    value=12,
    step=1,
    help="How far forward you want to account for turnover risk.",
)

t1, t2 = st.columns(2)

with t1:
    provider_turnover = st.number_input("Provider Turnover %", value=24.0, step=1.0) / 100
    psr_turnover = st.number_input("PSR Turnover %", value=30.0, step=1.0) / 100

with t2:
    ma_turnover = st.number_input("MA Turnover %", value=40.0, step=1.0) / 100
    xrt_turnover = st.number_input("XRT Turnover %", value=20.0, step=1.0) / 100


st.info("ℹ️ Enter turnover assumptions above, then click **Calculate Staffing** to generate turnover buffers.")

# -------------------------
# Provider Seasonality Inputs (GLOBAL)
# -------------------------

st.markdown("### Flu Season Settings")

flu_c1, flu_c2 = st.columns(2)

with flu_c1:
    flu_start_month = st.selectbox(
        "Flu Season Start Month",
        options=list(range(1, 13)),
        index=11,  # Default Dec
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
        key="flu_start_month_global"
    )

with flu_c2:
    flu_end_month = st.selectbox(
        "Flu Season End Month",
        options=list(range(1, 13)),
        index=1,  # Default Feb
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
        key="flu_end_month_global"
    )


st.markdown("### Provider Hiring Glidepath Inputs")

with st.expander("Provider Hiring Glidepath Assumptions", expanded=False):

    days_to_sign = st.number_input(
        "Days to Sign (Req Posted → Signed Offer)",
        min_value=1,
        value=90,
        step=5,
        key="days_to_sign_global"
    )

    days_to_credential = st.number_input(
        "Days to Credential (Signed → Fully Credentialed)",
        min_value=1,
        value=90,
        step=5,
        key="days_to_credential_global"
    )

    onboard_train_days = st.number_input(
        "Onboard / Train Days (Credentialed → Solo Ready)",
        min_value=0,
        value=30,
        step=5,
        key="onboard_train_days_global"
    )

    coverage_buffer_days = st.number_input(
        "Buffer Days (Planning Margin)",
        min_value=0,
        value=14,
        step=1,
        help="Conservative buffer so recruiting starts early.",
        key="coverage_buffer_days_global"
    )

    utilization_factor = st.number_input(
        "Hiring Effectiveness Factor",
        min_value=0.10,
        max_value=1.00,
        value=0.90,
        step=0.05,
        help="Accounts for onboarding inefficiency, vacancies, call-outs, and imperfect schedules.",
        key="utilization_factor_global"
    )

# -------------------------
# Calculate ✅ Button
# -------------------------

if st.button("Calculate Staffing"):

    st.session_state["calculated"] = True

    if "today" not in st.session_state:
        st.session_state["today"] = datetime.today()

    today = st.session_state["today"]

    daily_result = model.calculate(visits)

    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )

    # ✅ Save to session_state
    st.session_state["daily_result"] = daily_result
    st.session_state["fte_result"] = fte_result

    # ✅ Build df and save it too
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
# ✅ STEP 4: Staffing Summary + Productivity + Interpretation
# ============================================================

# ✅ This must be OUTSIDE the button, at the root indentation level
if st.session_state.get("calculated"):

    today = st.session_state["today"]
    daily_result = st.session_state["daily_result"]
    fte_result = st.session_state["fte_result"]
    fte_df = st.session_state["fte_df"]

    # ✅ display anytime after calculation
    st.subheader("Full-Time Employees (FTEs) Needed")
    st.dataframe(fte_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Staffing Summary + Interpretation")

    provider_day = daily_result["provider_day"]
    psr_day = daily_result["psr_day"]
    ma_day = daily_result["ma_day"]
    xrt_day = daily_result["xrt_day"]
    total_day = daily_result["total_day"]

    patients_per_provider = visits / provider_day if provider_day > 0 else 0
    patients_per_ma = visits / ma_day if ma_day > 0 else 0
    visits_per_total_staff = visits / total_day if total_day > 0 else 0

    interpretation = []
    interpretation.append(
        "Staffing outputs are intentionally conservative (rounded UP) to reduce under-coverage risk."
    )

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

    # ============================================================
    # ✅ STEP A2: Forecast & Scenario Planning
    # ============================================================

    st.markdown("---")
    st.subheader("Forecast Scenario Planning (Predictive View)")

    st.caption(
        "Use this section to model future volume scenarios and see how staffing needs may change. "
        "Baseline staffing comes from your current visits/day input."
    )

    # -------------------------
    # Scenario Controls
    # -------------------------

    mode = st.radio(
        "Forecast Method",
        ["Percent Change", "Visit Change (+/-)"],
        horizontal=True,
    )

    col_a, col_b = st.columns(2)

    if mode == "Percent Change":
        pct_change = col_a.number_input(
            "Forecast Volume Change (%)",
            value=10.0,
            step=1.0,
            format="%.1f",
        )
        forecast_visits = visits * (1 + pct_change / 100)

    else:
        visit_change = col_a.number_input(
            "Forecast Visit Change (+/- visits/day)",
            value=10.0,
            step=1.0,
            format="%.0f",
        )
        forecast_visits = visits + visit_change

    forecast_visits = max(forecast_visits, 1.0)

    col_b.metric("Forecast Visits / Day", f"{forecast_visits:.1f}")

    st.info(
        "✅ Forecast staffing is calculated using the same rules: "
        "**linear interpolation + daily staffing rounded UP to 0.25 increments.**"
    )

    # -------------------------
    # Forecast Staffing
    # -------------------------

    forecast_daily = model.calculate(forecast_visits)

    forecast_fte = model.calculate_fte_needed(
        visits_per_day=forecast_visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )

    # -------------------------
    # Build Comparison Tables
    # -------------------------

    baseline_daily = daily_result
    baseline_fte = fte_result

    compare_daily_df = pd.DataFrame(
        {
            "Role": ["Provider", "PSR", "MA", "XRT", "TOTAL"],
            "Baseline (FTE/Day)": [
                baseline_daily["provider_day"],
                baseline_daily["psr_day"],
                baseline_daily["ma_day"],
                baseline_daily["xrt_day"],
                baseline_daily["total_day"],
            ],
            "Forecast (FTE/Day)": [
                forecast_daily["provider_day"],
                forecast_daily["psr_day"],
                forecast_daily["ma_day"],
                forecast_daily["xrt_day"],
                forecast_daily["total_day"],
            ],
        }
    )

    compare_daily_df["Delta (FTE/Day)"] = (
        compare_daily_df["Forecast (FTE/Day)"] - compare_daily_df["Baseline (FTE/Day)"]
    )

    compare_fte_df = pd.DataFrame(
        {
            "Role": ["Provider", "PSR", "MA", "XRT", "TOTAL"],
            "Baseline (FTE Need)": [
                baseline_fte["provider_fte"],
                baseline_fte["psr_fte"],
                baseline_fte["ma_fte"],
                baseline_fte["xrt_fte"],
                baseline_fte["total_fte"],
            ],
            "Forecast (FTE Need)": [
                forecast_fte["provider_fte"],
                forecast_fte["psr_fte"],
                forecast_fte["ma_fte"],
                forecast_fte["xrt_fte"],
                forecast_fte["total_fte"],
            ],
        }
    )

    compare_fte_df["Delta (FTE Need)"] = (
        compare_fte_df["Forecast (FTE Need)"] - compare_fte_df["Baseline (FTE Need)"]
    )

    # Round display only
    compare_fte_df.iloc[:, 1:] = compare_fte_df.iloc[:, 1:].round(2)

    # -------------------------
    # Display Results
    # -------------------------

    st.markdown("### Staffing Change (Daily Output)")

    st.dataframe(compare_daily_df, hide_index=True, use_container_width=True)

    st.markdown("### Staffing Change (FTE Need)")

    st.dataframe(compare_fte_df, hide_index=True, use_container_width=True)

    # -------------------------
    # Simple Summary + Key Callouts
    # -------------------------

    st.markdown("### Forecast Summary")

    delta_total_daily = forecast_daily["total_day"] - baseline_daily["total_day"]
    delta_total_fte = forecast_fte["total_fte"] - baseline_fte["total_fte"]

    st.metric("Daily Staffing Change (TOTAL)", f"{delta_total_daily:+.2f} FTE/day")
    st.metric("FTE Need Change (TOTAL)", f"{delta_total_fte:+.2f} FTE")

    st.caption(
        "⚠️ **Daily staffing is always rounded UP**, which means forecast deltas may be conservative "
        "(they may show staffing need increasing sooner than expected — this is intentional)."
    )

    # ============================================================
    # ✅ STEP A3: Visuals (Baseline vs Forecast + Deltas)
    # ============================================================

    import matplotlib.pyplot as plt

    st.markdown("---")
    st.subheader("Visual Summary (Baseline vs Forecast)")

    st.caption(
        "These visuals make it easier to see staffing impact quickly. "
        "Daily staffing is rounded UP; FTE Need is calculated exactly."
    )

    # -------------------------
    # Helper function for charts
    # -------------------------

    def plot_baseline_forecast(df, baseline_col, forecast_col, title, y_label):
        roles = df["Role"].tolist()
        baseline_vals = df[baseline_col].tolist()
        forecast_vals = df[forecast_col].tolist()

        x = range(len(roles))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 3))

        ax.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Baseline")
        ax.bar([i + width / 2 for i in x], forecast_vals, width=width, label="Forecast")

        ax.set_xticks(x)
        ax.set_xticklabels(roles)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(frameon=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig)

    def plot_delta(df, delta_col, title, y_label):
        roles = df["Role"].tolist()
        deltas = df[delta_col].tolist()

        fig, ax = plt.subplots(figsize=(8, 2.8))

        ax.bar(roles, deltas)

        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.7)

        ax.set_ylabel(y_label)
        ax.set_title(title)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig)

    # -------------------------
    # Chart 1: Daily Staffing Comparison
    # -------------------------

    plot_baseline_forecast(
        compare_daily_df,
        baseline_col="Baseline (FTE/Day)",
        forecast_col="Forecast (FTE/Day)",
        title="Daily Staffing Targets (FTE/Day) — Baseline vs Forecast",
        y_label="FTE per Day",
    )

    # -------------------------
    # Chart 2: Daily Staffing Delta
    # -------------------------

    plot_delta(
        compare_daily_df,
        delta_col="Delta (FTE/Day)",
        title="Daily Staffing Change (Delta) — Forecast minus Baseline",
        y_label="Δ FTE per Day",
    )

    # -------------------------
    # Chart 3: FTE Need Comparison
    # -------------------------

    plot_baseline_forecast(
        compare_fte_df,
        baseline_col="Baseline (FTE Need)",
        forecast_col="Forecast (FTE Need)",
        title="FTE Need — Baseline vs Forecast",
        y_label="FTE Needed",
    )

    # -------------------------
    # Chart 4: FTE Need Delta
    # -------------------------

    plot_delta(
        compare_fte_df,
        delta_col="Delta (FTE Need)",
        title="FTE Need Change (Delta) — Forecast minus Baseline",
        y_label="Δ FTE Needed",
    )

    # -------------------------
    # Save Run Button
    # -------------------------

    if st.button("💾 Save This Run"):
        st.session_state["runs"].append(
            {
                "Run Name": run_name,
                "Baseline Visits/Day": visits,
                "Forecast Visits/Day": forecast_visits,

                # Daily staffing (rounded)
                "Baseline Provider/Day": baseline_daily["provider_day"],
                "Baseline PSR/Day": baseline_daily["psr_day"],
                "Baseline MA/Day": baseline_daily["ma_day"],
                "Baseline XRT/Day": baseline_daily["xrt_day"],
                "Baseline Total/Day": baseline_daily["total_day"],

                "Forecast Provider/Day": forecast_daily["provider_day"],
                "Forecast PSR/Day": forecast_daily["psr_day"],
                "Forecast MA/Day": forecast_daily["ma_day"],
                "Forecast XRT/Day": forecast_daily["xrt_day"],
                "Forecast Total/Day": forecast_daily["total_day"],

                # FTE Need (exact)
                "Baseline Provider FTE": baseline_fte["provider_fte"],
                "Baseline PSR FTE": baseline_fte["psr_fte"],
                "Baseline MA FTE": baseline_fte["ma_fte"],
                "Baseline XRT FTE": baseline_fte["xrt_fte"],
                "Baseline Total FTE": baseline_fte["total_fte"],

                "Forecast Provider FTE": forecast_fte["provider_fte"],
                "Forecast PSR FTE": forecast_fte["psr_fte"],
                "Forecast MA FTE": forecast_fte["ma_fte"],
                "Forecast XRT FTE": forecast_fte["xrt_fte"],
                "Forecast Total FTE": forecast_fte["total_fte"],
            }
        )

        st.success(f"✅ Saved: {run_name}")

    # -------------------------
    # Show Portfolio Table
    # -------------------------

    if len(st.session_state["runs"]) > 0:
        st.markdown("### Saved Runs Portfolio")

        portfolio_df = pd.DataFrame(st.session_state["runs"])

        # round display only
        for col in portfolio_df.columns:
            if "FTE" in col or "/Day" in col:
                portfolio_df[col] = portfolio_df[col].astype(float).round(2)

        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

        # Export as CSV
        csv = portfolio_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Portfolio CSV",
            data=csv,
            file_name="psm_saved_runs.csv",
            mime="text/csv",
        )

    # ============================================================
    # ✅ STEP A4: Save Runs + Export + Compare
    # ============================================================

    st.markdown("---")
    st.subheader("Save + Compare Runs (Portfolio)")

    st.caption(
        "Save this scenario to compare staffing needs across different volumes, "
        "growth scenarios, or operating models."
    )

    # -------------------------
    # Initialize Session State
    # -------------------------
    if "runs" not in st.session_state:
        st.session_state["runs"] = []

    # -------------------------
    # Name Run
    # -------------------------
    default_name = f"Run {len(st.session_state['runs']) + 1}"
    run_name = st.text_input("Name this run:", value=default_name)
    
    # -------------------------
    # Display Saved Runs
    # -------------------------
    if len(st.session_state["runs"]) > 0:

        st.markdown("### Saved Runs")

        runs_df = pd.DataFrame(st.session_state["runs"])

        st.dataframe(runs_df, hide_index=True, use_container_width=True)

        # -------------------------
        # Export Runs
        # -------------------------
        csv = runs_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Runs as CSV",
            data=csv,
            file_name="psm_saved_runs.csv",
            mime="text/csv",
        )

        # -------------------------
        # Clear Runs
        # -------------------------
        if st.button("🗑️ Clear All Saved Runs"):
            st.session_state["runs"] = []
            st.warning("All saved runs cleared.")
            st.rerun()
    else:
        st.info("No saved runs yet — click **Save This Run** to begin building your portfolio.")
   
    
    # ============================================================
    # ✅ STEP A5: Executive Summary Card (Baseline vs Forecast + Delta)
    # ============================================================

    st.markdown("---")
    st.subheader("Executive Summary (Baseline vs Forecast)")

    # -------------------------
    # Pull totals
    # -------------------------
    baseline_total_day = baseline_daily["total_day"]
    forecast_total_day = forecast_daily["total_day"]

    baseline_total_fte = baseline_fte["total_fte"]
    forecast_total_fte = forecast_fte["total_fte"]

    # Delta values
    delta_day = forecast_total_day - baseline_total_day
    delta_fte = forecast_total_fte - baseline_total_fte

    # -------------------------
    # Interpretation Logic
    # -------------------------
    interpretation_lines = []

    if delta_fte > 0:
        interpretation_lines.append(
            f"Forecast volume may require **+{delta_fte:.2f} additional FTEs** to maintain staffing coverage."
        )
    elif delta_fte < 0:
        interpretation_lines.append(
            f"Forecast volume may allow **{abs(delta_fte):.2f} fewer FTEs** while maintaining staffing coverage."
        )
    else:
        interpretation_lines.append(
            "Forecast volume does **not** materially change your staffing requirement."
        )

    interpretation_lines.append(
        "Daily staffing values are **rounded UP** to prevent under-staffing. FTE Need is **exact**."
    )

    # -------------------------
    # Productivity Ratios (optional but helpful)
    # -------------------------
    baseline_visits_per_staff = visits / baseline_total_day if baseline_total_day > 0 else 0
    forecast_visits_per_staff = forecast_visits / forecast_total_day if forecast_total_day > 0 else 0

    baseline_patients_per_provider = visits / baseline_daily["provider_day"] if baseline_daily["provider_day"] > 0 else 0
    forecast_patients_per_provider = forecast_visits / forecast_daily["provider_day"] if forecast_daily["provider_day"] > 0 else 0

    # -------------------------
    # Layout
    # -------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Baseline")
        st.metric("Visits / Day", f"{visits:.0f}")
        st.metric("Total Staff / Day", f"{baseline_total_day:.2f}")
        st.metric("FTE Need", f"{baseline_total_fte:.2f}")

    with c2:
        st.markdown("### Forecast")
        st.metric("Visits / Day", f"{forecast_visits:.0f}")
        st.metric("Total Staff / Day", f"{forecast_total_day:.2f}")
        st.metric("FTE Need", f"{forecast_total_fte:.2f}")

    with c3:
        st.markdown("### Change (Forecast - Baseline)")
        st.metric("Δ Staff / Day", f"{delta_day:+.2f}")
        st.metric("Δ FTE Need", f"{delta_fte:+.2f}")

    # -------------------------
    # Interpretation Box
    # -------------------------
    st.markdown("")
    st.info("✅ **Interpretation**\n\n" + "\n".join([f"- {x}" for x in interpretation_lines]))

    # -------------------------
    # Optional Productivity Snapshot
    # -------------------------
    with st.expander("Optional: Productivity Snapshot", expanded=False):
        st.caption("These are directional signals only. They help validate if staffing feels lean vs heavy.")

        p1, p2 = st.columns(2)

        with p1:
            st.markdown("**Baseline Productivity**")
            st.metric("Visits per Total Staff", f"{baseline_visits_per_staff:.1f}")
            st.metric("Patients per Provider", f"{baseline_patients_per_provider:.1f}")

        with p2:
            st.markdown("**Forecast Productivity**")
            st.metric("Visits per Total Staff", f"{forecast_visits_per_staff:.1f}")
            st.metric("Patients per Provider", f"{forecast_patients_per_provider:.1f}")

    # ============================================================
    # ✅ STEP A6: Provider Seasonality + Hiring Glidepath (Executive View)
    # ============================================================
    
    st.markdown("---")
    st.subheader("Provider Seasonality + Hiring Glidepath (Executive View)")
    
    st.caption(
        "This visualization shows the staffing seasonality curve, attrition risk if you do not backfill, "
        "and the recommended staffing plan to sustain provider coverage through flu season."
    )
    
    # ------------------------------------------------------------
    # ✅ Provider Hiring Glidepath Assumptions
    # ------------------------------------------------------------
    with st.expander("Provider Hiring Glidepath Assumptions", expanded=False):
    
        days_to_sign = st.number_input(
            "Days to Sign (Req Posted → Signed Offer)",
            min_value=1,
            value=90,
            step=5,
            key="days_to_sign_a6"
        )
    
        days_to_credential = st.number_input(
            "Days to Credential (Signed → Fully Credentialed)",
            min_value=1,
            value=90,
            step=5,
            key="days_to_credential_a6"
        )
    
        onboard_train_days = st.number_input(
            "Onboard / Train Days (Credentialed → Solo Ready)",
            min_value=0,
            value=30,
            step=5,
            key="onboard_train_days_a6"
        )
    
        coverage_buffer_days = st.number_input(
            "Buffer Days (Planning Margin)",
            min_value=0,
            value=14,
            step=1,
            help="Conservative buffer so recruiting starts early.",
            key="coverage_buffer_days_a6"
        )
    
        utilization_factor = st.number_input(
            "Hiring Effectiveness Factor",
            min_value=0.10,
            max_value=1.00,
            value=0.90,
            step=0.05,
            help="Accounts for onboarding inefficiency, vacancies, call-outs, and schedule imperfections.",
            key="utilization_factor_a6"
        )
    
    # ------------------------------------------------------------
    # ✅ Flu Season Inputs (provider-only)
    # ------------------------------------------------------------
    st.markdown("### Flu Season Settings")
    
    flu_c1, flu_c2 = st.columns(2)
    
    with flu_c1:
        flu_start_month = st.selectbox(
            "Flu Season Start Month",
            options=list(range(1, 13)),
            index=11,  # Dec default
            format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
            key="flu_start_month_a6"
        )
    
    with flu_c2:
        flu_end_month = st.selectbox(
            "Flu Season End Month",
            options=list(range(1, 13)),
            index=1,  # Feb default
            format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
            key="flu_end_month_a6"
        )
    
    # ------------------------------------------------------------
    # ✅ Flu season date logic (handles wrap across year)
    # ------------------------------------------------------------
    current_year = today.year
    
    flu_start_date = datetime(current_year, flu_start_month, 1)
    
    if flu_end_month < flu_start_month:
        flu_end_date = datetime(current_year + 1, flu_end_month, 1)
    else:
        flu_end_date = datetime(current_year, flu_end_month, 1)
    
    # set flu_end_date to last day of end month
    flu_end_date = flu_end_date + timedelta(days=32)
    flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)
    
    # ------------------------------------------------------------
    # ✅ Provider pipeline timeline anchored to flu_start_date
    # ------------------------------------------------------------
    staffing_needed_by = flu_start_date
    
    total_provider_lead_days = (
        days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days
    )
    
    req_post_date = staffing_needed_by - timedelta(days=total_provider_lead_days)
    signed_date = req_post_date + timedelta(days=days_to_sign)
    credentialed_date = signed_date + timedelta(days=days_to_credential)
    solo_ready_date = credentialed_date + timedelta(days=onboard_train_days)
    
    # ------------------------------------------------------------
    # ✅ Provider baseline and forecast
    # ------------------------------------------------------------
    baseline_provider_fte = baseline_fte["provider_fte"]
    forecast_provider_fte = forecast_fte["provider_fte"]
    
    provider_gap = max(forecast_provider_fte - baseline_provider_fte, 0)
    provider_gap_adjusted = provider_gap / utilization_factor if utilization_factor > 0 else provider_gap
    
    # ------------------------------------------------------------
    # ✅ Provider turnover rate
    # ------------------------------------------------------------
    provider_turnover_rate = provider_turnover
    
    # ------------------------------------------------------------
    # ✅ Auto Freeze Date Logic
    # Goal: Let attrition naturally pull staffing down to baseline by flu_end_date
    # ------------------------------------------------------------
    overhang_fte = max(forecast_provider_fte - baseline_provider_fte, 0)
    monthly_attrition_fte = baseline_provider_fte * (provider_turnover_rate / 12)
    
    months_to_burn_off = overhang_fte / max(monthly_attrition_fte, 0.01)
    
    freeze_start_date = flu_end_date - timedelta(days=int(months_to_burn_off * 30.4))
    freeze_end_date = flu_end_date
    
    # clip freeze start so it doesn't start before today
    freeze_start_date = max(freeze_start_date, today)
    
    # ------------------------------------------------------------
    # ✅ Chart window (12 months)
    # ------------------------------------------------------------
    chart_start = today
    chart_end = today + timedelta(days=365)
    
    dates = pd.date_range(start=chart_start, end=chart_end, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    
    # ------------------------------------------------------------
    # ✅ 3 REQUIRED LINES
    # 1) Staffing Target = Seasonality Curve
    # 2) Attrition Line = No Backfill Risk
    # 3) Recommended Plan = What to actually pursue
    # ------------------------------------------------------------
    
    baseline_level = 100
    
    # ✅ Seasonal Targets (matches your sample)
    WINTER_LEVEL = 120
    SPRING_LEVEL = 100
    SUMMER_LEVEL = 80
    FALL_LEVEL   = 100
    
    
    def seasonality_target(d):
        """Returns staffing % based on month seasonality."""
        m = d.month
        if m in [12, 1, 2]:
            return WINTER_LEVEL
        elif m in [3, 4, 5]:
            return SPRING_LEVEL
        elif m in [6, 7, 8]:
            return SUMMER_LEVEL
        else:
            return FALL_LEVEL
    
    
    def smooth_ramp(d, start_date, end_date, start_val, end_val):
        """Smooth ramp using linear interpolation."""
        if d <= start_date:
            return start_val
        if d >= end_date:
            return end_val
    
        pct = (d - start_date).days / max((end_date - start_date).days, 1)
        return start_val + pct * (end_val - start_val)
    
    
    # ✅ 1) Staffing Target (Seasonality Curve)
    staffing_target = [seasonality_target(d) for d in dates]
    
    
    # ✅ 2) Attrition Projection Line (No Backfill Risk)
    # What happens if you DO NOT backfill providers and allow turnover erosion?
    attrition_line = []
    for d in dates:
        months_elapsed = (d.year - chart_start.year) * 12 + (d.month - chart_start.month)
        attrition_loss = months_elapsed * (provider_turnover_rate / 12) * 100
        attrition_line.append(max(baseline_level - attrition_loss, 0))
    
    
    # ✅ 3) Recommended Plan
    # Recommended plan should respect BOTH:
    # - the seasonality staffing demand
    # - and the minimum coverage needed to avoid erosion collapse
    recommended_plan = [max(t, a) for t, a in zip(staffing_target, attrition_line)]
    
    # ------------------------------------------------------------
    # ✅ Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    
    line_target, = ax.plot(
        dates, staffing_target,
        linewidth=3, marker="o",
        label="Staffing Target (Seasonality Curve)"
    )
    
    line_attrition, = ax.plot(
        dates, attrition_line,
        linestyle="--", linewidth=2.0,
        label="Attrition Projection (No Backfill Risk)"
    )
    
    line_recommended, = ax.plot(
        dates, recommended_plan,
        linewidth=3,
        label="Recommended Plan"
    )

    
    # ------------------------------------------------------------
    # ✅ Shaded blocks (distinct, branded, executive-friendly)
    # ------------------------------------------------------------
    
    # 🎨 Color palette (brand-aware)
    COLOR_SIGNING       = "#4C78A8"   # muted blue
    COLOR_CREDENTIALING = "#72B7B2"   # teal
    COLOR_TRAINING      = "#54A24B"   # soft green
    COLOR_FLU_SEASON    = "#7a6200"   # ✅ Sunshine Gold (brand)
    COLOR_FREEZE        = "#B0B0B0"   # neutral gray
    
    # ------------------------------------------------------------
    # ✅ Shaded blocks (no labels so they stay OUT of legend)
    # ------------------------------------------------------------

    ax.axvspan(req_post_date, signed_date, color=COLOR_SIGNING, alpha=0.18)
    ax.axvspan(signed_date, credentialed_date, color=COLOR_CREDENTIALING, alpha=0.18)
    ax.axvspan(credentialed_date, solo_ready_date, color=COLOR_TRAINING, alpha=0.18)
    
    ax.axvspan(flu_start_date, flu_end_date, color=COLOR_FLU_SEASON, alpha=0.12)
    ax.axvspan(freeze_start_date, freeze_end_date, color=COLOR_FREEZE, alpha=0.22)


    # ------------------------------------------------------------
    # ✅ Formatting
    # ------------------------------------------------------------
    ax.set_title("Provider Seasonality Curve + Hiring Glidepath (Executive Summary)")
    ax.set_ylabel("Staffing Level (% of Baseline)")
    ax.set_ylim(40, max(recommended_plan) + 20)
    
    ax.set_xlim(chart_start, chart_end)
    ax.set_xticks(dates)
    ax.set_xticklabels(month_labels)
    
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    
    # ------------------------------------------------------------
    # ✅ Legend (LINES ONLY)
    # ------------------------------------------------------------
    ax.legend(
        handles=[line_target, line_attrition, line_recommended],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    
    # ------------------------------------------------------------
    # ✅ Timeline Summary (Executive)
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("Provider Timeline Summary (Auto-calculated)")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Req Posted By", req_post_date.strftime("%b %d, %Y"))
        st.metric("Signed By", signed_date.strftime("%b %d, %Y"))
    
    with c2:
        st.metric("Credentialed By", credentialed_date.strftime("%b %d, %Y"))
        st.metric("Solo Ready By", solo_ready_date.strftime("%b %d, %Y"))
    
    with c3:
        st.metric("Flu Starts", flu_start_date.strftime("%b %d, %Y"))
        st.metric("Freeze Starts (Auto)", freeze_start_date.strftime("%b %d, %Y"))
    
    st.info(
        """
    ✅ **Executive Interpretation**
    - The Seasonality Curve represents staffing need, not headcount policy.
    - Staffing may fall below baseline in summer because volume is lower and vacation is encouraged.
    - Hiring freezes work by allowing turnover to reduce staffing naturally (instead of layoffs).
    - Recruiting must start early because providers require signing + credentialing + training lead time.
    """
    )
    
    # ============================================================
    # ✅ Summary Outputs (executive explanation)
    # ============================================================
    st.markdown("---")
    st.subheader("Provider Timeline Summary (Auto-calculated)")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Req Posted By", req_post_date.strftime("%b %d, %Y"))
        st.metric("Signed By", signed_date.strftime("%b %d, %Y"))
    
    with c2:
        st.metric("Credentialed By", credentialed_date.strftime("%b %d, %Y"))
        st.metric("Solo Ready By", solo_ready_date.strftime("%b %d, %Y"))
    
    with c3:
        st.metric("Flu Starts", flu_start_date.strftime("%b %d, %Y"))
        st.metric("Freeze Starts (Auto)", freeze_start_date.strftime("%b %d, %Y"))
    
    st.info(
        """
    ✅ **Executive Interpretation**
    - Staffing rises before flu season because providers require long lead time before solo coverage.
    - Hiring freeze starts automatically so turnover naturally reduces staffing back to baseline by flu end.
    - Summer staffing can fall below baseline because:
      1) demand is lower,
      2) vacation is encouraged,
      3) hiring is paused and attrition is allowed to naturally reduce staffing.
    """
    )
    
    # ============================================================
    # ✅ Coverage Plan
    # ============================================================
    
    st.markdown("### Coverage Plan While You Hire")
    
    if final_hiring_target_fte <= 0.01:
        st.caption("No gap coverage plan needed based on current forecast vs baseline.")
    else:
        hours_gap_per_week = final_hiring_target_adjusted * fte_hours_per_week
    
        st.markdown(
            f"""
    **Estimated Coverage Gap**
    - Conservative hiring target: **{final_hiring_target_adjusted:.2f} FTE**
    - Approx coverage hours needed per week: **{hours_gap_per_week:.1f} hours/week**
    """
        )
    
        st.markdown("**Suggested coverage options:**")
        st.markdown(
            """
    1) **PRN / Float Coverage**
       - Use PRN coverage to cover peak clinic days
       - Protects core schedule while hiring
    
    2) **Extra Shift Incentives**
       - Target +1–2 shifts/week to reduce gap burn
       - Keep it time-limited (30–60 days) to prevent burnout
    
    3) **Template Shift Adjustments**
       - Move staffing to high-yield hours
       - Avoid full-day overstaffing during low-demand periods
    """
        )
    
        st.caption("This is designed to reduce undercoverage risk while your hiring pipeline catches up.")


    # ============================================================
    # ✅ Gap Coverage Plan (PRN + Extra Shifts)
    # ============================================================

    st.markdown("### Coverage Plan While You Hire")

    if final_hiring_target_fte <= 0.01:
        st.caption("No gap coverage plan needed based on current forecast vs baseline.")
    else:

        # Estimate hours gap/week
        hours_gap_per_week = final_hiring_target_adjusted * fte_hours_per_week

        st.markdown(
            f"""
**Estimated Coverage Gap**
- Conservative gap: **{final_hiring_target_adjusted:.2f} FTE**
- Approx coverage hours needed per week: **{hours_gap_per_week:.1f} hours/week**
"""
        )

        st.markdown("**Suggested coverage options:**")
        st.markdown(
            """
1) **PRN / Float Coverage**
   - Use PRN coverage to cover peak clinic days
   - Protects core schedule while hiring

2) **Extra Shift Incentives**
   - Target +1–2 shifts/week to reduce gap burn
   - Keep it time-limited (30–60 days) to prevent burnout

3) **Template Shift Adjustments**
   - Move staffing to high-yield hours
   - Avoid full-day overstaffing during low-demand periods
"""
        )

        st.caption(
            "This is designed to reduce undercoverage risk while your hiring pipeline catches up."
        )

    
    # ============================================================
    # ✅ STEP A7: Role-Specific Hiring Needs + Glidepath
    # ============================================================

    st.markdown("---")
    st.subheader("Role-Specific Hiring Needs + Glidepath")

    st.caption(
        "This breaks the staffing delta into role-specific FTE hiring needs and estimates when recruiting should begin."
    )

    # -------------------------
    # Helper: Role glidepath dates
    # -------------------------
    def glidepath_dates(today, time_to_hire_days, training_days, buffer_days):
        recruit_start = today - timedelta(days=(time_to_hire_days + training_days + buffer_days))
        hire_filled = today + timedelta(days=time_to_hire_days)
        fully_productive = today + timedelta(days=(time_to_hire_days + training_days))
        return recruit_start, hire_filled, fully_productive

    # -------------------------
    # ✅ Extract role-level FTEs (needed for A7 / A8)
    # -------------------------

    baseline_provider_fte = baseline_fte["provider_fte"]
    baseline_psr_fte = baseline_fte["psr_fte"]
    baseline_ma_fte = baseline_fte["ma_fte"]
    baseline_xrt_fte = baseline_fte["xrt_fte"]

    forecast_provider_fte = forecast_fte["provider_fte"]
    forecast_psr_fte = forecast_fte["psr_fte"]
    forecast_ma_fte = forecast_fte["ma_fte"]
    forecast_xrt_fte = forecast_fte["xrt_fte"]
    # -------------------------
    # ✅ Role-specific hiring config (A8)
    # -------------------------
    role_hiring_config = {
        "Provider": {"tth": provider_tth, "ramp": provider_ramp},
        "PSR": {"tth": psr_tth, "ramp": psr_ramp},
        "MA": {"tth": ma_tth, "ramp": ma_ramp},
        "XRT": {"tth": xrt_tth, "ramp": xrt_ramp},
    }

    # -------------------------
    # Role-specific gaps (raw + adjusted)
    # -------------------------
    role_gaps = []

    roles = [
        ("Provider", baseline_provider_fte, forecast_provider_fte),
        ("PSR", baseline_psr_fte, forecast_psr_fte),
        ("MA", baseline_ma_fte, forecast_ma_fte),
        ("XRT", baseline_xrt_fte, forecast_xrt_fte),
    ]

    for role_name, base, forecast in roles:
        gap = max(forecast - base, 0)

        # Conservative adjustment
        adj_gap = gap / utilization_factor if utilization_factor > 0 else gap

        # Convert to hours/week coverage gap
        gap_hours_week = adj_gap * fte_hours_per_week

        # Glidepath timing
        tth_days = role_hiring_config[role_name]["tth"]
        ramp_days = role_hiring_config[role_name]["ramp"]

        recruit_start, hire_filled, fully_productive = glidepath_dates(
            today,
            tth_days,
            ramp_days,
            coverage_buffer_days,
        )

        role_gaps.append(
            {
                "Role": role_name,
                "Baseline FTE": round(base, 2),
                "Forecast FTE": round(forecast, 2),
                "Gap FTE": round(gap, 2),
                "Adjusted Gap FTE": round(adj_gap, 2),
                "Coverage Hours/Wk": round(gap_hours_week, 1),
                "Recruit Start": recruit_start.strftime("%b %d, %Y"),
                "Hire Filled": hire_filled.strftime("%b %d, %Y"),
                "Fully Productive": fully_productive.strftime("%b %d, %Y"),
            }
        )

    role_gap_df = pd.DataFrame(role_gaps)

    # -------------------------
    # Display table
    # -------------------------
    st.markdown("### Hiring Need by Role")

    st.dataframe(
        role_gap_df,
        hide_index=True,
        use_container_width=True,
    )

    # -------------------------
    # Key callouts
    # -------------------------
    total_gap = role_gap_df["Gap FTE"].sum()
    total_adj_gap = role_gap_df["Adjusted Gap FTE"].sum()
    total_hours_week = role_gap_df["Coverage Hours/Wk"].sum()

    if total_gap <= 0.01:
        st.success("✅ No net new hiring required by role based on your baseline vs forecast assumptions.")
    else:
        st.warning("⚠️ Role-level staffing gaps detected — review recruiting and coverage timelines below.")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Total Gap (Raw FTE)", f"{total_gap:.2f}")

        with c2:
            st.metric("Total Gap (Adjusted FTE)", f"{total_adj_gap:.2f}")

        with c3:
            st.metric("Coverage Hours Needed / Week", f"{total_hours_week:.1f}")

    st.caption(
        "Adjusted gaps account for real-world utilization limits: onboarding ramp, imperfect schedules, call-outs, and vacancy drag."
    )
