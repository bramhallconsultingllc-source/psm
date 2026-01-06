    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    from datetime import datetime, timedelta
    from psm.staffing_model import StaffingModel
    
    if "today" not in st.session_state:
        st.session_state["today"] = datetime.today()
    today = st.session_state["today"]
    
    # -------------------------
    # Session State Init
    # -------------------------
    if "runs" not in st.session_state:
        st.session_state["runs"] = []
    
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
    
    # ✅ PLACE THIS LINE HERE
    st.info("ℹ️ Enter turnover assumptions above, then click **Calculate Staffing** to generate turnover buffers.")
    
    # -------------------------
    # Calculate
    # -------------------------
    
    if st.button("Calculate Staffing"):

    # ✅ FIX: define today ONCE here (global for all downstream steps)
    from datetime import datetime, timedelta
    today = datetime.today()

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
    # ✅ STEP A6: Hiring Glidepath + Coverage Plan (with Turnover Buffer + Timeline Chart)
    # ============================================================
    
    st.markdown("---")
    st.subheader("Hiring Glidepath + Coverage Plan")
    
    st.caption(
        "This converts the forecast staffing delta into a recommended recruiting start date "
        "and a short-term coverage plan while you hire."
    )
    
    # -------------------------
    # Adjustable assumptions
    # -------------------------
    with st.expander("Adjust Hiring Assumptions", expanded=False):
    
        avg_time_to_hire_days = st.number_input(
            "Average Time to Hire (days)",
            min_value=1,
            value=120,
            step=5,
        )
    
        training_ramp_days = st.number_input(
            "Training / Ramp Period (days)",
            min_value=0,
            value=14,
            step=1,
        )
    
        coverage_buffer_days = st.number_input(
            "Buffer Days (planning margin)",
            min_value=0,
            value=14,
            step=1,
            help="Adds a conservative buffer so you start recruiting earlier.",
        )
    
        utilization_factor = st.number_input(
            "Hiring Effectiveness Factor",
            min_value=0.10,
            max_value=1.00,
            value=0.90,
            step=0.05,
            help="Accounts for onboarding inefficiency, vacancies, call-outs, and imperfect schedules.",
        )
    
    # -------------------------
    # Staffing gap (FTE Need delta)
    # -------------------------
    raw_gap_fte = forecast_total_fte - baseline_total_fte
    gap_fte = max(raw_gap_fte, 0)
    
    # Adjusted gap (conservative)
    adjusted_gap_fte = gap_fte / utilization_factor if utilization_factor > 0 else gap_fte
    
    # -------------------------
    # ✅ DATE MATH (DEFINE FIRST so it can be reused everywhere)
    # -------------------------
                
    # -------------------------
    # ✅ DATE MATH (DEFINE FIRST so it can be reused everywhere)
    # -------------------------
    
    staffing_needed_by = today
    
    recruit_start_date = staffing_needed_by - timedelta(
        days=(avg_time_to_hire_days + training_ramp_days + coverage_buffer_days)
    )
    
    candidate_start_date = today + timedelta(days=avg_time_to_hire_days)
    
    full_productivity_date = today + timedelta(days=(avg_time_to_hire_days + training_ramp_days))
    
    turnover_end_date = today + timedelta(days=int(planning_months * 30.4))

    
    # -------------------------
    # Turnover Buffer (role specific)
    # -------------------------
    turnover_config = {
        "Provider": provider_turnover,
        "PSR": psr_turnover,
        "MA": ma_turnover,
        "XRT": xrt_turnover,
    }
    
    months_factor = planning_months / 12
    
    role_forecast_fte = {
        "Provider": forecast_fte["provider_fte"],
        "PSR": forecast_fte["psr_fte"],
        "MA": forecast_fte["ma_fte"],
        "XRT": forecast_fte["xrt_fte"],
    }
    
    turnover_buffer = {}
    for role, fte_needed in role_forecast_fte.items():
        turnover_buffer[role] = fte_needed * turnover_config[role] * months_factor
    
    turnover_buffer_total = sum(turnover_buffer.values())
    
    # -------------------------
    # Final hiring targets
    # -------------------------
    final_hiring_target_fte = gap_fte + turnover_buffer_total
    final_hiring_target_adjusted = final_hiring_target_fte / utilization_factor if utilization_factor > 0 else final_hiring_target_fte

    # -------------------------
    # Force chart window to only show 12 months
    # -------------------------
    chart_start = today
    chart_end = today + timedelta(days=365)
    
    # If req-post date is earlier than chart start, clip it so the chart stays clean
    plot_recruit_start = max(recruit_start_date, chart_start)
    plot_candidate_start = max(candidate_start_date, chart_start)
    plot_full_productive = max(full_productivity_date, chart_start)
    plot_turnover_end = min(turnover_end_date, chart_end)

    # ============================================================
    # ✅ Executive Summary View (Clean Seasonality Style)
    # ============================================================
    
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import timedelta
    import pandas as pd
    
    st.markdown("---")
    st.subheader("Seasonality Recommender – Executive Summary View")
    
    st.caption(
        "This chart shows (1) your staffing target, (2) how attrition erodes coverage if you do not backfill, "
        "and (3) the key windows where recruiting and onboarding should occur."
    )
    
    # -------------------------
    # Timeline setup (12 months, monthly points)
    # -------------------------
    dates = pd.date_range(start=chart_start, end=chart_end, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    
    # -------------------------
    # Staffing Target (% baseline)
    # -------------------------
    baseline_level = 100
    forecast_level = (forecast_total_fte / baseline_total_fte) * 100
    
    staffing_target = []
    for d in dates:
        if d < plot_full_productive:
            staffing_target.append(baseline_level)
        else:
            staffing_target.append(forecast_level)
    
    # Optional: return to baseline after turnover buffer ends
    for i, d in enumerate(dates):
        if d > plot_turnover_end:
            staffing_target[i] = baseline_level

    # -------------------------
    # Attrition / turnover erosion line (% baseline)
    # -------------------------
    turnover_drop_pct = (turnover_buffer_total / baseline_total_fte) * 100
    
    turnover_line = []
    for d in dates:
        if d <= plot_turnover_end:
            pct = baseline_level - (
                turnover_drop_pct * ((d - chart_start).days / max((plot_turnover_end - chart_start).days, 1))
            )
            turnover_line.append(pct)
        else:
            turnover_line.append(baseline_level)

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    
    # Staffing target step line
    ax.step(dates, staffing_target, where="post", linewidth=3, marker="o",
            label="Staffing Target (Forecast + Buffer)")
    
    # Attrition projection line
    ax.plot(dates, turnover_line, linestyle="--", linewidth=2.5,
            label="Attrition Projection (No Backfill)")
    
    # -------------------------
    # Shaded regions (Recruit / Ramp / Turnover)
    # -------------------------
    ax.axvspan(plot_recruit_start, plot_candidate_start, alpha=0.15, label="Recruiting Window")
    ax.axvspan(plot_candidate_start, plot_full_productive, alpha=0.10, label="Ramp Window")
    ax.axvspan(chart_start, plot_turnover_end, alpha=0.08, label="Turnover Buffer Window")
    
    # -------------------------
    # Vertical markers
    # -------------------------
    ax.axvline(plot_recruit_start, linestyle=":", linewidth=1)
    ax.axvline(plot_candidate_start, linestyle=":", linewidth=1)
    ax.axvline(plot_full_productive, linestyle=":", linewidth=1)
    ax.axvline(plot_turnover_end, linestyle=":", linewidth=1)
    
    # -------------------------
    # Labels / annotations
    # -------------------------
    ymax = max(staffing_target) + 8
    
    ax.annotate("Post Req",
                xy=(plot_recruit_start, ymax-6),
                xytext=(plot_recruit_start, ymax),
                arrowprops=dict(arrowstyle="->"),
                ha="center", fontsize=10)
    
    ax.annotate("Candidate Start",
                xy=(plot_candidate_start, ymax-6),
                xytext=(plot_candidate_start, ymax),
                arrowprops=dict(arrowstyle="->"),
                ha="center", fontsize=10)
    
    ax.annotate("Fully Productive",
                xy=(plot_full_productive, ymax-6),
                xytext=(plot_full_productive, ymax),
                arrowprops=dict(arrowstyle="->"),
                ha="center", fontsize=10)
    
    ax.annotate("Turnover Buffer Ends",
                xy=(plot_turnover_end, ymax-6),
                xytext=(plot_turnover_end, ymax),
                arrowprops=dict(arrowstyle="->"),
                ha="center", fontsize=10)
    
    # -------------------------
    # Formatting
    # -------------------------
    ax.set_title("Seasonality Recommender – Executive Summary View")
    ax.set_ylabel("Staffing Level (% of Baseline)")
    ax.set_ylim(60, ymax)
    
    # ✅ Force x-axis to only show 12 months
    ax.set_xlim(chart_start, chart_end)
    
    # ✅ Month labels exactly like your example
    ax.set_xticks(dates)
    ax.set_xticklabels(month_labels)
    
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, loc="lower left")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ============================================================
    # ✅ Output Summary Card
    # ============================================================
    
    st.markdown("---")
    st.subheader("Hiring Summary")
    
    if final_hiring_target_fte <= 0.01:
        st.success("✅ Forecast staffing does not require net new hiring based on current assumptions.")
        st.caption("If you are still feeling operational strain, the constraint is likely workflow, role clarity, or demand pattern—not headcount.")
    
    else:
        st.warning("⚠️ Forecast staffing likely requires additional staffing coverage.")
    
        c1, c2, c3 = st.columns(3)
    
        with c1:
            st.metric("Hiring Target (Gap + Turnover)", f"{final_hiring_target_fte:.2f}")
    
        with c2:
            st.metric("Hiring Target (Adjusted)", f"{final_hiring_target_adjusted:.2f}")
    
        with c3:
            st.metric("Recommended Recruiting Start", recruit_start_date.strftime("%b %d, %Y"))
    
        st.markdown("")
        st.info(
            f"""
    ✅ **Hiring Timeline Summary**
    - Hiring target begins now (forecast + turnover buffer)
    - Start recruiting by: **{recruit_start_date.strftime("%b %d, %Y")}**
    - Expected hire filled by: **{candidate_start_date.strftime("%b %d, %Y")}**
    - Fully productive by: **{full_productivity_date.strftime("%b %d, %Y")}**
    - Turnover buffer ends: **{turnover_end_date.strftime("%b %d, %Y")}**
    
    **Why conservative?**
    - We apply a utilization factor ({utilization_factor:.2f}) to reduce undercoverage risk.
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
