import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ============================================================
# ✅ Stable Today (so sliders don't reset chart window)
# ============================================================

if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()

today = st.session_state["today"]

# ============================================================
# ✅ Session State Init
# ============================================================

if "runs" not in st.session_state:
    st.session_state["runs"] = []

model = StaffingModel()

# ============================================================
# ✅ Page Setup
# ============================================================

st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")

st.title("Predictive Staffing Model (PSM)")
st.caption("A simple staffing calculator using linear interpolation + conservative rounding rules.")

st.info(
    "⚠️ **All daily staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)


# ============================================================
# ✅ INPUTS (GLOBAL — always at root level, never inside button)
# ============================================================

st.subheader("Baseline Inputs")

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


# ============================================================
# ✅ Role-Specific Hiring Assumptions
# ============================================================

st.markdown("---")
st.subheader("Role-Specific Hiring Assumptions")

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

st.markdown("---")
st.subheader("Role-Specific Turnover Assumptions")

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


# ============================================================
# ✅ Provider Seasonality Inputs (GLOBAL)
# ============================================================

st.markdown("---")
st.subheader("Provider Seasonality Inputs")

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
        help="Conservative buffer so recruiting starts earlier.",
        key="coverage_buffer_days_global"
    )

    utilization_factor = st.number_input(
        "Hiring Effectiveness Factor",
        min_value=0.10,
        max_value=1.00,
        value=0.90,
        step=0.05,
        help="Accounts for onboarding inefficiency, vacancy drag, imperfect scheduling.",
        key="utilization_factor_global"
    )


# ============================================================
# ✅ Calculate Button (store results only)
# ============================================================

st.markdown("---")

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
# ✅ Render Output AFTER calculation
# ============================================================

if st.session_state.get("calculated"):

    daily_result = st.session_state["daily_result"]
    fte_result = st.session_state["fte_result"]
    fte_df = st.session_state["fte_df"]

    # ============================================================
    # ✅ STEP 4: Staffing Output
    # ============================================================

    st.subheader("Full-Time Employees (FTEs) Needed")
    st.dataframe(fte_df, hide_index=True, use_container_width=True)

    # ============================================================
    # ✅ STEP A2: Forecast Scenario Planning
    # ============================================================

    st.markdown("---")
    st.subheader("Forecast Scenario Planning")

    mode = st.radio("Forecast Method", ["Percent Change", "Visit Change (+/-)"], horizontal=True)

    col_a, col_b = st.columns(2)

    if mode == "Percent Change":
        pct_change = col_a.number_input("Forecast Volume Change (%)", value=10.0, step=1.0)
        forecast_visits = visits * (1 + pct_change / 100)
    else:
        visit_change = col_a.number_input("Forecast Visit Change (+/- visits/day)", value=10.0, step=1.0)
        forecast_visits = visits + visit_change

    forecast_visits = max(forecast_visits, 1.0)
    col_b.metric("Forecast Visits / Day", f"{forecast_visits:.1f}")

    forecast_daily = model.calculate(forecast_visits)
    forecast_fte = model.calculate_fte_needed(forecast_visits, hours_of_operation, fte_hours_per_week)


    # ============================================================
    # ✅ STEP A3: Visual Summary (Baseline vs Forecast)
    # ============================================================

    st.markdown("---")
    st.subheader("Visual Summary (Baseline vs Forecast)")

    baseline_total_fte = fte_result["total_fte"]
    forecast_total_fte = forecast_fte["total_fte"]

    compare_df = pd.DataFrame(
        {
            "Metric": ["Total FTE Needed"],
            "Baseline": [baseline_total_fte],
            "Forecast": [forecast_total_fte],
        }
    )

    compare_df["Delta"] = compare_df["Forecast"] - compare_df["Baseline"]
    compare_df[["Baseline", "Forecast", "Delta"]] = compare_df[["Baseline", "Forecast", "Delta"]].round(2)

    st.dataframe(compare_df, hide_index=True, use_container_width=True)


    # ============================================================
    # ✅ STEP A6: PROVIDER SEASONALITY + GLIDEPATH (EXECUTIVE VIEW)
    # ============================================================

    st.markdown("---")
    st.subheader("Provider Seasonality + Hiring Glidepath (Executive View)")

    # -------------------------
    # ✅ Colors (distinct + includes Sunshine Gold)
    # -------------------------
    COLOR_SIGNING = "#7a6200"         # Sunshine Gold
    COLOR_CREDENTIALING = "#2e86de"   # strong blue
    COLOR_TRAINING = "#27ae60"        # green
    COLOR_FLU_SEASON = "#f4c542"      # warm flu highlight
    COLOR_FREEZE = "#7f8c8d"          # darker freeze gray

    # -------------------------
    # ✅ Flu season date logic (wrap across year)
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
    # ✅ Seasonality Curve Map (matches sample)
    # -------------------------
    seasonality_map = {
        12: 120, 1: 120, 2: 120,
        3: 100, 4: 100, 5: 100,
        6: 80, 7: 80, 8: 80,
        9: 100, 10: 100, 11: 100
    }

    chart_start = today
    chart_end = today + timedelta(days=365)

    dates = pd.date_range(start=chart_start, end=chart_end, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]

    staffing_target = [seasonality_map[d.month] for d in dates]
    baseline_level = 100

    # -------------------------
    # ✅ Provider hiring timeline (anchored to flu_start)
    # -------------------------
    staffing_needed_by = flu_start_date
    total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

    req_post_date = staffing_needed_by - timedelta(days=total_lead_days)
    signed_date = req_post_date + timedelta(days=days_to_sign)
    credentialed_date = signed_date + timedelta(days=days_to_credential)
    solo_ready_date = credentialed_date + timedelta(days=onboard_train_days)

    # -------------------------
    # ✅ Auto Freeze Logic (provider only)
    # -------------------------
    baseline_provider_fte = fte_result["provider_fte"]
    provider_turnover_rate = provider_turnover  # annual

    peak_overhang_pct = max(max(staffing_target) - baseline_level, 0)
    overhang_fte = (peak_overhang_pct / 100) * baseline_provider_fte

    monthly_attrition_fte = baseline_provider_fte * (provider_turnover_rate / 12)
    months_to_burn_off = overhang_fte / max(monthly_attrition_fte, 0.01)

    freeze_start_date = flu_end_date - timedelta(days=int(months_to_burn_off * 30.4))
    freeze_start_date = max(freeze_start_date, today)
    freeze_end_date = flu_end_date

    # -------------------------
    # ✅ Attrition Line (No backfill risk)
    # -------------------------
    attrition_line = []
    for d in dates:
        months_elapsed = (d.year - chart_start.year) * 12 + (d.month - chart_start.month)
        loss = months_elapsed * (provider_turnover_rate / 12) * 100
        attrition_line.append(max(baseline_level - loss, 0))

    # -------------------------
    # ✅ Recommended Plan = max(target, attrition)
    # -------------------------
    recommended_plan = [max(t, a) for t, a in zip(staffing_target, attrition_line)]

    # -------------------------
    # ✅ Plot (Legend is LINE ONLY)
    # -------------------------
    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(dates, staffing_target, linewidth=3, marker="o",
            label="Staffing Target (Seasonality Curve)")
    ax.plot(dates, attrition_line, linestyle="--", linewidth=2,
            label="Attrition Projection (No Backfill Risk)")
    ax.plot(dates, recommended_plan, linewidth=3,
            label="Recommended Plan")

    # ✅ Shading blocks (no legend entries)
    ax.axvspan(req_post_date, signed_date, color=COLOR_SIGNING, alpha=0.22)
    ax.axvspan(signed_date, credentialed_date, color=COLOR_CREDENTIALING, alpha=0.22)
    ax.axvspan(credentialed_date, solo_ready_date, color=COLOR_TRAINING, alpha=0.22)

    ax.axvspan(flu_start_date, flu_end_date, color=COLOR_FLU_SEASON, alpha=0.20)
    ax.axvspan(freeze_start_date, freeze_end_date, color=COLOR_FREEZE, alpha=0.18)

    ax.set_title("Provider Seasonality Curve + Hiring Glidepath (Executive Summary)")
    ax.set_ylabel("Staffing Level (% of Baseline)")
    ax.set_ylim(40, max(recommended_plan) + 20)

    ax.set_xlim(chart_start, chart_end)
    ax.set_xticks(dates)
    ax.set_xticklabels(month_labels)

    ax.grid(axis="y", linestyle=":", alpha=0.35)

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    st.pyplot(fig)

    # -------------------------
    # ✅ Block Key (Shaded Windows)
    # -------------------------
    st.markdown("### Block Key (Shaded Windows)")

    st.markdown(
        f"""
        <div style="font-size: 14px; line-height: 1.8;">
            <span style="background-color:{COLOR_SIGNING}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
            &nbsp; Req Posted → Signed Offer<br>

            <span style="background-color:{COLOR_CREDENTIALING}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
            &nbsp; Signed → Credentialed<br>

            <span style="background-color:{COLOR_TRAINING}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
            &nbsp; Credentialed → Solo Ready<br>

            <span style="background-color:{COLOR_FLU_SEASON}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
            &nbsp; Flu Season Peak<br>

            <span style="background-color:{COLOR_FREEZE}; padding:4px 10px; border-radius:3px;">&nbsp;</span>
            &nbsp; Hiring Freeze (Attrition Burn-Off)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------
    # ✅ Timeline Summary (Only ONCE)
    # -------------------------
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
        st.metric("Flu Ends", flu_end_date.strftime("%b %d, %Y"))

    st.info(
        """
✅ **Executive Interpretation**
- Staffing can fall below baseline in summer because census drops and vacations rise.
- Freeze hiring in winter, do not backfill attrition, and allow staffing to drift downward naturally.
- Unfreeze hiring in summer to rebuild pipeline for winter flu season.
- The Recommended Plan ensures you never fall below staffing viability while still following seasonality.
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
