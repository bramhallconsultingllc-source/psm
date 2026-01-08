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
# ✅ Page Setup
# ============================================================
st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")
st.title("Predictive Staffing Model (PSM)")
st.caption("Seasonality staffing + burnout protection + realism + finance case + float pool strategy.")

st.info(
    "⚠️ **All daily staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)

model = StaffingModel()

# ============================================================
# ✅ Helper Functions (Single Source of Truth)
# ============================================================
def base_seasonality_multiplier(month: int):
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00

def build_flu_window(current_year: int, flu_start_month: int, flu_end_month: int):
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
    return (d.year - anchor.year) * 12 + (d.month - anchor.month)

def clamp(x, lo, hi):
    return max(lo, min(x, hi))

def compute_seasonality_forecast(dates, baseline_visits, flu_start, flu_end, flu_uplift_pct):
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
# ✅ Burnout-Protective Staffing Curve (Recommended Target)
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
    vol_w, spike_w, debt_w = weights

    visits_arr = np.array(visits_by_month)
    mean_visits = np.mean(visits_arr)
    std_visits = np.std(visits_arr)
    cv = (std_visits / mean_visits) if mean_visits > 0 else 0.0
    p75 = np.percentile(visits_arr, 75)

    rdi = 0.0
    decay = 0.85
    lambda_debt = 0.10

    protective_curve = []
    prev_staff = max(base_demand_fte[0], provider_min_floor)

    volatility_buffer = []
    spike_buffer = []
    recovery_debt_buffer = []

    for d, v, base_fte in zip(dates, visits_by_month, base_demand_fte):

        vbuf = base_fte * cv
        sbuf = max(0.0, (v - p75) / mean_visits) * base_fte if mean_visits > 0 else 0

        visits_per_provider = v / max(prev_staff, 0.25)
        debt = max(0.0, visits_per_provider - safe_visits_per_provider_per_day)
        rdi = decay * rdi + debt
        dbuf = lambda_debt * rdi

        buffer_fte = burnout_slider * (vol_w * vbuf + spike_w * sbuf + debt_w * dbuf)

        raw_target = max(provider_min_floor, base_fte + buffer_fte)

        delta = raw_target - prev_staff
        if delta > 0:
            delta = clamp(delta, 0.0, smoothing_up)
        else:
            delta = clamp(delta, -smoothing_down, 0.0)

        final_staff = max(provider_min_floor, prev_staff + delta)

        protective_curve.append(final_staff)
        prev_staff = final_staff

        volatility_buffer.append(vbuf)
        spike_buffer.append(sbuf)
        recovery_debt_buffer.append(dbuf)

    buffers = {
        "volatility_buffer": volatility_buffer,
        "spike_buffer": spike_buffer,
        "recovery_debt_buffer": recovery_debt_buffer,
        "cv": cv,
        "p75": p75
    }

    return protective_curve, buffers

# ============================================================
# ✅ Realistic Staffing Supply Curve (Best-Case)
# ============================================================
def realistic_staffing_supply_curve(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    max_hiring_up_per_month=0.50,
    max_ramp_down_per_month=0.25,
):
    monthly_attrition_fte = baseline_fte * (annual_turnover_rate / 12)
    effective_attrition_start = today + timedelta(days=int(notice_days))

    staff = []
    prev = max(baseline_fte, provider_min_floor)

    for d, target in zip(dates, target_curve):

        delta = target - prev
        if delta > 0:
            delta = clamp(delta, 0.0, max_hiring_up_per_month)
        else:
            delta = clamp(delta, -max_ramp_down_per_month, 0.0)

        planned = prev + delta

        if d >= effective_attrition_start:
            months_elapsed = monthly_index(d, effective_attrition_start)
            attrition_loss = months_elapsed * monthly_attrition_fte
            planned = max(planned - attrition_loss, provider_min_floor)

        planned = max(planned, provider_min_floor)

        staff.append(planned)
        prev = planned

    return staff

# ============================================================
# ✅ Utility — gap in provider-days
# ============================================================
def provider_day_gap(target_curve, supply_curve, days_in_month):
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(t - s, 0) * dim
    return gap_days

# ============================================================
# ✅ Baseline Inputs
# ============================================================
st.markdown("## Baseline Inputs")

visits = st.number_input("Average Visits per Day (Annual Average)", min_value=1.0, value=45.0, step=1.0)

st.markdown("### Weekly Inputs (for FTE conversion)")

hours_of_operation = st.number_input("Hours of Operation per Week", min_value=1.0, value=70.0, step=1.0)
fte_hours_per_week = st.number_input("FTE Hours per Week (default 40)", min_value=1.0, value=40.0, step=1.0)

provider_min_floor = st.number_input(
    "Provider Minimum Floor (FTE)",
    min_value=0.25,
    value=1.00,
    step=0.25,
    help="Prevents unrealistic projections (e.g., demand curves dropping to 0)."
)

# ============================================================
# ✅ Burnout Protection Controls
# ============================================================
st.markdown("## Burnout Protection Controls")

burnout_slider = st.slider(
    "Burnout Protection Level",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="0.0 = Lean | 0.6 = Protective (recommended) | 1.0 = Max protection"
)

safe_visits_per_provider = st.number_input(
    "Safe Visits per Provider per Day Threshold",
    min_value=10,
    max_value=40,
    value=20,
    step=1,
    help="Used in recovery debt buffer. Higher = less protective."
)

# ============================================================
# ✅ Provider Turnover + Lead Time Inputs
# ============================================================
st.markdown("## Turnover + Hiring Lead Time Assumptions")

provider_turnover = st.number_input("Provider Turnover %", value=24.0, step=1.0) / 100

with st.expander("Hiring Pipeline Timing Inputs", expanded=False):
    days_to_sign = st.number_input("Days to Sign (Req → Signed Offer)", min_value=1, value=90, step=5)
    days_to_credential = st.number_input("Days to Credential (Signed → Credentialed)", min_value=1, value=90, step=5)
    onboard_train_days = st.number_input("Onboard/Train Days (Credentialed → Solo)", min_value=0, value=30, step=5)
    coverage_buffer_days = st.number_input("Planning Buffer Days", min_value=0, value=14, step=1)
    notice_days = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=75, step=5)

# ============================================================
# ✅ Flu Season Settings
# ============================================================
st.markdown("## Flu Season Settings")

flu_c1, flu_c2, flu_c3 = st.columns(3)

with flu_c1:
    flu_start_month = st.selectbox("Flu Start Month", options=list(range(1, 13)), index=11,
                                   format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
with flu_c2:
    flu_end_month = st.selectbox("Flu End Month", options=list(range(1, 13)), index=1,
                                 format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
with flu_c3:
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100

# ============================================================
# ✅ Calculate Button
# ============================================================
if st.button("Calculate Staffing"):

    # Baseline FTE requirement
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )

    baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

    fte_df = pd.DataFrame({
        "Role": ["Provider", "PSR", "MA", "XRT", "TOTAL"],
        "FTE Needed": [
            round(fte_result["provider_fte"], 2),
            round(fte_result["psr_fte"], 2),
            round(fte_result["ma_fte"], 2),
            round(fte_result["xrt_fte"], 2),
            round(fte_result["total_fte"], 2),
        ]
    })

    st.markdown("---")
    st.subheader("Baseline FTEs Needed")
    st.dataframe(fte_df, hide_index=True, use_container_width=True)

    # ============================================================
    # ✅ Timeline Setup
    # ============================================================
    current_year = today.year
    dates = pd.date_range(start=datetime(current_year, 1, 1), periods=12, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    flu_start_date, flu_end_date = build_flu_window(current_year, flu_start_month, flu_end_month)

    # ============================================================
    # ✅ Seasonality Forecast
    # ============================================================
    st.markdown("---")
    st.subheader("Seasonality Forecast (Visits/Day by Month)")

    forecast_visits_by_month = compute_seasonality_forecast(
        dates=dates,
        baseline_visits=visits,
        flu_start=flu_start_date,
        flu_end=flu_end_date,
        flu_uplift_pct=flu_uplift_pct,
    )

    forecast_df = pd.DataFrame({"Month": month_labels, "Forecast Visits/Day": np.round(forecast_visits_by_month, 1)})
    st.dataframe(forecast_df, hide_index=True, use_container_width=True)

    # ============================================================
    # ✅ Demand vs Recommended Curves
    # ============================================================
    provider_base_demand = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_by_month,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    protective_curve, buffers = burnout_protective_staffing_curve(
        dates=dates,
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # ============================================================
    # ✅ Realistic Supply
    # ============================================================
    realistic_supply_lean = realistic_staffing_supply_curve(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
    )

    realistic_supply_recommended = realistic_staffing_supply_curve(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=protective_curve,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
    )

    # ============================================================
    # ✅ A6 Executive View
    # ============================================================
    st.markdown("---")
    st.subheader("A6 — Executive View: Volume, Staffing Targets, Supply, Burnout Exposure")

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(dates, provider_base_demand, linestyle=":", linewidth=2, label="Lean Target (Volume Only)")
    ax1.plot(dates, protective_curve, linewidth=3.5, marker="o", label="Recommended Target (Burnout-Protective)")
    ax1.plot(dates, realistic_supply_recommended, linewidth=3, marker="o", label="Realistic Staffing Supply")

    ax1.fill_between(
        dates,
        realistic_supply_recommended,
        protective_curve,
        where=np.array(protective_curve) > np.array(realistic_supply_recommended),
        alpha=0.25,
        label="Burnout Exposure Zone"
    )

    ax1.set_ylabel("Provider FTE")
    ax1.set_xticks(dates)
    ax1.set_xticklabels(month_labels)
    ax1.grid(axis="y", linestyle=":", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(dates, forecast_visits_by_month, linestyle="-.", linewidth=2.5, label="Forecast Visits/Day")
    ax2.set_ylabel("Visits / Day")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.22), ncol=2)

    plt.tight_layout()
    st.pyplot(fig)

    burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    burnout_months = sum([1 for g in burnout_gap_fte if g > 0])

    k1, k2, k3 = st.columns(3)
    k1.metric("Peak Burnout Gap (FTE)", f"{max(burnout_gap_fte):.2f}")
    k2.metric("Avg Burnout Gap (FTE)", f"{np.mean(burnout_gap_fte):.2f}")
    k3.metric("Months Exposed", f"{burnout_months}/12")

    # ============================================================
    # ✅ A8 Fractional Buffer + Float Pool Builder (NEW)
    # ============================================================
    st.markdown("---")
    st.header("A8 — Fractional Staffing Buffer + Float Pool Builder")

    time_to_replace_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

    expected_departures_fte = baseline_provider_fte * provider_turnover
    coverage_leakage_fte = expected_departures_fte * (time_to_replace_days / 365)

    pipeline_target_fte = baseline_provider_fte + coverage_leakage_fte

    fte_per_hire = st.number_input(
        "Average FTE per Provider Hire",
        min_value=0.25,
        max_value=1.00,
        value=1.00,
        step=0.05,
        help="Used to translate turnover FTE into number of hires."
    )

    expected_hires_per_year = expected_departures_fte / max(fte_per_hire, 0.25)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline Provider FTE", f"{baseline_provider_fte:.2f}")
    k2.metric("Replacement Lag (Days)", f"{time_to_replace_days}")
    k3.metric("Coverage Leakage Buffer (FTE)", f"{coverage_leakage_fte:.2f}")
    k4.metric("Recruiting Pipeline Target (FTE)", f"{pipeline_target_fte:.2f}")

    st.caption(
        f"Expected provider hires/year ≈ {expected_hires_per_year:.2f} "
        f"(~1 hire every {12/max(expected_hires_per_year,0.01):.1f} months)."
    )

    st.subheader("Regional Float Pool Builder")

    fp1, fp2, fp3 = st.columns(3)

    with fp1:
        num_clinics = st.number_input("Number of Clinics in Region", min_value=1, value=5, step=1)
    with fp2:
        avg_baseline_fte_per_clinic = st.number_input("Avg Baseline Provider FTE per Clinic", min_value=0.5, value=float(baseline_provider_fte), step=0.1)
    with fp3:
        turnover_rate_region = st.number_input("Regional Provider Turnover %", min_value=0.0, max_value=100.0, value=float(provider_turnover*100), step=1.0) / 100

    region_departures_fte = num_clinics * avg_baseline_fte_per_clinic * turnover_rate_region
    region_leakage_fte = region_departures_fte * (time_to_replace_days / 365)

    region_float_fte_needed = region_leakage_fte
    region_float_providers_needed = region_float_fte_needed / max(fte_per_hire, 0.25)

    fk1, fk2, fk3 = st.columns(3)
    fk1.metric("Regional Departures (FTE/year)", f"{region_departures_fte:.2f}")
    fk2.metric("Regional Leakage Buffer (FTE)", f"{region_leakage_fte:.2f}")
    fk3.metric("Float Providers Needed", f"{region_float_providers_needed:.2f}")

    # ============================================================
    # ✅ Downloadable Float Pool Staffing Plan (NEW)
    # ============================================================
    st.subheader("Downloadable Float Pool Staffing Plan")

    float_plan_df = pd.DataFrame({
        "Clinic": [f"Clinic {i+1}" for i in range(int(num_clinics))],
        "Baseline Provider FTE": [avg_baseline_fte_per_clinic] * int(num_clinics),
        "Expected Coverage Leakage (FTE)": [(avg_baseline_fte_per_clinic * turnover_rate_region) * (time_to_replace_days / 365)] * int(num_clinics)
    })

    st.download_button(
        label="Download Float Pool Staffing Plan (CSV)",
        data=float_plan_df.to_csv(index=False),
        file_name="float_pool_staffing_plan.csv",
        mime="text/csv"
    )

    # ============================================================
    # ✅ A8.1 Float Pool ROI + Hybrid Slider + Confidence Interval (NEW)
    # ============================================================
    st.markdown("---")
    st.header("A8.1 — Float Pool ROI Comparison (EBITDA Impact)")
    st.caption("Compares full staffing vs float pool vs hybrid mix.")

    # ---- Financial Inputs ----
    st.subheader("Financial Inputs")
    f1, f2, f3 = st.columns(3)

    with f1:
        net_revenue_per_visit = st.number_input("Net Revenue per Visit ($)", min_value=0.0, value=180.0, step=10.0)
    with f2:
        contribution_margin_pct = st.number_input("Contribution Margin (%)", min_value=0.0, max_value=100.0, value=35.0, step=1.0) / 100
    with f3:
        loaded_cost_per_provider_fte = st.number_input("Loaded Cost per Provider FTE (Annual $)", min_value=0.0, value=230000.0, step=10000.0)

    margin_per_visit = net_revenue_per_visit * contribution_margin_pct
    annual_visits = visits * 365
    annual_net_revenue = annual_visits * net_revenue_per_visit

    # ---- Turnover Cost Confidence Interval ----
    st.subheader("Turnover Cost Confidence Interval")
    tc1, tc2 = st.columns(2)
    with tc1:
        turnover_cost_low = st.number_input("Turnover Cost Low ($)", min_value=0.0, value=100000.0, step=5000.0)
    with tc2:
        turnover_cost_high = st.number_input("Turnover Cost High ($)", min_value=0.0, value=200000.0, step=5000.0)

    turnover_cost_mid = (turnover_cost_low + turnover_cost_high) / 2

    # ---- Premium Labor Inputs ----
    st.subheader("Premium Labor (Optional)")
    use_premium_labor = st.checkbox("Include premium labor exposure on shortage days", value=False)

    premium_pct = 0.0
    provider_day_cost_basis = 0.0
    if use_premium_labor:
        p1, p2 = st.columns(2)
        with p1:
            premium_pct = st.number_input("Premium Pay Factor (%)", min_value=0.0, max_value=200.0, value=25.0, step=5.0) / 100
        with p2:
            provider_day_cost_basis = st.number_input("Provider Day Cost Basis ($)", min_value=0.0, value=float(loaded_cost_per_provider_fte / 260), step=50.0)

    # ---- Hybrid Slider ----
    st.subheader("Hybrid Strategy Slider (Fixed Staffing + Float Pool Mix)")
    hybrid_slider = st.slider(
        "Hybrid Mix: % of Gap Addressed via Fixed Hiring (rest via float coverage)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05
    )

    # ---- Float Assumptions ----
    st.subheader("Float Pool Assumptions")
    fp1, fp2, fp3 = st.columns(3)
    with fp1:
        float_effectiveness = st.number_input("Float Effectiveness (% of Gap Covered)", min_value=0.0, max_value=100.0, value=75.0, step=5.0) / 100
    with fp2:
        float_cost_fte = st.number_input("Loaded Cost per Float Provider FTE ($)", min_value=0.0, value=loaded_cost_per_provider_fte, step=10000.0)
    with fp3:
        float_overhead_pct = st.number_input("Float Program Overhead (%)", min_value=0.0, max_value=30.0, value=8.0, step=1.0) / 100

    float_program_cost_annual = region_float_fte_needed * float_cost_fte * (1 + float_overhead_pct)

    # ---- Gap Provider Days (Recommended vs Supply) ----
    gap_provider_days_total = provider_day_gap(protective_curve, realistic_supply_recommended, days_in_month)

    # ============================================================
    # ✅ Strategy Modeling
    # ============================================================
    def exposure_from_gap(gap_provider_days, turnover_cost):
        lost_visits = gap_provider_days * (visits / max(baseline_provider_fte, 0.25)) * 0.60
        lost_margin = lost_visits * margin_per_visit

        premium_exposure = 0.0
        if use_premium_labor:
            premium_exposure = gap_provider_days * provider_day_cost_basis * premium_pct

        expected_departures = baseline_provider_fte * provider_turnover
        turnover_exposure = expected_departures * turnover_cost

        return turnover_exposure + lost_margin + premium_exposure

    # Strategy A — full staffing (assume eliminates most gap)
    A_gap = gap_provider_days_total * 0.10
    A_cost = sum([(max(r - l, 0) * loaded_cost_per_provider_fte * (dim / 365)) for r, l, dim in zip(protective_curve, provider_base_demand, days_in_month)])
    A_exposure = exposure_from_gap(A_gap, turnover_cost_mid)
    A_net = -A_cost - A_exposure

    # Strategy B — lean + float
    B_gap = gap_provider_days_total * (1 - float_effectiveness)
    B_cost = float_program_cost_annual
    B_exposure = exposure_from_gap(B_gap, turnover_cost_mid)
    B_net = -B_cost - B_exposure

    # Strategy C — hybrid
    fixed_gap_reduction = hybrid_slider * 0.80
    float_gap_reduction = (1 - hybrid_slider) * float_effectiveness

    C_gap = gap_provider_days_total * (1 - fixed_gap_reduction - float_gap_reduction)
    C_cost = (hybrid_slider * A_cost) + ((1 - hybrid_slider) * float_program_cost_annual)
    C_exposure = exposure_from_gap(C_gap, turnover_cost_mid)
    C_net = -C_cost - C_exposure

    # ============================================================
    # ✅ Output Metrics
    # ============================================================
    st.subheader("Strategy Comparison (Annualized)")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Strategy A Net EBITDA", f"${A_net:,.0f}")
        st.caption("Staff each clinic to Recommended Target")
    with s2:
        st.metric("Strategy B Net EBITDA", f"${B_net:,.0f}")
        st.caption("Lean + Float Pool Coverage")
    with s3:
        st.metric("Strategy C Net EBITDA", f"${C_net:,.0f}")
        st.caption("Hybrid Mix")

    # Burnout risk reduction months (NEW)
    burnout_months_after_float = int(np.ceil(burnout_months * (1 - float_effectiveness)))
    st.info(
        f"""
✅ **Operational Impact Summary**
- Baseline burnout exposure months: **{burnout_months}**
- Burnout exposure months after float coverage: **~{burnout_months_after_float}**
- Float program cost: **${float_program_cost_annual:,.0f}/year**
"""
    )
else:
    st.info("Enter inputs above and click **Calculate Staffing**.")
