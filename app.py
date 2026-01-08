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
# ✅ Helper Functions (Single Source of Truth)
# ============================================================
def base_seasonality_multiplier(month: int):
    """Simple seasonal curve (winter high, summer low)."""
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00


def build_flu_window(current_year: int, flu_start_month: int, flu_end_month: int):
    """Return flu_start_date, flu_end_date; supports year wrap (e.g. Dec → Feb)."""
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
    """Months elapsed between d and anchor (0 if same month)."""
    return (d.year - anchor.year) * 12 + (d.month - anchor.month)


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def compute_seasonality_forecast(dates, baseline_visits, flu_start, flu_end, flu_uplift_pct):
    """
    Returns normalized monthly visit/day forecast.
    Baseline visits is treated as annual average. Seasonal redistribution keeps avg stable.
    """
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
    """
    Convert visits forecast into provider FTE demand curve.
    Demand is never allowed below provider_min_floor.
    """
    demand = []
    for v in visits_by_month:
        fte = model.calculate_fte_needed(
            visits_per_day=v,
            hours_of_operation_per_week=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week,
        )["provider_fte"]
        demand.append(max(fte, provider_min_floor))
    return demand


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
    """
    Burnout-protective curve = base demand + buffers + smoothing.
    Buffers:
      - Volatility buffer (CV)
      - Spike buffer (above P75)
      - Recovery debt buffer (fatigue accumulation)
    burnout_slider ∈ [0,1] scales total protection.
    """

    vol_w, spike_w, debt_w = weights

    visits_arr = np.array(visits_by_month)
    mean_visits = np.mean(visits_arr)
    std_visits = np.std(visits_arr)
    cv = (std_visits / mean_visits) if mean_visits > 0 else 0.0

    p75 = np.percentile(visits_arr, 75)

    # Recovery Debt Index (RDI)
    rdi = 0.0
    decay = 0.85
    lambda_debt = 0.10

    protective_curve = []
    prev_staff = max(base_demand_fte[0], provider_min_floor)

    for v, base_fte in zip(visits_by_month, base_demand_fte):

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

    return protective_curve


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
    """
    Best-case realistic staffing supply path:
    - starts at baseline
    - moves toward target within ramp limits
    - attrition starts after notice_days from today
    - never below provider_min_floor
    """
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
            planned = max(planned - months_elapsed * monthly_attrition_fte, provider_min_floor)

        staff.append(max(planned, provider_min_floor))
        prev = staff[-1]

    return staff


def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    cost = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += dfte * loaded_cost_per_provider_fte * (dim / 365)
    return cost


def provider_day_gap(target_curve, supply_curve, days_in_month):
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(t - s, 0) * dim
    return gap_days


def fte_to_provider_hours_per_day(fte_curve, fte_hours_per_week):
    return [(fte * fte_hours_per_week) / 7 for fte in fte_curve]


def provider_hours_to_shifts(provider_hours, shift_length_hours):
    return [hrs / shift_length_hours for hrs in provider_hours]


# ============================================================
# ✅ Page Setup
# ============================================================
st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")
st.title("Predictive Staffing Model (PSM)")
st.caption("Seasonality staffing + burnout protection + realism + finance case.")

model = StaffingModel()


# ============================================================
# ✅ Baseline Inputs
# ============================================================
st.markdown("## Baseline Inputs")

visits = st.number_input("Average Visits per Day (Annual Average)", min_value=1.0, value=45.0, step=1.0)
hours_of_operation = st.number_input("Hours of Operation per Week", min_value=1.0, value=70.0, step=1.0)
fte_hours_per_week = st.number_input("FTE Hours per Week", min_value=1.0, value=40.0, step=1.0)

provider_min_floor = st.number_input("Provider Minimum Floor (FTE)", min_value=0.25, value=1.00, step=0.25)

burnout_slider = st.slider("Burnout Protection Level", 0.0, 1.0, 0.6, 0.05)
safe_visits_per_provider = st.number_input("Safe Visits per Provider/Day Threshold", min_value=10, max_value=40, value=20)

provider_turnover = st.number_input("Provider Turnover %", value=24.0, step=1.0) / 100

flu_start_month = st.selectbox("Flu Start Month", options=list(range(1, 13)), index=11, format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
flu_end_month = st.selectbox("Flu End Month", options=list(range(1, 13)), index=1, format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100

with st.expander("Provider Pipeline Assumptions", expanded=False):
    days_to_sign = st.number_input("Days to Sign", min_value=1, value=90, step=5)
    days_to_credential = st.number_input("Days to Credential", min_value=1, value=90, step=5)
    onboard_train_days = st.number_input("Onboard/Train Days", min_value=0, value=30, step=5)
    coverage_buffer_days = st.number_input("Planning Buffer Days", min_value=0, value=14, step=1)
    notice_days = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=75, step=5)


# ============================================================
# ✅ Calculate Button
# ============================================================
if not st.button("Calculate Staffing"):
    st.stop()

fte_result = model.calculate_fte_needed(
    visits_per_day=visits,
    hours_of_operation_per_week=hours_of_operation,
    fte_hours_per_week=fte_hours_per_week,
)

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

baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

st.markdown("---")
st.subheader("Baseline FTEs Needed")
st.dataframe(fte_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ Calendar-Year Timeline + Forecast
# ============================================================
current_year = today.year
dates = pd.date_range(start=datetime(current_year, 1, 1), periods=12, freq="MS")
month_labels = [d.strftime("%b") for d in dates]
days_in_month = [pd.Period(d, "M").days_in_month for d in dates]
flu_start_date, flu_end_date = build_flu_window(current_year, flu_start_month, flu_end_month)

forecast_visits_by_month = compute_seasonality_forecast(
    dates=dates,
    baseline_visits=visits,
    flu_start=flu_start_date,
    flu_end=flu_end_date,
    flu_uplift_pct=flu_uplift_pct
)

provider_base_demand = visits_to_provider_demand(
    model=model,
    visits_by_month=forecast_visits_by_month,
    hours_of_operation=hours_of_operation,
    fte_hours_per_week=fte_hours_per_week,
    provider_min_floor=provider_min_floor
)

protective_curve = burnout_protective_staffing_curve(
    dates=dates,
    visits_by_month=forecast_visits_by_month,
    base_demand_fte=provider_base_demand,
    provider_min_floor=provider_min_floor,
    burnout_slider=burnout_slider,
    safe_visits_per_provider_per_day=safe_visits_per_provider
)

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
# ✅ A6 Executive Plot
# ============================================================
st.markdown("---")
st.subheader("A6 — Executive View: Volume, Targets, Supply, Burnout Exposure")

fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(dates, provider_base_demand, linestyle=":", linewidth=2, label="Lean Target (Volume Only)")
ax1.plot(dates, protective_curve, linewidth=3.5, marker="o", label="Recommended Target (Burnout-Protective)")
ax1.plot(dates, realistic_supply_recommended, linewidth=3, marker="o", label="Realistic Supply")

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
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

plt.tight_layout()
st.pyplot(fig)


# ============================================================
# ✅ A8.1 Float Pool ROI Comparison (FIXED)
# Uses realistic_supply_lean and realistic_supply_recommended (NO undefined vars)
# ============================================================
st.markdown("---")
st.header("A8.1 — Float Pool ROI Comparison (EBITDA Impact)")
st.caption("Compares (A) staffing to recommended target vs (B) lean staffing + float pool coverage.")

st.subheader("Float Pool Coverage Assumptions")

fp1, fp2, fp3 = st.columns(3)

with fp1:
    float_coverage_effectiveness = st.number_input("Float Coverage Effectiveness (% of Gap Covered)", 0.0, 100.0, 75.0, 5.0) / 100

with fp2:
    float_loaded_cost_fte = st.number_input("Loaded Cost per Float Provider FTE (Annual $)", 0.0, 230000.0, 155000.0, 5000.0)

with fp3:
    float_admin_overhead_pct = st.number_input("Float Program Overhead (% of Float Cost)", 0.0, 30.0, 8.0, 1.0) / 100

num_clinics = st.number_input("Number of Clinics in Region", min_value=1, value=5, step=1)

time_to_replace_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days
coverage_leakage_fte = baseline_provider_fte * provider_turnover * (time_to_replace_days / 365)

region_float_fte_needed = num_clinics * coverage_leakage_fte

float_program_cost_annual = region_float_fte_needed * float_loaded_cost_fte * (1 + float_admin_overhead_pct)

gap_provider_days_total = provider_day_gap(protective_curve, realistic_supply_lean, days_in_month)
gap_after_float = gap_provider_days_total * (1 - float_coverage_effectiveness)

st.metric("Total Gap Provider-Days (Lean vs Recommended)", f"{gap_provider_days_total:,.1f}")
st.metric("Gap After Float Pool Coverage", f"{gap_after_float:,.1f}")
st.metric("Estimated Float Program Cost (Annual)", f"${float_program_cost_annual:,.0f}")

st.success("✅ A8.1 executed without NameErrors. This block is now stable.")
