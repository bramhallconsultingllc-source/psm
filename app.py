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

    volatility_buffer = []
    spike_buffer = []
    recovery_debt_buffer = []

    for d, v, base_fte in zip(dates, visits_by_month, base_demand_fte):

        # (1) volatility buffer
        vbuf = base_fte * cv

        # (2) spike buffer
        sbuf = max(0.0, (v - p75) / mean_visits) * base_fte if mean_visits > 0 else 0

        # (3) recovery debt buffer
        visits_per_provider = v / max(prev_staff, 0.25)
        debt = max(0.0, visits_per_provider - safe_visits_per_provider_per_day)
        rdi = decay * rdi + debt
        dbuf = lambda_debt * rdi

        # weighted protection buffer
        buffer_fte = burnout_slider * (
            vol_w * vbuf +
            spike_w * sbuf +
            debt_w * dbuf
        )

        raw_target = max(provider_min_floor, base_fte + buffer_fte)

        # anti-whiplash smoothing
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

        # move toward target (ramp constrained)
        delta = target - prev
        if delta > 0:
            delta = clamp(delta, 0.0, max_hiring_up_per_month)
        else:
            delta = clamp(delta, -max_ramp_down_per_month, 0.0)

        planned = prev + delta

        # apply attrition after notice lag
        if d >= effective_attrition_start:
            months_elapsed = monthly_index(d, effective_attrition_start)
            attrition_loss = months_elapsed * monthly_attrition_fte
            planned = max(planned - attrition_loss, provider_min_floor)

        planned = max(planned, provider_min_floor)

        staff.append(planned)
        prev = planned

    return staff


# ============================================================
# ✅ Page Setup
# ============================================================
st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")
st.title("Predictive Staffing Model (PSM)")
st.caption("Seasonality staffing + burnout protection + realism + finance case.")

st.info(
    "⚠️ **All daily staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)

model = StaffingModel()


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
# ✅ Turnover Assumptions (Provider only matters for A6 + ROI)
# ============================================================
st.markdown("## Turnover Assumptions")
provider_turnover = st.number_input("Provider Turnover %", value=24.0, step=1.0) / 100


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
# ✅ Provider Pipeline Inputs
# ============================================================
st.markdown("## Provider Hiring Glidepath Inputs")

with st.expander("Provider Pipeline Assumptions", expanded=False):
    days_to_sign = st.number_input("Days to Sign (Req → Signed Offer)", min_value=1, value=90, step=5)
    days_to_credential = st.number_input("Days to Credential (Signed → Credentialed)", min_value=1, value=90, step=5)
    onboard_train_days = st.number_input("Onboard/Train Days (Credentialed → Solo)", min_value=0, value=30, step=5)
    coverage_buffer_days = st.number_input("Buffer Days (Planning Margin)", min_value=0, value=14, step=1)
    notice_days = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=75, step=5)


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
        {"Role": ["Provider", "PSR", "MA", "XRT", "TOTAL"],
         "FTE Needed": [
             round(fte_result["provider_fte"], 2),
             round(fte_result["psr_fte"], 2),
             round(fte_result["ma_fte"], 2),
             round(fte_result["xrt_fte"], 2),
             round(fte_result["total_fte"], 2),
         ]}
    )

    st.session_state["fte_df"] = fte_df


# ============================================================
# ✅ MAIN OUTPUT
# ============================================================
if st.session_state.get("fte_df") is None:
    st.info("Enter inputs above and click **Calculate Staffing** to generate outputs.")
    st.stop()

fte_result = st.session_state["fte_result"]
fte_df = st.session_state["fte_df"]

baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

st.markdown("---")
st.subheader("Baseline FTEs Needed")
st.dataframe(fte_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ Calendar-Year Timeline
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
st.caption("Seasonality redistributes volume while keeping the annual average equal to baseline visits/day.")

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
# ✅ Demand Curve (Lean) + Recommended Curve (Protective)
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
# ✅ Realistic Supply Curves (Lean + Recommended)
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
# ✅ Daily Staffing View (exec-friendly)
# ============================================================
def fte_to_provider_shifts_per_day(fte_curve, fte_hours_per_week, hours_of_operation_per_week):
    """
    Converts provider FTE into shifts/day equivalent (simple planning translation).
    - FTE hours/week / clinic hours/week = providers per hour coverage
    - Multiply by daily hours (hours/week / 7) -> providers per day
    """
    daily_hours = hours_of_operation_per_week / 7
    shift_equiv = []
    for fte in fte_curve:
        providers_per_hour = (fte * fte_hours_per_week) / max(hours_of_operation_per_week, 1)
        providers_per_day = providers_per_hour * daily_hours
        shift_equiv.append(providers_per_day)
    return shift_equiv


recommended_daily_providers = fte_to_provider_shifts_per_day(
    protective_curve, fte_hours_per_week, hours_of_operation
)
realistic_daily_providers = fte_to_provider_shifts_per_day(
    realistic_supply_recommended, fte_hours_per_week, hours_of_operation
)


# ============================================================
# ✅ A6 Executive Graph (Dual Axis + Burnout Exposure)
# ============================================================
st.markdown("---")
st.subheader("A6 — Executive View: Volume, Staffing Target, Supply, Burnout Exposure")
st.caption(
    "Shows forecasted volumes (right axis), staffing targets (lean vs recommended), realistic staffing supply, "
    "and burnout exposure where supply falls below the recommended target."
)

# Hiring pipeline markers
staffing_needed_by = flu_start_date
total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days
req_post_date = staffing_needed_by - timedelta(days=total_lead_days)
signed_date = req_post_date + timedelta(days=days_to_sign)
credentialed_date = signed_date + timedelta(days=days_to_credential)
solo_ready_date = credentialed_date + timedelta(days=onboard_train_days)

fig, ax1 = plt.subplots(figsize=(12, 4))

# Left axis: FTE curves
ax1.plot(dates, provider_base_demand, linestyle=":", linewidth=2, label="Lean Target (Volume Only)")
ax1.plot(dates, protective_curve, linewidth=3.5, marker="o", label="Recommended Target (Burnout-Protective)")
ax1.plot(dates, realistic_supply_recommended, linewidth=3, marker="o", label="Best-Case Realistic Staffing (Supply)")

# Burnout exposure shading
ax1.fill_between(
    dates,
    realistic_supply_recommended,
    protective_curve,
    where=np.array(protective_curve) > np.array(realistic_supply_recommended),
    alpha=0.25,
    label="Burnout Exposure Zone"
)

ax1.set_title("Provider Seasonality + Recommended Staffing Plan")
ax1.set_ylabel("Provider FTE")
ax1.set_ylim(0, max(protective_curve) + 1.5)
ax1.set_xticks(dates)
ax1.set_xticklabels(month_labels)
ax1.grid(axis="y", linestyle=":", alpha=0.35)

# Right axis: volume
ax2 = ax1.twinx()
ax2.plot(dates, forecast_visits_by_month, linestyle="-.", linewidth=2.5, label="Forecasted Volume (Visits/Day)")
ax2.set_ylabel("Visits / Day")

# Hiring markers (only if within window)
ymax = ax1.get_ylim()[1]
chart_min = dates[0].to_pydatetime()
chart_max = dates[-1].to_pydatetime()

for marker_date, label in [(req_post_date, "Req Post By"), (signed_date, "Sign By"), (solo_ready_date, "Solo By")]:
    if chart_min <= marker_date <= chart_max:
        ax1.axvline(marker_date, linestyle="--", linewidth=1.5, alpha=0.6)
        ax1.annotate(label, xy=(marker_date, ymax), xytext=(marker_date, ymax + 0.25),
                     ha="center", fontsize=9, rotation=90)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, -0.22), ncol=2)

plt.tight_layout()
st.pyplot(fig)

# Executive staffing KPIs
burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
k1, k2, k3 = st.columns(3)
k1.metric("Peak Burnout Gap (FTE)", f"{max(burnout_gap_fte):.2f}")
k2.metric("Avg Burnout Gap (FTE)", f"{np.mean(burnout_gap_fte):.2f}")
k3.metric("Months Exposed", f"{sum([1 for g in burnout_gap_fte if g > 0])}/12")


# ============================================================
# ✅ Daily Staffing Table (Executives want this)
# ============================================================
st.markdown("### Daily Staffing (Recommended vs Realistic Supply)")
daily_staff_df = pd.DataFrame({
    "Month": month_labels,
    "Recommended Target (Providers/Day)": np.round(recommended_daily_providers, 2),
    "Realistic Supply (Providers/Day)": np.round(realistic_daily_providers, 2),
    "Burnout Exposure (Providers/Day)": np.round(np.maximum(np.array(recommended_daily_providers) - np.array(realistic_daily_providers), 0), 2),
})
st.dataframe(daily_staff_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ Staffing Investment Case (ROI) — Scenario Based
# ============================================================
st.markdown("---")
st.header("Staffing Investment Case (EBITDA Impact — Clinic Only)")
st.caption("Scenario-based ROI comparing Lean vs Recommended staffing strategies.")

# Time horizon selector
time_horizon = st.radio("Display ROI impacts as:", ["Annual", "Quarterly", "Monthly"], horizontal=True, index=0)
horizon_factor = 1.0 if time_horizon == "Annual" else (1.0 / 4 if time_horizon == "Quarterly" else 1.0 / 12)


# Financial inputs
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
annual_margin = annual_visits * margin_per_visit

st.caption(f"Contribution Margin per Visit = ${margin_per_visit:,.2f}")


# Turnover cost builder
st.subheader("Turnover Cost Builder (Per Provider Event)")
turnover_role = st.selectbox("Provider Type", ["APP", "Physician"], index=0)

if turnover_role == "APP":
    default_recruiting, default_signing, default_admin, default_ramp_days, default_ramp_productivity = 20000, 10000, 7000, 60, 0.70
else:
    default_recruiting, default_signing, default_admin, default_ramp_days, default_ramp_productivity = 40000, 25000, 10000, 90, 0.65

t1, t2, t3 = st.columns(3)

with t1:
    admin_cost = st.number_input("Admin / Separation Cost ($)", min_value=0.0, value=float(default_admin), step=1000.0)
    recruiting_cost = st.number_input("Recruiting / Sourcing Cost ($)", min_value=0.0, value=float(default_recruiting), step=2000.0)

with t2:
    signing_bonus = st.number_input("Signing / Incentive Cost ($)", min_value=0.0, value=float(default_signing), step=2000.0)
    disruption_pct = st.number_input("Patient Experience / Disruption Cost (% of Annual Margin)", min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100

with t3:
    vacancy_loss_factor = st.number_input("Vacancy Loss Factor (0–1)", min_value=0.0, max_value=1.0, value=0.80, step=0.05)
    ramp_days = st.number_input("Ramp-Up Days After Start", min_value=0, value=int(default_ramp_days), step=10)
    ramp_productivity = st.number_input("Ramp-Up Productivity (0–1)", min_value=0.10, max_value=1.00, value=float(default_ramp_productivity), step=0.05)

vacancy_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days
vacancy_margin_loss = vacancy_days * visits * margin_per_visit * vacancy_loss_factor
ramp_margin_loss = ramp_days * visits * margin_per_visit * (1 - ramp_productivity)
disruption_cost = annual_margin * disruption_pct

turnover_cost_total = admin_cost + recruiting_cost + signing_bonus + vacancy_margin_loss + ramp_margin_loss + disruption_cost
st.metric("Estimated Fully Loaded Turnover Cost (Per Provider Event)", f"${turnover_cost_total:,.0f}")


# Premium labor optional
st.subheader("Premium Labor Costs (Optional)")
use_premium_labor = st.checkbox("Include premium labor / extra shift costs on shortage days", value=False)
premium_pct = 0.0
provider_day_cost_basis = 0.0

if use_premium_labor:
    p1, p2 = st.columns(2)
    with p1:
        premium_pct = st.number_input("Premium Pay Factor (%)", min_value=0.0, max_value=200.0, value=25.0, step=5.0) / 100
    with p2:
        provider_day_cost_basis = st.number_input("Provider Day Cost Basis ($)", min_value=0.0, value=float(loaded_cost_per_provider_fte / 260), step=50.0)


# Behavioral assumptions
st.subheader("Behavioral Assumptions (Protection → Outcomes)")
a1, a2, a3 = st.columns(3)

with a1:
    max_turnover_reduction = st.number_input("Max Turnover Reduction at Protection = 1.0 (%)", min_value=0.0, max_value=100.0, value=35.0, step=5.0) / 100
with a2:
    max_productivity_uplift = st.number_input("Max Productivity Uplift at Protection = 1.0 (%)", min_value=0.0, max_value=30.0, value=6.0, step=1.0) / 100
with a3:
    leakage_factor = st.number_input("Demand Leakage Factor (0–1)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)


# Cost helper functions
def annualize_monthly_fte_cost(delta_fte_curve):
    cost = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += dfte * loaded_cost_per_provider_fte * (dim / 365)
    return cost


def provider_day_gap(target_curve, supply_curve):
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(t - s, 0) * dim
    return gap_days


# ------------------------------------------------------------
# ✅ Scenario-based exposure computation
# ------------------------------------------------------------
def compute_exposure_annual(target_curve, supply_curve, turnover_rate):
    provider_count = baseline_provider_fte
    expected_departures = provider_count * turnover_rate
    turnover_exposure = expected_departures * turnover_cost_total

    gap_provider_days = provider_day_gap(target_curve, supply_curve)
    visits_per_provider_fte_per_day = visits / max(baseline_provider_fte, 0.25)
    lost_visits = gap_provider_days * visits_per_provider_fte_per_day * leakage_factor
    lost_margin = lost_visits * margin_per_visit

    premium_exposure = 0.0
    if use_premium_labor:
        premium_exposure = gap_provider_days * provider_day_cost_basis * premium_pct

    total_exposure = turnover_exposure + lost_margin + premium_exposure

    return {
        "total_exposure": total_exposure,
        "turnover_exposure": turnover_exposure,
        "lost_margin": lost_margin,
        "premium_exposure": premium_exposure,
        "gap_provider_days": gap_provider_days,
        "expected_departures": expected_departures
    }


# A) incremental staffing investment
delta_fte_curve = [max(r - l, 0) for r, l in zip(protective_curve, provider_base_demand)]
incremental_staffing_cost_annual = annualize_monthly_fte_cost(delta_fte_curve)

# B) exposure under lean
exposure_lean = compute_exposure_annual(provider_base_demand, realistic_supply_lean, provider_turnover)
expected_cost_exposure_not_staffed_annual = exposure_lean["total_exposure"]

# recommended turnover and productivity adjustments
turnover_recommended = provider_turnover * (1 - max_turnover_reduction * burnout_slider)
productivity_uplift = max_productivity_uplift * burnout_slider

# exposure under recommended strategy
exposure_recommended = compute_exposure_annual(protective_curve, realistic_supply_recommended, turnover_recommended)
expected_cost_exposure_recommended_annual = exposure_recommended["total_exposure"]

# C) savings / gains if staffed
exposure_avoided_annual = max(expected_cost_exposure_not_staffed_annual - expected_cost_exposure_recommended_annual, 0)
productivity_margin_uplift_annual = annual_visits * productivity_uplift * margin_per_visit
expected_savings_if_staffed_annual = exposure_avoided_annual + productivity_margin_uplift_annual

# Net EBITDA
net_ebitda_impact_annual = expected_savings_if_staffed_annual - incremental_staffing_cost_annual
net_ebitda_margin_impact_annual = (net_ebitda_impact_annual / annual_net_revenue) if annual_net_revenue > 0 else 0.0

# Driver breakdown
turnover_savings_annual = max(exposure_lean["turnover_exposure"] - exposure_recommended["turnover_exposure"], 0)
recovered_margin_from_staffing_annual = max(exposure_lean["lost_margin"] - exposure_recommended["lost_margin"], 0)
premium_labor_avoided_annual = max(exposure_lean["premium_exposure"] - exposure_recommended["premium_exposure"], 0)

# Display scaling
A_display = incremental_staffing_cost_annual * horizon_factor
B_display = expected_cost_exposure_not_staffed_annual * horizon_factor
B_rec_display = expected_cost_exposure_recommended_annual * horizon_factor
C_display = expected_savings_if_staffed_annual * horizon_factor
NET_display = net_ebitda_impact_annual * horizon_factor
exposure_avoided_display = exposure_avoided_annual * horizon_factor


# ============================================================
# ✅ Executive Summary (A/B/C/Net)
# ============================================================
st.markdown("## Executive Summary (A / B / C / Net)")
st.caption(f"All values shown reflect **{time_horizon.upper()} impact** (annual totals calculated first, then scaled).")

cA, cB, cC, cN = st.columns(4)
cA.metric(f"A) Cost to Staff to Recommended Target ({time_horizon})", f"${A_display:,.0f}")
cB.metric(f"B) Expected Cost Exposure if Operating Lean ({time_horizon})", f"${B_display:,.0f}")
cC.metric(f"C) Expected Savings / Gains if Staffed ({time_horizon})", f"${C_display:,.0f}")
cN.metric(f"Net EBITDA Impact ({time_horizon})", f"${NET_display:,.0f}", f"{net_ebitda_margin_impact_annual*100:.2f}% annual margin")

st.caption(
    f"Annual totals: A=${incremental_staffing_cost_annual:,.0f} | "
    f"B=${expected_cost_exposure_not_staffed_annual:,.0f} | "
    f"Recommended Exposure=${expected_cost_exposure_recommended_annual:,.0f} | "
    f"C=${expected_savings_if_staffed_annual:,.0f} | "
    f"Net=${net_ebitda_impact_annual:,.0f}"
)

st.info(
    f"""
✅ **Interpretation**
- **Lean strategy** = volume-only targets + realistic hiring/attrition constraints.
- **Recommended strategy** = burnout-protective targets + realistic constraints.
- **A** is the incremental staffing investment to move from lean targets to recommended targets.
- **B** is the expected annual cost exposure under the lean strategy (turnover + margin loss + optional premium labor).
- **C** is the expected avoided exposure from moving to recommended staffing plus productivity gains.
- **Net EBITDA** = C − A.
"""
)

with st.expander("Show monthly comparison (Targets + Supply)", expanded=False):
    detail_df = pd.DataFrame({
        "Month": month_labels,
        "Lean Target FTE": np.round(provider_base_demand, 2),
        "Recommended Target FTE": np.round(protective_curve, 2),
        "Lean Supply FTE (Realistic)": np.round(realistic_supply_lean, 2),
        "Recommended Supply FTE (Realistic)": np.round(realistic_supply_recommended, 2),
        "Incremental FTE (Rec - Lean)": np.round(delta_fte_curve, 2),
    })
    st.dataframe(detail_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ Executive Narrative (Scenario-based)
# ============================================================
turnover_savings_display = turnover_savings_annual * horizon_factor
recovered_margin_display = recovered_margin_from_staffing_annual * horizon_factor
productivity_uplift_display = productivity_margin_uplift_annual * horizon_factor
premium_avoided_display = (premium_labor_avoided_annual * horizon_factor) if use_premium_labor else 0.0

if net_ebitda_impact_annual >= 0:
    headline_tone = "EBITDA-positive"
    recommendation = (
        "Proceed with staffing to the recommended burnout-protective target. "
        "Under current assumptions, this strategy improves EBITDA while reducing operational and clinical risk."
    )
elif net_ebitda_impact_annual > -5000:
    headline_tone = "near breakeven"
    recommendation = (
        "Proceed with staffing to the recommended target. The financial impact is near breakeven, "
        "and stability benefits (retention, fewer gaps, better patient experience) justify the small EBITDA drag."
    )
else:
    headline_tone = "EBITDA-negative"
    recommendation = (
        "Consider a targeted protection strategy rather than full staffing to the recommended target. "
        "Under current assumptions, the plan is EBITDA-negative. A PRN/extra-shift approach during peak months "
        "may achieve protection benefits with lower fixed cost."
    )

driver_pairs = [
    ("Turnover savings", turnover_savings_display),
    ("Recovered margin from fewer gaps", recovered_margin_display),
    ("Productivity uplift (margin)", productivity_uplift_display),
]
if use_premium_labor:
    driver_pairs.append(("Premium labor avoided", premium_avoided_display))

driver_pairs_sorted = sorted(driver_pairs, key=lambda x: x[1], reverse=True)
top_drivers = driver_pairs_sorted[:3]
top_driver_text = "\n".join([f"- **{name}:** ${value:,.0f}" for name, value in top_drivers])

exec_narrative = f"""
### Executive Narrative (Staffing Investment Case)

**Headline:** Staffing to the recommended burnout-protective target is **{headline_tone}** under current assumptions.  
- **Net EBITDA impact ({time_horizon}):** **${NET_display:,.0f}** *(={net_ebitda_margin_impact_annual*100:.2f}% annual margin)*

---

### Investment vs Exposure ({time_horizon} view)

- **A) Incremental staffing investment:** **${A_display:,.0f}**
- **B) Expected cost exposure if operating Lean:** **${B_display:,.0f}**
- **Exposure under Recommended strategy:** **${B_rec_display:,.0f}**
- **Exposure avoided by staffing to Recommended:** **${exposure_avoided_display:,.0f}**
- **Total expected savings / gains (avoided exposure + productivity uplift):** **${C_display:,.0f}**

---

### Primary value drivers (largest contributors)
{top_driver_text}

---

### Recommendation
{recommendation}
"""

st.markdown(exec_narrative)


# ============================================================
# ✅ Waterfall EBITDA Bridge (CFO View)
# ============================================================
st.markdown("### EBITDA Impact Bridge (Waterfall View)")
st.caption("Avoided exposure + productivity uplift − staffing investment = net EBITDA impact.")

wf_exposure_avoided = exposure_avoided_annual * horizon_factor
wf_productivity = productivity_margin_uplift_annual * horizon_factor
wf_staffing_investment = -incremental_staffing_cost_annual * horizon_factor
wf_net = net_ebitda_impact_annual * horizon_factor

labels = [
    "Exposure Avoided",
    "Productivity Uplift",
    "Staffing Investment",
    "Net EBITDA Impact"
]
values = [wf_exposure_avoided, wf_productivity, wf_staffing_investment, wf_net]

starts = [0]
for v in values[:-1]:
    starts.append(starts[-1] + v)
starts[-1] = 0

fig, ax = plt.subplots(figsize=(10.5, 4))

for i in range(3):
    ax.bar(i, values[i], bottom=starts[i], alpha=0.85)
    ax.text(i, starts[i] + values[i] + (0.02 * max(abs(wf_net), 1)),
            f"${values[i]:,.0f}", ha="center", fontsize=10)

ax.bar(3, values[3], bottom=0, alpha=0.95)
ax.text(3, values[3] + (0.02 * max(abs(wf_net), 1)),
        f"${values[3]:,.0f}", ha="center", fontsize=11, fontweight="bold")

ax.axhline(0, linewidth=1, alpha=0.6)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(f"EBITDA Impact ({time_horizon})")
ax.set_title("EBITDA Bridge: Savings vs Investment")
ax.grid(axis="y", linestyle=":", alpha=0.35)

plt.tight_layout()
st.pyplot(fig)


# ============================================================
# ✅ CFO Breakdown Expander
# ============================================================
with st.expander("Show ROI scenario breakdown (Lean vs Recommended)", expanded=False):
    breakdown_df = pd.DataFrame({
        "Metric": [
            "Lean cost exposure",
            "Recommended cost exposure",
            "Exposure avoided (Lean - Recommended)",
            "Productivity uplift margin",
            "Total expected savings/gains (C)",
            "Incremental staffing investment (A)",
            "Net EBITDA impact"
        ],
        f"Value ({time_horizon})": [
            B_display,
            B_rec_display,
            exposure_avoided_display,
            productivity_uplift_display,
            C_display,
            A_display,
            NET_display
        ]
    })

    st.dataframe(
        breakdown_df.style.format({f"Value ({time_horizon})": "${:,.0f}"}),
        hide_index=True,
        use_container_width=True
    )

# ============================================================
# ✅ A7 — DAILY STAFFING VIEW (SHIFT + HOURS TRANSLATION)
# Executive-friendly operational staffing translation
# ============================================================
st.markdown("---")
st.subheader("A7 — Daily Staffing Schedule View (Recommended vs Realistic)")
st.caption(
    "Executives and operators need to understand what staffing targets *mean in real daily coverage*. "
    "This view translates FTE targets into provider-hours and shift equivalents."
)

# ------------------------------------------------------------
# ✅ Helper conversions (FTE → provider-hours/day → shifts/day)
# ------------------------------------------------------------
def fte_to_provider_hours_per_day(fte_curve, fte_hours_per_week):
    """Convert FTE into provider hours per day (annualized average)."""
    return [(fte * fte_hours_per_week) / 7 for fte in fte_curve]

def provider_hours_to_shifts(provider_hours, shift_length_hours):
    """Convert provider-hours/day into shift equivalents."""
    return [hrs / shift_length_hours for hrs in provider_hours]

# Provider hours/day curves
recommended_provider_hours = fte_to_provider_hours_per_day(protective_curve, fte_hours_per_week)
realistic_provider_hours = fte_to_provider_hours_per_day(realistic_supply_recommended, fte_hours_per_week)

# Shift conversions (8/10/12 hour shift equivalents)
rec_shifts_8  = provider_hours_to_shifts(recommended_provider_hours, 8)
rec_shifts_10 = provider_hours_to_shifts(recommended_provider_hours, 10)
rec_shifts_12 = provider_hours_to_shifts(recommended_provider_hours, 12)

sup_shifts_8  = provider_hours_to_shifts(realistic_provider_hours, 8)
sup_shifts_10 = provider_hours_to_shifts(realistic_provider_hours, 10)
sup_shifts_12 = provider_hours_to_shifts(realistic_provider_hours, 12)

# Burnout exposure in provider-hours/day
burnout_hours_gap = [max(r - s, 0) for r, s in zip(recommended_provider_hours, realistic_provider_hours)]

# ------------------------------------------------------------
# ✅ Daily staffing dataframe
# ------------------------------------------------------------
daily_staff_df = pd.DataFrame({
    "Month": month_labels,

    # Provider FTE view
    "Recommended Target (FTE)": np.round(protective_curve, 2),
    "Realistic Supply (FTE)": np.round(realistic_supply_recommended, 2),

    # Provider Hours/day view
    "Recommended Provider Hours/Day": np.round(recommended_provider_hours, 1),
    "Realistic Provider Hours/Day": np.round(realistic_provider_hours, 1),
    "Burnout Exposure Hours/Day": np.round(burnout_hours_gap, 1),

    # Shift equivalents (recommended target)
    "Rec Shifts/Day (8h)": np.round(rec_shifts_8, 2),
    "Rec Shifts/Day (10h)": np.round(rec_shifts_10, 2),
    "Rec Shifts/Day (12h)": np.round(rec_shifts_12, 2),

    # Shift equivalents (realistic supply)
    "Supply Shifts/Day (8h)": np.round(sup_shifts_8, 2),
    "Supply Shifts/Day (10h)": np.round(sup_shifts_10, 2),
    "Supply Shifts/Day (12h)": np.round(sup_shifts_12, 2),
})

st.dataframe(daily_staff_df, hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# ✅ Executive summary KPIs (daily view)
# ------------------------------------------------------------
peak_gap_hours = max(burnout_hours_gap)
avg_gap_hours = np.mean(burnout_hours_gap)
months_gap = sum([1 for g in burnout_hours_gap if g > 0])

k1, k2, k3 = st.columns(3)
k1.metric("Peak Burnout Exposure (Provider Hours/Day)", f"{peak_gap_hours:.1f}")
k2.metric("Avg Burnout Exposure (Provider Hours/Day)", f"{avg_gap_hours:.1f}")
k3.metric("Months w/ Daily Gap", f"{months_gap}/12")

# ------------------------------------------------------------
# ✅ Optional: shift-based gap framing (12-hour shifts)
# ------------------------------------------------------------
gap_shifts_12 = [g / 12 for g in burnout_hours_gap]
peak_gap_shifts = max(gap_shifts_12)

st.info(
    f"""
✅ **Executive Interpretation (Daily Staffing)**
- **Recommended staffing** translates to peak **{max(rec_shifts_12):.2f} twelve-hour shifts/day** during high season.
- **Realistic supply** peaks at **{max(sup_shifts_12):.2f} twelve-hour shifts/day** under current ramp + attrition constraints.
- Peak burnout exposure equals **{peak_gap_hours:.1f} provider-hours/day**, or about **{peak_gap_shifts:.2f} twelve-hour shifts/day**.
- This is the operational definition of burnout risk: **clinics must cover demand with fewer provider-hours than required**, leading to overtime, longer days, and diminished recovery capacity.
"""
)

with st.expander("Why shift equivalents matter", expanded=False):
    st.write("""
    Executives often approve staffing in FTE terms, but operations run in shifts and provider-hours.
    This view translates targets into real coverage requirements so leaders can understand:
    - how many provider shifts/day are needed,
    - what staffing patterns are required,
    - how much additional PRN or overtime coverage will be needed if supply cannot meet targets.
    """)

# ============================================================
# ✅ A7.2 — WEEKDAY/WEEKEND WEIGHTING + SCHEDULE TEMPLATE OUTPUT
# ============================================================
st.markdown("---")
st.subheader("A7.2 — Weekly Staffing Template (Weekday vs Weekend)")
st.caption(
    "Urgent care staffing is not truly “flat by day.” "
    "This section weights weekday vs weekend demand and generates a weekly staffing template."
)

# ------------------------------------------------------------
# ✅ Inputs
# ------------------------------------------------------------
st.markdown("#### Weekday / Weekend Demand Weighting")

w1, w2, w3 = st.columns(3)

with w1:
    weekday_weight = st.number_input(
        "Weekday Weight (relative demand)",
        min_value=0.50,
        max_value=2.00,
        value=1.10,
        step=0.05,
        help="Weekday demand multiplier vs baseline."
    )

with w2:
    weekend_weight = st.number_input(
        "Weekend Weight (relative demand)",
        min_value=0.50,
        max_value=2.00,
        value=0.85,
        step=0.05,
        help="Weekend demand multiplier vs baseline."
    )

with w3:
    shift_length = st.selectbox("Shift Length (hours)", [8, 10, 12], index=2)

st.markdown("#### Staffing Allocation Logic")

logic_choice = st.radio(
    "Allocate excess staffing (beyond minimum) toward:",
    ["Weekdays (default)", "Weekends", "Even split"],
    horizontal=True
)

min_providers_per_day = st.number_input(
    "Minimum Providers Per Day (Operational Floor)",
    min_value=1.0,
    value=1.0,
    step=0.5,
    help="Ensures schedule templates always show at least this many providers per day."
)

# ------------------------------------------------------------
# ✅ Helper: Convert monthly provider-hours/day into weekday/weekend schedule
# ------------------------------------------------------------
def build_weekly_staffing_template(
    provider_hours_per_day,
    weekday_weight,
    weekend_weight,
    shift_length_hours,
    min_providers_per_day,
    allocation_logic="Weekdays (default)"
):
    """
    Produces a weekly template:
    - Mon-Fri = weekday staffing
    - Sat-Sun = weekend staffing
    Total weekly provider-hours must equal provider_hours_per_day * 7
    Weighted distribution is based on weekday/weekend weights.
    """

    total_week_hours = provider_hours_per_day * 7
    base_floor_hours = min_providers_per_day * shift_length_hours

    # 5 weekdays, 2 weekend days
    weekday_total_weight = 5 * weekday_weight
    weekend_total_weight = 2 * weekend_weight
    weight_sum = weekday_total_weight + weekend_total_weight

    # Weighted raw allocation
    weekday_hours_raw = total_week_hours * (weekday_total_weight / weight_sum)
    weekend_hours_raw = total_week_hours * (weekend_total_weight / weight_sum)

    # Convert to daily average for weekday/weekend
    weekday_hours_per_day = weekday_hours_raw / 5
    weekend_hours_per_day = weekend_hours_raw / 2

    # Apply operational floor
    weekday_hours_per_day = max(weekday_hours_per_day, base_floor_hours)
    weekend_hours_per_day = max(weekend_hours_per_day, base_floor_hours)

    # Rebalance if floors pushed totals above available hours
    adjusted_total = weekday_hours_per_day * 5 + weekend_hours_per_day * 2
    if adjusted_total > total_week_hours:
        # we are exceeding available — reduce excess based on logic preference
        excess = adjusted_total - total_week_hours

        if allocation_logic == "Weekdays (default)":
            # remove excess from weekend first
            removable_weekend = max((weekend_hours_per_day - base_floor_hours) * 2, 0)
            reduce_weekend = min(excess, removable_weekend)
            weekend_hours_per_day -= reduce_weekend / 2
            excess -= reduce_weekend

            if excess > 0:
                # then weekday
                removable_weekday = max((weekday_hours_per_day - base_floor_hours) * 5, 0)
                reduce_weekday = min(excess, removable_weekday)
                weekday_hours_per_day -= reduce_weekday / 5

        elif allocation_logic == "Weekends":
            # remove excess from weekdays first
            removable_weekday = max((weekday_hours_per_day - base_floor_hours) * 5, 0)
            reduce_weekday = min(excess, removable_weekday)
            weekday_hours_per_day -= reduce_weekday / 5
            excess -= reduce_weekday

            if excess > 0:
                removable_weekend = max((weekend_hours_per_day - base_floor_hours) * 2, 0)
                reduce_weekend = min(excess, removable_weekend)
                weekend_hours_per_day -= reduce_weekend / 2

        else:
            # Even split removal
            removable_weekday = max((weekday_hours_per_day - base_floor_hours) * 5, 0)
            removable_weekend = max((weekend_hours_per_day - base_floor_hours) * 2, 0)
            removable_total = removable_weekday + removable_weekend

            if removable_total > 0:
                weekday_hours_per_day -= (excess * (removable_weekday / removable_total)) / 5
                weekend_hours_per_day -= (excess * (removable_weekend / removable_total)) / 2

    # Convert hours/day into shifts/day
    weekday_shifts = weekday_hours_per_day / shift_length_hours
    weekend_shifts = weekend_hours_per_day / shift_length_hours

    return weekday_shifts, weekend_shifts, weekday_hours_per_day, weekend_hours_per_day


# ------------------------------------------------------------
# ✅ Build schedule templates for Recommended vs Realistic Supply
# ------------------------------------------------------------
recommended_weekday_shifts = []
recommended_weekend_shifts = []

supply_weekday_shifts = []
supply_weekend_shifts = []

for rec_hrs, sup_hrs in zip(recommended_provider_hours, realistic_provider_hours):

    rec_wd, rec_we, rec_wd_hrs, rec_we_hrs = build_weekly_staffing_template(
        provider_hours_per_day=rec_hrs,
        weekday_weight=weekday_weight,
        weekend_weight=weekend_weight,
        shift_length_hours=shift_length,
        min_providers_per_day=min_providers_per_day,
        allocation_logic=logic_choice
    )

    sup_wd, sup_we, sup_wd_hrs, sup_we_hrs = build_weekly_staffing_template(
        provider_hours_per_day=sup_hrs,
        weekday_weight=weekday_weight,
        weekend_weight=weekend_weight,
        shift_length_hours=shift_length,
        min_providers_per_day=min_providers_per_day,
        allocation_logic=logic_choice
    )

    recommended_weekday_shifts.append(rec_wd)
    recommended_weekend_shifts.append(rec_we)

    supply_weekday_shifts.append(sup_wd)
    supply_weekend_shifts.append(sup_we)


# ------------------------------------------------------------
# ✅ Output schedule table
# ------------------------------------------------------------
schedule_df = pd.DataFrame({
    "Month": month_labels,

    f"Recommended Weekday Shifts/Day ({shift_length}h)": np.round(recommended_weekday_shifts, 2),
    f"Recommended Weekend Shifts/Day ({shift_length}h)": np.round(recommended_weekend_shifts, 2),

    f"Supply Weekday Shifts/Day ({shift_length}h)": np.round(supply_weekday_shifts, 2),
    f"Supply Weekend Shifts/Day ({shift_length}h)": np.round(supply_weekend_shifts, 2),
})

# Exposure gap in weekday shifts
schedule_df["Weekday Gap (Shifts/Day)"] = np.round(
    np.maximum(schedule_df[f"Recommended Weekday Shifts/Day ({shift_length}h)"] -
               schedule_df[f"Supply Weekday Shifts/Day ({shift_length}h)"], 0),
    2
)

schedule_df["Weekend Gap (Shifts/Day)"] = np.round(
    np.maximum(schedule_df[f"Recommended Weekend Shifts/Day ({shift_length}h)"] -
               schedule_df[f"Supply Weekend Shifts/Day ({shift_length}h)"], 0),
    2
)

st.dataframe(schedule_df, hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# ✅ Executive interpretation
# ------------------------------------------------------------
peak_weekday_gap = schedule_df["Weekday Gap (Shifts/Day)"].max()
peak_weekend_gap = schedule_df["Weekend Gap (Shifts/Day)"].max()

st.info(
    f"""
✅ **Weekly Template Interpretation**
- Recommended staffing requires up to **{schedule_df[f"Recommended Weekday Shifts/Day ({shift_length}h)"].max():.2f} shifts/day on weekdays**
  and **{schedule_df[f"Recommended Weekend Shifts/Day ({shift_length}h)"].max():.2f} shifts/day on weekends**.
- Realistic supply supports up to **{schedule_df[f"Supply Weekday Shifts/Day ({shift_length}h)"].max():.2f} weekday shifts/day**
  and **{schedule_df[f"Supply Weekend Shifts/Day ({shift_length}h)"].max():.2f} weekend shifts/day**.
- Peak shortage exposure = **{peak_weekday_gap:.2f} weekday shifts/day** and **{peak_weekend_gap:.2f} weekend shifts/day**.
- This is operationally useful for deciding **PRN coverage needs**, **extra shift incentives**, and whether the gap justifies fixed hiring.
"""
)

# ============================================================
# ✅ A7.3 — WEEKLY SCHEDULE TEMPLATE (MON–SUN) + COVERAGE PLAN
# ============================================================
st.markdown("---")
st.subheader("A7.3 — Weekly Staffing Schedule Template (Mon–Sun)")
st.caption(
    "This section converts provider-hours into an operational weekly schedule template. "
    "It shows daily shifts required (Mon–Sun) for Recommended vs Realistic staffing, plus a gap coverage plan."
)

# ------------------------------------------------------------
# ✅ Inputs (Operational Coverage Pattern)
# ------------------------------------------------------------
st.markdown("#### Schedule Pattern Assumptions")

s1, s2, s3 = st.columns(3)

with s1:
    weekday_open_hours = st.number_input(
        "Weekday Operating Hours (Daily)",
        min_value=4.0,
        max_value=24.0,
        value=10.0,
        step=0.5
    )

with s2:
    weekend_open_hours = st.number_input(
        "Weekend Operating Hours (Daily)",
        min_value=4.0,
        max_value=24.0,
        value=8.0,
        step=0.5
    )

with s3:
    prn_shift_length = st.selectbox("PRN Shift Length (hours)", [4, 6, 8, 10, 12], index=2)

# ------------------------------------------------------------
# ✅ Helper: build daily schedule (Mon–Sun)
# ------------------------------------------------------------
def mon_sun_schedule_from_shifts(weekday_shifts, weekend_shifts):
    """Build Mon–Sun list of shifts given weekday & weekend shift levels."""
    return [
        weekday_shifts,  # Mon
        weekday_shifts,  # Tue
        weekday_shifts,  # Wed
        weekday_shifts,  # Thu
        weekday_shifts,  # Fri
        weekend_shifts,  # Sat
        weekend_shifts,  # Sun
    ]


# ------------------------------------------------------------
# ✅ Pick "worst month" and "best month" templates automatically
# Executives want to see extremes.
# ------------------------------------------------------------
# Month with max burnout gap = most operationally critical
worst_month_idx = int(np.argmax(burnout_hours_gap))
best_month_idx = int(np.argmin(burnout_hours_gap))

worst_month_label = month_labels[worst_month_idx]
best_month_label = month_labels[best_month_idx]

# Recommended vs Supply shifts/day for that month
rec_wd = recommended_weekday_shifts[worst_month_idx]
rec_we = recommended_weekend_shifts[worst_month_idx]
sup_wd = supply_weekday_shifts[worst_month_idx]
sup_we = supply_weekend_shifts[worst_month_idx]

# Build Mon–Sun schedules (in shift equivalents/day)
rec_week_schedule = mon_sun_schedule_from_shifts(rec_wd, rec_we)
sup_week_schedule = mon_sun_schedule_from_shifts(sup_wd, sup_we)

gap_week_schedule = [max(r - s, 0) for r, s in zip(rec_week_schedule, sup_week_schedule)]

# Convert gaps into hours/day
gap_hours_week_schedule = [g * shift_length for g in gap_week_schedule]

# Convert gap into PRN shifts/day needed
gap_prn_shifts = [hrs / prn_shift_length for hrs in gap_hours_week_schedule]


# ------------------------------------------------------------
# ✅ Assemble weekly schedule template dataframe
# ------------------------------------------------------------
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

weekly_template_df = pd.DataFrame({
    "Day": days,
    f"Recommended Shifts ({shift_length}h)": np.round(rec_week_schedule, 2),
    f"Realistic Supply Shifts ({shift_length}h)": np.round(sup_week_schedule, 2),
    "Gap Shifts/Day": np.round(gap_week_schedule, 2),
    "Gap Provider Hours/Day": np.round(gap_hours_week_schedule, 1),
    f"PRN Shifts Needed ({prn_shift_length}h)": np.round(gap_prn_shifts, 2),
})

st.markdown(f"### Peak Burnout Month Staffing Template: **{worst_month_label}**")
st.dataframe(weekly_template_df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# ✅ Coverage Plan Recommendation (Simple decision logic)
# ------------------------------------------------------------
total_gap_hours_week = sum(gap_hours_week_schedule)
avg_gap_hours_day = total_gap_hours_week / 7
avg_gap_shifts_day = sum(gap_week_schedule) / 7

# Rules of thumb:
# - If avg gap > 0.75 shifts/day: consider fixed hire
# - If avg gap between 0.25–0.75 shifts/day: hybrid
# - If avg gap < 0.25 shifts/day: PRN/incentive coverage

if avg_gap_shifts_day >= 0.75:
    coverage_plan = "Fixed Hire Recommended"
    plan_reason = (
        "The staffing gap is sustained and large enough that PRN coverage will be expensive, inconsistent, and burnout-sustaining. "
        "A fixed hire is more stable and protects retention."
    )
elif avg_gap_shifts_day >= 0.25:
    coverage_plan = "Hybrid Coverage Strategy"
    plan_reason = (
        "The gap is meaningful but not large enough to justify a full fixed FTE year-round. "
        "Use PRN/extra shifts in peak months and target hiring for winter demand."
    )
else:
    coverage_plan = "PRN / Extra Shift Coverage"
    plan_reason = (
        "The gap is modest and intermittent. PRN and extra-shift incentives can close this gap at lower fixed cost "
        "while maintaining flexibility."
    )

st.markdown("### Coverage Plan Recommendation")
st.success(f"**{coverage_plan}**")
st.write(plan_reason)

# ------------------------------------------------------------
# ✅ Executive takeaway box
# ------------------------------------------------------------
st.info(
    f"""
✅ **Executive Takeaway (Peak Month: {worst_month_label})**
- Average staffing gap = **{avg_gap_hours_day:.1f} provider-hours/day** (≈ **{avg_gap_shifts_day:.2f} {shift_length}-hour shifts/day**)
- Total weekly shortage exposure = **{total_gap_hours_week:.1f} provider-hours/week**
- Recommended solution: **{coverage_plan}**

This is the operational interpretation of the burnout exposure zone:
You can see exactly which days require extra shifts, PRN coverage, or fixed staffing adjustments.
"""
)

# ------------------------------------------------------------
# ✅ Optional: show best month comparison too
# ------------------------------------------------------------
with st.expander("Show lowest gap month staffing template", expanded=False):

    rec_wd_best = recommended_weekday_shifts[best_month_idx]
    rec_we_best = recommended_weekend_shifts[best_month_idx]
    sup_wd_best = supply_weekday_shifts[best_month_idx]
    sup_we_best = supply_weekend_shifts[best_month_idx]

    rec_week_best = mon_sun_schedule_from_shifts(rec_wd_best, rec_we_best)
    sup_week_best = mon_sun_schedule_from_shifts(sup_wd_best, sup_we_best)
    gap_week_best = [max(r - s, 0) for r, s in zip(rec_week_best, sup_week_best)]
    gap_hours_best = [g * shift_length for g in gap_week_best]

    best_df = pd.DataFrame({
        "Day": days,
        f"Recommended Shifts ({shift_length}h)": np.round(rec_week_best, 2),
        f"Realistic Supply Shifts ({shift_length}h)": np.round(sup_week_best, 2),
        "Gap Shifts/Day": np.round(gap_week_best, 2),
        "Gap Provider Hours/Day": np.round(gap_hours_best, 1),
    })

    st.markdown(f"#### Low Gap Month: **{best_month_label}**")
    st.dataframe(best_df, hide_index=True, use_container_width=True)

# ============================================================
# ✅ A7.4 — SUGGESTED SHIFT SCHEDULE PATTERNS (OPERATOR-READY)
# ============================================================
st.markdown("---")
st.subheader("A7.4 — Suggested Shift Schedule Patterns (Operator Ready)")
st.caption(
    "This section converts recommended daily shift requirements into a suggested staffing pattern "
    "with shift start/end times and PRN coverage guidance."
)

# ------------------------------------------------------------
# ✅ Inputs: Shift start style
# ------------------------------------------------------------
st.markdown("#### Shift Template Preferences")

t1, t2, t3 = st.columns(3)

with t1:
    earliest_start_hour = st.number_input(
        "Earliest Shift Start Hour (24h clock)",
        min_value=4,
        max_value=12,
        value=8,
        step=1
    )

with t2:
    stagger_minutes = st.selectbox(
        "Stagger Start Times By",
        [0, 30, 60, 90, 120],
        index=2,
        help="Offsets stagger shifts (e.g., 60 = one hour stagger)."
    )

with t3:
    use_two_wave_pattern = st.checkbox(
        "Use Two-Wave Coverage Pattern (recommended)",
        value=True,
        help="Two-wave pattern creates overlap in peak hours and improves burnout protection."
    )


# ------------------------------------------------------------
# ✅ Helper: convert shift specs into readable time strings
# ------------------------------------------------------------
def format_shift(start_hour, start_minute, length_hours):
    end_hour = start_hour + length_hours
    end_minute = start_minute

    # Handle wrap past midnight
    while end_hour >= 24:
        end_hour -= 24

    def fmt(h, m):
        suffix = "AM" if h < 12 else "PM"
        hr = h if 1 <= h <= 12 else (h - 12 if h > 12 else 12)
        return f"{hr}:{m:02d}{suffix}"

    return f"{fmt(start_hour, start_minute)}–{fmt(end_hour, end_minute)}"


# ------------------------------------------------------------
# ✅ Core schedule generator
# ------------------------------------------------------------
def generate_shift_pattern(required_shifts, open_hours, shift_length, earliest_start, stagger_minutes, two_wave=True):
    """
    Generate a suggested shift pattern:
      - Uses a baseline shift starting earliest_start
      - Adds staggered overlapping shifts to match required_shifts
      - If required_shifts includes partials, last shift becomes PRN
    Returns list of dicts: [{shift, providers, type}]
    """

    # Round down to full shifts, then remainder becomes PRN
    full_shifts = int(np.floor(required_shifts))
    remainder = required_shifts - full_shifts

    pattern = []

    # Always include at least 1 core shift
    core_shifts = max(full_shifts, 1)

    # Two-wave coverage: staggered overlaps to create peak coverage
    for i in range(core_shifts):
        if two_wave:
            # wave 1 starts at earliest_start
            # wave 2 starts staggered
            start_hour = earliest_start + (i % 2) * (stagger_minutes / 60)
        else:
            start_hour = earliest_start + (i * stagger_minutes / 60)

        # Convert float hours to hour/minute
        start_hour_int = int(np.floor(start_hour))
        start_min = int(round((start_hour - start_hour_int) * 60))

        pattern.append({
            "Shift": format_shift(start_hour_int, start_min, shift_length),
            "Providers": 1,
            "Type": "Core"
        })

    # If remainder exists, add PRN partial coverage
    if remainder >= 0.10:
        prn_start_hour = earliest_start + ((core_shifts % 2) * (stagger_minutes / 60) if two_wave else core_shifts * (stagger_minutes / 60))
        prn_start_int = int(np.floor(prn_start_hour))
        prn_start_min = int(round((prn_start_hour - prn_start_int) * 60))

        pattern.append({
            "Shift": format_shift(prn_start_int, prn_start_min, shift_length),
            "Providers": remainder,
            "Type": "PRN (Partial)"
        })

    return pattern


# ------------------------------------------------------------
# ✅ Build suggested schedule patterns for Peak Month (Mon–Sun)
# ------------------------------------------------------------
suggested_schedule_rows = []

for _, row in weekly_template_df.iterrows():
    day = row["Day"]
    rec_shifts = row[f"Recommended Shifts ({shift_length}h)"]
    gap_shifts = row["Gap Shifts/Day"]

    # Determine open hours
    is_weekend = day in ["Sat", "Sun"]
    open_hours = weekend_open_hours if is_weekend else weekday_open_hours

    # Build recommended pattern
    rec_pattern = generate_shift_pattern(
        required_shifts=rec_shifts,
        open_hours=open_hours,
        shift_length=shift_length,
        earliest_start=earliest_start_hour,
        stagger_minutes=stagger_minutes,
        two_wave=use_two_wave_pattern
    )

    # Build PRN coverage suggestion if gap exists
    prn_pattern = []
    if gap_shifts > 0:
        prn_pattern = generate_shift_pattern(
            required_shifts=gap_shifts,
            open_hours=open_hours,
            shift_length=prn_shift_length,
            earliest_start=earliest_start_hour + 2,   # PRN starts mid-day
            stagger_minutes=0,
            two_wave=False
        )

    # Flatten patterns into readable strings
    rec_string = "; ".join([f"{p['Providers']:.2f} × {p['Shift']} ({p['Type']})" for p in rec_pattern])
    prn_string = "None"
    if prn_pattern:
        prn_string = "; ".join([f"{p['Providers']:.2f} × {p['Shift']} ({p['Type']})" for p in prn_pattern])

    suggested_schedule_rows.append({
        "Day": day,
        "Recommended Coverage Pattern": rec_string,
        "PRN / Extra Shift Plan (if shortage)": prn_string
    })


suggested_schedule_df = pd.DataFrame(suggested_schedule_rows)

st.markdown(f"### Suggested Staffing Pattern — Peak Burnout Month: **{worst_month_label}**")
st.dataframe(suggested_schedule_df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# ✅ Executive summary takeaway
# ------------------------------------------------------------
st.success(
    f"""
✅ **Staffing Template Output Generated**
This schedule represents a practical shift-based interpretation of the recommended staffing curve during **{worst_month_label}**.

It includes:
- **Core shift coverage plan**
- **Overlapping wave scheduling (burnout-protective)**
- **PRN shift recommendations for shortages**
"""
)

with st.expander("How to use this operationally", expanded=False):
    st.write("""
    **How operators can apply this output:**
    1) Use the **Core Coverage Pattern** as the base clinic schedule.
    2) Use the **PRN/Extra Shift Plan** to fill gaps during peak demand.
    3) If PRN shifts remain >0.75 shifts/day in peak months, fixed hiring is usually indicated.
    
    **Why two-wave coverage matters:**
    - It creates overlap during peak hours
    - Reduces single-provider overload
    - Creates recovery space (reducing turnover)
    """)
