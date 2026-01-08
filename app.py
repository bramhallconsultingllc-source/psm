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
