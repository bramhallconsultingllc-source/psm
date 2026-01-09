import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel

# ============================================================
# ✅ Page Setup
# ============================================================
st.set_page_config(page_title="PSM Staffing Model", layout="centered")
st.title("Predictive Staffing Model (PSM)")
st.caption("Operations → Reality → Finance → Strategy → Decision")

st.info(
    "⚠️ **All staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)

model = StaffingModel()

# ============================================================
# ✅ Stable 'today' for consistent reruns
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]

# ============================================================
# ✅ Helper Functions
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(x, hi))

def monthly_index(d: datetime, anchor: datetime):
    return (d.year - anchor.year) * 12 + (d.month - anchor.month)

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
# ✅ Burnout-Protective Curve (Recommended Target)
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

# ============================================================
# ✅ Realistic Staffing Supply Curve
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
            planned = max(planned - months_elapsed * monthly_attrition_fte, provider_min_floor)

        planned = max(planned, provider_min_floor)
        staff.append(planned)
        prev = planned

    return staff

# ============================================================
# ✅ Cost Helpers
# ============================================================
def provider_day_gap(target_curve, supply_curve, days_in_month):
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(t - s, 0) * dim
    return gap_days

def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    cost = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += dfte * loaded_cost_per_provider_fte * (dim / 365)
    return cost


# ============================================================
# ============================================================
# ✅ SECTION 1 — OPERATIONS
# ============================================================
# ============================================================
st.markdown("---")
st.header("1) Operations — Baseline + Seasonality + Recommended Target")
st.caption("Start with baseline clinic volume, then forecast seasonality and generate lean + recommended staffing targets.")

visits = st.number_input("Average Visits per Day (Annual Average)", min_value=1.0, value=45.0, step=1.0)

hours_of_operation = st.number_input("Hours of Operation per Week", min_value=1.0, value=70.0, step=1.0)
fte_hours_per_week = st.number_input("FTE Hours per Week", min_value=1.0, value=40.0, step=1.0)

provider_min_floor = st.number_input("Provider Minimum Floor (FTE)", min_value=0.25, value=1.00, step=0.25)

burnout_slider = st.slider(
    "Burnout Protection Level",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="0.0 = lean | 0.6 = recommended protective | 1.0 = max protection"
)

safe_visits_per_provider = st.number_input(
    "Safe Visits per Provider per Day Threshold",
    min_value=10,
    max_value=40,
    value=20,
    step=1,
    help="Used in recovery debt buffer. Higher = less protective."
)

provider_turnover = st.number_input("Provider Turnover % (Annual)", value=24.0, step=1.0) / 100

flu_c1, flu_c2, flu_c3 = st.columns(3)
with flu_c1:
    flu_start_month = st.selectbox("Flu Start Month", options=list(range(1, 13)), index=11,
                                   format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
with flu_c2:
    flu_end_month = st.selectbox("Flu End Month", options=list(range(1, 13)), index=1,
                                 format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
with flu_c3:
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100

# Pipeline inputs
with st.expander("Provider Hiring Pipeline Assumptions", expanded=False):
    days_to_sign = st.number_input("Days to Sign", min_value=0, value=90, step=5)
    days_to_credential = st.number_input("Days to Credential", min_value=0, value=90, step=5)
    onboard_train_days = st.number_input("Days to Train", min_value=0, value=30, step=5)
    coverage_buffer_days = st.number_input("Planning Buffer Days", min_value=0, value=14, step=1)
    notice_days = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=75, step=5)

# Calculate baseline provider FTE
fte_result = model.calculate_fte_needed(
    visits_per_day=visits,
    hours_of_operation_per_week=hours_of_operation,
    fte_hours_per_week=fte_hours_per_week,
)
baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

# Calendar timeline
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
    flu_uplift_pct=flu_uplift_pct,
)

provider_base_demand = visits_to_provider_demand(
    model=model,
    visits_by_month=forecast_visits_by_month,
    hours_of_operation=hours_of_operation,
    fte_hours_per_week=fte_hours_per_week,
    provider_min_floor=provider_min_floor,
)

protective_curve = burnout_protective_staffing_curve(
    dates=dates,
    visits_by_month=forecast_visits_by_month,
    base_demand_fte=provider_base_demand,
    provider_min_floor=provider_min_floor,
    burnout_slider=burnout_slider,
    safe_visits_per_provider_per_day=safe_visits_per_provider,
)

forecast_df = pd.DataFrame({"Month": month_labels, "Forecast Visits/Day": np.round(forecast_visits_by_month, 1)})
st.dataframe(forecast_df, hide_index=True, use_container_width=True)

# ✅ Section 1 micro-summary
st.success(
    f"✅ **Operations Summary:** Baseline demand is **{baseline_provider_fte:.2f} provider FTE**. "
    f"Seasonality shifts volume across the year and generates both a **Lean Target** and a **Recommended Target** "
    f"(burnout-protective) staffing curve."
)
# ============================================================
# ✅ A2.5 — Seasonality Staffing Requirements Table
# Visits/Day → Staff Needed per Day → FTE Needed (Monthly)
# ============================================================
st.markdown("---")
st.subheader("A2.5 — Seasonality Staffing Requirements (Monthly Table)")
st.caption(
    "This table translates forecasted visits/day into staffing needed per role per day "
    "and the corresponding FTE requirement for each month."
)

monthly_rows = []

for month_label, v in zip(month_labels, forecast_visits_by_month):

    # --- Daily staffing requirements (per day coverage)
    daily_staff = model.calculate(v)

    # --- FTE requirements (based on weekly hours)
    fte_staff = model.calculate_fte_needed(
        visits_per_day=v,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week
    )

    monthly_rows.append({
        "Month": month_label,
        "Forecast Visits/Day": round(v, 1),

        # Daily staffing
        "Providers Needed / Day": round(daily_staff["provider_daily"], 2),
        "PSR Needed / Day": round(daily_staff["psr_daily"], 2),
        "MA Needed / Day": round(daily_staff["ma_daily"], 2),
        "XRT Needed / Day": round(daily_staff["xrt_daily"], 2),

        # FTE staffing
        "Provider FTE": round(fte_staff["provider_fte"], 2),
        "PSR FTE": round(fte_staff["psr_fte"], 2),
        "MA FTE": round(fte_staff["ma_fte"], 2),
        "XRT FTE": round(fte_staff["xrt_fte"], 2),
        "Total FTE": round(fte_staff["total_fte"], 2),
    })

seasonality_staff_df = pd.DataFrame(monthly_rows)

st.dataframe(seasonality_staff_df, hide_index=True, use_container_width=True)

# --- Executive highlight metrics
peak_visits = seasonality_staff_df["Forecast Visits/Day"].max()
peak_total_fte = seasonality_staff_df["Total FTE"].max()
min_total_fte = seasonality_staff_df["Total FTE"].min()

k1, k2, k3 = st.columns(3)
k1.metric("Peak Visits/Day", f"{peak_visits:.1f}")
k2.metric("Peak Total FTE Needed", f"{peak_total_fte:.2f}")
k3.metric("Low Season Total FTE Needed", f"{min_total_fte:.2f}")

st.success(
    "✅ Executive takeaway: staffing needs increase and decrease throughout the year due to seasonality — "
    "this table makes that visible in both daily scheduling terms and FTE budgeting terms."
)

# ============================================================
# ============================================================
# ✅ SECTION 2 — REALITY
# ============================================================
# ============================================================
st.markdown("---")
st.header("2) Reality — Staffing Supply + Burnout Exposure")
st.caption("This shows what is realistically achievable given hiring ramp limits + attrition, and where burnout exposure remains.")

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

burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
months_exposed = sum([1 for g in burnout_gap_fte if g > 0])

fig, ax1 = plt.subplots(figsize=(12, 4))

ax1.plot(dates, provider_base_demand, linestyle=":", linewidth=2, label="Lean Target (Demand)")
ax1.plot(dates, protective_curve, linewidth=3, marker="o", label="Recommended Target (Protective)")
ax1.plot(dates, realistic_supply_recommended, linewidth=3, marker="o", label="Realistic Supply")

ax1.fill_between(
    dates,
    realistic_supply_recommended,
    protective_curve,
    where=np.array(protective_curve) > np.array(realistic_supply_recommended),
    alpha=0.25,
    label="Burnout Exposure Zone"
)

ax1.set_title("A6 — Volume, Targets, Supply & Burnout Exposure")
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

k1, k2, k3 = st.columns(3)
k1.metric("Peak Burnout Gap (FTE)", f"{max(burnout_gap_fte):.2f}")
k2.metric("Avg Burnout Gap (FTE)", f"{np.mean(burnout_gap_fte):.2f}")
k3.metric("Months Exposed", f"{months_exposed}/12")

st.success(
    f"✅ **Reality Summary:** Even with best-case hiring, you remain exposed in **{months_exposed}/12 months**, "
    f"with a peak shortage of **{max(burnout_gap_fte):.2f} FTE**. This is the operational burnout risk zone."
)

# ============================================================
# ============================================================
# ✅ SECTION 3 — FINANCE
# ============================================================
# ============================================================
st.markdown("---")
st.header("3) Finance — Investment Case (EBITDA Impact)")
st.caption("Quantifies the cost to staff to the recommended target vs the expected exposure if you do not.")

time_horizon = st.radio("Display ROI impacts as:", ["Annual", "Quarterly", "Monthly"], horizontal=True, index=0)
horizon_factor = 1.0 if time_horizon == "Annual" else (1.0/4 if time_horizon == "Quarterly" else 1.0/12)

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

# Turnover cost CI
st.subheader("Turnover Cost Confidence Interval (Per Provider Event)")
tci1, tci2 = st.columns(2)
with tci1:
    turnover_cost_low = st.number_input("Low Estimate ($)", min_value=0.0, value=60000.0, step=5000.0)
with tci2:
    turnover_cost_high = st.number_input("High Estimate ($)", min_value=0.0, value=120000.0, step=5000.0)
turnover_cost_mid = (turnover_cost_low + turnover_cost_high) / 2

use_premium_labor = st.checkbox("Include premium labor cost exposure", value=False)
premium_pct = 0.25
provider_day_cost_basis = loaded_cost_per_provider_fte / 260

leakage_factor = st.number_input("Demand Leakage Factor (0–1)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
max_productivity_uplift = st.number_input("Max Productivity Uplift at Protection=1.0 (%)", min_value=0.0, max_value=30.0, value=6.0, step=1.0) / 100
max_turnover_reduction = st.number_input("Max Turnover Reduction at Protection=1.0 (%)", min_value=0.0, max_value=100.0, value=35.0, step=5.0) / 100

# A = incremental staffing investment
delta_fte_curve = [max(r - l, 0) for r, l in zip(protective_curve, provider_base_demand)]
incremental_staffing_cost_annual = annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte)

# Exposure helper
def compute_exposure_annual(target_curve, supply_curve, turnover_rate, turnover_cost):
    provider_count = baseline_provider_fte
    expected_departures = provider_count * turnover_rate
    turnover_exposure = expected_departures * turnover_cost

    gap_provider_days = provider_day_gap(target_curve, supply_curve, days_in_month)
    visits_per_provider_fte_per_day = visits / max(baseline_provider_fte, 0.25)
    lost_visits = gap_provider_days * visits_per_provider_fte_per_day * leakage_factor
    lost_margin = lost_visits * margin_per_visit

    premium_exposure = 0.0
    if use_premium_labor:
        premium_exposure = gap_provider_days * provider_day_cost_basis * premium_pct

    return turnover_exposure + lost_margin + premium_exposure, turnover_exposure, lost_margin, premium_exposure

# B lean exposure using midpoint turnover cost
exposure_lean_total, exposure_lean_turn, exposure_lean_margin, exposure_lean_premium = compute_exposure_annual(
    provider_base_demand, realistic_supply_lean, provider_turnover, turnover_cost_mid
)

# Recommended exposure
turnover_recommended = provider_turnover * (1 - max_turnover_reduction * burnout_slider)
exposure_rec_total, exposure_rec_turn, exposure_rec_margin, exposure_rec_premium = compute_exposure_annual(
    protective_curve, realistic_supply_recommended, turnover_recommended, turnover_cost_mid
)

# C savings
exposure_avoided_annual = max(exposure_lean_total - exposure_rec_total, 0)
productivity_uplift = max_productivity_uplift * burnout_slider
productivity_margin_uplift_annual = annual_visits * productivity_uplift * margin_per_visit
expected_savings_if_staffed_annual = exposure_avoided_annual + productivity_margin_uplift_annual

net_ebitda_impact_annual = expected_savings_if_staffed_annual - incremental_staffing_cost_annual
net_margin_impact = net_ebitda_impact_annual / annual_net_revenue if annual_net_revenue > 0 else 0

A_display = incremental_staffing_cost_annual * horizon_factor
B_display = exposure_lean_total * horizon_factor
C_display = expected_savings_if_staffed_annual * horizon_factor
NET_display = net_ebitda_impact_annual * horizon_factor

cA, cB, cC, cN = st.columns(4)
cA.metric(f"A) Cost to Staff ({time_horizon})", f"${A_display:,.0f}")
cB.metric(f"B) Exposure if Lean ({time_horizon})", f"${B_display:,.0f}")
cC.metric(f"C) Savings if Staffed ({time_horizon})", f"${C_display:,.0f}")
cN.metric(f"Net EBITDA ({time_horizon})", f"${NET_display:,.0f}", f"{net_margin_impact*100:.2f}% annual margin")

st.success(
    f"✅ **Finance Summary:** Staffing to recommended costs **${A_display:,.0f}** and avoids "
    f"~**${C_display:,.0f}** in expected costs and lost margin, for a net EBITDA impact of **${NET_display:,.0f}**."
)

# ============================================================
# ============================================================
# ✅ SECTION 4 — STRATEGY (Recruiting Buffer + Float + Fractional + Hybrid)
# ============================================================
# ============================================================
st.markdown("---")
st.header("4) Strategy — Recruiting Buffer + Float Pool + Fractional Staffing")
st.caption("Makes turnover buffer explicit and shows how float + fractional staffing can cover demand without permanent overstaffing.")

time_to_replace_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days
expected_departures_fte = baseline_provider_fte * provider_turnover
coverage_leakage_fte = expected_departures_fte * (time_to_replace_days / 365)
pipeline_target_fte = baseline_provider_fte + coverage_leakage_fte

st.subheader("A8 — Recruiting Buffer (Turnover + Replacement Lag)")
k1, k2, k3 = st.columns(3)
k1.metric("Expected Departures (FTE/year)", f"{expected_departures_fte:.2f}")
k2.metric("Coverage Leakage Buffer (FTE)", f"{coverage_leakage_fte:.2f}")
k3.metric("Recruiting Pipeline Target (FTE)", f"{pipeline_target_fte:.2f}")

# Float pool builder
st.subheader("Float Pool Builder (Multi-Clinic)")
fc1, fc2, fc3 = st.columns(3)
with fc1:
    num_clinics = st.number_input("Clinics in Region", min_value=1, value=5, step=1)
with fc2:
    avg_baseline_fte = st.number_input("Avg Baseline FTE/Clinic", min_value=0.5, value=float(baseline_provider_fte), step=0.1)
with fc3:
    turnover_region = st.number_input("Regional Turnover %", min_value=0.0, max_value=100.0, value=float(provider_turnover*100), step=1.0) / 100

region_departures = num_clinics * avg_baseline_fte * turnover_region
region_leakage = region_departures * (time_to_replace_days / 365)

region_float_fte_needed = region_leakage
region_float_providers = region_float_fte_needed  # assumes 1.0 FTE/provider

fk1, fk2 = st.columns(2)
fk1.metric("Regional Leakage (FTE)", f"{region_leakage:.2f}")
fk2.metric("Float Pool Recommended (Providers)", f"{region_float_providers:.1f}")

# Hybrid slider
st.subheader("Hybrid Strategy Slider (Fixed + Float Mix)")
hybrid_pct_fixed = st.slider(
    "Percent of gap closed with fixed hiring (rest covered by float)",
    min_value=0,
    max_value=100,
    value=60,
    step=5
) / 100

float_pct = 1 - hybrid_pct_fixed

# Burnout gap provider-days (recommended vs realistic)
gap_provider_days_total = provider_day_gap(protective_curve, realistic_supply_recommended, days_in_month)

# Gap after hybrid: fixed closes portion permanently, float closes portion flexibly
gap_after_hybrid = gap_provider_days_total * (1 - hybrid_pct_fixed)

burnout_months_before = months_exposed
burnout_months_after = int(round(months_exposed * (1 - float_pct)))  # simplified proxy

st.success(
    f"✅ **Strategy Summary:** Leaders must recruit beyond baseline by **{coverage_leakage_fte:.2f} FTE** to offset turnover. "
    f"A regional float pool of **~{region_float_providers:.1f} providers** can cover predictable leakage and seasonality. "
    f"With a hybrid mix of **{hybrid_pct_fixed*100:.0f}% fixed / {float_pct*100:.0f}% float**, burnout exposure months drop "
    f"from **{burnout_months_before} → ~{burnout_months_after}**."
)

# Downloadable float pool plan
float_plan_df = pd.DataFrame({
    "Metric": ["Clinics", "Regional leakage FTE", "Float providers recommended", "Hybrid fixed %", "Hybrid float %"],
    "Value": [num_clinics, region_leakage, region_float_providers, hybrid_pct_fixed, float_pct]
})
csv = float_plan_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Float Pool Staffing Plan (CSV)", csv, "float_pool_plan.csv", "text/csv")

# ============================================================
# ============================================================
# ✅ SECTION 5 — DECISION SUMMARY (Big)
# ============================================================
# ============================================================
st.markdown("---")
st.header("5) Decision — PSM Final Recommendation")
st.caption("This is the decision-ready executive summary.")

if net_ebitda_impact_annual >= 0:
    decision_tone = "✅ EBITDA-Positive"
    decision = "Proceed with staffing to Recommended Target using a hybrid fixed + float strategy."
elif net_ebitda_impact_annual > -5000:
    decision_tone = "⚠️ Near Breakeven"
    decision = "Proceed with recommended staffing because stability benefits outweigh minimal EBITDA drag."
else:
    decision_tone = "❌ EBITDA-Negative"
    decision = "Use a float-heavy strategy and selective fixed hiring in peak months to reduce cost."

st.markdown(f"## {decision_tone}")
st.write(decision)

st.markdown("### Final PSM Decision Summary")

summary_df = pd.DataFrame({
    "PSM Decision Factor": [
        "Baseline Provider FTE Needed",
        "Recruiting Pipeline Target (Turnover Buffer)",
        "Peak Burnout Gap (FTE)",
        "Burnout Months Exposed",
        f"Cost to Staff (A) — {time_horizon}",
        f"Expected Savings (C) — {time_horizon}",
        f"Net EBITDA Impact — {time_horizon}",
        "Float Pool Recommended (Providers)",
        "Hybrid Mix (Fixed / Float)"
    ],
    "Value": [
        f"{baseline_provider_fte:.2f}",
        f"{pipeline_target_fte:.2f}",
        f"{max(burnout_gap_fte):.2f}",
        f"{months_exposed}/12",
        f"${A_display:,.0f}",
        f"${C_display:,.0f}",
        f"${NET_display:,.0f}",
        f"{region_float_providers:.1f}",
        f"{hybrid_pct_fixed*100:.0f}% / {float_pct*100:.0f}%"
    ]
})

st.dataframe(summary_df, hide_index=True, use_container_width=True)

st.success(
    "✅ **Decision Summary:** This final output aligns staffing reality, burnout exposure, financial ROI, recruiting buffer, "
    "and float pool strategy into one leadership-ready recommendation."
)
