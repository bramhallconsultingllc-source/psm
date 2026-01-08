import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ============================================================
# ✅ STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")
st.title("Predictive Staffing Model (PSM)")
st.caption("Seasonality + burnout protection + realism + financial ROI + float pool strategy")

st.info(
    "⚠️ **All daily staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)

model = StaffingModel()

# ============================================================
# ✅ Stable "today" for consistent chart windows across reruns
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]


# ============================================================
# ✅ Session State Init (Crash-Proof)
# ============================================================
if "calculated" not in st.session_state:
    st.session_state["calculated"] = False

if "locked_inputs" not in st.session_state:
    st.session_state["locked_inputs"] = {}

if "locked_outputs" not in st.session_state:
    st.session_state["locked_outputs"] = {}


# ============================================================
# ✅ Helper Functions (Single Source of Truth)
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
    return [v * (baseline_visits / avg_raw) for v in raw]


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
# ✅ Burnout-Protective Curve (Recommended)
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
            planned = max(planned - (months_elapsed * monthly_attrition_fte), provider_min_floor)

        planned = max(planned, provider_min_floor)
        staff.append(planned)
        prev = planned

    return staff


def provider_day_gap(target_curve, supply_curve, days_in_month):
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(t - s, 0) * dim
    return gap_days


def annualize_monthly_fte_cost(delta_fte_curve, loaded_cost, days_in_month):
    cost = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += dfte * loaded_cost * (dim / 365)
    return cost


# ============================================================
# ✅ INPUTS — ALWAYS VISIBLE (Option A architecture)
# ============================================================
st.markdown("## Baseline Inputs")

visits = st.number_input("Average Visits per Day (Annual Average)", min_value=1.0, value=45.0, step=1.0)

st.markdown("### Weekly Inputs")
hours_of_operation = st.number_input("Hours of Operation per Week", min_value=1.0, value=70.0, step=1.0)
fte_hours_per_week = st.number_input("FTE Hours per Week", min_value=1.0, value=40.0, step=1.0)

provider_min_floor = st.number_input(
    "Provider Minimum Floor (FTE)",
    min_value=0.25,
    value=1.00,
    step=0.25
)

st.markdown("## Burnout Protection Controls")
burnout_slider = st.slider("Burnout Protection Level", 0.0, 1.0, 0.6, 0.05)
safe_visits_per_provider = st.number_input("Safe Visits per Provider per Day Threshold", 10, 40, 20, 1)

st.markdown("## Provider Turnover Assumptions")
provider_turnover = st.number_input("Provider Turnover %", value=24.0, step=1.0) / 100

st.markdown("## Flu Season Settings")
flu_c1, flu_c2, flu_c3 = st.columns(3)

with flu_c1:
    flu_start_month = st.selectbox("Flu Start Month", list(range(1, 13)), index=11,
                                   format_func=lambda x: datetime(2000, x, 1).strftime("%B"))

with flu_c2:
    flu_end_month = st.selectbox("Flu End Month", list(range(1, 13)), index=1,
                                 format_func=lambda x: datetime(2000, x, 1).strftime("%B"))

with flu_c3:
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100

st.markdown("## Provider Hiring Lead Time")
with st.expander("Pipeline Inputs", expanded=False):
    days_to_sign = st.number_input("Days to Sign", min_value=0, value=90, step=5)
    days_to_credential = st.number_input("Days to Credential", min_value=0, value=120, step=5)
    onboard_train_days = st.number_input("Days to Train", min_value=0, value=30, step=5)
    coverage_buffer_days = st.number_input("Buffer Days", min_value=0, value=14, step=1)
    notice_days = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=75, step=5)


# ============================================================
# ✅ CALCULATE BUTTON (Locks Curves)
# ============================================================
if st.button("Calculate Staffing (Lock Curves)"):
    st.session_state["calculated"] = True

    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )

    baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

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

    realistic_supply_recommended = realistic_staffing_supply_curve(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=protective_curve,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
    )

    realistic_supply_lean = realistic_staffing_supply_curve(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
    )

    # lock results
    st.session_state["locked_outputs"] = {
        "fte_result": fte_result,
        "baseline_provider_fte": baseline_provider_fte,
        "dates": dates,
        "month_labels": month_labels,
        "days_in_month": days_in_month,
        "forecast_visits_by_month": forecast_visits_by_month,
        "provider_base_demand": provider_base_demand,
        "protective_curve": protective_curve,
        "realistic_supply_recommended": realistic_supply_recommended,
        "realistic_supply_lean": realistic_supply_lean,
        "flu_start_date": flu_start_date,
        "flu_end_date": flu_end_date,
    }


# ============================================================
# ✅ STOP if curves not locked yet
# ============================================================
if not st.session_state["calculated"]:
    st.info("Enter inputs above and click **Calculate Staffing (Lock Curves)** to generate outputs.")
    st.stop()


# ============================================================
# ✅ UNPACK LOCKED OUTPUTS
# ============================================================
o = st.session_state["locked_outputs"]

fte_result = o["fte_result"]
baseline_provider_fte = o["baseline_provider_fte"]
dates = o["dates"]
month_labels = o["month_labels"]
days_in_month = o["days_in_month"]
forecast_visits_by_month = o["forecast_visits_by_month"]
provider_base_demand = o["provider_base_demand"]
protective_curve = o["protective_curve"]
realistic_supply_recommended = o["realistic_supply_recommended"]
realistic_supply_lean = o["realistic_supply_lean"]
flu_start_date = o["flu_start_date"]


# ============================================================
# ✅ A6 EXECUTIVE GRAPH
# ============================================================
st.markdown("---")
st.subheader("A6 — Executive View: Volume, Targets, Supply, Burnout Exposure")

fig, ax1 = plt.subplots(figsize=(12, 4))

ax1.plot(dates, provider_base_demand, linestyle=":", linewidth=2, label="Lean Target (Volume Only)")
ax1.plot(dates, protective_curve, linewidth=3.5, marker="o", label="Recommended Target (Burnout-Protective)")
ax1.plot(dates, realistic_supply_recommended, linewidth=3, marker="o", label="Best-Case Realistic Supply")

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
ax2.plot(dates, forecast_visits_by_month, linestyle="-.", linewidth=2.5, label="Forecasted Visits/Day")
ax2.set_ylabel("Visits / Day")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, -0.22), ncol=2)

plt.tight_layout()
st.pyplot(fig)


# ============================================================
# ✅ A8 Recruiting Buffer + Float Pool Planning
# ============================================================
st.markdown("---")
st.header("A8 — Recruiting Buffer + Float Pool Planning")

time_to_replace_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days
expected_departures_fte = baseline_provider_fte * provider_turnover
coverage_leakage_fte = expected_departures_fte * (time_to_replace_days / 365)
pipeline_target_fte = baseline_provider_fte + coverage_leakage_fte

k1, k2, k3 = st.columns(3)
k1.metric("Baseline FTE Needed", f"{baseline_provider_fte:.2f}")
k2.metric("Coverage Leakage Buffer (FTE)", f"{coverage_leakage_fte:.2f}")
k3.metric("Pipeline Recruiting Target (FTE)", f"{pipeline_target_fte:.2f}")

st.success(
    f"""
✅ Leaders must recruit with a **turnover buffer**.
To maintain **{baseline_provider_fte:.2f} FTE**, the recruiting pipeline must support **{coverage_leakage_fte:.2f} FTE of leakage**
caused by turnover + replacement lag (~{time_to_replace_days} days).
"""
)

# ============================================================
# ✅ FRACTIONAL STAFFING POOL (explicit)
# ============================================================
st.subheader("Fractional Staffing Pooling (Multi-Clinic Buffer → Float FTE)")

c1, c2, c3 = st.columns(3)
with c1:
    num_clinics = st.number_input("Clinics in Region", min_value=1, value=5, step=1)
with c2:
    avg_fte_clinic = st.number_input("Avg Baseline Provider FTE per Clinic", min_value=0.5, value=float(baseline_provider_fte), step=0.1)
with c3:
    turnover_region = st.number_input("Regional Turnover %", min_value=0.0, max_value=100.0, value=float(provider_turnover*100), step=1.0) / 100

region_departures_fte = num_clinics * avg_fte_clinic * turnover_region
region_leakage_fte = region_departures_fte * (time_to_replace_days / 365)

st.metric("Regional Fractional Leakage (FTE)", f"{region_leakage_fte:.2f}")
st.info(
    f"""
✅ Fractional staffing buffers across clinics can be pooled:
Across **{num_clinics} clinics**, turnover + replacement lag creates **{region_leakage_fte:.2f} FTE** of predictable leakage.
Rather than each clinic carrying 0.1–0.3 FTE, you can build a **regional float pool**.
"""
)


# ============================================================
# ✅ A8.1 — Float Pool ROI Comparison + Hybrid Slider + CI
# ============================================================
st.markdown("---")
st.header("A8.1 — Float Pool ROI Comparison (EBITDA Impact)")
st.caption("Compares full staffing vs float pool vs hybrid mix.")


# --- Financial Inputs ---
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
annual_margin = annual_visits * margin_per_visit


# --- Turnover Cost CI ---
st.subheader("Turnover Cost Confidence Interval")
tc1, tc2 = st.columns(2)
with tc1:
    turnover_cost_low = st.number_input("Turnover Cost Low ($)", min_value=0.0, value=100000.0, step=5000.0)
with tc2:
    turnover_cost_high = st.number_input("Turnover Cost High ($)", min_value=0.0, value=200000.0, step=5000.0)

turnover_cost_mid = (turnover_cost_low + turnover_cost_high) / 2


# --- Hybrid Slider ---
st.subheader("Hybrid Mix Strategy")
hybrid_mix = st.slider(
    "Hybrid Mix: % Fixed Staffing (Recommended) vs % Float Pool",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="0% = all float pool, 100% = staff each clinic to full recommended target."
) / 100


# --- Float Pool Inputs ---
st.subheader("Float Pool Coverage Assumptions")
fp1, fp2, fp3 = st.columns(3)
with fp1:
    float_effectiveness = st.number_input("Float Coverage Effectiveness (% gap covered)", 0.0, 100.0, 75.0, 5.0) / 100
with fp2:
    float_cost_fte = st.number_input("Loaded Cost per Float FTE ($)", min_value=0.0, value=155000.0, step=5000.0)
with fp3:
    float_overhead_pct = st.number_input("Float Program Overhead (%)", 0.0, 30.0, 8.0, 1.0) / 100


# --- Burnout gap total under recommended supply ---
gap_provider_days_total = provider_day_gap(protective_curve, realistic_supply_recommended, days_in_month)
months_exposed = sum([1 for t, s in zip(protective_curve, realistic_supply_recommended) if t > s])


# --- Float pool covers some of the gap ---
gap_days_after_float = gap_provider_days_total * (1 - float_effectiveness)
lost_visits_after_float = gap_days_after_float * (visits / max(baseline_provider_fte, 0.25))
lost_margin_after_float = lost_visits_after_float * margin_per_visit


# --- Staffing investment cost (fixed recommended vs lean target) ---
delta_fte_curve = [max(r - l, 0) for r, l in zip(protective_curve, provider_base_demand)]
fixed_staffing_cost_annual = annualize_monthly_fte_cost(delta_fte_curve, loaded_cost_per_provider_fte, days_in_month)


# --- Float program cost (regional leakage sized) ---
float_program_fte = region_leakage_fte
float_program_cost_annual = float_program_fte * float_cost_fte * (1 + float_overhead_pct)


# --- Turnover savings modeled as exposure avoided, using mid CI ---
turnover_exposure_lean = baseline_provider_fte * provider_turnover * turnover_cost_mid
turnover_exposure_fixed = baseline_provider_fte * provider_turnover * (1 - burnout_slider * 0.35) * turnover_cost_mid
turnover_exposure_float = baseline_provider_fte * provider_turnover * (1 - burnout_slider * 0.35 * float_effectiveness) * turnover_cost_mid

turnover_savings_fixed = max(turnover_exposure_lean - turnover_exposure_fixed, 0)
turnover_savings_float = max(turnover_exposure_lean - turnover_exposure_float, 0)


# --- Margin recovered from closing gap ---
lost_margin_total = gap_provider_days_total * (visits / max(baseline_provider_fte, 0.25)) * margin_per_visit
recovered_margin_float = max(lost_margin_total - lost_margin_after_float, 0)


# --- Productivity uplift from stability (float reduces burnout partially) ---
productivity_uplift_fixed = burnout_slider * 0.06
productivity_uplift_float = burnout_slider * 0.06 * float_effectiveness

productivity_margin_fixed = annual_visits * productivity_uplift_fixed * margin_per_visit
productivity_margin_float = annual_visits * productivity_uplift_float * margin_per_visit


# ============================================================
# ✅ STRATEGIES
# ============================================================
# Strategy A: Fixed recommended staffing
strategyA_savings = turnover_savings_fixed + productivity_margin_fixed + lost_margin_total
strategyA_cost = fixed_staffing_cost_annual
strategyA_net = strategyA_savings - strategyA_cost

# Strategy B: Lean + float pool
strategyB_savings = turnover_savings_float + productivity_margin_float + recovered_margin_float
strategyB_cost = float_program_cost_annual
strategyB_net = strategyB_savings - strategyB_cost

# Strategy C: Hybrid mix
strategyC_savings = (hybrid_mix * strategyA_savings) + ((1 - hybrid_mix) * strategyB_savings)
strategyC_cost = (hybrid_mix * strategyA_cost) + ((1 - hybrid_mix) * strategyB_cost)
strategyC_net = strategyC_savings - strategyC_cost


# ============================================================
# ✅ DISPLAY RESULTS
# ============================================================
st.subheader("Strategy Comparison (Annual)")

d1, d2, d3 = st.columns(3)
d1.metric("A) Fixed Recommended (Net)", f"${strategyA_net:,.0f}")
d2.metric("B) Lean + Float Pool (Net)", f"${strategyB_net:,.0f}")
d3.metric("C) Hybrid Mix (Net)", f"${strategyC_net:,.0f}")

# Recommendation
best = max(strategyA_net, strategyB_net, strategyC_net)
if best == strategyA_net:
    best_label = "Strategy A — Fixed Recommended Staffing"
elif best == strategyB_net:
    best_label = "Strategy B — Lean + Float Pool"
else:
    best_label = "Strategy C — Hybrid Mix"

st.success(f"✅ Recommended: **{best_label}**")


# ============================================================
# ✅ Burnout Risk Reduction (Months)
# ============================================================
float_exposed_months = int(months_exposed * (1 - float_effectiveness))
st.info(
    f"""
✅ Burnout Risk Reduction
- Months exposed under Fixed Recommended: **{months_exposed}/12**
- Months exposed after Float Pool coverage (estimated): **{float_exposed_months}/12**
"""
)


# ============================================================
# ✅ DOWNLOADABLE FLOAT PLAN
# ============================================================
st.subheader("Downloadable Float Pool Staffing Plan")

float_plan_df = pd.DataFrame({
    "Region Clinics": [num_clinics],
    "Regional Leakage FTE": [round(region_leakage_fte, 2)],
    "Recommended Float FTE": [round(float_program_fte, 2)],
    "Float Cost per FTE": [round(float_cost_fte, 0)],
    "Overhead %": [round(float_overhead_pct * 100, 1)],
    "Annual Float Program Cost": [round(float_program_cost_annual, 0)],
    "Float Coverage Effectiveness": [round(float_effectiveness * 100, 1)],
    "Burnout Gap Provider Days (Annual)": [round(gap_provider_days_total, 1)],
    "Recovered Margin from Float Pool": [round(recovered_margin_float, 0)],
})

st.dataframe(float_plan_df, hide_index=True, use_container_width=True)

csv = float_plan_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Float Pool Staffing Plan (CSV)",
    data=csv,
    file_name="float_pool_staffing_plan.csv",
    mime="text/csv"
)
