import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ============================================================
# ✅ PAGE CONFIG + HYBRID WIDTH (Option B)
# ============================================================
st.set_page_config(page_title="PSM Staffing Calculator", layout="wide")

st.markdown("""
<style>
/* Hybrid layout: wide mode, but constrained readable width */
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Dataframe: avoid cramped mobile display */
div[data-testid="stDataFrame"] {
    overflow-x: auto;
}

/* Slightly tighter metric padding */
div[data-testid="metric-container"] {
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Predictive Staffing Model (PSM)")
st.caption("Operations → Reality → Finance → Strategy → Decision")

st.info(
    "⚠️ **All staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)

model = StaffingModel()


# ============================================================
# ✅ STABLE 'TODAY' FOR CONSISTENT CHART WINDOWS
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]


# ============================================================
# ✅ SESSION STATE (prevents resets & keeps results stable)
# ============================================================
STATE_KEYS = [
    "model_ran",
    "results",
]
for k in STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None


# ============================================================
# ✅ HELPERS
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
# ✅ BURNOUT-PROTECTIVE CURVE (Recommended Target)
# ============================================================
def burnout_protective_staffing_curve(
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
# ✅ PIPELINE SUPPLY CURVE + SEASONALITY RECRUITING RAMP
# ============================================================
def realistic_staffing_supply_curve_pipeline(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    req_post_date,
    solo_ready_date,
    max_hiring_up_per_month=0.50,
    max_hiring_up_after_pipeline=0.85,
    max_ramp_down_per_month=0.25,
    seasonality_ramp_enabled=True,
):
    """
    Pipeline-aware realistic staffing supply curve + post-pipeline acceleration.

    Rules:
    - If date < req_post_date → no hiring possible
    - If req_post_date <= date < solo_ready_date → hiring in pipeline (not visible yet)
    - If date >= solo_ready_date → hires show up and supply can accelerate toward target

    Attrition begins after notice_days from today.
    """

    monthly_attrition_fte = baseline_fte * (annual_turnover_rate / 12)
    effective_attrition_start = today + timedelta(days=int(notice_days))

    staff = []
    prev = max(baseline_fte, provider_min_floor)

    for d, target in zip(dates, target_curve):

        d_py = d.to_pydatetime()

        # -------------------------------
        # ✅ Determine ramp cap
        # -------------------------------
        ramp_up_cap = max_hiring_up_per_month

        if seasonality_ramp_enabled:
            if d_py < req_post_date:
                ramp_up_cap = 0.0
            elif d_py < solo_ready_date:
                ramp_up_cap = 0.0
            else:
                ramp_up_cap = max_hiring_up_after_pipeline

        # -------------------------------
        # ✅ Move supply toward target
        # -------------------------------
        delta = target - prev

        if delta > 0:
            delta = clamp(delta, 0.0, ramp_up_cap)
        else:
            delta = clamp(delta, -max_ramp_down_per_month, 0.0)

        planned = prev + delta

        # -------------------------------
        # ✅ Attrition after notice lag
        # -------------------------------
        if d_py >= effective_attrition_start:
            months_elapsed = monthly_index(d_py, effective_attrition_start)
            attrition_loss = months_elapsed * monthly_attrition_fte
            planned = max(planned - attrition_loss, provider_min_floor)

        planned = max(planned, provider_min_floor)

        staff.append(planned)
        prev = planned

    return staff


# ============================================================
# ✅ COST HELPERS
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
# ✅ SIDEBAR INPUTS
# ============================================================
with st.sidebar:
    st.header("Inputs")

    st.subheader("Baseline")
    visits = st.number_input("Avg Visits/Day (annual avg)", min_value=1.0, value=45.0, step=1.0)
    hours_of_operation = st.number_input("Hours of Operation / Week", min_value=1.0, value=70.0, step=1.0)
    fte_hours_per_week = st.number_input("FTE Hours / Week", min_value=1.0, value=40.0, step=1.0)

    st.subheader("Floors & Protection")
    provider_min_floor = st.number_input("Provider Minimum Floor (FTE)", min_value=0.25, value=1.00, step=0.25)
    burnout_slider = st.slider("Burnout Protection Level", 0.0, 1.0, 0.6, 0.05)
    safe_visits_per_provider = st.number_input("Safe Visits/Provider/Day", 10, 40, 20, 1)

    st.subheader("Turnover + Pipeline")
    provider_turnover = st.number_input("Provider Turnover % (annual)", value=24.0, step=1.0) / 100

    with st.expander("Provider Hiring Pipeline Assumptions", expanded=False):
        days_to_sign = st.number_input("Days to Sign", min_value=0, value=90, step=5)
        days_to_credential = st.number_input("Days to Credential", min_value=0, value=90, step=5)
        onboard_train_days = st.number_input("Days to Train", min_value=0, value=30, step=5)
        coverage_buffer_days = st.number_input("Planning Buffer Days", min_value=0, value=14, step=1)
        notice_days = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=75, step=5)

    st.subheader("Seasonality")
    flu_start_month = st.selectbox(
        "Flu Start Month",
        options=list(range(1, 13)),
        index=11,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    flu_end_month = st.selectbox(
        "Flu End Month",
        options=list(range(1, 13)),
        index=1,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100

    st.subheader("Seasonality Recruiting Ramp")
    enable_seasonality_ramp = st.checkbox(
        "Enable Seasonality Recruiting Ramp (recommended)",
        value=True,
        help="If ON: Supply cannot rise until requisitions post; after hires go solo, supply accelerates toward target."
    )

    max_hiring_up_per_month = st.number_input(
        "Base Hiring Ramp Limit (FTE/month)",
        min_value=0.05,
        value=0.50,
        step=0.05
    )

    max_hiring_up_after_pipeline = st.number_input(
        "Accelerated Ramp After Solo-Ready (FTE/month)",
        min_value=0.10,
        value=0.85,
        step=0.05,
        help="After solo_ready_date, supply can rise faster to meet seasonal peaks."
    )

    st.subheader("Run")
    run_model = st.button("Run Model")


# ============================================================
# ✅ RUN MODEL (only compute when user clicks)
# ============================================================
if run_model:

    current_year = today.year
    dates = pd.date_range(start=datetime(current_year, 1, 1), periods=12, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    flu_start_date, flu_end_date = build_flu_window(current_year, flu_start_month, flu_end_month)

    # Baseline provider FTE
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    baseline_provider_fte = max(fte_result["provider_fte"], provider_min_floor)

    # Forecast visits
    forecast_visits_by_month = compute_seasonality_forecast(
        dates=dates,
        baseline_visits=visits,
        flu_start=flu_start_date,
        flu_end=flu_end_date,
        flu_uplift_pct=flu_uplift_pct,
    )

    # Demand curve
    provider_base_demand = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_by_month,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    # Recommended curve
    protective_curve = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # Hiring timeline markers for flu ramp-up
    staffing_needed_by = flu_start_date
    total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days
    req_post_date = staffing_needed_by - timedelta(days=total_lead_days)
    solo_ready_date = staffing_needed_by

    # Supply curves (pipeline-aware + ramp flag + acceleration)
    realistic_supply_lean = realistic_staffing_supply_curve_pipeline(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        solo_ready_date=solo_ready_date,
        max_hiring_up_per_month=max_hiring_up_per_month,
        max_hiring_up_after_pipeline=max_hiring_up_after_pipeline,
        seasonality_ramp_enabled=enable_seasonality_ramp,
    )

    realistic_supply_recommended = realistic_staffing_supply_curve_pipeline(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=protective_curve,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        solo_ready_date=solo_ready_date,
        max_hiring_up_per_month=max_hiring_up_per_month,
        max_hiring_up_after_pipeline=max_hiring_up_after_pipeline,
        seasonality_ramp_enabled=enable_seasonality_ramp,
    )

    burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    months_exposed = sum([1 for g in burnout_gap_fte if g > 0])

    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        dates=dates,
        month_labels=month_labels,
        days_in_month=days_in_month,
        baseline_provider_fte=baseline_provider_fte,
        fte_result=fte_result,
        flu_start_date=flu_start_date,
        flu_end_date=flu_end_date,
        forecast_visits_by_month=forecast_visits_by_month,
        provider_base_demand=provider_base_demand,
        protective_curve=protective_curve,
        realistic_supply_lean=realistic_supply_lean,
        realistic_supply_recommended=realistic_supply_recommended,
        burnout_gap_fte=burnout_gap_fte,
        months_exposed=months_exposed,
        req_post_date=req_post_date,
        solo_ready_date=solo_ready_date,
        enable_seasonality_ramp=enable_seasonality_ramp,
        max_hiring_up_per_month=max_hiring_up_per_month,
        max_hiring_up_after_pipeline=max_hiring_up_after_pipeline,
    )


# ============================================================
# ✅ STOP IF NOT RAN YET
# ============================================================
if not st.session_state.get("model_ran"):
    st.info("Enter inputs in the sidebar and click **Run Model**.")
    st.stop()

R = st.session_state["results"]


# ============================================================
# ✅ SECTION 1 — OPERATIONS
# ============================================================
st.markdown("---")
st.header("1) Operations — Seasonality Staffing Requirements")
st.caption("Visits/day forecast → staff/day → FTE needed by month (all based on seasonality).")

monthly_rows = []
for month_label, v in zip(R["month_labels"], R["forecast_visits_by_month"]):
    fte_staff = model.calculate_fte_needed(
        visits_per_day=v,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week
    )

    provider_day = (fte_staff["provider_fte"] * fte_hours_per_week) / max(hours_of_operation, 1)
    psr_day      = (fte_staff["psr_fte"]      * fte_hours_per_week) / max(hours_of_operation, 1)
    ma_day       = (fte_staff["ma_fte"]       * fte_hours_per_week) / max(hours_of_operation, 1)
    xrt_day      = (fte_staff["xrt_fte"]      * fte_hours_per_week) / max(hours_of_operation, 1)

    monthly_rows.append({
        "Month": month_label,
        "Visits/Day (Forecast)": round(v, 1),

        "Providers Needed/Day": round(provider_day, 2),
        "PSR Needed/Day": round(psr_day, 2),
        "MA Needed/Day": round(ma_day, 2),
        "XRT Needed/Day": round(xrt_day, 2),

        "Provider FTE": round(fte_staff["provider_fte"], 2),
        "PSR FTE": round(fte_staff["psr_fte"], 2),
        "MA FTE": round(fte_staff["ma_fte"], 2),
        "XRT FTE": round(fte_staff["xrt_fte"], 2),
        "Total FTE": round(fte_staff["total_fte"], 2),
    })

ops_df = pd.DataFrame(monthly_rows)
st.dataframe(ops_df, hide_index=True, use_container_width=True)

peak_visits = ops_df["Visits/Day (Forecast)"].max()
peak_total_fte = ops_df["Total FTE"].max()
low_total_fte = ops_df["Total FTE"].min()

k1, k2, k3 = st.columns(3)
k1.metric("Peak Visits/Day", f"{peak_visits:.1f}")
k2.metric("Peak Total FTE Needed", f"{peak_total_fte:.2f}")
k3.metric("Low Season Total FTE Needed", f"{low_total_fte:.2f}")

st.success(
    "✅ **Ops Executive Summary:** Seasonal volume shifts create measurable changes in staff/day and FTE requirements. "
    "Use this table for budgeting and staffing pattern planning."
)


# ============================================================
# ✅ SECTION 2 — REALITY
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Compares lean vs recommended targets against realistic supply given hiring lead time + attrition.")

burnout_gap_fte = R["burnout_gap_fte"]
months_exposed = R["months_exposed"]

fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(R["dates"], R["provider_base_demand"], linestyle=":", linewidth=2, label="Lean Target (Demand)")
ax1.plot(R["dates"], R["protective_curve"], linewidth=3, marker="o", label="Recommended Target (Protective)")
ax1.plot(R["dates"], R["realistic_supply_recommended"], linewidth=3, marker="o", label="Realistic Supply (Pipeline + Ramp)")

ax1.fill_between(
    R["dates"],
    R["realistic_supply_recommended"],
    R["protective_curve"],
    where=np.array(R["protective_curve"]) > np.array(R["realistic_supply_recommended"]),
    alpha=0.25,
    label="Burnout Exposure Zone"
)

ax1.set_title("A6 — Volume, Targets, Supply & Burnout Exposure (Pipeline + Ramp)")
ax1.set_ylabel("Provider FTE")
ax1.set_xticks(R["dates"])
ax1.set_xticklabels(R["month_labels"])
ax1.grid(axis="y", linestyle=":", alpha=0.35)

ax2 = ax1.twinx()
ax2.plot(R["dates"], R["forecast_visits_by_month"], linestyle="-.", linewidth=2.5, label="Forecast Visits/Day")
ax2.set_ylabel("Visits / Day")

# Hiring markers
ymax = ax1.get_ylim()[1]
for marker_date, label in [(R["req_post_date"], "Req Post By"), (R["solo_ready_date"], "Solo By")]:
    if R["dates"][0].to_pydatetime() <= marker_date <= R["dates"][-1].to_pydatetime():
        ax1.axvline(marker_date, linestyle="--", linewidth=1.5, alpha=0.6)
        ax1.annotate(label, xy=(marker_date, ymax), xytext=(marker_date, ymax + 0.25),
                     ha="center", fontsize=9, rotation=90)

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

if R.get("enable_seasonality_ramp"):
    st.success(
        "✅ **Reality Executive Summary:** Supply assumes leaders post requisitions early enough to meet flu season demand. "
        "Hires do not appear until the **Solo-Ready date**, and supply then accelerates toward target."
    )
else:
    st.warning(
        "⚠️ **Reality Executive Summary:** Seasonality recruiting ramp is OFF. "
        "Supply reflects generic ramping only, which increases peak-month staffing exposure."
    )


# ============================================================
# ✅ SECTION 3 — FINANCE
# ============================================================
st.markdown("---")
st.header("3) Finance — Investment Case (EBITDA Impact)")
st.caption("Cost to staff to recommended vs expected exposure if operating lean.")

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

leakage_factor = st.number_input("Demand Leakage Factor (0–1)", 0.0, 1.0, 0.60, 0.05)
max_productivity_uplift = st.number_input("Max Productivity Uplift at Protection=1.0 (%)", 0.0, 30.0, 6.0, 1.0) / 100
max_turnover_reduction = st.number_input("Max Turnover Reduction at Protection=1.0 (%)", 0.0, 100.0, 35.0, 5.0) / 100

delta_fte_curve = [max(r - l, 0) for r, l in zip(R["protective_curve"], R["provider_base_demand"])]
incremental_staffing_cost_annual = annualize_monthly_fte_cost(delta_fte_curve, R["days_in_month"], loaded_cost_per_provider_fte)

def compute_exposure_annual(target_curve, supply_curve, turnover_rate, turnover_cost):
    provider_count = R["baseline_provider_fte"]
    expected_departures = provider_count * turnover_rate
    turnover_exposure = expected_departures * turnover_cost

    gap_provider_days = provider_day_gap(target_curve, supply_curve, R["days_in_month"])
    visits_per_provider_fte_per_day = visits / max(R["baseline_provider_fte"], 0.25)

    lost_visits = gap_provider_days * visits_per_provider_fte_per_day * leakage_factor
    lost_margin = lost_visits * margin_per_visit

    premium_exposure = 0.0
    if use_premium_labor:
        premium_exposure = gap_provider_days * provider_day_cost_basis * premium_pct

    return turnover_exposure + lost_margin + premium_exposure

exposure_lean_total = compute_exposure_annual(
    R["provider_base_demand"], R["realistic_supply_lean"], provider_turnover, turnover_cost_mid
)

turnover_recommended = provider_turnover * (1 - max_turnover_reduction * burnout_slider)
exposure_rec_total = compute_exposure_annual(
    R["protective_curve"], R["realistic_supply_recommended"], turnover_recommended, turnover_cost_mid
)

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
    f"✅ **Finance Executive Summary:** Staffing to recommended requires **${A_display:,.0f}**, "
    f"avoids **${C_display:,.0f}** in exposure + margin loss, and produces a net EBITDA impact of **${NET_display:,.0f}**."
)


# ============================================================
# ✅ SECTION 4 — STRATEGY
# ============================================================
st.markdown("---")
st.header("4) Strategy — Recruiting Buffer + Float Pool + Fractional Staffing")
st.caption("Explicitly models turnover buffer and converts fractional gaps into float pool strategy.")

time_to_replace_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days
expected_departures_fte = R["baseline_provider_fte"] * provider_turnover
coverage_leakage_fte = expected_departures_fte * (time_to_replace_days / 365)
pipeline_target_fte = R["baseline_provider_fte"] + coverage_leakage_fte

st.subheader("A8 — Recruiting Buffer (Turnover + Time-to-Replace)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Baseline Provider FTE", f"{R['baseline_provider_fte']:.2f}")
k2.metric("Expected Departures (FTE/Yr)", f"{expected_departures_fte:.2f}")
k3.metric("Coverage Leakage Buffer (FTE)", f"{coverage_leakage_fte:.2f}")
k4.metric("Recruiting Pipeline Target (FTE)", f"{pipeline_target_fte:.2f}")

st.subheader("Fractional Staffing Across Sister Clinics (Float Pool Builder)")
st.caption(
    "Instead of each clinic carrying fractional staffing buffer, a region can pool those fractions into a float provider team."
)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    num_clinics = st.number_input("Clinics in Region", min_value=1, value=5, step=1)
with fc2:
    avg_baseline_fte = st.number_input("Avg Baseline Provider FTE / Clinic", min_value=0.5, value=float(R["baseline_provider_fte"]), step=0.1)
with fc3:
    turnover_region = st.number_input("Regional Turnover %", 0.0, 100.0, float(provider_turnover*100), 1.0) / 100

region_departures = num_clinics * avg_baseline_fte * turnover_region
region_leakage = region_departures * (time_to_replace_days / 365)

region_float_fte_needed = region_leakage
region_float_providers = region_float_fte_needed

fk1, fk2, fk3 = st.columns(3)
fk1.metric("Regional Leakage (FTE)", f"{region_leakage:.2f}")
fk2.metric("Float Pool Recommended (FTE)", f"{region_float_fte_needed:.2f}")
fk3.metric("Float Providers (Headcount)", f"{region_float_providers:.1f}")

st.subheader("Hybrid Strategy Slider (Fixed + Float Mix)")
hybrid_pct_fixed = st.slider(
    "Percent of burnout gap closed with fixed hiring (remainder covered by float pool)",
    min_value=0, max_value=100, value=60, step=5
) / 100
float_pct = 1 - hybrid_pct_fixed

gap_provider_days_total = provider_day_gap(R["protective_curve"], R["realistic_supply_recommended"], R["days_in_month"])
burnout_months_before = R["months_exposed"]
burnout_months_after = int(round(burnout_months_before * (1 - float_pct)))

st.success(
    f"✅ **Strategy Executive Summary:** To maintain **{R['baseline_provider_fte']:.2f} FTE**, "
    f"leaders must recruit against predictable leakage of **{coverage_leakage_fte:.2f} FTE** caused by turnover + replacement lag. "
    f"A regional float pool of **~{region_float_providers:.1f} providers** can absorb these fractional gaps. "
    f"With a hybrid mix of **{hybrid_pct_fixed*100:.0f}% fixed / {float_pct*100:.0f}% float**, burnout exposure months drop from "
    f"**{burnout_months_before} → ~{burnout_months_after}**."
)

float_plan_df = pd.DataFrame({
    "Metric": [
        "Clinics in Region",
        "Regional Leakage (FTE)",
        "Float Providers Recommended",
        "Hybrid Fixed %",
        "Hybrid Float %"
    ],
    "Value": [
        num_clinics,
        round(region_leakage, 2),
        round(region_float_providers, 2),
        round(hybrid_pct_fixed, 2),
        round(float_pct, 2),
    ]
})

csv = float_plan_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Float Pool Staffing Plan (CSV)", csv, "float_pool_plan.csv", "text/csv")


# ============================================================
# ✅ SECTION 5 — DECISION (BIG EXEC SUMMARY)
# ============================================================
st.markdown("---")
st.header("5) Decision — PSM Final Recommendation")
st.caption("Decision-ready summary: operations → reality → finance → strategy.")

if net_ebitda_impact_annual >= 0:
    decision_tone = "✅ EBITDA-Positive"
    decision = "Proceed with staffing to Recommended Target using a hybrid fixed + float strategy."
elif net_ebitda_impact_annual > -5000:
    decision_tone = "⚠️ Near Breakeven"
    decision = "Proceed with recommended staffing — stability benefits outweigh minimal EBITDA drag."
else:
    decision_tone = "❌ EBITDA-Negative"
    decision = "Use a float-heavy strategy and selective fixed hiring in peak months to reduce cost."

st.markdown(f"## {decision_tone}")
st.write(decision)

summary_df = pd.DataFrame({
    "Decision Factor": [
        "Baseline Provider FTE",
        "Pipeline Recruiting Target (FTE)",
        "Peak Burnout Gap (FTE)",
        "Burnout Months Exposed",
        f"Cost to Staff (A) — {time_horizon}",
        f"Expected Savings (C) — {time_horizon}",
        f"Net EBITDA Impact — {time_horizon}",
        "Float Pool Recommended (Providers)",
        "Hybrid Mix (Fixed / Float)"
    ],
    "Value": [
        f"{R['baseline_provider_fte']:.2f}",
        f"{pipeline_target_fte:.2f}",
        f"{max(R['burnout_gap_fte']):.2f}",
        f"{R['months_exposed']}/12",
        f"${A_display:,.0f}",
        f"${C_display:,.0f}",
        f"${NET_display:,.0f}",
        f"{region_float_providers:.1f}",
        f"{hybrid_pct_fixed*100:.0f}% / {float_pct*100:.0f}%"
    ]
})

st.dataframe(summary_df, hide_index=True, use_container_width=True)

st.success(
    "✅ **Final Executive Summary:** This output ties together seasonality-driven staffing needs, realistic supply constraints, "
    "burnout exposure, financial ROI, recruiting buffer requirements, and float/fractional staffing strategy into a leadership-ready decision."
)
