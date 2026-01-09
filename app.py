import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ============================================================
# ✅ PAGE CONFIG + HYBRID WIDTH
# ============================================================
st.set_page_config(page_title="PSM Staffing Calculator", layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
div[data-testid="stDataFrame"] {
    overflow-x: auto;
}
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
# ✅ STABLE TODAY
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]


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

def build_flu_window(year: int, flu_start_month: int, flu_end_month: int):
    flu_start_date = datetime(year, flu_start_month, 1)

    if flu_end_month < flu_start_month:
        flu_end_date = datetime(year + 1, flu_end_month, 1)
    else:
        flu_end_date = datetime(year, flu_end_month, 1)

    flu_end_date = flu_end_date + timedelta(days=32)
    flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)

    return flu_start_date, flu_end_date

def in_window(d: datetime, start: datetime, end: datetime):
    return start <= d <= end

def compute_seasonality_forecast(dates, baseline_visits, flu_start_month, flu_end_month, flu_uplift_pct):
    raw = []
    for d in dates:
        d_py = d.to_pydatetime()
        mult = base_seasonality_multiplier(d_py.month)

        flu_start, flu_end = build_flu_window(d_py.year, flu_start_month, flu_end_month)

        if in_window(d_py, flu_start, flu_end):
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
# ✅ Burnout-Protective Curve
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
# ✅ PIPELINE SUPPLY CURVE (AUTO RAMP)
# ============================================================
def realistic_supply_pipeline_auto_ramp(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    req_post_date,
    solo_ready_date,
    ramp_after_solo,
    max_ramp_down_per_month=0.25,
    seasonality_ramp_enabled=True,
):
    monthly_attrition_fte = baseline_fte * (annual_turnover_rate / 12)
    effective_attrition_start = today + timedelta(days=int(notice_days))

    staff = []
    prev = max(baseline_fte, provider_min_floor)

    for d, target in zip(dates, target_curve):
        d_py = d.to_pydatetime()

        # Hiring logic
        if seasonality_ramp_enabled:
            if d_py < req_post_date:
                ramp_up_cap = 0.0
            elif d_py < solo_ready_date:
                ramp_up_cap = 0.0
            else:
                ramp_up_cap = ramp_after_solo
        else:
            ramp_up_cap = ramp_after_solo

        # Move toward target
        delta = target - prev
        if delta > 0:
            delta = clamp(delta, 0.0, ramp_up_cap)
        else:
            delta = clamp(delta, -max_ramp_down_per_month, 0.0)

        planned = prev + delta

        # Attrition
        if d_py >= effective_attrition_start:
            months_elapsed = monthly_index(d_py, effective_attrition_start)
            planned = max(planned - months_elapsed * monthly_attrition_fte, provider_min_floor)

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
    flu_start_month = st.selectbox("Flu Start Month", list(range(1, 13)), index=11,
                                  format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
    flu_end_month = st.selectbox("Flu End Month", list(range(1, 13)), index=1,
                                format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100

    st.subheader("Seasonality Recruiting Ramp")
    enable_seasonality_ramp = st.checkbox(
        "Enable Seasonality Recruiting Ramp (recommended)",
        value=True,
        help="If ON: Supply cannot rise until requisitions post; hires appear after solo-ready and supply accelerates."
    )

    run_model = st.button("Run Model")


# ============================================================
# ✅ RUN MODEL
# ============================================================
if run_model:

    # Flu start date for current year
    flu_start_date = datetime(today.year, flu_start_month, 1)

    # Timeline markers
    total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days
    req_post_date = flu_start_date - timedelta(days=total_lead_days)
    solo_ready_date = flu_start_date

    # ✅ Option A timeline window: start 6 months before req_post_date, show 18 months
    timeline_start = (req_post_date - timedelta(days=180)).replace(day=1)
    dates = pd.date_range(start=timeline_start, periods=18, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

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
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
        flu_uplift_pct=flu_uplift_pct
    )

    # Demand
    provider_base_demand = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_by_month,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    # Recommended target
    protective_curve = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # ✅ Auto-calc required ramp after solo-ready
    target_at_flu_start = protective_curve[list(dates).index(pd.Timestamp(flu_start_date))]
    gap_needed = max(target_at_flu_start - baseline_provider_fte, 0)

    months_available = max(1, monthly_index(flu_start_date, solo_ready_date) + 3)
    derived_ramp_after_solo = max(gap_needed / months_available, 0.25)

    # Supply curves
    realistic_supply_lean = realistic_supply_pipeline_auto_ramp(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        solo_ready_date=solo_ready_date,
        ramp_after_solo=derived_ramp_after_solo,
        seasonality_ramp_enabled=enable_seasonality_ramp,
    )

    realistic_supply_recommended = realistic_supply_pipeline_auto_ramp(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=protective_curve,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        solo_ready_date=solo_ready_date,
        ramp_after_solo=derived_ramp_after_solo,
        seasonality_ramp_enabled=enable_seasonality_ramp,
    )

    burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    months_exposed = sum([1 for g in burnout_gap_fte if g > 0])

    st.session_state["results"] = dict(
        dates=dates,
        month_labels=month_labels,
        days_in_month=days_in_month,
        baseline_provider_fte=baseline_provider_fte,
        flu_start_date=flu_start_date,
        forecast_visits_by_month=forecast_visits_by_month,
        provider_base_demand=provider_base_demand,
        protective_curve=protective_curve,
        realistic_supply_recommended=realistic_supply_recommended,
        burnout_gap_fte=burnout_gap_fte,
        months_exposed=months_exposed,
        req_post_date=req_post_date,
        solo_ready_date=solo_ready_date,
        derived_ramp_after_solo=derived_ramp_after_solo,
        enable_seasonality_ramp=enable_seasonality_ramp,
    )

# ============================================================
# ✅ STOP IF NOT RAN
# ============================================================
if "results" not in st.session_state or st.session_state["results"] is None:
    st.info("Enter inputs in the sidebar and click **Run Model**.")
    st.stop()

R = st.session_state["results"]


# ============================================================
# ✅ SECTION 2 (Reality) — Chart First
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Shows whether recruiting in advance can realistically meet protective targets during flu season peaks.")

fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(R["dates"], R["provider_base_demand"], linestyle=":", linewidth=2, label="Lean Target (Demand)")
ax1.plot(R["dates"], R["protective_curve"], linewidth=3, marker="o", label="Recommended Target (Protective)")
ax1.plot(R["dates"], R["realistic_supply_recommended"], linewidth=3, marker="o", label="Realistic Supply (Pipeline + Auto Ramp)")

ax1.fill_between(
    R["dates"],
    R["realistic_supply_recommended"],
    R["protective_curve"],
    where=np.array(R["protective_curve"]) > np.array(R["realistic_supply_recommended"]),
    alpha=0.25,
    label="Burnout Exposure Zone"
)

ax1.set_title("A6 — Volume, Targets, Supply & Burnout Exposure (Pipeline + Auto Ramp)")
ax1.set_ylabel("Provider FTE")
ax1.set_xticks(R["dates"])
ax1.set_xticklabels(R["month_labels"])
ax1.grid(axis="y", linestyle=":", alpha=0.35)

ax2 = ax1.twinx()
ax2.plot(R["dates"], R["forecast_visits_by_month"], linestyle="-.", linewidth=2.5, label="Forecast Visits/Day")
ax2.set_ylabel("Visits / Day")

for marker_date, label in [(R["req_post_date"], "Req Post By"), (R["solo_ready_date"], "Solo By")]:
    ax1.axvline(marker_date, linestyle="--", linewidth=1.5, alpha=0.6)
    ax1.text(marker_date, ax1.get_ylim()[1], label, rotation=90, fontsize=8, va="bottom")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, -0.22), ncol=2)

plt.tight_layout()
st.pyplot(fig)

st.success(
    f"✅ **Reality Executive Summary:** The model assumes leaders post requisitions by **{R['req_post_date'].strftime('%b %d')}**, "
    f"hires go solo by **{R['solo_ready_date'].strftime('%b %d')}**, and supply ramps at ~**{R['derived_ramp_after_solo']:.2f} FTE/month** after solo-ready. "
    f"Burnout exposure remains in **{R['months_exposed']}/18 months**."
)
