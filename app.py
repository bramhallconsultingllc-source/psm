import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ============================================================
# ✅ PAGE CONFIG (OPTION A — Centered)
# ============================================================
st.set_page_config(page_title="PSM Staffing Calculator", layout="centered")

st.title("Predictive Staffing Model (PSM)")
st.caption("Operations → Reality → Finance → Strategy → Decision")

st.info(
    "⚠️ **All staffing outputs round UP to the nearest 0.25 FTE/day.** "
    "This is intentional to prevent under-staffing."
)

model = StaffingModel()


# ============================================================
# ✅ STABLE TODAY (prevents moving windows on reruns)
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]


# ============================================================
# ✅ SESSION STATE
# ============================================================
for key in ["model_ran", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None


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

    # last day of flu_end month
    flu_end_date = flu_end_date + timedelta(days=32)
    flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)

    return flu_start_date, flu_end_date

def in_window(d: datetime, start: datetime, end: datetime):
    return start <= d <= end

def compute_seasonality_forecast(dates, baseline_visits, flu_start, flu_end, flu_uplift_pct):
    raw = []
    for d in dates:
        mult = base_seasonality_multiplier(d.month)
        if in_window(d.to_pydatetime(), flu_start, flu_end):
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
            fte_hours_per_week=fte_hours_per_week
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
# ✅ PIPELINE SUPPLY CURVE + AUTO-SEASONALITY RAMP
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
    max_hiring_up_after_pipeline,
    max_ramp_down_per_month=0.25,
    seasonality_ramp_enabled=True,
):
    """
    Pipeline-aware realistic staffing supply curve.

    - date < req_post_date → supply cannot grow
    - req_post_date <= date < solo_ready_date → hiring in pipeline (not visible)
    - date >= solo_ready_date → hires show up and supply accelerates toward target
    """

    monthly_attrition_fte = baseline_fte * (annual_turnover_rate / 12)
    effective_attrition_start = today + timedelta(days=int(notice_days))

    staff = []
    prev = max(baseline_fte, provider_min_floor)

    for d, target in zip(dates, target_curve):
        d_py = d.to_pydatetime()

        # -------------------------------
        # Determine ramp cap
        # -------------------------------
        if seasonality_ramp_enabled:
            if d_py < req_post_date:
                ramp_up_cap = 0.0
            elif d_py < solo_ready_date:
                ramp_up_cap = 0.0
            else:
                ramp_up_cap = max_hiring_up_after_pipeline
        else:
            ramp_up_cap = 0.35  # generic ramp

        # -------------------------------
        # Move supply toward target
        # -------------------------------
        delta = target - prev
        if delta > 0:
            delta = clamp(delta, 0.0, ramp_up_cap)
        else:
            delta = clamp(delta, -max_ramp_down_per_month, 0.0)

        planned = prev + delta

        # -------------------------------
        # Attrition after notice lag
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
        help="If ON: supply cannot rise until requisitions post; hires appear at solo-ready and supply accelerates."
    )

    st.subheader("Run")
    run_model = st.button("Run Model")


# ============================================================
# ✅ RUN MODEL
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

    # ✅ Pipeline timeline: req post date is based on flu start + lead time
    staffing_needed_by = flu_start_date
    total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

    req_post_date = staffing_needed_by - timedelta(days=total_lead_days)

    # ✅ Solo-ready is flu_start_date (the “need by” point)
    solo_ready_date = staffing_needed_by

    # ============================================================
    # ✅ AUTO-DERIVE ACCELERATED RAMP SPEED AFTER SOLO READY
    # ============================================================
    flu_month_idx = next(i for i, d in enumerate(dates) if d.to_pydatetime().month == flu_start_date.month)

    # Determine last month index inside flu window
    flu_end_idx = flu_month_idx
    for i, d in enumerate(dates):
        if d.to_pydatetime() <= flu_end_date:
            flu_end_idx = i

    months_in_flu_window = max(flu_end_idx - flu_month_idx + 1, 1)

    target_at_flu = protective_curve[flu_month_idx]
    supply_at_solo = baseline_provider_fte
    fte_gap_to_close = max(target_at_flu - supply_at_solo, 0)

    derived_ramp_after_solo = fte_gap_to_close / months_in_flu_window
    derived_ramp_after_solo = min(derived_ramp_after_solo, 1.25)

    # Supply curves
    realistic_supply_lean = realistic_staffing_supply_curve_pipeline(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        solo_ready_date=solo_ready_date,
        max_hiring_up_after_pipeline=derived_ramp_after_solo,
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
        max_hiring_up_after_pipeline=derived_ramp_after_solo,
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
        derived_ramp_after_solo=derived_ramp_after_solo,
        months_in_flu_window=months_in_flu_window,
        fte_gap_to_close=fte_gap_to_close,
    )


# ============================================================
# ✅ STOP IF NOT RUN
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


# ============================================================
# ✅ SECTION 2 — REALITY
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Compares lean vs recommended targets against realistic supply given hiring lead time + attrition.")

burnout_gap_fte = R["burnout_gap_fte"]
months_exposed = R["months_exposed"]

fig, ax1 = plt.subplots(figsize=(10, 4))

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

ax1.set_title("A6 — Volume, Targets, Supply & Burnout Exposure")
ax1.set_ylabel("Provider FTE")
ax1.set_xticks(R["dates"])
ax1.set_xticklabels(R["month_labels"])
ax1.grid(axis="y", linestyle=":", alpha=0.35)

ax2 = ax1.twinx()
ax2.plot(R["dates"], R["forecast_visits_by_month"], linestyle="-.", linewidth=2.5, label="Forecast Visits/Day")
ax2.set_ylabel("Visits / Day")

# Markers
ymax = ax1.get_ylim()[1]
for marker_date, label in [(R["req_post_date"], "Req Post By"), (R["solo_ready_date"], "Solo By")]:
    if R["dates"][0].to_pydatetime() <= marker_date <= R["dates"][-1].to_pydatetime():
        ax1.axvline(marker_date, linestyle="--", linewidth=1.5, alpha=0.6)
        ax1.annotate(label, xy=(marker_date, ymax), xytext=(marker_date, ymax + 0.2),
                     ha="center", fontsize=9, rotation=90)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, -0.25), ncol=2)

plt.tight_layout()
st.pyplot(fig)

k1, k2, k3 = st.columns(3)
k1.metric("Peak Burnout Gap (FTE)", f"{max(burnout_gap_fte):.2f}")
k2.metric("Avg Burnout Gap (FTE)", f"{np.mean(burnout_gap_fte):.2f}")
k3.metric("Months Exposed", f"{months_exposed}/12")

if R.get("enable_seasonality_ramp"):
    st.success(
        f"✅ **Reality Summary:** To meet flu demand, requisitions must post by **{R['req_post_date'].strftime('%b %d')}**, "
        f"so providers go solo by **{R['solo_ready_date'].strftime('%b %d')}**. "
        f"Model derives a required ramp speed of **{R.get('derived_ramp_after_solo', 0.0):.2f} FTE/month**."
    )
else:
    st.warning("⚠️ Seasonality Recruiting Ramp is OFF — supply uses a generic ramp only.")


st.markdown("---")
st.success("✅ **Next:** Finance + Strategy + Decision sections remain unchanged and can be reattached once you confirm the supply behavior looks correct.")
