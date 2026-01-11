import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from psm.staffing_model import StaffingModel


# ============================================================
# ✅ PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM)", layout="centered")

st.markdown(
    """
    <style>
      .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# ✅ SESSION STATE
# ============================================================
for key in ["model_ran", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ============================================================
# ✅ BRAND COLORS
# ============================================================
BRAND_BLACK = "#000000"
BRAND_GOLD  = "#7a6200"
GRAY        = "#B0B0B0"
LIGHT_GRAY  = "#EAEAEA"


# ============================================================
# ✅ HELPERS
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def base_seasonality_multiplier(month: int):
    """Baseline seasonality curve outside flu uplift."""
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00


def build_flu_window(current_year: int, flu_start_month: int, flu_end_month: int):
    """Build flu season window between start month and end month (can cross year boundary)."""
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
    """Apply seasonality + flu uplift then normalize back to annual baseline."""
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
    """Convert visits/day forecast to provider FTE demand by month."""
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
# ✅ BURNOUT-PROTECTIVE CURVE
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
# ✅ LOOPED SUPPLY CURVE (FIXED Dec→Jan CARRYOVER)
# ============================================================
def in_any_freeze_window(d, freeze_windows):
    if not freeze_windows:
        return False
    for start, end in freeze_windows:
        if start <= d <= end:
            return True
    return False


def pipeline_supply_curve_looped(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    req_post_date,
    pipeline_lead_days,
    max_hiring_up_after_pipeline,
    confirmed_hire_month=None,
    confirmed_hire_fte=0.0,
    max_ramp_down_per_month=0.25,
    seasonality_ramp_enabled=True,
    freeze_windows=None,
    max_iters=25,
    tol=0.01,
):
    """
    ✅ LOOPED equilibrium supply curve so Dec carries into Jan.
    """

    if freeze_windows is None:
        freeze_windows = []

    monthly_attrition_fte = baseline_fte * (annual_turnover_rate / 12)
    effective_attrition_start = today + timedelta(days=int(notice_days))
    hire_visible_date = req_post_date + timedelta(days=int(pipeline_lead_days))

    confirmed_hire_dt = None
    if confirmed_hire_month:
        for d in dates:
            if d.month == confirmed_hire_month:
                confirmed_hire_dt = d.to_pydatetime()
                break

    start_supply = max(baseline_fte, provider_min_floor)

    for _ in range(max_iters):

        hire_applied = False
        staff = []
        prev = start_supply

        for d, target in zip(dates, target_curve):
            d_py = d.to_pydatetime()
            in_freeze = in_any_freeze_window(d_py, freeze_windows)

            if seasonality_ramp_enabled:
                ramp_up_cap = 0.0 if (in_freeze or d_py < hire_visible_date) else max_hiring_up_after_pipeline
            else:
                ramp_up_cap = 0.35

            delta = target - prev
            if delta > 0:
                delta = clamp(delta, 0.0, ramp_up_cap)
            else:
                delta = clamp(delta, -max_ramp_down_per_month, 0.0)

            planned = prev + delta

            if d_py >= effective_attrition_start:
                planned -= monthly_attrition_fte

            if (not hire_applied) and confirmed_hire_dt and (d_py >= confirmed_hire_dt):
                planned += confirmed_hire_fte
                hire_applied = True

            planned = max(planned, provider_min_floor)
            staff.append(planned)
            prev = planned

        end_supply = staff[-1]

        if abs(end_supply - start_supply) < tol:
            return staff

        start_supply = end_supply

    return staff


# ============================================================
# ✅ AUTO-FREEZE STRATEGY V3
# ============================================================
def auto_freeze_v3(dates, demand_curve, notice_days, threshold=0.10):
    """
    Auto-freeze when demand declines sharply.
    Freeze months = any month where demand drops > threshold from previous month.
    Freeze windows include those months + notice lag awareness.
    """

    freeze_months = []
    for i in range(1, len(demand_curve)):
        prev = demand_curve[i - 1]
        cur = demand_curve[i]
        if prev > 0 and ((prev - cur) / prev) >= threshold:
            freeze_months.append(dates[i].month)

    freeze_windows = []
    for d in dates:
        if d.month in freeze_months:
            start = d.to_pydatetime()
            end = (pd.Timestamp(start) + pd.offsets.MonthEnd(1)).to_pydatetime()
            freeze_windows.append((start, end))

    return freeze_months, freeze_windows


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
        notice_days = st.number_input("Resignation Notice Period (days)", min_value=0, max_value=180, value=90, step=5)

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

    st.subheader("Confirmed Hiring")
    confirmed_hire_month = st.selectbox(
        "Confirmed Hire Start Month (Independent)",
        options=list(range(1, 13)),
        index=10,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B")
    )
    confirmed_hire_fte = st.number_input("Confirmed Hire FTE", min_value=0.0, value=1.0, step=0.25)

    enable_seasonality_ramp = st.checkbox(
        "Enable Recruiting Ramp Logic",
        value=True,
        help="If ON: supply cannot rise until hires become visible (pipeline completion)."
    )

    st.divider()
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

    # Forecast visits (seasonality + flu uplift)
    forecast_visits_by_month = compute_seasonality_forecast(
        dates=dates,
        baseline_visits=visits,
        flu_start=flu_start_date,
        flu_end=flu_end_date,
        flu_uplift_pct=flu_uplift_pct,
    )

    # Lean demand curve
    provider_base_demand = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_by_month,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    # Protective curve
    protective_curve = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # Pipeline lead time
    total_lead_days = days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days

    # Independent-ready = flu start
    independent_ready_date = flu_start_date
    req_post_date = independent_ready_date - timedelta(days=total_lead_days)

    # Auto-freeze v3 (based on demand declines)
    freeze_months, freeze_windows = auto_freeze_v3(
        dates=dates,
        demand_curve=protective_curve,
        notice_days=notice_days,
        threshold=0.10
    )

    # Derived ramp speed (gap at flu start)
    flu_idx = next(i for i, d in enumerate(dates) if d.month == flu_start_month)
    target_at_flu = protective_curve[flu_idx]
    fte_gap_to_close = max(target_at_flu - baseline_provider_fte, 0)
    derived_ramp_after_independent = min(fte_gap_to_close / 3, 1.25)  # assume 3-month flu window avg

    # Supply curves (looped!)
    realistic_supply_recommended = pipeline_supply_curve_looped(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=protective_curve,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        pipeline_lead_days=total_lead_days,
        max_hiring_up_after_pipeline=derived_ramp_after_independent,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        freeze_windows=freeze_windows,
    )

    realistic_supply_lean = pipeline_supply_curve_looped(
        dates=dates,
        baseline_fte=baseline_provider_fte,
        target_curve=provider_base_demand,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_date=req_post_date,
        pipeline_lead_days=total_lead_days,
        max_hiring_up_after_pipeline=derived_ramp_after_independent,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        freeze_windows=freeze_windows,
    )

    burnout_gap_fte = [max(t - s, 0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    months_exposed = sum(1 for g in burnout_gap_fte if g > 0)

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
        independent_ready_date=independent_ready_date,
        derived_ramp_after_independent=derived_ramp_after_independent,
        pipeline_lead_days=total_lead_days,
        freeze_months=freeze_months,
        freeze_windows=freeze_windows,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
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
st.caption("Visits/day forecast → staff/day → FTE needed by month (based on seasonality).")

monthly_rows = []
for month_label, d, v in zip(R["month_labels"], R["dates"], R["forecast_visits_by_month"]):
    fte_staff = model.calculate_fte_needed(
        visits_per_day=v,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week
    )

    provider_day = (fte_staff["provider_fte"] * fte_hours_per_week) / max(hours_of_operation, 1)

    monthly_rows.append({
        "Month": month_label,
        "Visits/Day (Forecast)": round(v, 1),
        "Providers Needed/Day": round(provider_day, 2),
        "Provider FTE": round(fte_staff["provider_fte"], 2),
        "Total FTE": round(fte_staff["total_fte"], 2),
    })

ops_df = pd.DataFrame(monthly_rows)
st.dataframe(ops_df, hide_index=True, use_container_width=True)


# ============================================================
# ✅ SECTION 2 — REALITY (Presentation Ready)
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Lean vs recommended targets compared to looped realistic supply (no Dec→Jan reset).")

peak_gap = max(R["burnout_gap_fte"])
avg_gap = float(np.mean(R["burnout_gap_fte"]))

m1, m2, m3 = st.columns(3)
m1.metric("Peak Burnout Gap (FTE)", f"{peak_gap:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
m3.metric("Months Exposed", f"{R['months_exposed']}/12")

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Shade freeze months
for d in R["dates"]:
    if d.month in R["freeze_months"]:
        ax1.axvspan(d, d + timedelta(days=27), alpha=0.10, color=BRAND_GOLD, linewidth=0)

# Lines
ax1.plot(R["dates"], R["provider_base_demand"], linestyle=":", linewidth=1.2, color=GRAY, label="Lean Target")
ax1.plot(R["dates"], R["protective_curve"], linewidth=2.0, color=BRAND_GOLD, marker="o", markersize=4, label="Protective Target")
ax1.plot(R["dates"], R["realistic_supply_recommended"], linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=4, label="Realistic Supply")

ax1.fill_between(
    R["dates"],
    R["realistic_supply_recommended"],
    R["protective_curve"],
    where=np.array(R["protective_curve"]) > np.array(R["realistic_supply_recommended"]),
    color=BRAND_GOLD,
    alpha=0.10,
    label="Burnout Exposure",
)

ax1.set_title("Reality — Targets vs Pipeline-Constrained Supply", fontsize=16, fontweight="bold")
ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold")
ax1.set_xticks(R["dates"])
ax1.set_xticklabels(R["month_labels"], fontsize=11)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

# Secondary axis: visits/day
ax2 = ax1.twinx()
ax2.plot(R["dates"], R["forecast_visits_by_month"], linestyle="-.", linewidth=1.4, color="#666666", label="Forecast Visits/Day")
ax2.set_ylabel("Visits / Day", fontsize=12, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=11)

plt.tight_layout()
st.pyplot(fig)

st.success(
    f"**Reality Summary:** To be flu-ready by **{R['independent_ready_date'].strftime('%b')}**, post requisitions by "
    f"**{R['req_post_date'].strftime('%b %d')}**. Supply is solved as a month-loop equilibrium (Dec carries into Jan)."
)


# ============================================================
# ✅ SECTION 3 — FINANCE
# ============================================================
st.markdown("---")
st.header("3) Finance — ROI Investment Case")
st.caption("Investment required to meet protective targets and revenue at risk from provider-day gaps.")

colA, colB, colC = st.columns(3)
with colA:
    loaded_cost_per_provider_fte = st.number_input("Loaded Cost per Provider FTE (annual)", value=260000, step=5000)
with colB:
    net_revenue_per_visit = st.number_input("Net Revenue per Visit", value=140.0, step=5.0)
with colC:
    visits_lost_per_provider_day_gap = st.number_input("Visits Lost per 1.0 Provider-Day Gap", value=18.0, step=1.0)

delta_fte_curve = [max(t - R["baseline_provider_fte"], 0) for t in R["protective_curve"]]
annual_investment = annualize_monthly_fte_cost(delta_fte_curve, R["days_in_month"], loaded_cost_per_provider_fte)

gap_days = provider_day_gap(R["protective_curve"], R["realistic_supply_recommended"], R["days_in_month"])
est_visits_lost = gap_days * visits_lost_per_provider_day_gap
est_revenue_lost = est_visits_lost * net_revenue_per_visit

roi = (est_revenue_lost / annual_investment) if annual_investment > 0 else np.nan

f1, f2, f3 = st.columns(3)
f1.metric("Annual Investment (Protective)", f"${annual_investment:,.0f}")
f2.metric("Est. Net Revenue at Risk", f"${est_revenue_lost:,.0f}")
f3.metric("ROI (Revenue ÷ Investment)", f"{roi:,.2f}x" if np.isfinite(roi) else "—")


# ============================================================
# ✅ SECTION 4 — STRATEGY
# ============================================================
st.markdown("---")
st.header("4) Strategy — Closing the Gap with Flexible Coverage")
st.caption("Flex coverage options reduce burnout exposure without full permanent hiring.")

s1, s2, s3, s4 = st.columns(4)
with s1:
    buffer_pct = st.slider("Buffer Coverage %", 0, 100, 25, 5)
with s2:
    float_pool_fte = st.slider("Float Pool (FTE)", 0.0, 5.0, 1.0, 0.25)
with s3:
    fractional_fte = st.slider("Fractional Add (FTE)", 0.0, 5.0, 0.5, 0.25)
with s4:
    hybrid_slider = st.slider("Hybrid (flex → perm)", 0.0, 1.0, 0.5, 0.05)

gap_fte_curve = [max(t - s, 0) for t, s in zip(R["protective_curve"], R["realistic_supply_recommended"])]

effective_gap_curve = []
for g in gap_fte_curve:
    g2 = g * (1 - buffer_pct / 100)
    g2 = max(g2 - float_pool_fte, 0)
    g2 = max(g2 - fractional_fte, 0)
    effective_gap_curve.append(g2)

reduced_gap_days = provider_day_gap([0]*12, effective_gap_curve, R["days_in_month"])
reduced_gap_days = max(gap_days - reduced_gap_days, 0)

est_visits_saved = reduced_gap_days * visits_lost_per_provider_day_gap
est_revenue_saved = est_visits_saved * net_revenue_per_visit

hybrid_investment = annual_investment * hybrid_slider

sA, sB, sC = st.columns(3)
sA.metric("Provider-Day Gap Reduced", f"{reduced_gap_days:,.0f}")
sB.metric("Est. Revenue Saved", f"${est_revenue_saved:,.0f}")
sC.metric("Hybrid Investment Share", f"${hybrid_investment:,.0f}")


# ============================================================
# ✅ SECTION 5 — DECISION
# ============================================================
st.markdown("---")
st.header("5) Decision — Executive Summary")

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Peak Gap (FTE)", f"{peak_gap:.2f}")
    st.metric("Avg Gap (FTE)", f"{avg_gap:.2f}")
    st.metric("Months Exposed", f"{R['months_exposed']}/12")
with col2:
    st.write(
        f"**Independent-ready by {R['independent_ready_date'].strftime('%b %d')}:**\n"
        f"- Post requisitions by **{R['req_post_date'].strftime('%b %d')}**\n"
        f"- Pipeline lead time: **{R['pipeline_lead_days']} days**\n"
        f"- Derived ramp speed: **{R['derived_ramp_after_independent']:.2f} FTE/month**\n"
        f"- Annual protective investment: **${annual_investment:,.0f}**\n"
        f"- Estimated revenue at risk: **${est_revenue_lost:,.0f}**\n"
        f"- ROI: **{roi:,.2f}x**\n\n"
        f"**With flex strategy applied:**\n"
        f"- Provider-day gap reduced: **{reduced_gap_days:,.0f} days**\n"
        f"- Estimated revenue saved: **${est_revenue_saved:,.0f}**\n"
    )

st.success(
    "✅ **Decision Summary:** Seasonality drives demand. Pipeline timing drives reality. "
    "This model converts both into executive-level staffing strategy with ROI-based decision support."
)
