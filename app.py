import math
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from psm.staffing_model import StaffingModel

# ============================================================
# APP VERSION (prevents stale session_state schema issues)
# ============================================================
APP_VERSION = "v4.2"

# ============================================================
# PAGE CONFIG
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
# STABLE TODAY (prevents moving windows on reruns)
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]

# ============================================================
# SESSION STATE
# ============================================================
for key in ["model_ran", "results", "results_version"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# BRAND COLORS
# ============================================================
BRAND_BLACK = "#000000"
BRAND_GOLD = "#7a6200"
GRAY = "#B0B0B0"
LIGHT_GRAY = "#EAEAEA"
MID_GRAY = "#666666"


# ============================================================
# HELPERS
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def base_seasonality_multiplier(month: int) -> float:
    """Baseline seasonality curve outside flu uplift."""
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00


def build_flu_window(current_year: int, flu_start_month: int, flu_end_month: int):
    """Builds flu season window between start month and end month (can cross year boundary)."""
    flu_start_date = datetime(current_year, flu_start_month, 1)

    if flu_end_month < flu_start_month:
        flu_end_date = datetime(current_year + 1, flu_end_month, 1)
    else:
        flu_end_date = datetime(current_year, flu_end_month, 1)

    # last day of flu_end month
    flu_end_date = flu_end_date + timedelta(days=32)
    flu_end_date = flu_end_date.replace(day=1) - timedelta(days=1)
    return flu_start_date, flu_end_date


def in_window(d: datetime, start: datetime, end: datetime) -> bool:
    return start <= d <= end


def compute_seasonality_forecast(
    dates, baseline_visits: float, flu_start: datetime, flu_end: datetime, flu_uplift_pct: float
):
    """
    Applies monthly seasonality multipliers + flu uplift, then normalizes to annual baseline.
    Output stays on the same annual average as baseline_visits.
    """
    raw = []
    for d in dates:
        mult = base_seasonality_multiplier(d.month)
        if in_window(d.to_pydatetime(), flu_start, flu_end):
            mult *= (1 + flu_uplift_pct)
        raw.append(baseline_visits * mult)

    avg_raw = float(np.mean(raw)) if len(raw) else baseline_visits
    if avg_raw <= 0:
        return [baseline_visits for _ in dates]

    normalized = [v * (baseline_visits / avg_raw) for v in raw]
    return normalized


def visits_to_provider_demand(
    model_obj: StaffingModel,
    visits_by_month,
    hours_of_operation: float,
    fte_hours_per_week: float,
    provider_min_floor: float,
):
    """Converts visits/day forecast to provider FTE demand by month."""
    demand = []
    for v in visits_by_month:
        fte = model_obj.calculate_fte_needed(
            visits_per_day=v,
            hours_of_operation_per_week=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week,
        )["provider_fte"]
        demand.append(max(float(fte), float(provider_min_floor)))
    return demand


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
    """
    Recommended (burnout-protective) target curve.

    - Buffer driven by: volume variability, spikes, and cumulative workload debt.
    - Smoothing caps month-to-month changes.
    """
    vol_w, spike_w, debt_w = weights

    visits_arr = np.array(visits_by_month, dtype=float)
    mean_visits = float(np.mean(visits_arr)) if len(visits_arr) else 0.0
    std_visits = float(np.std(visits_arr)) if len(visits_arr) else 0.0
    cv = (std_visits / mean_visits) if mean_visits > 0 else 0.0
    p75 = float(np.percentile(visits_arr, 75)) if len(visits_arr) else 0.0

    rdi = 0.0
    decay = 0.85
    lambda_debt = 0.10

    protective_curve = []
    prev_staff = max(float(base_demand_fte[0]), float(provider_min_floor))

    for v, base_fte in zip(visits_by_month, base_demand_fte):
        base_fte = float(base_fte)
        v = float(v)

        vbuf = base_fte * cv
        sbuf = max(0.0, (v - p75) / mean_visits) * base_fte if mean_visits > 0 else 0.0

        visits_per_provider = v / max(prev_staff, 0.25)
        debt = max(0.0, visits_per_provider - float(safe_visits_per_provider_per_day))

        rdi = decay * rdi + debt
        dbuf = lambda_debt * rdi

        buffer_fte = float(burnout_slider) * (vol_w * vbuf + spike_w * sbuf + debt_w * dbuf)
        raw_target = max(float(provider_min_floor), base_fte + buffer_fte)

        delta = raw_target - prev_staff
        if delta > 0:
            delta = clamp(delta, 0.0, float(smoothing_up))
        else:
            delta = clamp(delta, -float(smoothing_down), 0.0)

        final_staff = max(float(provider_min_floor), prev_staff + delta)
        protective_curve.append(final_staff)
        prev_staff = final_staff

    return protective_curve


def months_between(start_month: int, end_month: int):
    """Returns month numbers in a wrapped window. Example: Dec(12)→Feb(2) = [12,1,2]."""
    months = []
    m = int(start_month)
    while True:
        months.append(m)
        if m == int(end_month):
            break
        m = 1 if m == 12 else m + 1
    return months


def shift_month(month: int, shift: int) -> int:
    """Shift month integer forward/backward with wraparound."""
    return ((int(month) - 1 + int(shift)) % 12) + 1


def lead_days_to_months(days: int, avg_days_per_month: float = 30.4) -> int:
    """Convert days to months (ceiling), using an average month length."""
    return max(0, int(math.ceil(float(days) / float(avg_days_per_month))))


def month_index_map(dates):
    """month_num -> index in dates (Jan..Dec)."""
    return {d.month: i for i, d in enumerate(dates)}


def rotate_by_month(dates, series_list, anchor_month: int):
    """
    Rotate all series so the chart starts at anchor_month.
    Returns rotated_dates, rotated_labels, rotated_series_list, and rotation index.
    """
    m2i = month_index_map(dates)
    anchor_month = int(anchor_month)
    if anchor_month not in m2i:
        anchor_month = 1
    k = m2i[anchor_month]

    rot_dates = list(dates[k:]) + list(dates[:k])
    rot_labels = [d.strftime("%b") for d in rot_dates]
    rot_series = []
    for s in series_list:
        s = list(s)
        rot_series.append(s[k:] + s[:k])
    return rot_dates, rot_labels, rot_series, k


def month_range_label(months):
    """Ex: [4,5,6,7] => Apr–Jul; [] => —"""
    if not months:
        return "—"
    start = datetime(2000, int(months[0]), 1).strftime("%b")
    end = datetime(2000, int(months[-1]), 1).strftime("%b")
    return f"{start}–{end}" if start != end else start


def dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ============================================================
# AUTO-FREEZE v3 (month-loop safe)
# ============================================================
def auto_freeze_strategy_v3(
    dates,
    protective_curve,
    flu_start_month,
    flu_end_month,
    pipeline_lead_days,
    notice_days,
    freeze_buffer_months=1,
):
    """
    Auto-Freeze v3 (month-loop safe, no year dependency):

    Goals:
    - Be "Independent by" flu_start_month.
    - Post req early enough to cover pipeline lead time.
    - Freeze during the decline window after flu ends (avoid over-hiring into demand drop),
      with optional buffer past the trough.
    - Recruiting window opens ahead of req posting.

    Note:
    - This is a planning heuristic: stable, interpretable, executive-friendly.
    """
    lead_months = lead_days_to_months(int(pipeline_lead_days))
    notice_months = lead_days_to_months(int(notice_days))

    independent_ready_month = int(flu_start_month)
    req_post_month = shift_month(independent_ready_month, -lead_months)
    hire_visible_month = shift_month(req_post_month, lead_months)

    # Trough month = min protective demand
    trough_idx = int(np.argmin(np.array(protective_curve, dtype=float)))
    trough_month = int(dates[trough_idx].month)

    # Decline window: month after flu_end → trough
    decline_months = months_between(shift_month(int(flu_end_month), 1), trough_month)

    # Freeze months = decline window + buffer months after trough
    freeze_months = list(decline_months)
    for i in range(1, int(freeze_buffer_months) + 1):
        freeze_months.append(shift_month(trough_month, i))

    # Recruiting open window = months leading into req_post_month (lead_months + 1)
    recruiting_open_months = []
    for i in range(lead_months + 1):
        recruiting_open_months.append(shift_month(req_post_month, -i))

    freeze_months = dedupe_keep_order(freeze_months)
    recruiting_open_months = dedupe_keep_order(recruiting_open_months)

    # If overlap occurs, recruiting-open wins
    freeze_months = [m for m in freeze_months if m not in recruiting_open_months]

    return dict(
        lead_months=lead_months,
        notice_months=notice_months,
        independent_ready_month=independent_ready_month,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        trough_month=trough_month,
        freeze_months=freeze_months,
        recruiting_open_months=recruiting_open_months,
        flu_months=months_between(int(flu_start_month), int(flu_end_month)),
    )


# ============================================================
# PIPELINE SUPPLY (cyclic steady-state, month-loop safe)
# ============================================================
def pipeline_supply_curve_cyclic(
    dates,
    baseline_fte,
    target_curve,
    provider_min_floor,
    annual_turnover_rate,
    hire_visible_month,
    max_hiring_up_after_visible,
    freeze_months,
    confirmed_hire_month=None,
    confirmed_hire_fte=0.0,
    max_ramp_down_per_month=0.25,
    seasonality_ramp_enabled=True,
    smooth_start_iters=25,
):
    """
    Cyclic month-loop supply curve.

    Key fix:
    - Solve for a steady-state annual cycle by iterating the starting supply until
      start-of-cycle and end-of-cycle converge. This prevents the "Dec → Jan reset"
      artifact without needing 24 months.

    Attrition:
    - Monthly attrition scales with current supply (prev), not baseline.

    Hiring ramp:
    - Ramp-up blocked before hire_visible_month and during freeze_months (if enabled).
    - Confirmed hire applies once when month is reached.
    """
    m2i = month_index_map(dates)
    hv_idx = m2i.get(int(hire_visible_month), 0)
    freeze_set = set(int(m) for m in (freeze_months or []))

    confirmed_idx = None
    if confirmed_hire_month is not None and int(confirmed_hire_month) in m2i:
        confirmed_idx = m2i[int(confirmed_hire_month)]

    monthly_turnover_rate = float(annual_turnover_rate) / 12.0

    def simulate_once(start_supply: float):
        staff = []
        prev = max(float(start_supply), float(provider_min_floor))
        hire_applied = False

        for i, (d, target) in enumerate(zip(dates, target_curve)):
            month_num = int(d.month)
            in_freeze = month_num in freeze_set

            if seasonality_ramp_enabled:
                if in_freeze or (i < hv_idx):
                    ramp_up_cap = 0.0
                else:
                    ramp_up_cap = float(max_hiring_up_after_visible)
            else:
                ramp_up_cap = 0.35

            delta = float(target) - prev
            if delta > 0:
                delta = clamp(delta, 0.0, ramp_up_cap)
            else:
                delta = clamp(delta, -float(max_ramp_down_per_month), 0.0)

            planned = prev + delta

            # Attrition scales with current supply
            planned -= prev * monthly_turnover_rate

            # Confirmed hire applies once when month reached
            if (confirmed_idx is not None) and (not hire_applied) and (i >= confirmed_idx):
                planned += float(confirmed_hire_fte)
                hire_applied = True

            planned = max(planned, float(provider_min_floor))
            staff.append(planned)
            prev = planned

        end_supply = staff[-1]
        return staff, end_supply

    # Fixed-point iteration for steady-state
    start = max(float(baseline_fte), float(provider_min_floor))
    staff = None
    for _ in range(int(smooth_start_iters)):
        staff, end_supply = simulate_once(start)
        start = 0.6 * start + 0.4 * float(end_supply)

    staff, _ = simulate_once(start)
    return staff


# ============================================================
# COST HELPERS
# ============================================================
def provider_day_gap(target_curve, supply_curve, days_in_month):
    """Total provider-days of under-staffing (area between curves)."""
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(float(t) - float(s), 0.0) * float(dim)
    return float(gap_days)


def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    """Annualized cost of added FTE (pro-rated by days in month)."""
    cost = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += float(dfte) * float(loaded_cost_per_provider_fte) * (float(dim) / 365.0)
    return float(cost)


# ============================================================
# SIDEBAR INPUTS
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
    provider_turnover = st.number_input("Provider Turnover % (annual)", value=24.0, step=1.0) / 100.0

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
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
    )
    flu_end_month = st.selectbox(
        "Flu End Month",
        options=list(range(1, 13)),
        index=1,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
    )
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0) / 100.0

    st.subheader("Confirmed Hiring (Month-Based)")
    confirmed_hire_month = st.selectbox(
        "Confirmed Hire Start Month (Independent)",
        options=list(range(1, 13)),
        index=10,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
    )
    confirmed_hire_fte = st.number_input("Confirmed Hire FTE", min_value=0.0, value=1.0, step=0.25)

    enable_seasonality_ramp = st.checkbox(
        "Enable Seasonality Recruiting Ramp",
        value=True,
        help="If ON: supply cannot rise until hires are visible and recruiting is not frozen.",
    )

    st.divider()

    st.caption(f"App Version: **{APP_VERSION}**")
    col_reset_a, _ = st.columns([1, 1])
    with col_reset_a:
        if st.button("Reset / Clear Results"):
            for k in ["model_ran", "results", "results_version"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    run_model = st.button("Run Model")


# ============================================================
# RUN MODEL (v4.2)
# ============================================================
if run_model:
    current_year = today.year
    dates = pd.date_range(start=datetime(current_year, 1, 1), periods=12, freq="MS")
    month_labels = [d.strftime("%b") for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    # Flu window for forecast math only
    flu_start_date, flu_end_date = build_flu_window(current_year, int(flu_start_month), int(flu_end_month))

    # Baseline provider FTE
    fte_result = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    baseline_provider_fte = max(float(fte_result["provider_fte"]), float(provider_min_floor))

    # Forecast visits (normalized)
    forecast_visits_by_month = compute_seasonality_forecast(
        dates=dates,
        baseline_visits=float(visits),
        flu_start=flu_start_date,
        flu_end=flu_end_date,
        flu_uplift_pct=float(flu_uplift_pct),
    )

    # Lean demand curve
    provider_base_demand = visits_to_provider_demand(
        model_obj=model,
        visits_by_month=forecast_visits_by_month,
        hours_of_operation=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week),
        provider_min_floor=float(provider_min_floor),
    )

    # Protective curve
    protective_curve = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_by_month,
        base_demand_fte=provider_base_demand,
        provider_min_floor=float(provider_min_floor),
        burnout_slider=float(burnout_slider),
        safe_visits_per_provider_per_day=float(safe_visits_per_provider),
    )

    # Pipeline lead time
    total_lead_days = int(days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days)

    # Auto-freeze strategy v3 (months, loop-safe)
    strategy = auto_freeze_strategy_v3(
        dates=dates,
        protective_curve=protective_curve,
        flu_start_month=int(flu_start_month),
        flu_end_month=int(flu_end_month),
        pipeline_lead_days=total_lead_days,
        notice_days=int(notice_days),
        freeze_buffer_months=1,
    )

    req_post_month = int(strategy["req_post_month"])
    hire_visible_month = int(strategy["hire_visible_month"])
    independent_ready_month = int(strategy["independent_ready_month"])
    freeze_months = strategy["freeze_months"]
    recruiting_open_months = strategy["recruiting_open_months"]
    lead_months = int(strategy["lead_months"])
    flu_months = strategy["flu_months"]

    # Derived ramp (gap at flu start / months in flu window)
    flu_month_idx = month_index_map(dates).get(int(flu_start_month), 0)
    months_in_flu_window = max(len(flu_months), 1)

    target_at_flu = float(protective_curve[flu_month_idx])
    fte_gap_to_close = max(target_at_flu - float(baseline_provider_fte), 0.0)
    derived_ramp_after_visible = min(fte_gap_to_close / float(months_in_flu_window), 1.25)

    # Supply curves (cyclic steady-state, no Jan reset)
    realistic_supply_lean = pipeline_supply_curve_cyclic(
        dates=dates,
        baseline_fte=float(baseline_provider_fte),
        target_curve=provider_base_demand,
        provider_min_floor=float(provider_min_floor),
        annual_turnover_rate=float(provider_turnover),
        hire_visible_month=int(hire_visible_month),
        max_hiring_up_after_visible=float(derived_ramp_after_visible),
        freeze_months=(freeze_months if enable_seasonality_ramp else []),
        confirmed_hire_month=int(confirmed_hire_month),
        confirmed_hire_fte=float(confirmed_hire_fte),
        seasonality_ramp_enabled=bool(enable_seasonality_ramp),
    )

    realistic_supply_recommended = pipeline_supply_curve_cyclic(
        dates=dates,
        baseline_fte=float(baseline_provider_fte),
        target_curve=protective_curve,
        provider_min_floor=float(provider_min_floor),
        annual_turnover_rate=float(provider_turnover),
        hire_visible_month=int(hire_visible_month),
        max_hiring_up_after_visible=float(derived_ramp_after_visible),
        freeze_months=(freeze_months if enable_seasonality_ramp else []),
        confirmed_hire_month=int(confirmed_hire_month),
        confirmed_hire_fte=float(confirmed_hire_fte),
        seasonality_ramp_enabled=bool(enable_seasonality_ramp),
    )

    # Burnout exposure
    burnout_gap_fte = [max(float(t) - float(s), 0.0) for t, s in zip(protective_curve, realistic_supply_recommended)]
    months_exposed = int(sum(1 for g in burnout_gap_fte if g > 0))

    # Anchor rotation: Req Post month (Option C1)
    rotation_anchor_month = int(req_post_month)

    # Store results (include version)
    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        dates=dates,
        month_labels=month_labels,
        days_in_month=days_in_month,
        baseline_provider_fte=baseline_provider_fte,
        flu_start_date=flu_start_date,
        flu_end_date=flu_end_date,
        flu_start_month=int(flu_start_month),
        flu_end_month=int(flu_end_month),
        flu_months=flu_months,
        forecast_visits_by_month=forecast_visits_by_month,
        provider_base_demand=provider_base_demand,
        protective_curve=protective_curve,
        realistic_supply_lean=realistic_supply_lean,
        realistic_supply_recommended=realistic_supply_recommended,
        burnout_gap_fte=burnout_gap_fte,
        months_exposed=months_exposed,
        pipeline_lead_days=total_lead_days,
        lead_months=lead_months,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        independent_ready_month=independent_ready_month,
        freeze_months=freeze_months,
        recruiting_open_months=recruiting_open_months,
        derived_ramp_after_visible=derived_ramp_after_visible,
        fte_gap_to_close=fte_gap_to_close,
        months_in_flu_window=months_in_flu_window,
        confirmed_hire_month=int(confirmed_hire_month),
        confirmed_hire_fte=float(confirmed_hire_fte),
        enable_seasonality_ramp=bool(enable_seasonality_ramp),
        rotation_anchor_month=rotation_anchor_month,
    )
    st.session_state["results_version"] = APP_VERSION


# ============================================================
# STOP IF NOT RUN
# ============================================================
if not st.session_state.get("model_ran"):
    st.info("Enter inputs in the sidebar and click **Run Model**.")
    st.stop()

# Version guard (prevents KeyErrors after code updates)
if st.session_state.get("results_version") != APP_VERSION:
    st.warning("Model updated — please click **Run Model** to refresh results.")
    st.stop()

R = st.session_state["results"]

# ============================================================
# SECTION 1 — OPERATIONS
# ============================================================
st.markdown("---")
st.header("1) Operations — Seasonality Staffing Requirements")
st.caption("Visits/day forecast → FTE needed by month (seasonality + flu uplift).")

monthly_rows = []
for month_label, d, v in zip(R["month_labels"], R["dates"], R["forecast_visits_by_month"]):
    fte_staff = model.calculate_fte_needed(
        visits_per_day=float(v),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week),
    )

    monthly_rows.append(
        {
            "Month": month_label,
            "Visits/Day (Forecast)": round(float(v), 1),
            "Provider FTE": round(float(fte_staff["provider_fte"]), 2),
            "PSR FTE": round(float(fte_staff["psr_fte"]), 2),
            "MA FTE": round(float(fte_staff["ma_fte"]), 2),
            "XRT FTE": round(float(fte_staff["xrt_fte"]), 2),
            "Total FTE": round(float(fte_staff["total_fte"]), 2),
        }
    )

ops_df = pd.DataFrame(monthly_rows)
st.dataframe(ops_df, hide_index=True, use_container_width=True)

st.success(
    "**Operations Summary:** This is the seasonality-adjusted demand signal. "
    "Lean demand is minimum coverage; the protective target adds a burnout buffer to protect throughput and quality."
)

# ============================================================
# SECTION 2 — REALITY (Presentation-ready, Rotated Display)
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption("Targets vs realistic supply when hiring is constrained by lead time, freezes, and turnover.")

freeze_months = R.get("freeze_months", []) or []
recruiting_open_months = R.get("recruiting_open_months", []) or []
req_post_month = int(R.get("req_post_month", 1))
hire_visible_month = int(R.get("hire_visible_month", 1))
independent_ready_month = int(R.get("independent_ready_month", 1))

freeze_label = month_range_label(freeze_months)
recruit_label = month_range_label(recruiting_open_months)
req_post_label = datetime(2000, req_post_month, 1).strftime("%b")
hire_visible_label = datetime(2000, hire_visible_month, 1).strftime("%b")
independent_label = datetime(2000, independent_ready_month, 1).strftime("%b")

st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; gap:16px;
                padding:12px 16px; background:#F7F7F7; border-radius:10px;
                border:1px solid #E0E0E0; font-size:15px;">
        <div><b>Freeze:</b> {freeze_label}</div>
        <div><b>Recruiting Window:</b> {recruit_label}</div>
        <div><b>Post Req:</b> {req_post_label}</div>
        <div><b>Hires Visible:</b> {hire_visible_label}</div>
        <div><b>Independent By:</b> {independent_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

peak_gap = float(max(R["burnout_gap_fte"])) if R.get("burnout_gap_fte") else 0.0
avg_gap = float(np.mean(R["burnout_gap_fte"])) if R.get("burnout_gap_fte") else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Peak Burnout Gap (FTE)", f"{peak_gap:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
m3.metric("Months Exposed", f"{int(R.get('months_exposed', 0))}/12")

# Rotate chart (Option C1): start at req post month
anchor_month = int(R.get("rotation_anchor_month", req_post_month))
rot_dates, rot_labels, rot_series, _ = rotate_by_month(
    R["dates"],
    [
        R["provider_base_demand"],
        R["protective_curve"],
        R["realistic_supply_recommended"],
        R["forecast_visits_by_month"],
    ],
    anchor_month=anchor_month,
)
rot_lean, rot_prot, rot_supply, rot_visits = rot_series

freeze_set = set(int(m) for m in (R.get("freeze_months", []) or []))

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Shade freeze months
for d in rot_dates:
    if int(d.month) in freeze_set:
        ax1.axvspan(d, d + timedelta(days=27), alpha=0.12, color=BRAND_GOLD, linewidth=0)

# Lines
ax1.plot(rot_dates, rot_lean, linestyle=":", linewidth=1.2, color=GRAY, label="Lean Target (Demand)")
ax1.plot(
    rot_dates,
    rot_prot,
    linewidth=2.0,
    color=BRAND_GOLD,
    marker="o",
    markersize=4,
    label="Recommended Target (Protective)",
)
ax1.plot(
    rot_dates,
    rot_supply,
    linewidth=2.0,
    color=BRAND_BLACK,
    marker="o",
    markersize=4,
    label="Realistic Supply (Pipeline)",
)

# Burnout zone fill
ax1.fill_between(
    rot_dates,
    rot_supply,
    rot_prot,
    where=np.array(rot_prot, dtype=float) > np.array(rot_supply, dtype=float),
    color=BRAND_GOLD,
    alpha=0.12,
    label="Burnout Exposure Zone",
)

ax1.set_title("Reality — Targets vs Pipeline-Constrained Supply", fontsize=16, fontweight="bold", pad=16, color=BRAND_BLACK)
ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax1.set_xticks(rot_dates)
ax1.set_xticklabels(rot_labels, fontsize=11, color=BRAND_BLACK)
ax1.tick_params(axis="y", labelsize=11, colors=BRAND_BLACK)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

# Secondary axis: visits
ax2 = ax1.twinx()
ax2.plot(rot_dates, rot_visits, linestyle="-.", linewidth=1.4, color=MID_GRAY, label="Forecast Visits/Day")
ax2.set_ylabel("Visits / Day", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax2.tick_params(axis="y", labelsize=11, colors=BRAND_BLACK)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False,
    fontsize=11,
)

plt.tight_layout()
st.pyplot(fig)

st.success(
    f"**Reality Summary:** To be flu-ready by **{independent_label}**, requisitions must post by **{req_post_label}** "
    f"so hires are visible by **{hire_visible_label}**. "
    f"Required protective ramp speed: **{float(R.get('derived_ramp_after_visible', 0.0)):.2f} FTE/month**."
)

st.info(
    "🧠 **Auto-Hiring Strategy (v3)**\n\n"
    f"- Freeze months: **{', '.join([datetime(2000,int(m),1).strftime('%b') for m in freeze_months]) or '—'}**\n"
    f"- Recruiting window: **{', '.join([datetime(2000,int(m),1).strftime('%b') for m in recruiting_open_months]) or '—'}**\n"
    f"- Post req: **{req_post_label}** | Hires visible: **{hire_visible_label}** | Independent by: **{independent_label}**\n"
    f"- Lead time: **{int(R.get('pipeline_lead_days', 0))} days (~{int(R.get('lead_months', 0))} months)**\n"
)

# ============================================================
# SECTION 3 — FINANCE
# ============================================================
st.markdown("---")
st.header("3) Finance — ROI Investment Case")
st.caption("Quantifies the investment required to close the gap and the economic value of reducing provider-day shortages.")

st.subheader("Finance Inputs")
colA, colB, colC = st.columns(3)
with colA:
    loaded_cost_per_provider_fte = st.number_input("Loaded Cost per Provider FTE (annual)", value=260000, step=5000)
with colB:
    net_revenue_per_visit = st.number_input("Net Revenue per Visit", value=140.0, step=5.0)
with colC:
    visits_lost_per_provider_day_gap = st.number_input("Visits Lost per 1.0 Provider-Day Gap", value=18.0, step=1.0)

delta_fte_curve = [max(float(t) - float(R["baseline_provider_fte"]), 0.0) for t in R["protective_curve"]]
annual_investment = annualize_monthly_fte_cost(delta_fte_curve, R["days_in_month"], float(loaded_cost_per_provider_fte))

gap_days = provider_day_gap(R["protective_curve"], R["realistic_supply_recommended"], R["days_in_month"])
est_visits_lost = float(gap_days) * float(visits_lost_per_provider_day_gap)
est_revenue_lost = float(est_visits_lost) * float(net_revenue_per_visit)

roi = (est_revenue_lost / annual_investment) if annual_investment > 0 else np.nan

f1, f2, f3 = st.columns(3)
f1.metric("Annual Investment (Protective)", f"${annual_investment:,.0f}")
f2.metric("Est. Net Revenue at Risk", f"${est_revenue_lost:,.0f}")
f3.metric("ROI (Revenue ÷ Investment)", f"{roi:,.2f}x" if np.isfinite(roi) else "—")

st.success(
    "**Finance Summary:** The investment is the cost of staffing to the protective curve. "
    "The value is the revenue protected by reducing provider-day shortages during peak demand."
)

# ============================================================
# SECTION 4 — STRATEGY
# ============================================================
st.markdown("---")
st.header("4) Strategy — Closing the Gap with Flexible Coverage")
st.caption("Use flex coverage strategies to reduce burnout exposure without hiring full permanent FTE.")

st.subheader("Strategy Levers")
s1, s2, s3, s4 = st.columns(4)
with s1:
    buffer_pct = st.slider("Buffer Coverage %", 0, 100, 25, 5)
with s2:
    float_pool_fte = st.slider("Float Pool (FTE)", 0.0, 5.0, 1.0, 0.25)
with s3:
    fractional_fte = st.slider("Fractional Add (FTE)", 0.0, 5.0, 0.5, 0.25)
with s4:
    hybrid_slider = st.slider("Hybrid (flex → perm)", 0.0, 1.0, 0.5, 0.05)

gap_fte_curve = [max(float(t) - float(s), 0.0) for t, s in zip(R["protective_curve"], R["realistic_supply_recommended"])]

effective_gap_curve = []
for g in gap_fte_curve:
    g2 = float(g) * (1 - float(buffer_pct) / 100.0)
    g2 = max(g2 - float(float_pool_fte), 0.0)
    g2 = max(g2 - float(fractional_fte), 0.0)
    effective_gap_curve.append(g2)

remaining_gap_days = provider_day_gap([0] * 12, effective_gap_curve, R["days_in_month"])
reduced_gap_days = max(float(gap_days) - float(remaining_gap_days), 0.0)

est_visits_saved = float(reduced_gap_days) * float(visits_lost_per_provider_day_gap)
est_revenue_saved = float(est_visits_saved) * float(net_revenue_per_visit)

hybrid_investment = float(annual_investment) * float(hybrid_slider)

sA, sB, sC = st.columns(3)
sA.metric("Provider-Day Gap Reduced", f"{reduced_gap_days:,.0f}")
sB.metric("Est. Revenue Saved", f"${est_revenue_saved:,.0f}")
sC.metric("Hybrid Investment Share", f"${hybrid_investment:,.0f}")

st.success(
    "**Strategy Summary:** Flex levers can reduce exposure faster than permanent hiring. "
    "Use hybrid to transition temporary coverage into permanent staffing once demand proves durable."
)

# ============================================================
# SECTION 5 — DECISION
# ============================================================
st.markdown("---")
st.header("5) Decision — Executive Summary")

st.subheader("Decision Snapshot")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Peak Gap (FTE)", f"{peak_gap:.2f}")
    st.metric("Avg Gap (FTE)", f"{avg_gap:.2f}")
    st.metric("Months Exposed", f"{int(R.get('months_exposed', 0))}/12")
with col2:
    st.write(
        f"**To be flu-ready by {independent_label}:**\n"
        f"- Post requisitions by **{req_post_label}** (lead time: {int(R.get('pipeline_lead_days', 0))} days ≈ {int(R.get('lead_months', 0))} months)\n"
        f"- Hiring becomes visible by **{hire_visible_label}**\n"
        f"- Protective ramp required: **{float(R.get('derived_ramp_after_visible', 0.0)):.2f} FTE/month**\n\n"
        f"**Financial framing:**\n"
        f"- Annual protective investment: **${annual_investment:,.0f}**\n"
        f"- Estimated revenue at risk if not closed: **${est_revenue_lost:,.0f}**\n"
        f"- ROI: **{roi:,.2f}x**\n\n"
        f"**With strategy levers applied:**\n"
        f"- Provider-day gap reduced: **{reduced_gap_days:,.0f} days**\n"
        f"- Estimated revenue saved: **${est_revenue_saved:,.0f}**\n"
    )

st.success(
    "✅ **Decision Summary:** This model converts seasonality into staffing demand, converts pipeline timing into realistic supply, "
    "and quantifies both burnout exposure and the ROI of closing gaps. "
    "Use the auto-hiring strategy + flex levers to move from reactive coverage to decision-ready staffing."
)

# ============================================================
# SECTION 6 — IN-DEPTH EXECUTIVE SUMMARY (Narrative)
# ============================================================
st.markdown("---")
st.header("6) In-Depth Executive Summary")
st.caption("A plain-language explanation of what the model is doing, what it found, and how to act on it.")

flu_range = month_range_label(R.get("flu_months", []))
summary_md = f"""
### What this model is solving
This Predictive Staffing Model (PSM) answers one operational question:

**“Given seasonal volume swings and a constrained hiring pipeline, what staffing plan prevents burnout exposure — and when must recruiting actions occur to make that plan feasible?”**

The model treats months as a **repeatable annual cycle** (Jan–Dec as recurring seasonal states). Staffing capacity does not “reset” at year-end, so the supply curve is computed as a **steady-state annual cycle**.

---

### How Operations works (demand signal)
1. You enter an annual average visits/day.
2. The model applies a baseline seasonality curve and a flu uplift over **{flu_range}**.
3. It normalizes the result so the **annual average stays equal to your baseline**.
4. That forecast converts into a lean monthly provider demand curve (minimum coverage).

Then the model builds a **recommended (protective) target** that adds a burnout buffer. The buffer grows when:
- volume variability increases,
- months spike above typical,
- workload per provider exceeds your “Safe Visits/Provider/Day.”

---

### How Reality works (pipeline-constrained supply)
Supply is constrained by:
- Hiring lead time (sign + credential + train + buffer): **{int(R.get('pipeline_lead_days', 0))} days** (~{int(R.get('lead_months', 0))} months)
- Recruiting freezes during selected months
- Turnover, applied monthly as a percentage of current staffing

**Important:** supply is computed as a **cyclic steady-state**, iterating the annual loop until start and end converge. This prevents misleading “Dec → Jan” reset artifacts without requiring a 24-month view.

---

### Auto-Hiring Strategy (v3) — what it’s recommending
- **Freeze window:** {freeze_label}  
- **Recruiting window:** {recruit_label}  
- **Post requisition:** {req_post_label}  
- **Hires visible:** {hire_visible_label}  
- **Independent by:** {independent_label}

Interpretation: avoid adding capacity during the post-peak decline window, and open recruiting far enough in advance to hit flu with independent capacity.

---

### What it found (burnout exposure)
- **Peak burnout gap:** {peak_gap:.2f} FTE  
- **Average burnout gap:** {avg_gap:.2f} FTE  
- **Months exposed:** {int(R.get('months_exposed', 0))}/12

---

### What to do next
To reduce burnout exposure you have three practical levers:
1. **Timing:** post requisitions earlier.
2. **Ramp speed:** increase achievable monthly ramp-up (pipeline/process improvement or contingent coverage).
3. **Flex coverage:** float pool, fractional staffing, or buffer coverage to bridge gaps before permanent staff arrives.

---

### How to read the Reality chart (important)
The chart is **rotated** to start at the **requisition posting month ({req_post_label})** so the visual story aligns to “actions → outcomes” and the year boundary never looks like a staffing reset.
"""
st.markdown(summary_md)

st.success(
    "✅ Executive Summary complete. Next step: tie feasibility to VVI (RF/LF) so staffing plans can be tested against operational AND financial performance."
)
