# app.py (v5) — 36-month engine + 12-month display window + Monte Carlo (P50/P90) + SWB/Visit feasibility
# Brand: Black #000000, Sunshine Gold #7a6200

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

from psm.staffing_model import StaffingModel


# =========================
# PAGE CONFIG + STYLE
# =========================
st.set_page_config(page_title="Predictive Staffing Model (PSM)", layout="centered")
st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 2.5rem; }
      .small-note { font-size: 0.92rem; color: #555; }
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

# Stable today
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]

# Session state
for key in ["model_ran", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =========================
# BRAND COLORS
# =========================
BRAND_BLACK = "#000000"
BRAND_GOLD = "#7a6200"
GRAY = "#B0B0B0"
LIGHT_GRAY = "#EAEAEA"
MID_GRAY = "#666666"


# =========================
# HELPERS
# =========================
def monthly_hours_from_fte(fte: float, fte_hours_per_week: float, days_in_month: int) -> float:
    """
    Converts FTE to monthly hours assuming FTE hours are defined per week.
    Uses days_in_month/7 to convert week-based hours to month.
    """
    return float(fte) * float(fte_hours_per_week) * (float(days_in_month) / 7.0)


def loaded_hourly_rate(base_hourly: float, benefits_load_pct: float, ot_sick_pct: float) -> float:
    """
    Applies benefits load and OT/Sick/PTO load to hourly rate.
    Example: benefits=30% and OT=4% => hourly * 1.30 * 1.04
    """
    return float(base_hourly) * (1.0 + float(benefits_load_pct)) * (1.0 + float(ot_sick_pct))


def compute_role_mix_ratios(model: StaffingModel, visits_per_day: float, hours_of_operation: float, fte_hours_per_week: float):
    """
    Establishes baseline role-mix ratios vs provider FTE using StaffingModel outputs.
    These ratios are used to scale support staffing off provider supply FTE.
    """
    f = model.calculate_fte_needed(
        visits_per_day=visits_per_day,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week
    )
    prov = max(float(f.get("provider_fte", 0.0)), 0.25)

    return {
        "psr_per_provider": float(f.get("psr_fte", 0.0)) / prov,
        "ma_per_provider": float(f.get("ma_fte", 0.0)) / prov,
        "xrt_per_provider": float(f.get("xrt_fte", 0.0)) / prov,
    }


def compute_monthly_swb_per_visit_fte_based(
    provider_supply_curve,
    visits_per_day_curve,
    days_in_month,
    fte_hours_per_week,
    role_mix,
    hourly_rates,              # dict: {"apc":, "ma":, "psr":, "rt":, "supervisor":, "physician":}
    benefits_load_pct,
    ot_sick_pct,
    physician_supervision_hours_per_month=0.0,   # optional: flat monthly supervision hours
    supervisor_hours_per_month=0.0,              # optional: flat monthly supervisor hours
):
    """
    Policy A: Build staffing cost off provider FTE supply + role-mix ratios.

    Provider supply -> APC labor.
    Support roles scale off provider supply using baseline ratios.
    Optional flat monthly hours for physician supervision and supervisor.
    """
    out_rows = []

    for i in range(12):
        prov_fte = float(provider_supply_curve[i])
        vpd = float(visits_per_day_curve[i])
        dim = int(days_in_month[i])
        month_visits = max(vpd * dim, 1.0)

        # Scale support staffing from provider supply using baseline mix
        psr_fte = prov_fte * float(role_mix["psr_per_provider"])
        ma_fte  = prov_fte * float(role_mix["ma_per_provider"])
        rt_fte  = prov_fte * float(role_mix["xrt_per_provider"])

        # Monthly hours
        apc_hours = monthly_hours_from_fte(prov_fte, fte_hours_per_week, dim)
        psr_hours = monthly_hours_from_fte(psr_fte, fte_hours_per_week, dim)
        ma_hours  = monthly_hours_from_fte(ma_fte,  fte_hours_per_week, dim)
        rt_hours  = monthly_hours_from_fte(rt_fte,  fte_hours_per_week, dim)

        # Loaded rates
        apc_rate = loaded_hourly_rate(hourly_rates["apc"], benefits_load_pct, ot_sick_pct)
        psr_rate = loaded_hourly_rate(hourly_rates["psr"], benefits_load_pct, ot_sick_pct)
        ma_rate  = loaded_hourly_rate(hourly_rates["ma"],  benefits_load_pct, ot_sick_pct)
        rt_rate  = loaded_hourly_rate(hourly_rates["rt"],  benefits_load_pct, ot_sick_pct)

        # Optional flat-hours roles (supervision / supervisor)
        phys_rate = loaded_hourly_rate(hourly_rates["physician"], benefits_load_pct, ot_sick_pct)
        sup_rate  = loaded_hourly_rate(hourly_rates["supervisor"], benefits_load_pct, ot_sick_pct)

        apc_cost = apc_hours * apc_rate
        psr_cost = psr_hours * psr_rate
        ma_cost  = ma_hours  * ma_rate
        rt_cost  = rt_hours  * rt_rate

        phys_cost = float(physician_supervision_hours_per_month) * phys_rate
        sup_cost  = float(supervisor_hours_per_month) * sup_rate

        total_swb = apc_cost + psr_cost + ma_cost + rt_cost + phys_cost + sup_cost
        swb_per_visit = total_swb / month_visits

        out_rows.append({
            "Provider_FTE_Supply": prov_fte,
            "PSR_FTE": psr_fte,
            "MA_FTE": ma_fte,
            "RT_FTE": rt_fte,
            "Visits": month_visits,
            "SWB_$": total_swb,
            "SWB_per_Visit_$": swb_per_visit,
        })

    return pd.DataFrame(out_rows)


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def lead_days_to_months(days: float, avg_days_per_month: float = 30.4) -> int:
    return max(0, int(math.ceil(days / avg_days_per_month)))


def shift_month(month: int, shift: int) -> int:
    return ((month - 1 + shift) % 12) + 1


def months_between(start_month: int, end_month: int):
    """Wrap-safe month list: Dec(12)→Feb(2) => [12,1,2]."""
    months = []
    m = start_month
    while True:
        months.append(m)
        if m == end_month:
            break
        m = 1 if m == 12 else m + 1
    return months


def month_name(m: int) -> str:
    return datetime(2000, m, 1).strftime("%b")


def month_range_label(months):
    if not months:
        return "—"
    start = month_name(months[0])
    end = month_name(months[-1])
    return f"{start}–{end}" if start != end else start


def base_seasonality_multiplier(month: int):
    """Baseline seasonality curve outside flu uplift."""
    if month in [12, 1, 2]:
        return 1.20
    if month in [6, 7, 8]:
        return 0.80
    return 1.00


def compute_seasonality_forecast_months(
    month_nums, baseline_visits, flu_months_set, flu_uplift_pct
):
    """
    Month-loop-safe:
    - baseline multiplier + flu uplift on specified months
    - normalized so average across the provided months equals baseline_visits
    """
    raw = []
    for m in month_nums:
        mult = base_seasonality_multiplier(m)
        if m in flu_months_set:
            mult *= (1.0 + flu_uplift_pct)
        raw.append(baseline_visits * mult)

    avg_raw = float(np.mean(raw)) if raw else baseline_visits
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
        vbuf = float(base_fte) * cv
        sbuf = max(0.0, (float(v) - p75) / mean_visits) * float(base_fte) if mean_visits > 0 else 0.0

        visits_per_provider = float(v) / max(prev_staff, 0.25)
        debt = max(0.0, visits_per_provider - float(safe_visits_per_provider_per_day))

        rdi = decay * rdi + debt
        dbuf = lambda_debt * rdi

        buffer_fte = float(burnout_slider) * (vol_w * vbuf + spike_w * sbuf + debt_w * dbuf)
        raw_target = max(float(provider_min_floor), float(base_fte) + buffer_fte)

        delta = raw_target - prev_staff
        if delta > 0:
            delta = clamp(delta, 0.0, float(smoothing_up))
        else:
            delta = clamp(delta, -float(smoothing_down), 0.0)

        final_staff = max(float(provider_min_floor), prev_staff + delta)
        protective_curve.append(final_staff)
        prev_staff = final_staff

    return protective_curve


# =========================
# AUTO-FREEZE (v3) — month-loop safe
# =========================
def auto_freeze_strategy_v3(
    month_nums_12,
    protective_curve_12,
    flu_start_month,
    flu_end_month,
    pipeline_lead_days,
    notice_days,
    freeze_buffer_months=1,
):
    lead_months = lead_days_to_months(pipeline_lead_days)
    notice_months = lead_days_to_months(notice_days)

    independent_ready_month = flu_start_month
    req_post_month = shift_month(independent_ready_month, -lead_months)
    hire_visible_month = shift_month(req_post_month, lead_months)

    trough_idx = int(np.argmin(np.array(protective_curve_12, dtype=float)))
    trough_month = int(month_nums_12[trough_idx])

    decline_months = months_between(shift_month(flu_end_month, 1), trough_month)

    freeze_months = list(decline_months)
    for i in range(1, int(freeze_buffer_months) + 1):
        freeze_months.append(shift_month(trough_month, i))

    recruiting_open_months = []
    for i in range(lead_months + 1):
        recruiting_open_months.append(shift_month(req_post_month, -i))

    # Dedupe keep order
    def dedupe_keep_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    freeze_months = dedupe_keep_order(freeze_months)
    recruiting_open_months = dedupe_keep_order(recruiting_open_months)

    # Recruiting-open wins over freeze
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
        flu_months=months_between(flu_start_month, flu_end_month),
    )


# =========================
# 36-MONTH SUPPLY SIM (one run)
# =========================
def simulate_supply_36(
    target_curve_36,
    provider_min_floor,
    annual_turnover_rate,
    hire_visible_idx,               # month index (0..35) after which ramp-up allowed
    freeze_months_set,              # set of month numbers (1..12) that block ramp-up
    max_ramp_up_per_month,
    max_ramp_down_per_month,
    seasonality_ramp_enabled,
    confirmed_hire_month=None,      # 1..12
    confirmed_hire_fte=0.0,
    confirmed_once=True,
    start_supply=None,
):
    """
    Month-by-month evolution (true timeline M1..M36):
    - Ramp toward target (cap up/down)
    - Attrition scales with current supply (prev)
    - Ramp-up blocked before hire_visible_idx and during freeze months if enabled
    - Confirmed hire applied once at first occurrence of confirmed_hire_month
    """
    monthly_turnover_rate = float(annual_turnover_rate) / 12.0

    staff = []
    prev = float(start_supply) if start_supply is not None else float(target_curve_36[0])
    prev = max(prev, float(provider_min_floor))

    hire_applied = False

    for i, target in enumerate(target_curve_36):
        # month number inferred from i (Jan=1 on i=0, etc.)
        month_num = (i % 12) + 1

        in_freeze = month_num in freeze_months_set

        if seasonality_ramp_enabled:
            if (i < int(hire_visible_idx)) or in_freeze:
                ramp_up_cap = 0.0
            else:
                ramp_up_cap = float(max_ramp_up_per_month)
        else:
            ramp_up_cap = float(max_ramp_up_per_month)

        delta = float(target) - prev
        if delta > 0:
            delta = clamp(delta, 0.0, ramp_up_cap)
        else:
            delta = clamp(delta, -float(max_ramp_down_per_month), 0.0)

        planned = prev + delta

        # Attrition on current supply
        planned -= prev * monthly_turnover_rate

        # Confirmed hire (one-time, first time we hit that month)
        if (confirmed_hire_month is not None) and (not hire_applied):
            if int(month_num) == int(confirmed_hire_month):
                planned += float(confirmed_hire_fte)
                hire_applied = True if confirmed_once else False

        planned = max(planned, float(provider_min_floor))
        staff.append(planned)
        prev = planned

    return staff


# =========================
# MONTE CARLO (N runs) → P50/P90
# =========================
def run_monte_carlo_supply(
    target_curve_36,
    provider_min_floor,
    provider_turnover_mean,
    provider_turnover_sd,
    hire_visible_idx_mean,
    hire_visible_idx_sd,
    ramp_up_mean,
    ramp_up_sd,
    ramp_down,
    freeze_months_set,
    seasonality_ramp_enabled,
    confirmed_hire_month,
    confirmed_hire_fte,
    n_runs,
    seed=7,
):
    rng = np.random.default_rng(int(seed))

    all_runs = np.zeros((int(n_runs), len(target_curve_36)), dtype=float)

    for r in range(int(n_runs)):
        # Turnover sample (truncate to [0, 0.8] for sanity)
        turnover = float(rng.normal(provider_turnover_mean, provider_turnover_sd))
        turnover = float(clamp(turnover, 0.0, 0.80))

        # Hire-visible index sample (truncate to [0, 35])
        hv = float(rng.normal(hire_visible_idx_mean, hire_visible_idx_sd))
        hv = int(clamp(int(round(hv)), 0, 35))

        # Ramp-up sample (truncate to [0.05, 1.5])
        ramp_up = float(rng.normal(ramp_up_mean, ramp_up_sd))
        ramp_up = float(clamp(ramp_up, 0.05, 1.50))

        # Start supply: start near first target but don’t assume perfect
        start_supply = float(target_curve_36[0]) * float(rng.normal(1.0, 0.03))
        start_supply = max(start_supply, float(provider_min_floor))

        run = simulate_supply_36(
            target_curve_36=target_curve_36,
            provider_min_floor=provider_min_floor,
            annual_turnover_rate=turnover,
            hire_visible_idx=hv,
            freeze_months_set=freeze_months_set,
            max_ramp_up_per_month=ramp_up,
            max_ramp_down_per_month=ramp_down,
            seasonality_ramp_enabled=seasonality_ramp_enabled,
            confirmed_hire_month=confirmed_hire_month,
            confirmed_hire_fte=confirmed_hire_fte,
            start_supply=start_supply,
        )
        all_runs[r, :] = np.array(run, dtype=float)

    p50 = np.percentile(all_runs, 50, axis=0)
    p90 = np.percentile(all_runs, 90, axis=0)
    return p50.tolist(), p90.tolist()


# =========================
# SWB / VISIT FEASIBILITY
# =========================
def monthly_hours_from_fte(role_fte: float, fte_hours_per_week: float) -> float:
    # Approx hours per month (52/12 weeks)
    return float(role_fte) * float(fte_hours_per_week) * (52.0 / 12.0)


def labor_cost_month(role_fte: float, hourly: float, fte_hours_per_week: float, benefits_pct: float, pto_ot_pct: float) -> float:
    hrs = monthly_hours_from_fte(role_fte, fte_hours_per_week)
    loaded_hourly = float(hourly) * (1.0 + float(benefits_pct))
    base = hrs * loaded_hourly
    return base * (1.0 + float(pto_ot_pct))


def compute_swb_visit_12(
    visits_per_day_12,
    days_in_month_12,
    fte_by_role_12,                 # dict role -> list[12] role_fte
    rates,                          # dict role -> hourly
    fte_hours_per_week,
    benefits_pct,
    pto_ot_pct,
):
    rows = []
    for i in range(12):
        visits_month = float(visits_per_day_12[i]) * float(days_in_month_12[i])
        visits_month = max(visits_month, 1.0)

        total_labor = 0.0
        for role, fte_list in fte_by_role_12.items():
            hourly = float(rates.get(role, 0.0))
            total_labor += labor_cost_month(
                role_fte=float(fte_list[i]),
                hourly=hourly,
                fte_hours_per_week=fte_hours_per_week,
                benefits_pct=benefits_pct,
                pto_ot_pct=pto_ot_pct,
            )

        swb_per_visit = total_labor / visits_month
        rows.append((total_labor, swb_per_visit))
    return rows


# =========================
# SIDEBAR INPUTS
# =========================
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

    st.subheader("Probability / Accuracy")
    n_runs = st.number_input("Simulation Runs (Monte Carlo)", min_value=100, max_value=5000, value=1000, step=100)
    seed = st.number_input("Random Seed", min_value=1, max_value=999999, value=7, step=1)

    # Uncertainty knobs (kept simple, executive-friendly)
    st.markdown("<div class='small-note'><b>Uncertainty ranges</b> (for “near-certain” planning):</div>", unsafe_allow_html=True)
    turnover_sd = st.slider("Turnover SD (±)", 0.00, 0.20, 0.05, 0.01)
    hv_sd_months = st.slider("Hire-visible timing SD (months)", 0.0, 6.0, 2.0, 0.5)
    ramp_sd = st.slider("Ramp-up SD (FTE/mo)", 0.0, 0.75, 0.20, 0.05)

    st.divider()

    st.subheader("SWB/Visit Feasibility (FTE-based)")
    target_swb_visit = st.number_input("Target SWB / Visit ($)", min_value=0.0, value=85.0, step=1.0)

    with st.expander("Hourly Rates (baseline assumptions)", expanded=False):
        benefits_pct = st.number_input("Benefits Load (%)", min_value=0.0, value=30.0, step=5.0) / 100.0
        pto_ot_pct = st.number_input("OT + Sick/PTO (%)", min_value=0.0, value=4.0, step=1.0) / 100.0

        rate_phys_supervision = st.number_input("Physician (Supervision) $/hr", min_value=0.0, value=135.79, step=1.0)
        rate_apc = st.number_input("APC $/hr", min_value=0.0, value=62.00, step=0.5)
        rate_ma = st.number_input("MA $/hr", min_value=0.0, value=24.14, step=0.25)
        rate_psr = st.number_input("PSR $/hr", min_value=0.0, value=21.23, step=0.25)
        rate_rt = st.number_input("RT $/hr", min_value=0.0, value=31.36, step=0.25)
        rate_supervisor = st.number_input("Supervisor $/hr", min_value=0.0, value=28.25, step=0.25)

    st.subheader("SWB/Visit Feasibility (FTE-based)")
    target_swb_per_visit = st.number_input("Target SWB / Visit ($)", value=85.00, step=1.00)
    
    with st.expander("Hourly Rates (baseline assumptions)", expanded=False):
        benefits_load_pct = st.number_input("Benefits Load (%)", value=30.00, step=1.00) / 100.0
        ot_sick_pct = st.number_input("OT + Sick/PTO (%)", value=4.00, step=0.50) / 100.0
    
        physician_hr = st.number_input("Physician (Supervision) $/hr", value=135.79, step=1.00)
        apc_hr = st.number_input("APC $/hr", value=62.00, step=1.00)
        ma_hr = st.number_input("MA $/hr", value=24.14, step=0.50)
        psr_hr = st.number_input("PSR $/hr", value=21.23, step=0.50)
        rt_hr = st.number_input("RT $/hr", value=31.36, step=0.50)
        supervisor_hr = st.number_input("Supervisor $/hr", value=28.25, step=0.50)
    
    with st.expander("Optional: fixed monthly hours", expanded=False):
        physician_supervision_hours_per_month = st.number_input("Physician supervision hours/month", value=0.0, step=1.0)
        supervisor_hours_per_month = st.number_input("Supervisor hours/month", value=0.0, step=1.0)

    
    run_model = st.button("Run Model")


# =========================
# RUN MODEL (36-month engine, show 12 months)
# =========================
if run_model:
    # Core timeline: 36 months (month numbers only + day counts for Jan..Dec repeated)
    month_nums_36 = [(i % 12) + 1 for i in range(36)]
    month_nums_12 = [(i % 12) + 1 for i in range(12)]

    # day counts for a representative year (current_year) – repeated is fine for planning
    current_year = today.year
    dates_12 = pd.date_range(start=datetime(current_year, 1, 1), periods=12, freq="MS")
    days_in_month_12 = [pd.Period(d, "M").days_in_month for d in dates_12]
    days_in_month_36 = [days_in_month_12[i % 12] for i in range(36)]

    # Visits forecast (36 months) – normalized over the 12-month seasonal cycle, repeated
    flu_months = months_between(flu_start_month, flu_end_month)
    flu_months_set = set(flu_months)

    visits_12 = compute_seasonality_forecast_months(
        month_nums_12, baseline_visits=visits, flu_months_set=flu_months_set, flu_uplift_pct=flu_uplift_pct
    )
    visits_36 = [visits_12[i % 12] for i in range(36)]

    # Baseline provider FTE (from the StaffingModel at baseline visits)
    fte_result_baseline = model.calculate_fte_needed(
        visits_per_day=visits,
        hours_of_operation_per_week=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    baseline_provider_fte = max(float(fte_result_baseline["provider_fte"]), float(provider_min_floor))

    # Demand curves (lean + protective) computed for 12, then repeated for 36
    provider_base_demand_12 = visits_to_provider_demand(
        model=model,
        visits_by_month=visits_12,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    protective_curve_12 = burnout_protective_staffing_curve(
        visits_by_month=visits_12,
        base_demand_fte=provider_base_demand_12,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    provider_base_demand_36 = [provider_base_demand_12[i % 12] for i in range(36)]
    protective_curve_36 = [protective_curve_12[i % 12] for i in range(36)]

    # Pipeline lead time (mean)
    pipeline_lead_days_mean = float(days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days)

    # Auto-freeze v3 (month loop strategy based on the 12-month protective curve)
    strategy = auto_freeze_strategy_v3(
        month_nums_12=month_nums_12,
        protective_curve_12=protective_curve_12,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
        pipeline_lead_days=pipeline_lead_days_mean,
        notice_days=notice_days,
        freeze_buffer_months=1,
    )

    req_post_month = int(strategy["req_post_month"])
    hire_visible_month = int(strategy["hire_visible_month"])
    independent_ready_month = int(strategy["independent_ready_month"])
    freeze_months = list(strategy["freeze_months"])
    recruiting_open_months = list(strategy["recruiting_open_months"])
    lead_months = int(strategy["lead_months"])

    # Convert hire-visible month into a *timeline index* within 36 months:
    # "hire-visible begins at the first occurrence of hire_visible_month after the first occurrence of req_post_month"
    # We anchor “story” at req_post_month.
    def first_occurrence_idx(month_nums, month):
        for i, m in enumerate(month_nums):
            if m == month:
                return i
        return 0

    req_post_idx0 = first_occurrence_idx(month_nums_36, req_post_month)
    # hire visible idx = req_post_idx0 + lead_months (expected)
    hire_visible_idx_mean = int(clamp(req_post_idx0 + lead_months, 0, 35))

    # Derived ramp-up mean (protective gap at flu start / months in flu window)
    flu_months_list = months_between(flu_start_month, flu_end_month)
    months_in_flu_window = max(len(flu_months_list), 1)

    # Use protective at flu-start (in the 12-month cycle)
    flu_start_idx_12 = month_nums_12.index(flu_start_month) if flu_start_month in month_nums_12 else 0
    target_at_flu = float(protective_curve_12[flu_start_idx_12])
    fte_gap_to_close = max(target_at_flu - baseline_provider_fte, 0.0)
    derived_ramp_mean = min(fte_gap_to_close / months_in_flu_window, 1.25)
    derived_ramp_mean = max(derived_ramp_mean, 0.10)

    # Monte Carlo: compute P50/P90 supply for protective target
    p50_supply_36, p90_supply_36 = run_monte_carlo_supply(
        target_curve_36=protective_curve_36,
        provider_min_floor=provider_min_floor,
        provider_turnover_mean=float(provider_turnover),
        provider_turnover_sd=float(turnover_sd),
        hire_visible_idx_mean=float(hire_visible_idx_mean),
        hire_visible_idx_sd=float(hv_sd_months),
        ramp_up_mean=float(derived_ramp_mean),
        ramp_up_sd=float(ramp_sd),
        ramp_down=0.25,
        freeze_months_set=set(freeze_months) if enable_seasonality_ramp else set(),
        seasonality_ramp_enabled=enable_seasonality_ramp,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        n_runs=int(n_runs),
        seed=int(seed),
    )

    # Burnout exposure (use P50 as “most likely” exposure and P90 for conservative exposure)
    gap_p50_36 = [max(t - s, 0.0) for t, s in zip(protective_curve_36, p50_supply_36)]
    gap_p90_36 = [max(t - s, 0.0) for t, s in zip(protective_curve_36, p90_supply_36)]

    # Choose 12-month display window (anchor at req-post month)
    window_start = req_post_idx0
    window_end = window_start + 12

    # If the anchor is too late (e.g., req_post in Dec and start hits last cycle), wrap window within 36
    # We’ll just take 12 sequential months; if it spills past 36, shift earlier so we still show 12.
    if window_end > 36:
        window_start = 36 - 12
        window_end = 36

    # Slice window
    win_months = month_nums_36[window_start:window_end]
    win_labels = [month_name(m) for m in win_months]

    win_visits = visits_36[window_start:window_end]
    win_days = days_in_month_36[window_start:window_end]

    win_lean = provider_base_demand_36[window_start:window_end]
    win_prot = protective_curve_36[window_start:window_end]
    win_p50_supply = p50_supply_36[window_start:window_end]
    win_p90_supply = p90_supply_36[window_start:window_end]

    win_gap_p50 = [max(t - s, 0.0) for t, s in zip(win_prot, win_p50_supply)]
    win_gap_p90 = [max(t - s, 0.0) for t, s in zip(win_prot, win_p90_supply)]

    # Compute “months exposed” in the window (most likely and near-certain)
    months_exposed_p50 = int(sum(1 for g in win_gap_p50 if g > 1e-6))
    months_exposed_p90 = int(sum(1 for g in win_gap_p90 if g > 1e-6))

    # SWB/Visit feasibility (FTE-based), using the protective target FTEs by role for the window
    # We calculate role FTE requirements from StaffingModel month-by-month.
    role_keys = ["provider_fte", "psr_fte", "ma_fte", "xrt_fte"]
    role_map = {
        "Physician (Supervision)": None,  # optional / separate input
        "APC": "provider_fte",
        "PSR": "psr_fte",
        "MA": "ma_fte",
        "RT": "xrt_fte",
        "Supervisor": None,               # optional
    }

    rates = {
        "Physician (Supervision)": rate_phys_supervision,
        "APC": rate_apc,
        "MA": rate_ma,
        "PSR": rate_psr,
        "RT": rate_rt,
        "Supervisor": rate_supervisor,
    }

    # Build role FTE lists for the 12-month window
    fte_by_role_12 = {role: [] for role in rates.keys()}
    for vpd in win_visits:
        fte_staff = model.calculate_fte_needed(
            visits_per_day=float(vpd),
            hours_of_operation_per_week=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week,
        )
        # Required FTE by role
        fte_by_role_12["APC"].append(float(fte_staff["provider_fte"]))
        fte_by_role_12["PSR"].append(float(fte_staff["psr_fte"]))
        fte_by_role_12["MA"].append(float(fte_staff["ma_fte"]))
        fte_by_role_12["RT"].append(float(fte_staff["xrt_fte"]))

        # Optional roles default to 0 unless you later define logic
        fte_by_role_12["Physician (Supervision)"].append(0.0)
        fte_by_role_12["Supervisor"].append(0.0)

    swb_rows = compute_swb_visit_12(
        visits_per_day_12=win_visits,
        days_in_month_12=win_days,
        fte_by_role_12=fte_by_role_12,
        rates=rates,
        fte_hours_per_week=fte_hours_per_week,
        benefits_pct=benefits_pct,
        pto_ot_pct=pto_ot_pct,
    )
    swb_monthly_costs = [x[0] for x in swb_rows]
    swb_per_visit = [x[1] for x in swb_rows]
    swb_feasible = [sv <= float(target_swb_visit) for sv in swb_per_visit]
    swb_months_over = int(sum(1 for ok in swb_feasible if not ok))

    # Store results
    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        # window
        win_labels=win_labels,
        win_months=win_months,
        win_visits=win_visits,
        win_days=win_days,
        win_lean=win_lean,
        win_prot=win_prot,
        win_p50_supply=win_p50_supply,
        win_p90_supply=win_p90_supply,
        win_gap_p50=win_gap_p50,
        win_gap_p90=win_gap_p90,
        months_exposed_p50=months_exposed_p50,
        months_exposed_p90=months_exposed_p90,

        # strategy
        flu_months=flu_months,
        freeze_months=freeze_months,
        recruiting_open_months=recruiting_open_months,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        independent_ready_month=independent_ready_month,
        lead_months=lead_months,

        # probability settings
        n_runs=int(n_runs),
        seed=int(seed),

        # ramp + gap
        derived_ramp_mean=derived_ramp_mean,
        fte_gap_to_close=fte_gap_to_close,
        months_in_flu_window=months_in_flu_window,

        # swb
        target_swb_visit=float(target_swb_visit),
        swb_monthly_costs=swb_monthly_costs,
        swb_per_visit=swb_per_visit,
        swb_feasible=swb_feasible,
        swb_months_over=swb_months_over,

        # model inputs snapshot
        baseline_provider_fte=baseline_provider_fte,
        provider_turnover=float(provider_turnover),
        pipeline_lead_days_mean=float(pipeline_lead_days_mean),
        enable_seasonality_ramp=bool(enable_seasonality_ramp),
    )


# =========================
# STOP IF NOT RUN
# =========================
if not st.session_state.get("model_ran"):
    st.info("Enter inputs in the sidebar and click **Run Model**.")
    st.stop()

R = st.session_state["results"]


# =========================
# SECTION 1 — OPERATIONS
# =========================
st.markdown("---")
st.header("1) Operations — Seasonality Demand (12-Month View)")
st.caption("Visits/day (seasonality + flu uplift) → lean demand → protective target (burnout-protective).")

ops_df = pd.DataFrame({
    "Month": R["win_labels"],
    "Visits/Day (Forecast)": np.round(R["win_visits"], 1),
    "Lean Target (FTE)": np.round(R["win_lean"], 2),
    "Protective Target (FTE)": np.round(R["win_prot"], 2),
})
st.dataframe(ops_df, hide_index=True, use_container_width=True)

st.success(
    "**Operations Summary:** The app forecasts visits/day (seasonality + flu uplift) and converts that into "
    "a lean staffing target. The protective target adds a burnout buffer."
)


# =========================
# SECTION 2 — REALITY (probabilistic supply)
# =========================
st.markdown("---")
st.header("2) Reality — Probabilistic Supply vs Targets (Most-likely + Near-certain)")
st.caption("36-month simulation engine → show a 12-month window. P50 = most likely. P90 = near-certain planning line.")

flu_label = month_range_label(R["flu_months"])
freeze_label = month_range_label(R["freeze_months"])
recruit_label = month_range_label(R["recruiting_open_months"])

req_post_label = month_name(int(R["req_post_month"]))
hire_visible_label = month_name(int(R["hire_visible_month"]))
independent_label = month_name(int(R["independent_ready_month"]))

st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; gap:16px;
                padding:12px 16px; background:#F7F7F7; border-radius:10px;
                border:1px solid #E0E0E0; font-size:15px;">
        <div><b>Flu:</b> {flu_label}</div>
        <div><b>Freeze:</b> {freeze_label}</div>
        <div><b>Recruiting:</b> {recruit_label}</div>
        <div><b>Post Req:</b> {req_post_label}</div>
        <div><b>Hires Visible:</b> {hire_visible_label}</div>
        <div><b>Independent By:</b> {independent_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

peak_gap_p50 = float(max(R["win_gap_p50"])) if R["win_gap_p50"] else 0.0
avg_gap_p50 = float(np.mean(R["win_gap_p50"])) if R["win_gap_p50"] else 0.0
peak_gap_p90 = float(max(R["win_gap_p90"])) if R["win_gap_p90"] else 0.0
avg_gap_p90 = float(np.mean(R["win_gap_p90"])) if R["win_gap_p90"] else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Peak Gap (P50)", f"{peak_gap_p50:.2f} FTE")
m2.metric("Avg Gap (P50)", f"{avg_gap_p50:.2f} FTE")
m3.metric("Months Exposed (P50)", f"{R['months_exposed_p50']}/12")
m4.metric("Months Exposed (P90)", f"{R['months_exposed_p90']}/12")

# Plot: targets + P50/P90 supply
fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

x = np.arange(12)

# Shade freeze months in the displayed window
freeze_set = set(R["freeze_months"])
for i, m in enumerate(R["win_months"]):
    if int(m) in freeze_set:
        ax1.axvspan(i - 0.45, i + 0.45, alpha=0.12, color=BRAND_GOLD, linewidth=0)

# Lines
ax1.plot(x, R["win_lean"], linestyle=":", linewidth=1.2, color=GRAY, label="Lean Target (Demand)")
ax1.plot(x, R["win_prot"], linewidth=2.0, color=BRAND_GOLD, marker="o", markersize=4, label="Protective Target")

ax1.plot(x, R["win_p50_supply"], linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=4, label="Supply (P50 most likely)")
ax1.plot(x, R["win_p90_supply"], linewidth=1.6, color=BRAND_BLACK, linestyle="--", label="Supply (P90 near-certain)")

# Exposure zones
prot = np.array(R["win_prot"], dtype=float)
p50 = np.array(R["win_p50_supply"], dtype=float)
p90 = np.array(R["win_p90_supply"], dtype=float)

ax1.fill_between(x, p50, prot, where=prot > p50, color=BRAND_GOLD, alpha=0.12, label="Exposure (P50)")
ax1.fill_between(x, p90, prot, where=prot > p90, color=BRAND_GOLD, alpha=0.06, label="Exposure (P90)")

ax1.set_title("Reality — Targets vs Supply (Probabilistic)", fontsize=16, fontweight="bold", pad=16, color=BRAND_BLACK)
ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax1.set_xticks(x)
ax1.set_xticklabels(R["win_labels"], fontsize=11, color=BRAND_BLACK)
ax1.tick_params(axis="y", labelsize=11, colors=BRAND_BLACK)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

# Secondary axis (visits/day)
ax2 = ax1.twinx()
ax2.plot(x, R["win_visits"], linestyle="-.", linewidth=1.4, color=MID_GRAY, label="Forecast Visits/Day")
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
    f"**Reality Summary:** This view is a 12-month slice from a 36-month simulation (no year reset). "
    f"P50 is the most likely supply path; P90 is the near-certain planning line. "
    f"Expected ramp baseline: **{R['derived_ramp_mean']:.2f} FTE/month**. "
    f"Simulations: **{R['n_runs']}** (seed {R['seed']})."
)

st.info(
    "🧠 **Auto-Hiring Strategy (v3)**\n\n"
    f"- Freeze months: **{', '.join([month_name(m) for m in R['freeze_months']]) or '—'}**\n"
    f"- Recruiting window: **{', '.join([month_name(m) for m in R['recruiting_open_months']]) or '—'}**\n"
    f"- Post req: **{req_post_label}** | Hires visible: **{hire_visible_label}** | Independent by: **{independent_label}**\n"
    f"- Pipeline lead: **{R['pipeline_lead_days_mean']:.0f} days (~{R['lead_months']} months)**\n"
)


# =========================
# SECTION 3 — SWB/Visit FEASIBILITY
# =========================
st.markdown("---")
st.header("3) Financial Feasibility — SWB/Visit vs Target (FTE-based)")
st.caption("Uses staffing FTE requirements (by role) + hourly rates + benefits + PTO/OT load to estimate SWB/Visit.")

feas_df = pd.DataFrame({
    "Month": R["win_labels"],
    "Est. Labor $/Month": np.round(R["swb_monthly_costs"], 0).astype(int),
    "Est. SWB / Visit ($)": np.round(R["swb_per_visit"], 2),
    "Target SWB / Visit ($)": [round(R["target_swb_visit"], 2)] * 12,
    "Feasible?": ["✅" if ok else "❌" for ok in R["swb_feasible"]],
})
st.dataframe(feas_df, hide_index=True, use_container_width=True)

if R["swb_months_over"] == 0:
    st.success("✅ **Feasibility:** All 12 months meet the target SWB/Visit under these assumptions.")
else:
    st.error(
        f"❌ **Feasibility:** {R['swb_months_over']}/12 months exceed the target SWB/Visit. "
        f"To reconcile, we can adjust: rates/benefits assumptions, demand/throughput assumptions, or the staffing target."
    )


# =========================
# SECTION 4 — DECISION SUMMARY
# =========================
st.markdown("---")
st.header("4) Decision — Executive Summary")

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Peak Gap (P50)", f"{peak_gap_p50:.2f} FTE")
    st.metric("Months Exposed (P50)", f"{R['months_exposed_p50']}/12")
    st.metric("Months Exposed (P90)", f"{R['months_exposed_p90']}/12")
    st.metric("SWB/Visit Over Target", f"{R['swb_months_over']}/12")
with col2:
    st.write(
        f"**What you’re seeing:** a 12-month view pulled from a continuous 36-month simulation, so **Dec → Jan is continuous**.\n\n"
        f"**Operational plan:**\n"
        f"- **Post req:** {req_post_label}\n"
        f"- **Hires visible:** {hire_visible_label}\n"
        f"- **Independent by:** {independent_label}\n"
        f"- **Freeze months:** {month_range_label(R['freeze_months'])}\n\n"
        f"**Probability framing:**\n"
        f"- P50 supply = most likely outcome\n"
        f"- P90 supply = near-certain planning line (conservative)\n\n"
        f"**Financial feasibility (SWB/Visit):**\n"
        f"- Target SWB/Visit: **${R['target_swb_visit']:.2f}**\n"
        f"- Months over target: **{R['swb_months_over']}/12**\n"
    )

st.success(
    "✅ **Decision Summary:** This version eliminates the Jan reset by simulating a true timeline (36 months) "
    "and only *displaying* a 12-month window. It also adds near-certain planning (P90) and tests feasibility "
    "against SWB/Visit using FTE-based labor costing."
)


# =========================
# SECTION 5 — IN-DEPTH EXECUTIVE SUMMARY (Narrative)
# =========================
st.markdown("---")
st.header("5) In-Depth Executive Summary")
st.caption("Plain-language explanation of what the model is doing and how to use it.")

summary_md = f"""
### Why the “December → January” issue is now solved
We run the staffing math over a continuous **36-month timeline**, then show a **12-month slice**.
That means January on the chart is simply the next month in the same simulation, not a reset.

---

### What the model does (in order)
1) **Demand:** Forecasts visits/day using seasonality and flu uplift (**{month_range_label(R['flu_months'])}**), normalized to your annual baseline.  
2) **Targets:** Converts visits into a **lean target** and a **protective target** (burnout protection).  
3) **Reality (probabilistic):** Simulates provider supply month-to-month under:
   - pipeline timing (req → visible → independent),
   - hiring freezes (v3 strategy),
   - ramp limits,
   - turnover.

We run that simulation **{R['n_runs']}** times with uncertainty in turnover, hire-visible timing, and ramp-up feasibility.
- **P50** = most likely supply path
- **P90** = near-certain planning line (conservative)

---

### The auto-hiring strategy (v3) it’s recommending
- **Post req:** {req_post_label}  
- **Hires visible:** {hire_visible_label}  
- **Independent by:** {independent_label}  
- **Freeze:** {month_range_label(R['freeze_months'])}  
- **Recruiting window:** {month_range_label(R['recruiting_open_months'])}

---

### How we test financial feasibility (SWB/Visit)
For the 12-month window shown, we estimate labor dollars using:
- role FTE requirements (from the staffing model),
- your hourly rates,
- benefits load, and PTO/OT load,
then compute **SWB/Visit** and compare to your target.

If SWB/Visit is over target in a month, the plan may be operationally correct but financially infeasible under the current assumptions.

---
"""
st.markdown(summary_md)

st.success("Next: we’ll tie this into VVI (RF/LF + targets) so staffing feasibility is evaluated in the same language as performance (VVI).")
