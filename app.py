import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from collections import deque

from psm.staffing_model import StaffingModel

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM)", layout="centered")

st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Predictive Staffing Model (PSM)")
st.caption("Operations → Reality → Finance → Strategy → Decision")

model = StaffingModel()

# ============================================================
# STABLE TODAY
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]

# ============================================================
# SESSION STATE (top-level)
# ============================================================
for k in ["model_ran", "results", "suggested_confirmed_hire_fte"]:
    st.session_state.setdefault(k, None)

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

def lead_days_to_months(days: int, avg_days_per_month: float = 30.4) -> int:
    return max(0, int(math.ceil(float(days) / float(avg_days_per_month))))

def shift_month(month: int, shift: int) -> int:
    return ((int(month) - 1 + int(shift)) % 12) + 1

def months_between(start_month: int, end_month: int):
    """Wrapped month list inclusive. Example: Dec->Feb = [12,1,2]."""
    months = []
    m = int(start_month)
    end_month = int(end_month)
    while True:
        months.append(m)
        if m == end_month:
            break
        m = 1 if m == 12 else m + 1
    return months

def month_range_label(months):
    if not months:
        return "—"
    start = datetime(2000, int(months[0]), 1).strftime("%b")
    end = datetime(2000, int(months[-1]), 1).strftime("%b")
    return f"{start}–{end}" if start != end else start

# ============================================================
# SEASONALITY (NEW)
# ============================================================
def compute_seasonality_forecast_multiyear(
    dates,
    baseline_visits,
    seasonal_start_month,
    seasonal_end_month,
    seasonal_change_pct,
    summer_downcycle=True,
    summer_months=(6, 7, 8),
    normalize=True,
):
    """
    Baseline = user's annual average visits/day.

    Seasonal window (start->end): baseline * (1 + pct)
    Summer downcycle (default Jun-Aug): baseline * (1 - pct) if enabled
    Shoulder months: baseline

    If normalize=True, scales series so mean(visits) == baseline across full horizon.
    """
    seasonal_months = set(months_between(int(seasonal_start_month), int(seasonal_end_month)))
    summer_set = set(int(m) for m in summer_months)

    raw = []
    for d in dates:
        m = int(d.month)
        if m in seasonal_months:
            v = float(baseline_visits) * (1.0 + float(seasonal_change_pct))
        elif summer_downcycle and (m in summer_set):
            v = float(baseline_visits) * (1.0 - float(seasonal_change_pct))
        else:
            v = float(baseline_visits)
        raw.append(v)

    if not normalize:
        return raw

    avg_raw = float(np.mean(raw)) if len(raw) else float(baseline_visits)
    if avg_raw <= 0:
        return [float(baseline_visits) for _ in raw]

    scale = float(baseline_visits) / avg_raw
    return [v * scale for v in raw]

def visits_to_provider_demand(model, visits_by_month, hours_of_operation, fte_hours_per_week, provider_min_floor):
    demand = []
    for v in visits_by_month:
        fte = model.calculate_fte_needed(
            visits_per_day=float(v),
            hours_of_operation_per_week=float(hours_of_operation),
            fte_hours_per_week=float(fte_hours_per_week),
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
    Protective target curve (demand + buffer).
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
        v = float(v)
        base_fte = float(base_fte)

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

def typical_12_month_curve(dates_full, values_full):
    """Average multi-year values by month-of-year -> 12 values Jan..Dec."""
    df = pd.DataFrame({"date": dates_full, "val": values_full})
    df["month"] = df["date"].dt.month
    g = df.groupby("month")["val"].mean()
    return [float(g.loc[m]) for m in range(1, 13)]

def planned_hires_from_typical_target(
    dates_full,
    target_typical_12,
    max_hiring_up_after_visible,
    req_post_month,
    hire_visible_month,
    freeze_months,
    lead_months,
    baseline_provider_fte,              # ✅ add this
    seasonality_ramp_enabled=True,
):
    freeze_set = set(int(m) for m in (freeze_months or []))

    # Visible blackout: don't show visible hires before hire_visible_month
    blackout_months = set(months_between(int(req_post_month), shift_month(int(hire_visible_month), -1)))

    hires_plan = []
    backlog = 0.0

    for d in dates_full:
        visible_m = int(d.month)

        # ✅ desired staffing LEVEL for the month (not delta)
        # (don’t plan to hire below baseline)
        desired_level = max(float(target_typical_12[visible_m - 1]), float(baseline_provider_fte))

        prev_m = shift_month(visible_m, -1)
        prev_level = max(float(target_typical_12[prev_m - 1]), float(baseline_provider_fte))

        # How much the desired level increased this month
        delta_up = max(desired_level - prev_level, 0.0)

        # backlog carries missed “delta_up” from blocked months
        need_visible = backlog + delta_up

        if seasonality_ramp_enabled:
            # Freeze blocks the REQ month that would create this visible month
            req_m = shift_month(visible_m, -int(lead_months))
            if req_m in freeze_set:
                # can't start this req → backlog it
                hires_plan.append(0.0)
                backlog = need_visible
                continue

            # no visible hires before visibility starts
            if visible_m in blackout_months:
                hires_plan.append(0.0)
                backlog = need_visible
                continue

        # allowed month: schedule as much as possible (cap), keep remaining backlog
        hires = clamp(need_visible, 0.0, float(max_hiring_up_after_visible))
        hires_plan.append(hires)
        backlog = max(need_visible - hires, 0.0)

    return hires_plan

# ============================================================
# AUTO-FREEZE v3 (built from typical seasonal curve)
# ============================================================
def auto_freeze_strategy_v3_from_typical(
    dates_template_12,
    protective_typical_12,
    seasonal_start_month,
    seasonal_end_month,
    pipeline_lead_days,
    notice_days,
    freeze_buffer_months=1,
):
    lead_months = lead_days_to_months(pipeline_lead_days)
    notice_months = lead_days_to_months(notice_days)

    independent_ready_month = int(seasonal_start_month)
    req_post_month = shift_month(independent_ready_month, -lead_months)
    hire_visible_month = shift_month(req_post_month, lead_months)

    trough_idx = int(np.argmin(np.array(protective_typical_12, dtype=float)))
    trough_month = int(dates_template_12[trough_idx].month)

    decline_months = months_between(shift_month(int(seasonal_end_month), 1), trough_month)

    freeze_months = list(decline_months)
    for i in range(1, int(freeze_buffer_months) + 1):
        freeze_months.append(shift_month(trough_month, i))

    recruiting_open_months = []
    for i in range(lead_months + 1):
        recruiting_open_months.append(shift_month(req_post_month, -i))

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

    # Recruiting-open wins
    freeze_months = [m for m in freeze_months if m not in set(recruiting_open_months)]

    return dict(
        lead_months=lead_months,
        notice_months=notice_months,
        independent_ready_month=independent_ready_month,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        trough_month=trough_month,
        freeze_months=freeze_months,
        recruiting_open_months=recruiting_open_months,
        seasonal_months=months_between(int(seasonal_start_month), int(seasonal_end_month)),
    )

# ============================================================
# SUPPLY SIM (multi-year continuous, attrition always; hiring only when allowed)
# ============================================================
def simulate_supply_multiyear_best_case(
    dates_full,
    baseline_provider_fte,
    target_curve_full,
    provider_min_floor,
    annual_turnover_rate,
    notice_days,
    req_post_month,
    hire_visible_month,
    freeze_months,
    max_hiring_up_after_visible,
    lead_months,
    confirmed_hire_month=None,
    confirmed_hire_fte=0.0,
    confirmed_apply_start_idx=0,
    seasonality_ramp_enabled=True,
    hiring_mode="reactive",
    planned_hires_visible_full=None,
):
    notice_months = lead_days_to_months(int(notice_days))
    monthly_turnover_rate = float(annual_turnover_rate) / 12.0

    freeze_set = set(int(m) for m in (freeze_months or []))
    req_post_month = int(req_post_month)

    # First index in the timeline where we can POST requisitions (req_post_month)
    req_start_idx = next(
        (idx for idx, dt in enumerate(dates_full) if int(dt.month) == req_post_month),
        0
    )
    
    # ✅ req-based blackout: months before req_post_month (wrap-safe)
    
    q = deque([0.0] * notice_months, maxlen=notice_months) if notice_months > 0 else None

    if hiring_mode == "planned":
        if planned_hires_visible_full is None or len(planned_hires_visible_full) != len(dates_full):
            raise ValueError("planned_hires_visible_full must be provided and match dates_full length when hiring_mode='planned'")

    staff = []
    prev = max(float(baseline_provider_fte), float(provider_min_floor))
    hire_applied = False

    for i, d in enumerate(dates_full):
        month_num = int(d.month)
        target = float(target_curve_full[i])

        # 1) Attrition (notice-lag)
        resignations = prev * monthly_turnover_rate
        if notice_months <= 0:
            separations = resignations
        else:
            q.append(resignations)
            separations = q.popleft()

        after_attrition = max(prev - separations, float(provider_min_floor))

        # 2) Hiring allowed? (freeze blocks POSTING reqs, not in-flight hires)
        if seasonality_ramp_enabled:
            req_i = i - int(lead_months)

            # Can't post reqs before the req-start point in the simulated timeline
            if req_i < req_start_idx or req_i < 0:
                hiring_allowed = False
            else:
                req_month_num = int(dates_full[req_i].month)
                hiring_allowed = (req_month_num not in freeze_set)
        else:
            hiring_allowed = True

        # 3) Hiring
        if hiring_allowed:
            needed = max(target - after_attrition, 0.0)

            if hiring_mode == "planned":
                planned_visible = float(planned_hires_visible_full[i])
                hires = min(planned_visible, needed)
                hires = clamp(hires, 0.0, float(max_hiring_up_after_visible))
            else:
                hires = clamp(needed, 0.0, float(max_hiring_up_after_visible))
        else:
            hires = 0.0

        planned = after_attrition + hires

        # 4) Confirmed hire (one-time)
        if (
            (not hire_applied)
            and (confirmed_hire_month is not None)
            and (int(month_num) == int(confirmed_hire_month))
            and (i >= int(confirmed_apply_start_idx))
        ):
            planned += float(confirmed_hire_fte)
            hire_applied = True

        # ✅ finalize month
        planned = max(planned, float(provider_min_floor))
        staff.append(planned)
        prev = planned

    return staff

# ============================================================
# COST / SWB-VISIT (FTE-based)
# ============================================================
def monthly_hours_from_fte(fte: float, fte_hours_per_week: float, days_in_month: int) -> float:
    return float(fte) * float(fte_hours_per_week) * (float(days_in_month) / 7.0)

def loaded_hourly_rate(base_hourly: float, benefits_load_pct: float, ot_sick_pct: float) -> float:
    return float(base_hourly) * (1.0 + float(benefits_load_pct)) * (1.0 + float(ot_sick_pct))

def compute_role_mix_ratios(model: StaffingModel, visits_per_day: float, hours_of_operation: float, fte_hours_per_week: float):
    f = model.calculate_fte_needed(
        visits_per_day=float(visits_per_day),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week),
    )
    prov = max(float(f.get("provider_fte", 0.0)), 0.25)
    return {
        "psr_per_provider": float(f.get("psr_fte", 0.0)) / prov,
        "ma_per_provider": float(f.get("ma_fte", 0.0)) / prov,
        "xrt_per_provider": float(f.get("xrt_fte", 0.0)) / prov,
    }

def compute_monthly_swb_per_visit_fte_based(
    provider_supply_curve_12,
    visits_per_day_curve_12,
    days_in_month_12,
    fte_hours_per_week,
    role_mix,
    hourly_rates,
    benefits_load_pct,
    ot_sick_pct,
    physician_supervision_hours_per_month=0.0,
    supervisor_hours_per_month=0.0,
):
    out_rows = []
    for i in range(12):
        prov_fte = float(provider_supply_curve_12[i])
        vpd = float(visits_per_day_curve_12[i])
        dim = int(days_in_month_12[i])
        month_visits = max(vpd * dim, 1.0)

        psr_fte = prov_fte * float(role_mix["psr_per_provider"])
        ma_fte  = prov_fte * float(role_mix["ma_per_provider"])
        rt_fte  = prov_fte * float(role_mix["xrt_per_provider"])

        apc_hours = monthly_hours_from_fte(prov_fte, fte_hours_per_week, dim)
        psr_hours = monthly_hours_from_fte(psr_fte, fte_hours_per_week, dim)
        ma_hours  = monthly_hours_from_fte(ma_fte,  fte_hours_per_week, dim)
        rt_hours  = monthly_hours_from_fte(rt_fte,  fte_hours_per_week, dim)

        apc_rate = loaded_hourly_rate(hourly_rates["apc"], benefits_load_pct, ot_sick_pct)
        psr_rate = loaded_hourly_rate(hourly_rates["psr"], benefits_load_pct, ot_sick_pct)
        ma_rate  = loaded_hourly_rate(hourly_rates["ma"],  benefits_load_pct, ot_sick_pct)
        rt_rate  = loaded_hourly_rate(hourly_rates["rt"],  benefits_load_pct, ot_sick_pct)

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

# ============================================================
# GAP + COST HELPERS
# ============================================================
def provider_day_gap(target_curve, supply_curve, days_in_month):
    gap_days = 0.0
    for t, s, dim in zip(target_curve, supply_curve, days_in_month):
        gap_days += max(float(t) - float(s), 0.0) * float(dim)
    return float(gap_days)

def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    cost = 0.0
    for dfte, dim in zip(delta_fte_curve, days_in_month):
        cost += float(dfte) * float(loaded_cost_per_provider_fte) * (float(dim) / 365.0)
    return float(cost)

def annual_swb_feasibility(provider_supply_12, visits_12, days_in_month_12, swb_df_12, target_swb_per_visit):
    total_swb = float(swb_df_12["SWB_$"].sum())
    total_visits = float(sum(float(v) * float(dim) for v, dim in zip(visits_12, days_in_month_12)))
    total_visits = max(total_visits, 1.0)
    annual_swb = total_swb / total_visits
    feasible = annual_swb <= float(target_swb_per_visit)
    return annual_swb, feasible

# ============================================================
# DEFAULTS + COMP PACKAGES
# ============================================================
RECOMMENDED = {
    # Hiring / pipeline
    "pipeline_total_days": 150,
    "notice_days": 90,
    "pipeline_use_breakdown": False,
    "days_to_sign": 90,
    "days_to_credential": 90,
    "onboard_train_days": 30,
    "coverage_buffer_days": 14,

    # Recruiting strategy
    "enable_seasonality_ramp": True,

    # Burnout
    "safe_visits_per_provider": 20,

    # Finance targets
    "target_swb_per_visit": 85.00,

    # ROI assumptions
    "loaded_cost_per_provider_fte": 260000,
    "net_revenue_per_visit": 140.0,
    "visits_lost_per_provider_day_gap": 18.0,

    # Compensation package
    "comp_package": "Expected (Recommended)",
    "benefits_load_pct": 0.30,
    "ot_sick_pct": 0.04,
    "physician_hr": 135.79,
    "apc_hr": 62.00,
    "ma_hr": 24.14,
    "psr_hr": 21.23,
    "rt_hr": 31.36,
    "supervisor_hr": 28.25,
    "physician_supervision_hours_per_month": 0.0,
    "supervisor_hours_per_month": 0.0,
}

COMP_PACKAGES = {
    "Lean": dict(benefits_load_pct=0.25, ot_sick_pct=0.03, physician_hr=125.0, apc_hr=58.0, ma_hr=22.0, psr_hr=19.0, rt_hr=29.0, supervisor_hr=26.0),
    "Expected (Recommended)": dict(benefits_load_pct=0.30, ot_sick_pct=0.04, physician_hr=135.79, apc_hr=62.0, ma_hr=24.14, psr_hr=21.23, rt_hr=31.36, supervisor_hr=28.25),
    "Conservative": dict(benefits_load_pct=0.35, ot_sick_pct=0.06, physician_hr=150.0, apc_hr=66.0, ma_hr=26.0, psr_hr=23.0, rt_hr=34.0, supervisor_hr=30.0),
}

def _ensure_state_defaults():
    for k, v in RECOMMENDED.items():
        st.session_state.setdefault(f"psm_{k}", v)

    # Confirmed hire controls
    st.session_state.setdefault("psm_has_confirmed_hire", False)
    st.session_state.setdefault("psm_confirmed_hire_month", 12)
    st.session_state.setdefault("psm_confirmed_hire_fte", 1.00)

    # Optional override toggles
    st.session_state.setdefault("psm_manual_rates", False)

_ensure_state_defaults()

def apply_recommended_defaults():
    for k, v in RECOMMENDED.items():
        st.session_state[f"psm_{k}"] = v
    st.session_state["psm_has_confirmed_hire"] = False
    st.session_state["psm_manual_rates"] = False

def apply_comp_package(package_name: str):
    if bool(st.session_state.get("psm_manual_rates", False)):
        return
    pkg = COMP_PACKAGES.get(package_name, COMP_PACKAGES["Expected (Recommended)"])
    st.session_state["psm_benefits_load_pct"] = float(pkg["benefits_load_pct"])
    st.session_state["psm_ot_sick_pct"] = float(pkg["ot_sick_pct"])
    st.session_state["psm_physician_hr"] = float(pkg["physician_hr"])
    st.session_state["psm_apc_hr"] = float(pkg["apc_hr"])
    st.session_state["psm_ma_hr"] = float(pkg["ma_hr"])
    st.session_state["psm_psr_hr"] = float(pkg["psr_hr"])
    st.session_state["psm_rt_hr"] = float(pkg["rt_hr"])
    st.session_state["psm_supervisor_hr"] = float(pkg["supervisor_hr"])

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Inputs")

    st.subheader("Clinic Demand")
    visits = st.number_input("Avg Visits/Day (annual avg)", min_value=1.0, value=float(st.session_state.get("psm_visits", 36.0)), step=1.0, key="psm_visits")
    hours_of_operation = st.number_input("Hours of Operation / Week", min_value=1.0, value=float(st.session_state.get("psm_hours", 70.0)), step=1.0, key="psm_hours")
    fte_hours_per_week = st.number_input("FTE Hours / Week", min_value=1.0, value=float(st.session_state.get("psm_fte_hours", 40.0)), step=1.0, key="psm_fte_hours")

    st.subheader("Coverage Safety")
    provider_min_floor = st.number_input("Provider Minimum Floor (FTE)", min_value=0.25, value=float(st.session_state.get("psm_floor", 1.0)), step=0.25, key="psm_floor")
    burnout_slider = st.slider("Burnout Protection Level", 0.0, 1.0, float(st.session_state.get("psm_burnout", 0.60)), 0.05, key="psm_burnout")

    st.subheader("Workforce Reality")
    provider_turnover = st.number_input("Provider Turnover % (annual)", value=float(st.session_state.get("psm_turnover_pct_ui", 24.0)), step=1.0, key="psm_turnover_pct_ui") / 100.0

    st.subheader("Seasonality")
    seasonal_start_month = st.selectbox(
        "Seasonal Peak Start",
        options=list(range(1, 13)),
        index=int(st.session_state.get("psm_seasonal_start", 12)) - 1,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
        key="psm_seasonal_start",
    )
    seasonal_end_month = st.selectbox(
        "Seasonal Peak End",
        options=list(range(1, 13)),
        index=int(st.session_state.get("psm_seasonal_end", 2)) - 1,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
        key="psm_seasonal_end",
    )
    seasonal_change_pct = st.number_input(
        "Seasonal Volume Change (%)",
        min_value=0.0,
        value=float(st.session_state.get("psm_seasonal_change_ui", 20.0)),
        step=5.0,
        key="psm_seasonal_change_ui",
    ) / 100.0

    st.divider()

    with st.expander("⚙️ Advanced Assumptions", expanded=False):
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("✅ Use Recommended Defaults", use_container_width=True):
                apply_recommended_defaults()
                st.rerun()
        with cols[1]:
            st.caption("Optional. Defaults are designed for first-run success.")

        with st.expander("Hiring Pipeline", expanded=False):
            pipeline_use_breakdown = st.checkbox(
                "Customize pipeline breakdown (optional)",
                value=bool(st.session_state["psm_pipeline_use_breakdown"]),
                key="psm_pipeline_use_breakdown",
            )

            if not pipeline_use_breakdown:
                pipeline_total_days = st.number_input(
                    "Total Time to Independent Provider (days)",
                    min_value=0,
                    value=int(st.session_state["psm_pipeline_total_days"]),
                    step=5,
                    key="psm_pipeline_total_days",
                )
                days_to_sign = int(st.session_state["psm_days_to_sign"])
                days_to_credential = int(st.session_state["psm_days_to_credential"])
                onboard_train_days = int(st.session_state["psm_onboard_train_days"])
                coverage_buffer_days = int(st.session_state["psm_coverage_buffer_days"])
                total_lead_days = int(pipeline_total_days)
            else:
                days_to_sign = st.number_input("Days to Sign", min_value=0, value=int(st.session_state["psm_days_to_sign"]), step=5, key="psm_days_to_sign")
                days_to_credential = st.number_input("Days to Credential", min_value=0, value=int(st.session_state["psm_days_to_credential"]), step=5, key="psm_days_to_credential")
                onboard_train_days = st.number_input("Days to Train", min_value=0, value=int(st.session_state["psm_onboard_train_days"]), step=5, key="psm_onboard_train_days")
                coverage_buffer_days = st.number_input("Planning Buffer Days", min_value=0, value=int(st.session_state["psm_coverage_buffer_days"]), step=1, key="psm_coverage_buffer_days")
                total_lead_days = int(days_to_sign + days_to_credential + onboard_train_days + coverage_buffer_days)

            notice_days = st.number_input(
                "Resignation Notice Period (days)",
                min_value=0, max_value=180,
                value=int(st.session_state["psm_notice_days"]),
                step=5,
                key="psm_notice_days",
            )

        with st.expander("Recruiting Strategy", expanded=False):
            enable_seasonality_ramp = st.checkbox(
                "Enable Seasonality Recruiting Ramp",
                value=bool(st.session_state["psm_enable_seasonality_ramp"]),
                key="psm_enable_seasonality_ramp",
                help="If ON: freeze + pipeline blackout blocks HIRING only. Attrition always continues.",
            )

        with st.expander("Burnout Assumptions", expanded=False):
            safe_visits_per_provider = st.number_input(
                "Safe Visits/Provider/Day",
                min_value=10, max_value=40,
                value=int(st.session_state["psm_safe_visits_per_provider"]),
                step=1,
                key="psm_safe_visits_per_provider",
            )

        with st.expander("Confirmed Hire (Optional)", expanded=False):
            has_confirmed_hire = st.checkbox(
                "I have a confirmed hire",
                value=bool(st.session_state["psm_has_confirmed_hire"]),
                key="psm_has_confirmed_hire",
            )
            if has_confirmed_hire:
                confirmed_hire_month = st.selectbox(
                    "Confirmed Hire Month (shown in display window)",
                    options=list(range(1, 13)),
                    index=int(st.session_state["psm_confirmed_hire_month"]) - 1,
                    format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
                    key="psm_confirmed_hire_month",
                )
                default_confirmed = st.session_state.get("suggested_confirmed_hire_fte")
                if default_confirmed is None:
                    default_confirmed = float(st.session_state["psm_confirmed_hire_fte"])
                confirmed_hire_fte = st.number_input(
                    "Confirmed Hire FTE",
                    min_value=0.0,
                    value=float(default_confirmed),
                    step=0.25,
                    key="psm_confirmed_hire_fte",
                )

        with st.expander("Finance Targets (Optional)", expanded=False):
            target_swb_per_visit = st.number_input(
                "Target SWB / Visit ($)",
                value=float(st.session_state["psm_target_swb_per_visit"]),
                step=1.00,
                key="psm_target_swb_per_visit",
            )
            with st.expander("ROI Assumptions (Optional)", expanded=False):
                loaded_cost_per_provider_fte = st.number_input(
                    "Loaded Cost per Provider FTE (annual)",
                    value=float(st.session_state["psm_loaded_cost_per_provider_fte"]),
                    step=5000.0,
                    key="psm_loaded_cost_per_provider_fte",
                )
                net_revenue_per_visit = st.number_input(
                    "Net Revenue per Visit",
                    value=float(st.session_state["psm_net_revenue_per_visit"]),
                    step=5.0,
                    key="psm_net_revenue_per_visit",
                )
                visits_lost_per_provider_day_gap = st.number_input(
                    "Visits Lost per 1.0 Provider-Day Gap",
                    value=float(st.session_state["psm_visits_lost_per_provider_day_gap"]),
                    step=1.0,
                    key="psm_visits_lost_per_provider_day_gap",
                )

        with st.expander("Compensation Package (Optional)", expanded=False):
            cols_pkg = st.columns([2, 1])
            with cols_pkg[0]:
                comp_package = st.selectbox(
                    "Compensation Package",
                    options=list(COMP_PACKAGES.keys()),
                    key="psm_comp_package",
                )
                if st.button("Apply Package", use_container_width=True):
                    apply_comp_package(st.session_state["psm_comp_package"])
                    st.rerun()
            with cols_pkg[1]:
                manual_rates = st.checkbox(
                    "Manually override rates",
                    key="psm_manual_rates",
                )

            benefits_load_pct = float(st.session_state["psm_benefits_load_pct"])
            ot_sick_pct = float(st.session_state["psm_ot_sick_pct"])
            physician_hr = float(st.session_state["psm_physician_hr"])
            apc_hr = float(st.session_state["psm_apc_hr"])
            ma_hr = float(st.session_state["psm_ma_hr"])
            psr_hr = float(st.session_state["psm_psr_hr"])
            rt_hr = float(st.session_state["psm_rt_hr"])
            supervisor_hr = float(st.session_state["psm_supervisor_hr"])

            if manual_rates:
                benefits_load_pct = st.number_input("Benefits Load (%)", value=float(benefits_load_pct * 100.0), step=1.0) / 100.0
                ot_sick_pct = st.number_input("OT + Sick/PTO (%)", value=float(ot_sick_pct * 100.0), step=0.5) / 100.0
                physician_hr = st.number_input("Physician (Supervision) $/hr", value=float(physician_hr), step=1.0)
                apc_hr = st.number_input("APC $/hr", value=float(apc_hr), step=1.0)
                ma_hr = st.number_input("MA $/hr", value=float(ma_hr), step=0.5)
                psr_hr = st.number_input("PSR $/hr", value=float(psr_hr), step=0.5)
                rt_hr = st.number_input("RT $/hr", value=float(rt_hr), step=0.5)
                supervisor_hr = st.number_input("Supervisor $/hr", value=float(supervisor_hr), step=0.5)

                st.session_state["psm_benefits_load_pct"] = float(benefits_load_pct)
                st.session_state["psm_ot_sick_pct"] = float(ot_sick_pct)
                st.session_state["psm_physician_hr"] = float(physician_hr)
                st.session_state["psm_apc_hr"] = float(apc_hr)
                st.session_state["psm_ma_hr"] = float(ma_hr)
                st.session_state["psm_psr_hr"] = float(psr_hr)
                st.session_state["psm_rt_hr"] = float(rt_hr)
                st.session_state["psm_supervisor_hr"] = float(supervisor_hr)

            with st.expander("Optional: fixed monthly hours", expanded=False):
                physician_supervision_hours_per_month = st.number_input(
                    "Physician supervision hours/month",
                    value=float(st.session_state["psm_physician_supervision_hours_per_month"]),
                    step=1.0,
                    key="psm_physician_supervision_hours_per_month",
                )
                supervisor_hours_per_month = st.number_input(
                    "Supervisor hours/month",
                    value=float(st.session_state["psm_supervisor_hours_per_month"]),
                    step=1.0,
                    key="psm_supervisor_hours_per_month",
                )

    # Pull defaults if advanced expander wasn’t opened
    total_lead_days = int(st.session_state["psm_pipeline_total_days"])
    notice_days = int(st.session_state["psm_notice_days"])
    enable_seasonality_ramp = bool(st.session_state["psm_enable_seasonality_ramp"])
    safe_visits_per_provider = int(st.session_state["psm_safe_visits_per_provider"])

    target_swb_per_visit = float(st.session_state["psm_target_swb_per_visit"])
    loaded_cost_per_provider_fte = float(st.session_state["psm_loaded_cost_per_provider_fte"])
    net_revenue_per_visit = float(st.session_state["psm_net_revenue_per_visit"])
    visits_lost_per_provider_day_gap = float(st.session_state["psm_visits_lost_per_provider_day_gap"])

    benefits_load_pct = float(st.session_state["psm_benefits_load_pct"])
    ot_sick_pct = float(st.session_state["psm_ot_sick_pct"])
    physician_hr = float(st.session_state["psm_physician_hr"])
    apc_hr = float(st.session_state["psm_apc_hr"])
    ma_hr = float(st.session_state["psm_ma_hr"])
    psr_hr = float(st.session_state["psm_psr_hr"])
    rt_hr = float(st.session_state["psm_rt_hr"])
    supervisor_hr = float(st.session_state["psm_supervisor_hr"])
    physician_supervision_hours_per_month = float(st.session_state["psm_physician_supervision_hours_per_month"])
    supervisor_hours_per_month = float(st.session_state["psm_supervisor_hours_per_month"])

    st.divider()
    run_model = st.button("▶️ Run PSM", use_container_width=True)

# ============================================================
# RUN MODEL
# ============================================================
if run_model:
    sim_months = 36
    start_date = datetime(today.year, 1, 1)
    dates_full = pd.date_range(start=start_date, periods=sim_months, freq="MS")
    days_in_month_full = [pd.Period(d, "M").days_in_month for d in dates_full]

    # Baseline provider FTE derived from annual average visits input
    fte_result = model.calculate_fte_needed(
        visits_per_day=float(visits),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week),
    )
    baseline_provider_fte = max(float(fte_result["provider_fte"]), float(provider_min_floor))

    forecast_visits_full = compute_seasonality_forecast_multiyear(
        dates=dates_full,
        baseline_visits=visits,
        seasonal_start_month=seasonal_start_month,
        seasonal_end_month=seasonal_end_month,
        seasonal_change_pct=seasonal_change_pct,
        summer_downcycle=True,
        normalize=True,
    )

    provider_base_demand_full = visits_to_provider_demand(
        model=model,
        visits_by_month=forecast_visits_full,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
        provider_min_floor=provider_min_floor,
    )

    protective_full = burnout_protective_staffing_curve(
        visits_by_month=forecast_visits_full,
        base_demand_fte=provider_base_demand_full,
        provider_min_floor=provider_min_floor,
        burnout_slider=burnout_slider,
        safe_visits_per_provider_per_day=safe_visits_per_provider,
    )

    # Strategy built from typical 12-month protective curve
    dates_template_12 = pd.date_range(start=datetime(2000, 1, 1), periods=12, freq="MS")
    protective_typical_12 = typical_12_month_curve(dates_full, protective_full)

    strategy = auto_freeze_strategy_v3_from_typical(
        dates_template_12=dates_template_12,
        protective_typical_12=protective_typical_12,
        seasonal_start_month=seasonal_start_month,
        seasonal_end_month=seasonal_end_month,
        pipeline_lead_days=int(total_lead_days),
        notice_days=int(notice_days),
        freeze_buffer_months=1,
    )

    req_post_month = strategy["req_post_month"]
    hire_visible_month = strategy["hire_visible_month"]
    independent_ready_month = strategy["independent_ready_month"]
    freeze_months = strategy["freeze_months"]
    recruiting_open_months = strategy["recruiting_open_months"]
    lead_months = strategy["lead_months"]

    # Derived visible ramp (anchor to PEAK in seasonal window)
    months_in_peak_window = max(len(strategy["seasonal_months"]), 1)
    peak_idxs = [(m - 1) for m in strategy["seasonal_months"]]
    target_peak_typical = max(float(protective_typical_12[i]) for i in peak_idxs)
    fte_gap_to_close = max(target_peak_typical - baseline_provider_fte, 0.0)
    derived_ramp_after_visible = min(fte_gap_to_close / float(months_in_peak_window), 1.25)

    planned_hires_visible_full = planned_hires_from_typical_target(
    dates_full=dates_full,
    target_typical_12=protective_typical_12,
    max_hiring_up_after_visible=derived_ramp_after_visible,
    req_post_month=req_post_month,
    hire_visible_month=hire_visible_month,
    freeze_months=(freeze_months if enable_seasonality_ramp else []),
    lead_months=lead_months,  # ✅ NEW
    baseline_provider_fte=baseline_provider_fte,
    seasonality_ramp_enabled=enable_seasonality_ramp,
)

    # Choose a 12-month display window from stabilized portion (months 13–24)
    stabilized_start = 12
    stabilized_end = min(24, sim_months - 12) if sim_months > 24 else max(12, sim_months - 12)
    anchor_idx = stabilized_start
    for i in range(stabilized_start, stabilized_end):
        if int(dates_full[i].month) == int(req_post_month):
            anchor_idx = i
            break

    display_idx = list(range(anchor_idx, anchor_idx + 12))
    if display_idx[-1] >= len(dates_full):
        display_idx = list(range(len(dates_full) - 12, len(dates_full)))

    confirmed_apply_start_idx = int(display_idx[0])

    # Confirmed hire inputs
    if st.session_state.get("psm_has_confirmed_hire", False):
        confirmed_hire_month = int(st.session_state["psm_confirmed_hire_month"])
        confirmed_hire_fte = float(st.session_state["psm_confirmed_hire_fte"])
    else:
        confirmed_hire_month = None
        confirmed_hire_fte = 0.0

    # Supply sims (ALWAYS run)
    supply_rec_full = simulate_supply_multiyear_best_case(
        dates_full=dates_full,
        baseline_provider_fte=baseline_provider_fte,
        target_curve_full=protective_full,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        freeze_months=(freeze_months if enable_seasonality_ramp else []),
        max_hiring_up_after_visible=derived_ramp_after_visible,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        confirmed_apply_start_idx=confirmed_apply_start_idx,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        lead_months=lead_months,
        hiring_mode="planned",
        planned_hires_visible_full=planned_hires_visible_full,
    )

    supply_lean_full = simulate_supply_multiyear_best_case(
        dates_full=dates_full,
        baseline_provider_fte=baseline_provider_fte,
        target_curve_full=provider_base_demand_full,
        provider_min_floor=provider_min_floor,
        annual_turnover_rate=provider_turnover,
        notice_days=notice_days,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        freeze_months=(freeze_months if enable_seasonality_ramp else []),
        max_hiring_up_after_visible=derived_ramp_after_visible,
        confirmed_hire_month=confirmed_hire_month,
        confirmed_hire_fte=confirmed_hire_fte,
        confirmed_apply_start_idx=confirmed_apply_start_idx,
        seasonality_ramp_enabled=enable_seasonality_ramp,
        lead_months=lead_months,
        hiring_mode="reactive",
    )

    # 12-mo view
    dates_12 = [dates_full[i] for i in display_idx]
    days_in_month_12 = [days_in_month_full[i] for i in display_idx]
    month_labels_12 = [d.strftime("%b") for d in dates_12]

    visits_12 = [float(forecast_visits_full[i]) for i in display_idx]
    demand_lean_12 = [float(provider_base_demand_full[i]) for i in display_idx]
    target_prot_12 = [float(protective_full[i]) for i in display_idx]
    supply_lean_12 = [float(supply_lean_full[i]) for i in display_idx]
    supply_rec_12 = [float(supply_rec_full[i]) for i in display_idx]

    # Suggested confirmed hire FTE default: typical month-to-month lift at hire-visible month
    hv_m = int(hire_visible_month)
    prev_m = shift_month(hv_m, -1)
    hv_delta = float(protective_typical_12[hv_m - 1]) - float(protective_typical_12[prev_m - 1])
    suggested_confirmed = max(hv_delta, 0.0)
    if suggested_confirmed <= 0:
        suggested_confirmed = 1.0
    st.session_state["suggested_confirmed_hire_fte"] = float(suggested_confirmed)

    burnout_gap_fte_12 = [max(float(t) - float(s), 0.0) for t, s in zip(target_prot_12, supply_rec_12)]
    months_exposed_12 = int(sum(1 for g in burnout_gap_fte_12 if g > 0))

    role_mix = compute_role_mix_ratios(
        model=model,
        visits_per_day=visits,
        hours_of_operation=hours_of_operation,
        fte_hours_per_week=fte_hours_per_week,
    )
    hourly_rates = {
        "physician": physician_hr,
        "apc": apc_hr,
        "ma": ma_hr,
        "psr": psr_hr,
        "rt": rt_hr,
        "supervisor": supervisor_hr,
    }

    swb_df = compute_monthly_swb_per_visit_fte_based(
        provider_supply_curve_12=supply_rec_12,
        visits_per_day_curve_12=visits_12,
        days_in_month_12=days_in_month_12,
        fte_hours_per_week=fte_hours_per_week,
        role_mix=role_mix,
        hourly_rates=hourly_rates,
        benefits_load_pct=benefits_load_pct,
        ot_sick_pct=ot_sick_pct,
        physician_supervision_hours_per_month=physician_supervision_hours_per_month,
        supervisor_hours_per_month=supervisor_hours_per_month,
    )

    avg_swb_per_visit_modeled = float(swb_df["SWB_per_Visit_$"].mean())
    annual_swb_per_visit_modeled, annual_feasible = annual_swb_feasibility(
        provider_supply_12=supply_rec_12,
        visits_12=visits_12,
        days_in_month_12=days_in_month_12,
        swb_df_12=swb_df,
        target_swb_per_visit=target_swb_per_visit,
    )
    labor_factor = (float(target_swb_per_visit) / float(annual_swb_per_visit_modeled)) if annual_swb_per_visit_modeled > 0 else np.nan

    st.session_state["model_ran"] = True
    st.session_state["results"] = dict(
        dates=dates_12,
        month_labels=month_labels_12,
        days_in_month=days_in_month_12,

        forecast_visits_by_month=visits_12,
        provider_base_demand=demand_lean_12,
        protective_curve=target_prot_12,
        realistic_supply_lean=supply_lean_12,
        realistic_supply_recommended=supply_rec_12,

        burnout_gap_fte=burnout_gap_fte_12,
        months_exposed=months_exposed_12,

        pipeline_lead_days=int(total_lead_days),
        lead_months=lead_months,
        req_post_month=req_post_month,
        hire_visible_month=hire_visible_month,
        independent_ready_month=independent_ready_month,
        freeze_months=freeze_months,
        recruiting_open_months=recruiting_open_months,
        seasonal_months=strategy["seasonal_months"],
        months_in_peak_window=months_in_peak_window,
        fte_gap_to_close=fte_gap_to_close,
        derived_ramp_after_visible=derived_ramp_after_visible,

        baseline_provider_fte=baseline_provider_fte,

        target_swb_per_visit=float(target_swb_per_visit),
        avg_swb_per_visit_modeled=float(avg_swb_per_visit_modeled),
        annual_swb_per_visit_modeled=float(annual_swb_per_visit_modeled),
        swb_feasible_monthly_avg=(avg_swb_per_visit_modeled <= float(target_swb_per_visit)),
        swb_feasible_annual=bool(annual_feasible),
        labor_factor=float(labor_factor) if np.isfinite(labor_factor) else None,
        swb_df=swb_df,

        confirmed_hire_month=(int(confirmed_hire_month) if confirmed_hire_month else 0),
        confirmed_hire_fte=float(confirmed_hire_fte),

        sim_months=int(sim_months),
    )

# ============================================================
# STOP IF NOT RUN
# ============================================================
if not st.session_state.get("model_ran"):
    st.info("Enter your core inputs in the sidebar and click **Run PSM**.")
    st.stop()

R = st.session_state["results"]

# ============================================================
# SECTION 1 — OPERATIONS
# ============================================================
st.markdown("---")
st.header("1) Operations — Seasonality Staffing Requirements")
st.caption("Visits/day forecast → FTE needed by month (seasonality behavior).")

monthly_rows = []
for month_label, d, v in zip(R["month_labels"], R["dates"], R["forecast_visits_by_month"]):
    fte_staff = model.calculate_fte_needed(
        visits_per_day=float(v),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week),
    )
    monthly_rows.append({
        "Month": month_label,
        "Visits/Day (Forecast)": round(float(v), 1),
        "Provider FTE (Lean)": round(float(fte_staff["provider_fte"]), 2),
        "PSR FTE": round(float(fte_staff["psr_fte"]), 2),
        "MA FTE": round(float(fte_staff["ma_fte"]), 2),
        "XRT FTE": round(float(fte_staff["xrt_fte"]), 2),
        "Total FTE": round(float(fte_staff["total_fte"]), 2),
    })

ops_df = pd.DataFrame(monthly_rows)
st.dataframe(ops_df, hide_index=True, use_container_width=True)

st.success(
    "**Operations Summary:** This is the seasonality-adjusted demand signal. "
    "Lean demand is minimum coverage; the protective target adds a burnout buffer."
)

# ============================================================
# SECTION 2 — REALITY
# ============================================================
st.markdown("---")
st.header("2) Reality — Pipeline-Constrained Supply + Burnout Exposure")
st.caption(
    "Targets vs realistic supply when hiring is constrained by lead time + freezes. "
    "**Attrition continues every month** (including freeze months). **No forced ramp-down**: down = attrition only."
)

freeze_label = month_range_label(R.get("freeze_months", []))
recruit_label = month_range_label(R.get("recruiting_open_months", []))
req_post_label = datetime(2000, int(R["req_post_month"]), 1).strftime("%b")
hire_visible_label = datetime(2000, int(R["hire_visible_month"]), 1).strftime("%b")
independent_label = datetime(2000, int(R["independent_ready_month"]), 1).strftime("%b")

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

peak_gap = float(max(R["burnout_gap_fte"])) if R["burnout_gap_fte"] else 0.0
avg_gap = float(np.mean(R["burnout_gap_fte"])) if R["burnout_gap_fte"] else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Peak Burnout Gap (FTE)", f"{peak_gap:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
m3.metric("Months Exposed", f"{R['months_exposed']}/12")

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

freeze_set = set(int(m) for m in (R.get("freeze_months", []) or []))
for d in R["dates"]:
    if int(d.month) in freeze_set:
        ax1.axvspan(d, d + timedelta(days=27), alpha=0.12, color=BRAND_GOLD, linewidth=0)

ax1.plot(R["dates"], R["provider_base_demand"], linestyle=":", linewidth=1.2, color=GRAY, label="Lean Target (Demand)")
ax1.plot(R["dates"], R["protective_curve"], linewidth=2.0, color=BRAND_GOLD, marker="o", markersize=4,
         label="Recommended Target (Protective)")
ax1.plot(R["dates"], R["realistic_supply_recommended"], linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=4,
         label="Realistic Supply (Best-Case, Constrained)")

ax1.fill_between(
    R["dates"],
    np.array(R["realistic_supply_recommended"], dtype=float),
    np.array(R["protective_curve"], dtype=float),
    where=np.array(R["protective_curve"], dtype=float) > np.array(R["realistic_supply_recommended"], dtype=float),
    color=BRAND_GOLD,
    alpha=0.12,
    label="Burnout Exposure Zone",
)

confirmed_month = int(R.get("confirmed_hire_month", 0) or 0)
confirmed_fte = float(R.get("confirmed_hire_fte", 0.0) or 0.0)
if confirmed_month in range(1, 13) and confirmed_fte > 0:
    confirmed_date = None
    for d in R["dates"]:
        if int(d.month) == confirmed_month:
            confirmed_date = d
            break
    if confirmed_date is not None:
        ax1.axvline(confirmed_date, color=BRAND_BLACK, linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)

ax1.set_title("Reality — Targets vs Pipeline-Constrained Supply", fontsize=16, fontweight="bold", pad=16, color=BRAND_BLACK)
ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax1.set_xticks(R["dates"])
ax1.set_xticklabels(R["month_labels"], fontsize=11, color=BRAND_BLACK)
ax1.tick_params(axis="y", labelsize=11, colors=BRAND_BLACK)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

ax2 = ax1.twinx()
ax2.plot(R["dates"], R["forecast_visits_by_month"], linestyle="-.", linewidth=1.4, color=MID_GRAY, label="Forecast Visits/Day")
ax2.set_ylabel("Visits / Day", fontsize=12, fontweight="bold", color=BRAND_BLACK)
ax2.tick_params(axis="y", labelsize=11, colors=BRAND_BLACK)

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
    f"**Reality Summary:** This 12-month view is taken from a continuous simulation. "
    f"To be peak-ready by **{independent_label}**, requisitions must post by **{req_post_label}** so hires are visible by **{hire_visible_label}**. "
    f"Best-case visible hiring ramp cap: **{R['derived_ramp_after_visible']:.2f} FTE/month**."
)

# ============================================================
# SECTION 3 — FINANCE
# ============================================================
st.markdown("---")
st.header("3) Finance — ROI Investment Case")
st.caption("Quantifies investment to close the gap and the economic value of reducing provider-day shortages.")

delta_fte_curve = [max(float(t) - float(R["baseline_provider_fte"]), 0.0) for t in R["protective_curve"]]
annual_investment = annualize_monthly_fte_cost(delta_fte_curve, R["days_in_month"], loaded_cost_per_provider_fte)

gap_days = provider_day_gap(R["protective_curve"], R["realistic_supply_recommended"], R["days_in_month"])
est_visits_lost = gap_days * float(visits_lost_per_provider_day_gap)
est_revenue_lost = est_visits_lost * float(net_revenue_per_visit)

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
# SECTION 3B — VVI FEASIBILITY (SWB/Visit) — ANNUAL CONSTRAINT
# ============================================================
st.markdown("---")
st.header("3B) VVI Feasibility — SWB/Visit (FTE-based)")
st.caption("Feasibility is judged on the year (annual SWB/Visit vs target).")

swb_df = R["swb_df"].copy()
swb_df.insert(0, "Month", R["month_labels"])

colX, colY, colZ, colW = st.columns(4)
colX.metric("Target SWB/Visit", f"${R['target_swb_per_visit']:.2f}")
colY.metric("Modeled SWB/Visit (avg month)", f"${R['avg_swb_per_visit_modeled']:.2f}")
colZ.metric("Modeled SWB/Visit (annual)", f"${R['annual_swb_per_visit_modeled']:.2f}")
colW.metric("Annual Feasible?", "YES" if R["swb_feasible_annual"] else "NO")

st.dataframe(
    swb_df[["Month", "Provider_FTE_Supply", "Visits", "SWB_$", "SWB_per_Visit_$"]],
    hide_index=True,
    use_container_width=True,
)

fig2, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(R["dates"], swb_df["SWB_per_Visit_$"].astype(float).values, linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=3, label="Modeled SWB/Visit (monthly)")
ax.axhline(R["target_swb_per_visit"], linewidth=2.0, linestyle="--", color=BRAND_GOLD, label="Target SWB/Visit")
ax.set_title("SWB/Visit — Monthly Diagnostic (Annual is the constraint)", fontsize=14, fontweight="bold")
ax.set_ylabel("$/Visit")
ax.set_xticks(R["dates"])
ax.set_xticklabels(R["month_labels"])
ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
st.pyplot(fig2)

months_above = int((swb_df["SWB_per_Visit_$"].astype(float) > float(R["target_swb_per_visit"])).sum())
lf = R.get("labor_factor", None)
lf_str = f"{lf:.2f}" if isinstance(lf, float) else "—"
st.info(
    f"- Months above target SWB/Visit: **{months_above}/12** (expected if volumes fall faster than staffing).\n"
    f"- **Labor Factor (LF)** = Target SWB/Visit ÷ **Annual** Modeled SWB/Visit = **{lf_str}** "
    "(LF > 1.0 favorable; LF < 1.0 means labor runs heavier than target on the year)."
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
    g2 = float(g) * (1.0 - float(buffer_pct) / 100.0)
    g2 = max(g2 - float(float_pool_fte), 0.0)
    g2 = max(g2 - float(fractional_fte), 0.0)
    effective_gap_curve.append(g2)

remaining_gap_days = provider_day_gap([0] * 12, effective_gap_curve, R["days_in_month"])
reduced_gap_days = max(gap_days - remaining_gap_days, 0.0)

est_visits_saved = reduced_gap_days * float(visits_lost_per_provider_day_gap)
est_revenue_saved = est_visits_saved * float(net_revenue_per_visit)

hybrid_investment = annual_investment * float(hybrid_slider)

sA, sB, sC = st.columns(3)
sA.metric("Provider-Day Gap Reduced", f"{reduced_gap_days:,.0f}")
sB.metric("Est. Revenue Saved", f"${est_revenue_saved:,.0f}")
sC.metric("Hybrid Investment Share", f"${hybrid_investment:,.0f}")

st.success(
    "**Strategy Summary:** Flex levers reduce burnout exposure faster than permanent hiring alone. "
    "Use hybrid to bridge peak months while the permanent pipeline catches up."
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
    st.metric("Months Exposed", f"{R['months_exposed']}/12")
with col2:
    st.write(
        f"**To be flu-ready by {independent_label}:**\n"
        f"- Post requisitions by **{req_post_label}** (lead time: {R['pipeline_lead_days']} days ≈ {R['lead_months']} months)\n"
        f"- Hiring becomes visible by **{hire_visible_label}**\n"
        f"- Best-case visible hiring ramp cap: **{R['derived_ramp_after_visible']:.2f} FTE/month**\n\n"
        f"**Financial feasibility (annual):**\n"
        f"- Annual SWB/Visit: **${R['annual_swb_per_visit_modeled']:.2f}** vs Target **${R['target_swb_per_visit']:.2f}** → **{'PASS' if R['swb_feasible_annual'] else 'FAIL'}**\n"
        f"- ROI: **{roi:,.2f}x**\n\n"
        f"**Key realism rule:**\n"
        f"- No forced ramp-down. Staffing reduces only via turnover (with notice lag). Freeze blocks hiring only.\n\n"
        f"**With strategy levers applied:**\n"
        f"- Provider-day gap reduced: **{reduced_gap_days:,.0f} days**\n"
        f"- Estimated revenue saved: **${est_revenue_saved:,.0f}**\n"
    )

st.success(
    "✅ **Decision Summary:** This model produces best-case staffing recommendations that track seasonality demand "
    "as closely as possible given pipeline constraints, while judging SWB/Visit feasibility on the year (not each month)."
)

# ============================================================
# SECTION 6 — IN-DEPTH EXECUTIVE SUMMARY (Narrative)
# ============================================================
st.markdown("---")
st.header("6) In-Depth Executive Summary")
st.caption("What the model is doing, what it found, and how to act on it.")

flu_range = month_range_label(R["flu_months"])
freeze_range = freeze_label
recruit_range = recruit_label

summary_md = f"""
### What this model is solving
This Predictive Staffing Model (PSM) answers:

**“Given seasonal volume swings and a constrained hiring pipeline, what staffing plan minimizes burnout exposure — while staying financially feasible on SWB/Visit?”**

To keep the output accurate, the model does **not** treat Jan–Dec as a boundary where staffing “resets.”
It runs a **{R['sim_months']}-month continuous simulation** and then displays a representative 12-month window.

---

### How Operations works (demand signal)
- A repeatable seasonality curve + flu uplift over **{flu_range}**
- Normalized so the average over the simulation still matches your baseline
- Converted into:
  - Lean demand (minimum)
  - Protective target (lean + burnout buffer based on variability, spikes, and safe visits/provider)

---

### How Reality works (supply)
Supply is “best-case realistic,” meaning it assumes you will:
- Hire toward the target whenever hiring is allowed, **up to a ramp cap**
- Continue to lose staff through turnover every month (with notice lag)
- **Never force a ramp-down** (no firing to target). Downward movement is **attrition-only**.
- Freeze months block **hiring only**, which is exactly why supply can drift down into slower months.

This is why December naturally carries into January—there is no year-end reset.

---

### Finance & VVI feasibility (Policy A — SWB/Visit, annual constraint)
Monthly SWB/Visit will vary. The correct feasibility test is the **annual** number:

- Annual SWB/Visit: **${R['annual_swb_per_visit_modeled']:.2f}**
- Target SWB/Visit: **${R['target_swb_per_visit']:.2f}**
- Result: **{"PASS" if R["swb_feasible_annual"] else "FAIL"}**

Burnout protection is the executive dial:
- More protection → less exposure, higher labor per visit
- Less protection → more exposure, lower labor per visit
"""
st.markdown(summary_md)
