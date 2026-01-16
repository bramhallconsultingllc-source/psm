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

st.info(
    "⚠️ **All staffing outputs round UP to the nearest 0.25 FTE.** "
    "This is intentional to prevent under-staffing."
)

model = StaffingModel()

# ============================================================
# STABLE TODAY
# ============================================================
if "today" not in st.session_state:
    st.session_state["today"] = datetime.today()
today = st.session_state["today"]

# ============================================================
# SESSION STATE
# ============================================================
for k in ["model_ran", "results", "suggested_confirmed_hire_fte"]:
    if k not in st.session_state:
        st.session_state[k] = None

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

def round_up_quarter(x: float) -> float:
    return math.ceil(float(x) * 4.0) / 4.0

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

def base_seasonality_multiplier(month: int):
    if int(month) in [12, 1, 2]:
        return 1.20
    if int(month) in [6, 7, 8]:
        return 0.80
    return 1.00

def compute_seasonality_forecast_multiyear(dates, baseline_visits, flu_months, flu_uplift_pct):
    """
    Seasonality uses month-of-year only (repeatable cycle).
    Flu uplift applied to any month in flu_months.
    Normalize so mean(visits) equals baseline across full horizon.
    """
    flu_set = set(int(m) for m in flu_months)
    raw = []
    for d in dates:
        mult = base_seasonality_multiplier(d.month)
        if int(d.month) in flu_set:
            mult *= (1.0 + float(flu_uplift_pct))
        raw.append(float(baseline_visits) * mult)

    avg_raw = float(np.mean(raw)) if len(raw) else float(baseline_visits)
    if avg_raw <= 0:
        return [float(baseline_visits) for _ in raw]
    return [v * (float(baseline_visits) / avg_raw) for v in raw]

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
    This is *not* hiring logic; it's the recommended staffing target to avoid burnout exposure.
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

# ============================================================
# AUTO-FREEZE v3 (built from typical seasonal curve)
# ============================================================
def auto_freeze_strategy_v3_from_typical(
    dates_template_12,            # Jan..Dec template dates
    protective_typical_12,        # 12 values
    flu_start_month,
    flu_end_month,
    pipeline_lead_days,
    notice_days,
    freeze_buffer_months=1,
):
    lead_months = lead_days_to_months(pipeline_lead_days)
    notice_months = lead_days_to_months(notice_days)

    independent_ready_month = int(flu_start_month)
    req_post_month = shift_month(independent_ready_month, -lead_months)
    hire_visible_month = shift_month(req_post_month, lead_months)

    trough_idx = int(np.argmin(np.array(protective_typical_12, dtype=float)))
    trough_month = int(dates_template_12[trough_idx].month)

    decline_months = months_between(shift_month(int(flu_end_month), 1), trough_month)

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
        flu_months=months_between(int(flu_start_month), int(flu_end_month)),
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
    confirmed_hire_month=None,
    confirmed_hire_fte=0.0,
    confirmed_apply_start_idx=0,
    seasonality_ramp_enabled=True,
):
    """
    Best-case realistic supply simulation:

    - Continuous across multiple years (no reset).
    - Attrition always reduces staffing (including during freeze months).
    - NO artificial ramp-down: we do not "fire to target". Downward movement is attrition-only.
    - Hiring is the only upward force and is constrained by:
        * freeze months (blocks hiring only),
        * pipeline visibility blackout (blocks hiring only),
        * a max visible hiring ramp (FTE/month) once hiring is allowed.
    - Hiring policy (best-case): when hiring is allowed, add enough FTE to move as close as possible to target,
      up to the monthly ramp cap.
    - Notice-lag: resignations now, separations occur notice_months later.

    Confirmed hire:
    - Applies once, but NOT earlier than confirmed_apply_start_idx. This ensures a "December hire" intended
      for the displayed/stabilized year actually appears in that year’s December.
    """
    notice_months = lead_days_to_months(int(notice_days))
    monthly_turnover_rate = float(annual_turnover_rate) / 12.0

    freeze_set = set(int(m) for m in (freeze_months or []))
    req_post_month = int(req_post_month)
    hire_visible_month = int(hire_visible_month)

    # Months where hiring is "blackout" (no visible hires yet)
    blackout_months = set(months_between(req_post_month, shift_month(hire_visible_month, -1)))

    # Attrition lag queue
    if notice_months <= 0:
        q = None
    else:
        q = deque([0.0] * notice_months, maxlen=notice_months)

    staff = []
    prev = max(float(baseline_provider_fte), float(provider_min_floor))
    hire_applied = False

    for i, d in enumerate(dates_full):
        month_num = int(d.month)
        target = float(target_curve_full[i])

        # 1) Schedule attrition (notice-lag)
        resignations = prev * monthly_turnover_rate
        if notice_months <= 0:
            separations = resignations
        else:
            q.append(resignations)
            separations = q.popleft()

        after_attrition = max(prev - separations, float(provider_min_floor))

        # 2) Hiring allowed?
        if seasonality_ramp_enabled:
            in_freeze = month_num in freeze_set
            in_blackout = month_num in blackout_months
            hiring_allowed = (not in_freeze) and (not in_blackout)
        else:
            hiring_allowed = True

        # 3) Best-case hiring: fill toward target (up to cap)
        if hiring_allowed:
            needed = max(target - after_attrition, 0.0)
            hires = clamp(needed, 0.0, float(max_hiring_up_after_visible))
        else:
            hires = 0.0

        planned = after_attrition + hires

        # 4) Confirmed hire (one-time), applied in/after the display anchor region
        if (
            (not hire_applied)
            and (confirmed_hire_month is not None)
            and (int(month_num) == int(confirmed_hire_month))
            and (i >= int(confirmed_apply_start_idx))
        ):
            planned += float(confirmed_hire_fte)
            hire_applied = True

        planned = max(planned, float(provider_min_floor))
        staff.append(planned)
        prev = planned

    return staff

# ============================================================
# COST / SWB-VISIT (Policy A: FTE-based)
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
    """Annual feasibility: total SWB / total visits."""
    total_swb = float(swb_df_12["SWB_$"].sum())
    total_visits = float(sum(float(v) * float(dim) for v, dim in zip(visits_12, days_in_month_12)))
    total_visits = max(total_visits, 1.0)
    annual_swb = total_swb / total_visits
    feasible = annual_swb <= float(target_swb_per_visit)
    return annual_swb, feasible

# ============================================================
# MONTE CARLO (Near-certain)
# ============================================================
def _lognormal_noise(mean=1.0, cv=0.10, size=1, rng=None):
    """
    Multiplicative noise with target coefficient of variation (approx).
    If X~LogNormal(mu, sigma), then CV = sqrt(exp(sigma^2)-1)
    """
    rng = rng or np.random.default_rng()
    if cv <= 0:
        return np.ones(size) * mean
    sigma2 = math.log(1.0 + cv**2)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=sigma, size=size)

def monte_carlo_quantiles(
    dates_full,
    days_in_month_full,
    baseline_visits,
    flu_months,
    flu_uplift_pct,
    hours_of_operation,
    fte_hours_per_week,
    provider_min_floor,
    burnout_slider,
    safe_visits_per_provider,
    total_lead_days,
    notice_days,
    provider_turnover,
    enable_seasonality_ramp,
    max_hiring_up_after_visible,
    strategy_template_12,
    role_mix,
    hourly_rates,
    benefits_load_pct,
    ot_sick_pct,
    physician_supervision_hours_per_month,
    supervisor_hours_per_month,
    confirmed_hire_month,
    confirmed_hire_fte,
    confirmed_apply_start_idx,
    display_idx,
    confidence_level,
    mc_runs,
    visits_cv,
    turnover_var,
    pipeline_var_days,
    rng_seed=7,
):
    """
    Returns quantile curves (12 months) for:
      - demand_lean
      - target_protective
      - supply_recommended
      - visits
      - swb_per_visit (monthly)
    For near-certain view:
      - Use protective target at q=confidence (higher demand)
      - Use supply at q=(1-confidence) (lower supply)
    """
    rng = np.random.default_rng(rng_seed)

    runs_visits, runs_demand, runs_prot, runs_supply, runs_swb = [], [], [], [], []

    for _ in range(int(mc_runs)):
        v_noise = _lognormal_noise(mean=1.0, cv=float(visits_cv), size=len(dates_full), rng=rng)
        turnover_draw = float(provider_turnover) * (1.0 + rng.uniform(-float(turnover_var), float(turnover_var)))
        turnover_draw = max(turnover_draw, 0.0)

        lead_days_draw = int(round(float(total_lead_days) + rng.uniform(-float(pipeline_var_days), float(pipeline_var_days))))
        lead_days_draw = max(0, lead_days_draw)

        visits_full = compute_seasonality_forecast_multiyear(
            dates=dates_full,
            baseline_visits=baseline_visits,
            flu_months=flu_months,
            flu_uplift_pct=flu_uplift_pct,
        )
        visits_full = (np.array(visits_full, dtype=float) * v_noise).tolist()

        demand_full = visits_to_provider_demand(
            model=model,
            visits_by_month=visits_full,
            hours_of_operation=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week,
            provider_min_floor=provider_min_floor,
        )
        prot_full = burnout_protective_staffing_curve(
            visits_by_month=visits_full,
            base_demand_fte=demand_full,
            provider_min_floor=provider_min_floor,
            burnout_slider=burnout_slider,
            safe_visits_per_provider_per_day=safe_visits_per_provider,
        )

        dates_template_12 = pd.date_range(start=datetime(2000, 1, 1), periods=12, freq="MS")
        prot_typical_12 = typical_12_month_curve(dates_full, prot_full)
        strategy = auto_freeze_strategy_v3_from_typical(
            dates_template_12=dates_template_12,
            protective_typical_12=prot_typical_12,
            flu_start_month=strategy_template_12["independent_ready_month"],
            flu_end_month=strategy_template_12["flu_months"][-1],
            pipeline_lead_days=lead_days_draw,
            notice_days=notice_days,
            freeze_buffer_months=1,
        )

        supply_full = simulate_supply_multiyear_best_case(
            dates_full=dates_full,
            baseline_provider_fte=strategy_template_12["baseline_provider_fte"],
            target_curve_full=prot_full,
            provider_min_floor=provider_min_floor,
            annual_turnover_rate=turnover_draw,
            notice_days=notice_days,
            req_post_month=strategy["req_post_month"],
            hire_visible_month=strategy["hire_visible_month"],
            freeze_months=(strategy["freeze_months"] if enable_seasonality_ramp else []),
            max_hiring_up_after_visible=max_hiring_up_after_visible,
            confirmed_hire_month=confirmed_hire_month,
            confirmed_hire_fte=confirmed_hire_fte,
            confirmed_apply_start_idx=confirmed_apply_start_idx,
            seasonality_ramp_enabled=enable_seasonality_ramp,
        )

        v12 = [float(visits_full[i]) for i in display_idx]
        d12 = [float(demand_full[i]) for i in display_idx]
        p12 = [float(prot_full[i]) for i in display_idx]
        s12 = [float(supply_full[i]) for i in display_idx]

        swb_df = compute_monthly_swb_per_visit_fte_based(
            provider_supply_curve_12=s12,
            visits_per_day_curve_12=v12,
            days_in_month_12=[days_in_month_full[i] for i in display_idx],
            fte_hours_per_week=fte_hours_per_week,
            role_mix=role_mix,
            hourly_rates=hourly_rates,
            benefits_load_pct=benefits_load_pct,
            ot_sick_pct=ot_sick_pct,
            physician_supervision_hours_per_month=physician_supervision_hours_per_month,
            supervisor_hours_per_month=supervisor_hours_per_month,
        )
        swb12 = swb_df["SWB_per_Visit_$"].astype(float).values.tolist()

        runs_visits.append(v12)
        runs_demand.append(d12)
        runs_prot.append(p12)
        runs_supply.append(s12)
        runs_swb.append(swb12)

    def qcurve(runs, q):
        arr = np.array(runs, dtype=float)
        return np.quantile(arr, q, axis=0).tolist()

    c = float(confidence_level)
    return {
        "visits_q50": qcurve(runs_visits, 0.50),
        "demand_qc": qcurve(runs_demand, c),
        "prot_qc": qcurve(runs_prot, c),
        "supply_qlo": qcurve(runs_supply, 1.0 - c),
        "swb_q50": qcurve(runs_swb, 0.50),
    }

# ============================================================
# RECOMMENDED DEFAULTS + PRESETS
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

    # Near-certain defaults
    "enable_probability": False,
    "confidence_level": 0.90,
    "sim_horizon_months": 36,
    "mc_runs": 1000,
    "visits_cv": 0.10,
    "turnover_var": 0.20,
    "pipeline_var_days": 15,

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
    """Ensure session_state has keys used by widgets (so the defaults button can set them)."""
    for k, v in RECOMMENDED.items():
        st.session_state.setdefault(f"psm_{k}", v)
    # Confirmed hire controls
    st.session_state.setdefault("psm_has_confirmed_hire", False)
    st.session_state.setdefault("psm_confirmed_hire_month", 12)
    st.session_state.setdefault("psm_confirmed_hire_fte", 1.00)
    # Optional override toggles
    st.session_state.setdefault("psm_manual_rates", False)

def apply_recommended_defaults():
    for k, v in RECOMMENDED.items():
        st.session_state[f"psm_{k}"] = v
    st.session_state["psm_has_confirmed_hire"] = False
    st.session_state["psm_manual_rates"] = False

def apply_comp_package(package_name: str):
    pkg = COMP_PACKAGES.get(package_name, COMP_PACKAGES["Expected (Recommended)"])

    # IMPORTANT: do NOT set st.session_state["psm_comp_package"] here
    # That key belongs to the selectbox widget.

    st.session_state["psm_benefits_load_pct"] = float(pkg["benefits_load_pct"])
    st.session_state["psm_ot_sick_pct"] = float(pkg["ot_sick_pct"])
    st.session_state["psm_physician_hr"] = float(pkg["physician_hr"])
    st.session_state["psm_apc_hr"] = float(pkg["apc_hr"])
    st.session_state["psm_ma_hr"] = float(pkg["ma_hr"])
    st.session_state["psm_psr_hr"] = float(pkg["psr_hr"])
    st.session_state["psm_rt_hr"] = float(pkg["rt_hr"])
    st.session_state["psm_supervisor_hr"] = float(pkg["supervisor_hr"])

_ensure_state_defaults()

# ============================================================
# SIDEBAR (SIMPLIFIED CORE + ADVANCED)
# ============================================================
with st.sidebar:
    st.header("Inputs")

    # --- Core (first-run friendly) ---
    st.subheader("Clinic Demand")
    visits = st.number_input("Avg Visits/Day (annual avg)", min_value=1.0, value=45.0, step=1.0, key="psm_visits")
    hours_of_operation = st.number_input("Hours of Operation / Week", min_value=1.0, value=70.0, step=1.0, key="psm_hours")
    fte_hours_per_week = st.number_input("FTE Hours / Week", min_value=1.0, value=40.0, step=1.0, key="psm_fte_hours")

    st.subheader("Coverage Safety")
    provider_min_floor = st.number_input("Provider Minimum Floor (FTE)", min_value=0.25, value=1.00, step=0.25, key="psm_floor")
    burnout_slider = st.slider("Burnout Protection Level", 0.0, 1.0, 0.60, 0.05, key="psm_burnout")

    st.subheader("Workforce Reality")
    provider_turnover = st.number_input("Provider Turnover % (annual)", value=24.0, step=1.0, key="psm_turnover_pct") / 100.0

    st.subheader("Seasonality")
    flu_start_month = st.selectbox(
        "Flu Season Start",
        options=list(range(1, 13)),
        index=11,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
        key="psm_flu_start",
    )
    flu_end_month = st.selectbox(
        "Flu Season End",
        options=list(range(1, 13)),
        index=1,
        format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
        key="psm_flu_end",
    )
    flu_uplift_pct = st.number_input("Flu Uplift (%)", min_value=0.0, value=20.0, step=5.0, key="psm_flu_uplift") / 100.0

    st.divider()

    # --- Advanced (collapsed) ---
    with st.expander("⚙️ Advanced Assumptions", expanded=False):
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("✅ Use Recommended Defaults", use_container_width=True):
                apply_recommended_defaults()
                st.rerun()
        with cols[1]:
            st.caption("Optional. Defaults are designed for first-run success.")

        # Hiring pipeline (simplified)
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
                    help="Single number for signing + credentialing + training + buffer. Use breakdown only if you need it.",
                )
                # Keep components available internally (not used unless breakdown enabled)
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
                help="Used for notice-lag (separations occur after notice period).",
            )

        # Recruiting strategy
        with st.expander("Recruiting Strategy", expanded=False):
            enable_seasonality_ramp = st.checkbox(
                "Enable Seasonality Recruiting Ramp",
                value=bool(st.session_state["psm_enable_seasonality_ramp"]),
                key="psm_enable_seasonality_ramp",
                help="If ON: freeze + pipeline blackout blocks HIRING only. Attrition always continues.",
            )

        # Burnout assumptions
        with st.expander("Burnout Assumptions", expanded=False):
            safe_visits_per_provider = st.number_input(
                "Safe Visits/Provider/Day",
                min_value=10, max_value=40,
                value=int(st.session_state["psm_safe_visits_per_provider"]),
                step=1,
                key="psm_safe_visits_per_provider",
            )

        # Confirmed hire (conditional)
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
            else:
                confirmed_hire_month = None
                confirmed_hire_fte = 0.0

        # Finance targets + ROI assumptions
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

        # Compensation package
        with st.expander("Compensation Package (Optional)", expanded=False):
            comp_package = st.selectbox(
                "Compensation Package",
                options=list(COMP_PACKAGES.keys()),
                index=list(COMP_PACKAGES.keys()).index(str(st.session_state["psm_comp_package"]))
                if str(st.session_state["psm_comp_package"]) in COMP_PACKAGES
                else list(COMP_PACKAGES.keys()).index("Expected (Recommended)"),
                key="psm_comp_package",
            )
            cols_pkg = st.columns([1, 1])
            with cols_pkg[0]:
                if st.button("Apply Package", use_container_width=True):
                    apply_comp_package(comp_package)
                    st.rerun()
            with cols_pkg[1]:
                manual_rates = st.checkbox(
                    "Manually override rates",
                    value=bool(st.session_state["psm_manual_rates"]),
                    key="psm_manual_rates",
                )

            # Defaults come from session_state (set by package apply)
            benefits_load_pct = float(st.session_state["psm_benefits_load_pct"])
            ot_sick_pct = float(st.session_state["psm_ot_sick_pct"])
            physician_hr = float(st.session_state["psm_physician_hr"])
            apc_hr = float(st.session_state["psm_apc_hr"])
            ma_hr = float(st.session_state["psm_ma_hr"])
            psr_hr = float(st.session_state["psm_psr_hr"])
            rt_hr = float(st.session_state["psm_rt_hr"])
            supervisor_hr = float(st.session_state["psm_supervisor_hr"])

            if manual_rates:
                benefits_load_pct = st.number_input("Benefits Load (%)", value=float(benefits_load_pct * 100.0), step=1.0, key="psm_benefits_load_pct_ui") / 100.0
                ot_sick_pct = st.number_input("OT + Sick/PTO (%)", value=float(ot_sick_pct * 100.0), step=0.5, key="psm_ot_sick_pct_ui") / 100.0

                physician_hr = st.number_input("Physician (Supervision) $/hr", value=float(physician_hr), step=1.0, key="psm_physician_hr_ui")
                apc_hr = st.number_input("APC $/hr", value=float(apc_hr), step=1.0, key="psm_apc_hr_ui")
                ma_hr = st.number_input("MA $/hr", value=float(ma_hr), step=0.5, key="psm_ma_hr_ui")
                psr_hr = st.number_input("PSR $/hr", value=float(psr_hr), step=0.5, key="psm_psr_hr_ui")
                rt_hr = st.number_input("RT $/hr", value=float(rt_hr), step=0.5, key="psm_rt_hr_ui")
                supervisor_hr = st.number_input("Supervisor $/hr", value=float(supervisor_hr), step=0.5, key="psm_supervisor_hr_ui")

                # Sync back to internal canonical keys so the rest of the app uses the overridden values
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

        # Near-certain planning (Monte Carlo)
        with st.expander("Near-Certain Planning Mode (Pro)", expanded=False):
            enable_probability = st.checkbox(
                "Enable Near-Certain Mode",
                value=bool(st.session_state["psm_enable_probability"]),
                key="psm_enable_probability",
                help="Runs Monte Carlo simulations with uncertainty in visits, turnover, and pipeline time.",
            )
            confidence_level = st.slider(
                "Near-Certain Confidence Level",
                min_value=0.50, max_value=0.95,
                value=float(st.session_state["psm_confidence_level"]),
                step=0.05,
                key="psm_confidence_level",
                help="Example: 0.90 shows NEED at ~90th percentile demand and HAVE at ~10th percentile supply.",
            )
            with st.expander("Tuning (Optional)", expanded=False):
                sim_horizon_months = st.slider(
                    "Simulation Horizon (months)",
                    min_value=24, max_value=60,
                    value=int(st.session_state["psm_sim_horizon_months"]),
                    step=12,
                    key="psm_sim_horizon_months",
                )
                mc_runs = st.slider(
                    "Monte Carlo Runs",
                    min_value=200, max_value=3000,
                    value=int(st.session_state["psm_mc_runs"]),
                    step=100,
                    key="psm_mc_runs",
                )
                visits_cv = st.slider(
                    "Visits Forecast Variability (CV %)",
                    min_value=0.0, max_value=25.0,
                    value=float(st.session_state["psm_visits_cv"] * 100.0),
                    step=1.0,
                    key="psm_visits_cv_ui",
                ) / 100.0
                turnover_var = st.slider(
                    "Turnover Variability (± % of annual turnover)",
                    min_value=0.0, max_value=50.0,
                    value=float(st.session_state["psm_turnover_var"] * 100.0),
                    step=5.0,
                    key="psm_turnover_var_ui",
                ) / 100.0
                pipeline_var_days = st.slider(
                    "Pipeline Duration Variability (± days)",
                    min_value=0, max_value=60,
                    value=int(st.session_state["psm_pipeline_var_days"]),
                    step=5,
                    key="psm_pipeline_var_days",
                )
                # Sync back canonical
                st.session_state["psm_visits_cv"] = float(visits_cv)
                st.session_state["psm_turnover_var"] = float(turnover_var)

            # Defaults if tuning expander never opened
            sim_horizon_months = int(st.session_state["psm_sim_horizon_months"])
            mc_runs = int(st.session_state["psm_mc_runs"])
            visits_cv = float(st.session_state["psm_visits_cv"])
            turnover_var = float(st.session_state["psm_turnover_var"])
            pipeline_var_days = int(st.session_state["psm_pipeline_var_days"])

    # Pull any advanced values not defined when the expander is never opened
    # (Needed because we use them below even on first run.)
    # Hiring
    if "total_lead_days" not in locals():
        total_lead_days = int(st.session_state["psm_pipeline_total_days"])
    if "notice_days" not in locals():
        notice_days = int(st.session_state["psm_notice_days"])
    if "enable_seasonality_ramp" not in locals():
        enable_seasonality_ramp = bool(st.session_state["psm_enable_seasonality_ramp"])
    if "safe_visits_per_provider" not in locals():
        safe_visits_per_provider = int(st.session_state["psm_safe_visits_per_provider"])

    # Finance
    target_swb_per_visit = float(st.session_state["psm_target_swb_per_visit"])
    loaded_cost_per_provider_fte = float(st.session_state["psm_loaded_cost_per_provider_fte"])
    net_revenue_per_visit = float(st.session_state["psm_net_revenue_per_visit"])
    visits_lost_per_provider_day_gap = float(st.session_state["psm_visits_lost_per_provider_day_gap"])

    # Compensation
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

    # Near-certain (defaults)
    enable_probability = bool(st.session_state["psm_enable_probability"])
    confidence_level = float(st.session_state["psm_confidence_level"])
    sim_horizon_months = int(st.session_state["psm_sim_horizon_months"])
    mc_runs = int(st.session_state["psm_mc_runs"])
    visits_cv = float(st.session_state["psm_visits_cv"])
    turnover_var = float(st.session_state["psm_turnover_var"])
    pipeline_var_days = int(st.session_state["psm_pipeline_var_days"])

    st.divider()
    run_model = st.button("▶️ Run PSM", use_container_width=True)

# ============================================================
# RUN MODEL (v6 — continuous multi-year + annual SWB constraint)
# ============================================================
if run_model:
    sim_months = int(sim_horizon_months)
    start_date = datetime(today.year, 1, 1)
    dates_full = pd.date_range(start=start_date, periods=sim_months, freq="MS")
    days_in_month_full = [pd.Period(d, "M").days_in_month for d in dates_full]

    flu_months = months_between(int(flu_start_month), int(flu_end_month))

    fte_result = model.calculate_fte_needed(
        visits_per_day=float(visits),
        hours_of_operation_per_week=float(hours_of_operation),
        fte_hours_per_week=float(fte_hours_per_week),
    )
    baseline_provider_fte = max(float(fte_result["provider_fte"]), float(provider_min_floor))

    forecast_visits_full = compute_seasonality_forecast_multiyear(
        dates=dates_full,
        baseline_visits=visits,
        flu_months=flu_months,
        flu_uplift_pct=flu_uplift_pct,
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

    # Strategy built from typical 12-month protective curve (stable seasonality)
    dates_template_12 = pd.date_range(start=datetime(2000, 1, 1), periods=12, freq="MS")
    protective_typical_12 = typical_12_month_curve(dates_full, protective_full)

    strategy = auto_freeze_strategy_v3_from_typical(
        dates_template_12=dates_template_12,
        protective_typical_12=protective_typical_12,
        flu_start_month=flu_start_month,
        flu_end_month=flu_end_month,
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

    # Derived visible ramp (FTE/month)
    flu_month_idx_typical = int(flu_start_month) - 1
    months_in_flu_window = max(len(strategy["flu_months"]), 1)
    target_at_flu_typical = float(protective_typical_12[flu_month_idx_typical])
    fte_gap_to_close = max(target_at_flu_typical - baseline_provider_fte, 0.0)
    derived_ramp_after_visible = min(fte_gap_to_close / float(months_in_flu_window), 1.25)

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
    )

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
    suggested_confirmed = round_up_quarter(max(hv_delta, 0.0))
    if suggested_confirmed <= 0:
        suggested_confirmed = 1.0
    st.session_state["suggested_confirmed_hire_fte"] = float(suggested_confirmed)

    # Probability mode (optional)
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

    if enable_probability:
        template = dict(strategy)
        template["baseline_provider_fte"] = baseline_provider_fte

        qout = monte_carlo_quantiles(
            dates_full=dates_full,
            days_in_month_full=days_in_month_full,
            baseline_visits=visits,
            flu_months=flu_months,
            flu_uplift_pct=flu_uplift_pct,
            hours_of_operation=hours_of_operation,
            fte_hours_per_week=fte_hours_per_week,
            provider_min_floor=provider_min_floor,
            burnout_slider=burnout_slider,
            safe_visits_per_provider=safe_visits_per_provider,
            total_lead_days=int(total_lead_days),
            notice_days=notice_days,
            provider_turnover=provider_turnover,
            enable_seasonality_ramp=enable_seasonality_ramp,
            max_hiring_up_after_visible=derived_ramp_after_visible,
            strategy_template_12=template,
            role_mix=role_mix,
            hourly_rates=hourly_rates,
            benefits_load_pct=benefits_load_pct,
            ot_sick_pct=ot_sick_pct,
            physician_supervision_hours_per_month=physician_supervision_hours_per_month,
            supervisor_hours_per_month=supervisor_hours_per_month,
            confirmed_hire_month=confirmed_hire_month,
            confirmed_hire_fte=confirmed_hire_fte,
            confirmed_apply_start_idx=confirmed_apply_start_idx,
            display_idx=display_idx,
            confidence_level=confidence_level,
            mc_runs=mc_runs,
            visits_cv=visits_cv,
            turnover_var=turnover_var,
            pipeline_var_days=pipeline_var_days,
        )

        visits_12 = qout["visits_q50"]
        demand_lean_12 = qout["demand_qc"]
        target_prot_12 = qout["prot_qc"]
        supply_rec_12 = qout["supply_qlo"]

    # Apply rounding — after probability adjustments
    demand_lean_12 = [round_up_quarter(x) for x in demand_lean_12]
    target_prot_12 = [round_up_quarter(x) for x in target_prot_12]
    supply_rec_12 = [round_up_quarter(x) for x in supply_rec_12]
    supply_lean_12 = [round_up_quarter(x) for x in supply_lean_12]

    burnout_gap_fte_12 = [max(float(t) - float(s), 0.0) for t, s in zip(target_prot_12, supply_rec_12)]
    months_exposed_12 = int(sum(1 for g in burnout_gap_fte_12 if g > 0))

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
        flu_months=strategy["flu_months"],
        months_in_flu_window=months_in_flu_window,
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
        enable_probability=bool(enable_probability),
        confidence_level=float(confidence_level),
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
st.caption("Visits/day forecast → FTE needed by month (seasonality + flu uplift).")

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
        "Provider FTE (Lean)": round_up_quarter(fte_staff["provider_fte"]),
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
        y_top = ax1.get_ylim()[1]
        ax1.text(
            confirmed_date, y_top,
            f" Confirmed Hire (+{confirmed_fte:.2f} FTE)",
            rotation=90, va="top", ha="left",
            fontsize=9, color=BRAND_BLACK, alpha=0.8,
        )

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

prob_note = ""
if R.get("enable_probability"):
    prob_note = f" (Near-Certain Mode ON: {int(R['confidence_level']*100)}% confidence.)"

st.success(
    f"**Reality Summary:** This 12-month view is taken from a **{R['sim_months']}-month continuous simulation** "
    f"(no year-end reset).{prob_note} To be flu-ready by **{independent_label}**, requisitions must post by **{req_post_label}** "
    f"so hires are visible by **{hire_visible_label}**. "
    f"Best-case visible hiring ramp cap: **{R['derived_ramp_after_visible']:.2f} FTE/month**."
)

st.info(
    "🧠 **Auto-Hiring Strategy (v3)**\n\n"
    f"- Freeze months (blocks hiring only): **{', '.join([datetime(2000,m,1).strftime('%b') for m in R['freeze_months']]) or '—'}**\n"
    f"- Recruiting window: **{', '.join([datetime(2000,m,1).strftime('%b') for m in R['recruiting_open_months']]) or '—'}**\n"
    f"- Post req: **{req_post_label}** | Hires visible: **{hire_visible_label}** | Independent by: **{independent_label}**\n"
    f"- Lead time: **{R['pipeline_lead_days']} days (~{R['lead_months']} months)**\n"
    f"- Notice lag modeled: **{lead_days_to_months(int(notice_days))} months** (separations occur after notice period)\n"
)

# ============================================================
# SECTION 3 — FINANCE
# ============================================================
st.markdown("---")
st.header("3) Finance — ROI Investment Case")
st.caption("Quantifies the investment required to close the gap and the economic value of reducing provider-day shortages.")

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

with st.expander("Show ROI assumptions used", expanded=False):
    st.write(
        f"- Loaded cost per provider FTE (annual): **${loaded_cost_per_provider_fte:,.0f}**\n"
        f"- Net revenue per visit: **${net_revenue_per_visit:,.2f}**\n"
        f"- Visits lost per 1.0 provider-day gap: **{visits_lost_per_provider_day_gap:,.1f}**\n"
    )

st.success(
    "**Finance Summary:** The investment is the cost of staffing to the protective curve. "
    "The value is the revenue protected by reducing provider-day shortages during peak demand."
)

# ============================================================
# SECTION 3B — VVI FEASIBILITY (SWB/Visit) — ANNUAL CONSTRAINT
# ============================================================
st.markdown("---")
st.header("3B) VVI Feasibility — SWB/Visit (FTE-based)")
st.caption(
    "Monthly SWB/Visit will vary (especially in downcycle months where attrition lags volume). "
    "**Feasibility is judged on the year (annual SWB/Visit vs target).**"
)

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
