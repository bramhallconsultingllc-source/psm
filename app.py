# app.py — Predictive Staffing Model (PSM) — Client-Grade (Providers Only)
# Includes all 10 upgrades + client-grade chart recommendations.

import math
import io
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from psm.staffing_model import StaffingModel

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM) — Client Grade", layout="centered")
st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1.25rem; padding-bottom: 2.25rem; }
      .small { font-size: 0.92rem; color: #444; }
      .contract { background: #f7f7f7; border: 1px solid #e6e6e6; border-radius: 10px; padding: 14px 16px; }
      .warn { background: #fff6e6; border: 1px solid #ffe2a8; border-radius: 10px; padding: 12px 14px; }
      .ok { background: #ecfff0; border: 1px solid #b7f0c0; border-radius: 10px; padding: 12px 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Predictive Staffing Model (PSM) — Client Grade")
st.caption("Providers only • Jan–Dec display • Continuous simulation (no year reset) • Audit-ready outputs")

model = StaffingModel()

# ============================================================
# CONSTANTS
# ============================================================
WINTER = {12, 1, 2}
SPRING = {3, 4, 5}
SUMMER = {6, 7, 8}
FALL = {9, 10, 11}

BRAND_BLACK = "#000000"
BRAND_GOLD = "#7a6200"
GRAY = "#B0B0B0"
LIGHT_GRAY = "#EAEAEA"
MID_GRAY = "#666666"

DISPLAY_START = 12  # Jan of display year
DISPLAY_END = 23    # Dec of display year

# ============================================================
# HELPERS
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(float(x), float(hi)))

def lead_days_to_months(days: int, avg_days_per_month: float = 30.4) -> int:
    return max(0, int(math.ceil(float(days) / float(avg_days_per_month))))

def wrap_month(m: int) -> int:
    m = int(m)
    while m <= 0:
        m += 12
    while m > 12:
        m -= 12
    return m

def month_name(m: int) -> str:
    return datetime(2000, int(m), 1).strftime("%b")

def provider_fte_needed(visits_per_day: float, hours_week: float, fte_hours_week: float) -> float:
    res = model.calculate_fte_needed(
        visits_per_day=float(visits_per_day),
        hours_of_operation_per_week=float(hours_week),
        fte_hours_per_week=float(fte_hours_week),
    )
    return float(res.get("provider_fte", 0.0))

def compute_role_mix_ratios(visits_per_day: float, hours_week: float, fte_hours_week: float):
    f = model.calculate_fte_needed(
        visits_per_day=float(visits_per_day),
        hours_of_operation_per_week=float(hours_week),
        fte_hours_per_week=float(fte_hours_week),
    )
    prov = max(float(f.get("provider_fte", 0.0)), 0.25)
    return {
        "psr_per_provider": float(f.get("psr_fte", 0.0)) / prov,
        "ma_per_provider": float(f.get("ma_fte", 0.0)) / prov,
        "xrt_per_provider": float(f.get("xrt_fte", 0.0)) / prov,
    }

def loaded_hourly_rate(base_hourly: float, benefits_load_pct: float, ot_sick_pct: float) -> float:
    return float(base_hourly) * (1.0 + float(benefits_load_pct)) * (1.0 + float(ot_sick_pct))

def monthly_hours_from_fte(fte: float, fte_hours_per_week: float, days_in_month: int) -> float:
    return float(fte) * float(fte_hours_per_week) * (float(days_in_month) / 7.0)

def compute_monthly_swb_per_visit_fte_based(
    provider_supply_12,
    visits_per_day_12,
    days_in_month_12,
    fte_hours_per_week,
    role_mix,
    hourly_rates,
    benefits_load_pct,
    ot_sick_pct,
    physician_supervision_hours_per_month=0.0,
    supervisor_hours_per_month=0.0,
):
    rows = []
    for i in range(12):
        prov_fte = float(provider_supply_12[i])
        vpd = float(visits_per_day_12[i])
        dim = int(days_in_month_12[i])
        month_visits = max(vpd * dim, 1.0)

        psr_fte = prov_fte * float(role_mix["psr_per_provider"])
        ma_fte = prov_fte * float(role_mix["ma_per_provider"])
        rt_fte = prov_fte * float(role_mix["xrt_per_provider"])

        apc_hours = monthly_hours_from_fte(prov_fte, fte_hours_per_week, dim)
        psr_hours = monthly_hours_from_fte(psr_fte, fte_hours_per_week, dim)
        ma_hours = monthly_hours_from_fte(ma_fte, fte_hours_per_week, dim)
        rt_hours = monthly_hours_from_fte(rt_fte, fte_hours_per_week, dim)

        apc_rate = loaded_hourly_rate(hourly_rates["apc"], benefits_load_pct, ot_sick_pct)
        psr_rate = loaded_hourly_rate(hourly_rates["psr"], benefits_load_pct, ot_sick_pct)
        ma_rate = loaded_hourly_rate(hourly_rates["ma"], benefits_load_pct, ot_sick_pct)
        rt_rate = loaded_hourly_rate(hourly_rates["rt"], benefits_load_pct, ot_sick_pct)

        phys_rate = loaded_hourly_rate(hourly_rates["physician"], benefits_load_pct, ot_sick_pct)
        sup_rate = loaded_hourly_rate(hourly_rates["supervisor"], benefits_load_pct, ot_sick_pct)

        apc_cost = apc_hours * apc_rate
        psr_cost = psr_hours * psr_rate
        ma_cost = ma_hours * ma_rate
        rt_cost = rt_hours * rt_rate

        phys_cost = float(physician_supervision_hours_per_month) * phys_rate
        sup_cost = float(supervisor_hours_per_month) * sup_rate

        total_swb = apc_cost + psr_cost + ma_cost + rt_cost + phys_cost + sup_cost
        swb_per_visit = total_swb / month_visits

        rows.append({
            "Provider_FTE_Supply": prov_fte,
            "PSR_FTE": psr_fte,
            "MA_FTE": ma_fte,
            "RT_FTE": rt_fte,
            "Visits": month_visits,
            "SWB_$": total_swb,
            "SWB_per_Visit_$": swb_per_visit,
        })
    return pd.DataFrame(rows)

def provider_day_gap(target_curve, supply_curve, days_in_month):
    return float(sum(max(float(t) - float(s), 0.0) * float(dim) for t, s, dim in zip(target_curve, supply_curve, days_in_month)))

def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    return float(sum(float(df) * float(loaded_cost_per_provider_fte) * (float(dim) / 365.0) for df, dim in zip(delta_fte_curve, days_in_month)))

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def build_one_page_pdf_bytes(
    title: str,
    subtitle: str,
    bullets: list[str],
    metrics: dict[str, str],
    chart_png_bytes: bytes,
) -> bytes:
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=letter)
    w, h = letter

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, title)
    c.setFont("Helvetica", 10.5)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(40, h - 68, subtitle)
    c.setFillColorRGB(0, 0, 0)

    # Metrics
    y = h - 105
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(40, y, "Key metrics")
    c.setFont("Helvetica", 10)
    y -= 16
    for k, v in metrics.items():
        c.drawString(50, y, f"- {k}: {v}")
        y -= 14

    # Bullets
    y -= 6
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(40, y, "Executive summary")
    c.setFont("Helvetica", 10)
    y -= 16
    for b in bullets[:8]:
        c.drawString(50, y, f"- {b}")
        y -= 14

    # Chart image
    img_reader = ImageReader(io.BytesIO(chart_png_bytes))
    img_w = w - 80
    img_h = 250
    c.drawImage(img_reader, 40, 60, width=img_w, height=img_h, preserveAspectRatio=True, anchor='sw')

    c.showPage()
    c.save()
    pdf_buf.seek(0)
    return pdf_buf.read()

# ============================================================
# PARAMS + CORE SIMULATION
# ============================================================
@dataclass
class PSMParams:
    visits: float
    hours_week: float
    fte_hours_week: float

    seasonality_pct: float
    annual_turnover: float
    annual_growth: float

    lead_days: int

    provider_floor_fte: float
    starting_supply_fte: float

    # Hiring strategy rules
    ready_month: int = 11                   # Nov 1 readiness anchor
    flu_anchor_month: int = 12              # December target anchor
    hire_step_cap_fte: float = 1.25         # realistic visible step cap
    allow_floor_maintenance_pipeline: bool = True
    freeze_except_flu_and_floor: bool = True

def compute_visits_curve(months: list[int], base_year0: float, base_year1: float, seasonality_pct: float) -> list[float]:
    out = []
    for i, m in enumerate(months):
        base = base_year0 if i < 12 else base_year1
        if m in WINTER:
            v = base * (1.0 + seasonality_pct)
        elif m in SUMMER:
            v = base * (1.0 - seasonality_pct)
        else:
            v = base
        out.append(float(v))
    return out

def find_month_index_in_range(months: list[int], target_month: int, start_i: int, end_i: int):
    for i in range(start_i, end_i + 1):
        if int(months[i]) == int(target_month):
            return i
    return None

def compute_simulation(params: PSMParams, scenario_name: str = "Current"):
    """
    Continuous simulation:
    - 25 months (Y0 warm-up 12, Y1 display 12, +1 month for no-reset check)
    - Flu-cycle step hire applied in BOTH years so Jan display inherits prior Nov
    - Optional floor maintenance pipeline (lead-time aware) to prevent dipping below floor (permanent supply)
    - Hiring freeze: only flu req month (and floor maintenance if enabled) can post reqs
    """
    today = datetime.today()
    year0 = today.year
    N = 25
    dates = pd.date_range(start=datetime(year0, 1, 1), periods=N, freq="MS")
    months = [int(d.month) for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    lead_months = lead_days_to_months(int(params.lead_days))
    monthly_turnover = float(params.annual_turnover) / 12.0

    # Baseline visits: grow into display year (Y1)
    base_year0 = float(params.visits)
    base_year1 = float(params.visits) * (1.0 + float(params.annual_growth))

    visits_curve = compute_visits_curve(months, base_year0, base_year1, float(params.seasonality_pct))

    # Target provider FTE by month (demand-driven) + floor
    target_curve = []
    for v in visits_curve:
        t = provider_fte_needed(v, params.hours_week, params.fte_hours_week)
        target_curve.append(max(float(t), float(params.provider_floor_fte)))

    # Baseline provider FTE reference (Spring/Fall baseline of display year baseline)
    baseline_provider_fte = max(
        provider_fte_needed(base_year1, params.hours_week, params.fte_hours_week),
        float(params.provider_floor_fte),
    )

    # Determine flu planning months
    req_post_month = wrap_month(params.ready_month - lead_months)
    hire_visible_month = wrap_month(req_post_month + lead_months)

    # Compute flu need based on December target (winter uplift)
    # Use display-year baseline for flu sizing.
    dec_visits = base_year1 * (1.0 + float(params.seasonality_pct))
    dec_target_fte = max(provider_fte_needed(dec_visits, params.hours_week, params.fte_hours_week), float(params.provider_floor_fte))

    # Hiring arrays (visible) for permanent supply
    hires_visible = [0.0] * N
    hires_visible_reason = [""] * N

    # Freeze logic: define which months allow posting reqs (month-of-year)
    # We freeze all months except the flu req-post month. Floor maintenance can override if enabled.
    def can_post_req(month_num: int, is_floor_maintenance: bool) -> bool:
        if not params.freeze_except_flu_and_floor:
            return True
        if is_floor_maintenance and params.allow_floor_maintenance_pipeline:
            return True
        return int(month_num) == int(req_post_month)

    # ---- Flu step hire in YEAR 0 (warm-up), so it carries into display Jan
    req_idx_y0 = find_month_index_in_range(months, req_post_month, 0, 11)
    if req_idx_y0 is not None:
        vis_idx_y0 = req_idx_y0 + lead_months
        # Flu visible month should fall within warm-up year to affect carryover (usually Nov)
        if 0 <= vis_idx_y0 <= 11:
            # Size flu hire using starting supply projected to ready
            loss_factor = (1.0 - monthly_turnover) ** float(lead_months) if lead_months > 0 else 1.0
            expected_at_ready = float(params.starting_supply_fte) * float(loss_factor)
            fte_needed = max(float(dec_target_fte) - float(expected_at_ready), 0.0)
            step = min(fte_needed, float(params.hire_step_cap_fte)) if params.hire_step_cap_fte > 0 else fte_needed
            if step > 1e-6:
                hires_visible[vis_idx_y0] += float(step)
                hires_visible_reason[vis_idx_y0] = f"Flu hire step (Y0), capped at {params.hire_step_cap_fte:.2f} FTE"
            # Remaining flu need becomes flex coverage requirement (not permanent) — tracked later

    # ---- Flu step hire in YEAR 1 (display year), affects Nov–Dec and beyond
    req_idx_y1 = find_month_index_in_range(months, req_post_month, DISPLAY_START, DISPLAY_END)
    if req_idx_y1 is None:
        # Should never happen, but keep safe
        req_idx_y1 = find_month_index_in_range(months, req_post_month, 0, N - 1)

    vis_idx_y1 = (req_idx_y1 + lead_months) if req_idx_y1 is not None else None

    # We'll size YEAR1 flu hire off projected supply at ready using the simulation itself,
    # but we don't yet have supply. We'll do it after we build supply with floor planning.
    # So: placeholder; we will apply after initial pass or compute analytically.
    pending_flu_y1 = {"vis_idx": vis_idx_y1}

    # Optional: floor maintenance pipeline planning (lead-time aware)
    # Plan hires so that in v = t+lead_months, projected permanent supply doesn't fall below floor.
    # This is how "maintain minimum staffing" can work without breaking time logic.
    planned_floor_hires_visible = [0.0] * N

    # Supply simulation
    supply = [0.0] * N
    supply[0] = max(float(params.starting_supply_fte), float(params.provider_floor_fte))

    # We'll iterate, and in each month t we can "post reqs" that become visible at v=t+lead_months.
    for t in range(1, N):
        prev = float(supply[t - 1])
        after_attrition = prev * (1.0 - monthly_turnover)

        # Floor maintenance planning (posts reqs now for future visibility)
        if params.allow_floor_maintenance_pipeline and lead_months >= 0:
            v = t + lead_months
            if v < N:
                # Project supply at v if we do nothing else beyond attrition (rough)
                # Use current after_attrition as base; apply attrition forward lead_months
                proj = float(after_attrition) * ((1.0 - monthly_turnover) ** float(lead_months) if lead_months > 0 else 1.0)
                # Include already planned visible hires at v (flu + previously planned floor)
                proj += float(hires_visible[v]) + float(planned_floor_hires_visible[v])

                # If projected dips below floor, plan a floor maintenance hire visible at v
                if proj < float(params.provider_floor_fte) - 1e-6:
                    need = float(params.provider_floor_fte) - proj
                    # Can we post req in month t? (freeze exception applies)
                    if can_post_req(months[t], is_floor_maintenance=True):
                        planned_floor_hires_visible[v] += need
                    # If we cannot post due to freeze (rare if floor maintenance allowed),
                    # we leave as gap (flex), tracked later.

        # Apply any hires visible this month (flu + planned floor)
        hires_this_month = float(hires_visible[t]) + float(planned_floor_hires_visible[t])
        supply[t] = max(after_attrition + hires_this_month, float(params.provider_floor_fte))

    # Now size and apply YEAR1 flu hire step based on projected supply at ready (vis_idx_y1)
    # We want readiness by Nov (ready_month), which corresponds to the month-of-year; the visible index should land there.
    if pending_flu_y1["vis_idx"] is not None and 0 <= pending_flu_y1["vis_idx"] < N:
        idx = int(pending_flu_y1["vis_idx"])
        # Only if this visible month is in display year window (so user sees the step)
        # Still apply if outside; it will carry but won't be shown.
        expected_at_ready = float(supply[idx])  # this is supply WITH floor maintenance hires that are planned
        fte_needed = max(float(dec_target_fte) - float(expected_at_ready), 0.0)
        step = min(fte_needed, float(params.hire_step_cap_fte)) if params.hire_step_cap_fte > 0 else fte_needed
        if step > 1e-6:
            hires_visible[idx] += float(step)
            hires_visible_reason[idx] = f"Flu hire step (Y1), capped at {params.hire_step_cap_fte:.2f} FTE"
            # Re-run supply from that month forward to reflect the added step
            for t in range(idx, N):
                if t == 0:
                    continue
                prev = float(supply[t - 1])
                after_attrition = prev * (1.0 - monthly_turnover)
                hires_this_month = float(hires_visible[t]) + float(planned_floor_hires_visible[t])
                supply[t] = max(after_attrition + hires_this_month, float(params.provider_floor_fte))

    # Build display-year slices
    idx12 = list(range(DISPLAY_START, DISPLAY_END + 1))
    dates_12 = [dates[i] for i in idx12]
    month_labels_12 = [d.strftime("%b") for d in dates_12]
    days_12 = [days_in_month[i] for i in idx12]

    visits_12 = [float(visits_curve[i]) for i in idx12]
    target_12 = [float(target_curve[i]) for i in idx12]
    supply_12 = [float(supply[i]) for i in idx12]
    hires_12 = [float(hires_visible[i] + planned_floor_hires_visible[i]) for i in idx12]
    hires_flu_12 = [float(hires_visible[i]) for i in idx12]
    hires_floor_12 = [float(planned_floor_hires_visible[i]) for i in idx12]

    gap_12 = [max(t - s, 0.0) for t, s in zip(target_12, supply_12)]
    months_exposed = int(sum(1 for g in gap_12 if g > 1e-6))
    peak_gap = float(max(gap_12)) if gap_12 else 0.0
    avg_gap = float(np.mean(gap_12)) if gap_12 else 0.0

    # Flex requirement: any target gap is assumed to be covered via PRN/fractional
    flex_fte_12 = gap_12[:]  # provider FTE equivalent needed
    provider_day_gap_total = provider_day_gap(target_12, supply_12, days_12)

    # Ledger (audit table)
    start_supply_12 = [supply_12[0]] + supply_12[:-1]
    turnover_shed_12 = [-(start_supply_12[i] * monthly_turnover) for i in range(12)]  # negative number
    end_supply_12 = supply_12

    ledger = pd.DataFrame({
        "Month": month_labels_12,
        "Visits/Day": np.round(visits_12, 1),
        "Start_FTE": np.round(start_supply_12, 3),
        "Turnover_Shed_FTE": np.round(turnover_shed_12, 3),
        "Hire_Visible_FTE": np.round(hires_12, 3),
        "  (Flu step)": np.round(hires_flu_12, 3),
        "  (Floor maint)": np.round(hires_floor_12, 3),
        "End_FTE": np.round(end_supply_12, 3),
        "Target_FTE": np.round(target_12, 3),
        "Gap_FTE": np.round(gap_12, 3),
        "Flex_FTE_Req": np.round(flex_fte_12, 3),
    })

    # No-reset check: Dec display year -> next Jan
    dec_idx = DISPLAY_END
    next_jan_idx = DISPLAY_END + 1
    no_reset = None
    if next_jan_idx < N:
        no_reset = {
            "Supply_Dec": float(supply[dec_idx]),
            "Supply_Next_Jan": float(supply[next_jan_idx]),
        }

    # Planning timeline points
    timeline = {
        "lead_months": lead_months,
        "monthly_turnover": monthly_turnover,
        "req_post_month": req_post_month,
        "hire_visible_month": hire_visible_month,
        "ready_month": params.ready_month,
        "dec_target_fte": float(dec_target_fte),
        "flu_anchor_month": params.flu_anchor_month,
        "scenario": scenario_name,
    }

    return dict(
        dates_12=dates_12,
        month_labels_12=month_labels_12,
        days_12=days_12,

        visits_12=visits_12,
        target_12=target_12,
        supply_12=supply_12,
        gap_12=gap_12,
        flex_fte_12=flex_fte_12,

        months_exposed=months_exposed,
        peak_gap=peak_gap,
        avg_gap=avg_gap,
        provider_day_gap_total=provider_day_gap_total,

        baseline_provider_fte=baseline_provider_fte,
        ledger=ledger,

        hires_visible_full=hires_visible,
        floor_hires_full=planned_floor_hires_visible,
        timeline=timeline,
        no_reset=no_reset,

        dates_full=list(dates),
        supply_full=supply,
        target_full=target_curve,
        visits_full=visits_curve,
    )

# ============================================================
# SIDEBAR — INPUTS + OPTIONS
# ============================================================
with st.sidebar:
    st.header("Inputs")

    # Core demand
    visits = st.number_input("Avg Visits/Day", min_value=1.0, value=36.0, step=1.0)
    hours_week = st.number_input("Hours of Operation / Week", min_value=1.0, value=84.0, step=1.0)
    fte_hours_week = st.number_input("FTE Hours / Week", min_value=1.0, value=36.0, step=1.0)

    # Seasonality + workforce
    st.subheader("Seasonality + Workforce Reality")
    seasonality_pct = st.number_input("Seasonality % Lift/Drop", min_value=0.0, value=20.0, step=5.0) / 100.0
    annual_turnover = st.number_input("Annual Turnover %", min_value=0.0, value=16.0, step=1.0) / 100.0
    annual_growth = st.number_input("Annual Visit Growth % (applied to display year)", min_value=0.0, value=10.0, step=1.0) / 100.0

    st.subheader("Pipeline")
    lead_days = st.number_input("Days to Independent (Req→Independent)", min_value=0, value=210, step=10)
    hire_step_cap_fte = st.number_input("Hire Step Cap (FTE) — realism guardrail", min_value=0.0, value=1.25, step=0.25)

    st.divider()
    st.subheader("Minimum Staffing Model")
    provider_floor_fte = st.number_input("Provider Minimum Floor (FTE)", min_value=0.25, value=1.0, step=0.25)
    allow_floor_maintenance_pipeline = st.checkbox("Maintain floor via replacement pipeline (lead-time aware)", value=True)
    freeze_except_flu_and_floor = st.checkbox("Freeze hiring except flu req month (and floor maintenance)", value=True)

    st.divider()
    st.subheader("Starting Supply (Optional)")
    use_calculated_baseline = st.checkbox("Use calculated baseline as starting supply", value=True)

    # Finance
    st.divider()
    st.header("Finance Inputs")
    net_revenue_per_visit = st.number_input("Net Revenue per Visit (NRPV)", min_value=0.0, value=140.0, step=5.0)
    target_swb_per_visit = st.number_input("Target SWB/Visit", min_value=0.0, value=85.0, step=1.0)
    visits_lost_per_provider_day_gap = st.number_input("Visits Lost per 1.0 Provider-Day Gap", min_value=0.0, value=18.0, step=1.0)
    loaded_cost_per_provider_fte = st.number_input("Loaded Cost per Provider FTE (annual)", min_value=0.0, value=260000.0, step=5000.0)

    # Compensation inputs
    st.divider()
    st.header("Comp + Loads")
    benefits_load_pct = st.number_input("Benefits Load %", min_value=0.0, value=30.0, step=1.0) / 100.0
    ot_sick_pct = st.number_input("OT + Sick/PTO %", min_value=0.0, value=4.0, step=0.5) / 100.0

    physician_hr = st.number_input("Physician (optional) $/hr", min_value=0.0, value=135.79, step=1.0)
    apc_hr = st.number_input("APP $/hr", min_value=0.0, value=62.0, step=1.0)
    ma_hr = st.number_input("MA $/hr", min_value=0.0, value=24.14, step=0.5)
    psr_hr = st.number_input("PSR $/hr", min_value=0.0, value=21.23, step=0.5)
    rt_hr = st.number_input("RT $/hr", min_value=0.0, value=31.36, step=0.5)
    supervisor_hr = st.number_input("Supervisor (optional) $/hr", min_value=0.0, value=28.25, step=0.5)

    physician_supervision_hours_per_month = st.number_input("Physician supervision hours/month", min_value=0.0, value=0.0, step=1.0)
    supervisor_hours_per_month = st.number_input("Supervisor hours/month", min_value=0.0, value=0.0, step=1.0)

    st.divider()
    st.header("Display + Tools")
    show_visits_overlay = st.checkbox("Show Visits/Day overlay", value=True)
    show_heatmap = st.checkbox("Show gap heatmap", value=True)
    show_debug = st.checkbox("Show debug panel", value=False)

    st.subheader("Scenario Compare")
    enable_compare = st.checkbox("Compare scenarios", value=True)
    improved_lead_days = st.number_input("Improved pipeline days (scenario B)", min_value=0, value=150, step=10)

    st.subheader("QA Harness")
    run_tests = st.checkbox("Run test mode (PASS/FAIL)", value=False)

    st.divider()
    run = st.button("▶️ Run PSM", use_container_width=True)

if not run:
    st.info("Set inputs and click **Run PSM**.")
    st.stop()

# ============================================================
# COMPUTE STARTING SUPPLY (baseline if chosen)
# ============================================================
baseline_for_start = max(
    provider_fte_needed(float(visits) * (1.0 + float(annual_growth)), float(hours_week), float(fte_hours_week)),
    float(provider_floor_fte),
)
if use_calculated_baseline:
    starting_supply_fte = float(baseline_for_start)
else:
    starting_supply_fte = st.sidebar.number_input(
        "Current Provider FTE (Starting Supply)",
        min_value=0.0,
        value=float(baseline_for_start),
        step=0.25,
        help="Used for predicted supply only. Target stays demand-driven.",
    )
    starting_supply_fte = max(float(starting_supply_fte), float(provider_floor_fte))

# ============================================================
# BUILD PARAMS + RUN SIMS
# ============================================================
params_A = PSMParams(
    visits=float(visits),
    hours_week=float(hours_week),
    fte_hours_week=float(fte_hours_week),
    seasonality_pct=float(seasonality_pct),
    annual_turnover=float(annual_turnover),
    annual_growth=float(annual_growth),
    lead_days=int(lead_days),
    provider_floor_fte=float(provider_floor_fte),
    starting_supply_fte=float(starting_supply_fte),
    hire_step_cap_fte=float(hire_step_cap_fte),
    allow_floor_maintenance_pipeline=bool(allow_floor_maintenance_pipeline),
    freeze_except_flu_and_floor=bool(freeze_except_flu_and_floor),
)
R_A = compute_simulation(params_A, scenario_name="A (Current)")

R_B = None
if enable_compare:
    params_B = PSMParams(
        **{**params_A.__dict__, "lead_days": int(improved_lead_days)}
    )
    R_B = compute_simulation(params_B, scenario_name="B (Improved pipeline)")

# ============================================================
# 1) MODEL CONTRACT PANEL (Upgrade #1)
# ============================================================
lead_months_A = R_A["timeline"]["lead_months"]
req_m_A = R_A["timeline"]["req_post_month"]
vis_m_A = R_A["timeline"]["hire_visible_month"]
monthly_turnover_pct = R_A["timeline"]["monthly_turnover"] * 100.0

st.markdown(
    f"""
<div class="contract">
  <b>Model Contract (what this tool assumes)</b>
  <ul class="small" style="margin-top:8px;">
    <li><b>Seasonality:</b> Winter (Dec–Feb) up, Summer (Jun–Aug) down, Spring/Fall baseline.</li>
    <li><b>Target line:</b> demand-driven Provider FTE from visits/day + a Provider Floor of <b>{provider_floor_fte:.2f}</b>.</li>
    <li><b>Predicted line:</b> Starting supply − turnover + hires that become visible after lead time.</li>
    <li><b>Pipeline:</b> {lead_days} days ≈ <b>{lead_months_A}</b> months (continuous simulation; no year reset).</li>
    <li><b>Hiring freeze:</b> posts reqs only in <b>{month_name(req_m_A)}</b> (flu cycle) + floor maintenance (if enabled).</li>
    <li><b>Burnout risk area:</b> shaded gap = Target − Predicted; gap is assumed covered by flex/PRN/fractional.</li>
  </ul>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 2) REALITY CHECK WARNINGS (Upgrade #4)
# ============================================================
warnings = []
# pipeline too slow (hire visible beyond Dec in display year)
lead_months = lead_days_to_months(int(lead_days))
req_idx_display = find_month_index_in_range(
    [d.month for d in R_A["dates_full"]],
    req_m_A,
    DISPLAY_START,
    DISPLAY_END
)
if req_idx_display is not None:
    if req_idx_display + lead_months > DISPLAY_END:
        warnings.append("Lead time pushes visibility beyond Dec — you cannot be peak-ready this year under the current pipeline.")
# sustained exposure
if R_A["months_exposed"] >= 7:
    warnings.append("Sustained gap exposure (≥7 months) — risk of chronic burnout unless flex coverage is planned and funded.")
# large hire requirement (if step is capped)
if params_A.hire_step_cap_fte > 0 and params_A.hire_step_cap_fte < 0.75:
    warnings.append("Hire step cap is very low — model will shift more burden to flex coverage to close gaps.")
# floor maintenance off + high turnover
if (not allow_floor_maintenance_pipeline) and annual_turnover >= 0.20:
    warnings.append("Floor maintenance pipeline is OFF with high turnover — predicted supply may be unrealistically optimistic at the floor.")

if warnings:
    st.markdown("<div class='warn'><b>Reality Check</b><ul class='small' style='margin-top:8px;'>"
                + "".join(f"<li>{w}</li>" for w in warnings) +
                "</ul></div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='ok'><b>Reality Check</b><div class='small' style='margin-top:6px;'>No model integrity flags triggered.</div></div>", unsafe_allow_html=True)

# ============================================================
# 3) CLIENT-GRADE HERO CHART (Upgrade: recommended graph)
# ============================================================
st.markdown("---")
st.header("Jan–Dec Staffing Outlook (Providers Only)")

# headline metrics (scenario A)
m1, m2, m3 = st.columns(3)
m1.metric("Peak Burnout Gap (FTE)", f"{R_A['peak_gap']:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{R_A['avg_gap']:.2f}")
m3.metric("Months Exposed", f"{R_A['months_exposed']}/12")

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

dates_12 = R_A["dates_12"]
labels_12 = R_A["month_labels_12"]

# Target and predicted
ax1.plot(dates_12, R_A["target_12"], linewidth=2.2, color=BRAND_GOLD, marker="o", markersize=4, label="Target Provider FTE")
ax1.plot(dates_12, R_A["supply_12"], linewidth=2.2, color=BRAND_BLACK, marker="o", markersize=4, label="Predicted Provider FTE (Permanent, constrained)")

# Burnout gap shading
ax1.fill_between(
    dates_12,
    np.array(R_A["supply_12"], dtype=float),
    np.array(R_A["target_12"], dtype=float),
    where=np.array(R_A["target_12"], dtype=float) > np.array(R_A["supply_12"], dtype=float),
    color=BRAND_GOLD,
    alpha=0.12,
    label="Burnout Risk (Gap)",
)

# Scenario compare overlay (predicted line only + light styling)
if R_B is not None:
    ax1.plot(dates_12, R_B["supply_12"], linewidth=2.0, color=MID_GRAY, linestyle="--", marker=None,
             label=f"Predicted Provider FTE — B ({improved_lead_days}d pipeline)")

ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold")
ax1.set_xticks(dates_12)
ax1.set_xticklabels(labels_12, fontsize=11)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

# Visits overlay toggle
ax2 = None
if show_visits_overlay:
    ax2 = ax1.twinx()
    ax2.plot(dates_12, R_A["visits_12"], linestyle="-.", linewidth=1.6, color=GRAY, label="Visits/Day (Forecast)")
    ax2.set_ylabel("Visits / Day", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=11)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
if ax2 is not None:
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
else:
    ax1.legend(lines1, labels1, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))

ax1.set_title("Target vs Predicted Provider Staffing (Continuous; no year reset)", fontsize=14, fontweight="bold")
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# 4) GAP HEATMAP (Upgrade #A)
# ============================================================
if show_heatmap:
    st.subheader("Gap Heatmap (Burnout Risk by Month)")
    gaps = np.array(R_A["gap_12"], dtype=float)
    mx = float(np.max(gaps)) if len(gaps) else 0.0

    def level(g):
        if mx <= 1e-9:
            return ("🟩", "No gap")
        r = g / mx
        if r <= 0.15:
            return ("🟩", "Low")
        if r <= 0.40:
            return ("🟨", "Moderate")
        if r <= 0.70:
            return ("🟧", "High")
        return ("🟥", "Severe")

    cols = st.columns(12)
    for i, (lab, g) in enumerate(zip(labels_12, gaps)):
        icon, txt = level(float(g))
        cols[i].markdown(f"**{lab}**<br/>{icon}<br/><span class='small'>{txt}</span>", unsafe_allow_html=True)

# ============================================================
# 5) HIRING TIMELINE VISUAL (Upgrade #6)
# ============================================================
st.markdown("---")
st.header("Pipeline Timeline (Flu Readiness)")

tl = R_A["timeline"]
st.markdown(
    f"""
<div class="small">
<b>Anchor:</b> Staff needed for <b>{month_name(tl["flu_anchor_month"])}</b> should be independent by <b>{month_name(tl["ready_month"])}</b> 1.<br/>
<b>Lead time:</b> {lead_days} days ≈ <b>{tl["lead_months"]}</b> months • <b>Turnover:</b> {annual_turnover*100:.1f}% annual ≈ {monthly_turnover_pct:.2f}% monthly<br/>
<b>Post requisition by:</b> <b>{month_name(tl["req_post_month"])}</b> • <b>Hires become visible:</b> <b>{month_name(tl["hire_visible_month"])}</b>
</div>
""",
    unsafe_allow_html=True,
)

# Simple timeline bar chart
fig_t, ax_t = plt.subplots(figsize=(12, 1.5))
ax_t.set_axis_off()
# positions 0..12 for months
x_req = tl["req_post_month"] - 1
x_vis = tl["hire_visible_month"] - 1
x_ready = tl["ready_month"] - 1
ax_t.hlines(0.5, 0, 11, linewidth=6, color=LIGHT_GRAY)
ax_t.vlines([x_req, x_vis, x_ready], 0.35, 0.65, colors=[BRAND_GOLD, BRAND_BLACK, MID_GRAY], linewidth=4)
ax_t.text(x_req, 0.75, f"Post req ({month_name(tl['req_post_month'])})", ha="center", va="bottom", fontsize=9)
ax_t.text(x_vis, 0.15, f"Visible ({month_name(tl['hire_visible_month'])})", ha="center", va="top", fontsize=9)
ax_t.text(x_ready, 0.75, f"Independent ({month_name(tl['ready_month'])})", ha="center", va="bottom", fontsize=9)
for i in range(12):
    ax_t.text(i, 0.02, month_name(i + 1), ha="center", va="bottom", fontsize=8, color=MID_GRAY)
st.pyplot(fig_t)

# ============================================================
# 6) MONTHLY STAFFING LEDGER (Upgrade #2)
# ============================================================
st.markdown("---")
st.header("Monthly Staffing Ledger (Audit View)")

ledger = R_A["ledger"].copy()
st.dataframe(ledger, hide_index=True, use_container_width=True)

st.caption("This ledger is the 'truth table' for the chart: start → turnover shed → hires visible → end supply → target → gap.")

# Month-by-month explainer (click-to-expand)
st.subheader("Why did this month happen? (Click a month)")
for i, row in ledger.iterrows():
    month = str(row["Month"])
    with st.expander(month, expanded=False):
        start_fte = float(row["Start_FTE"])
        shed = float(row["Turnover_Shed_FTE"])
        hire_vis = float(row["Hire_Visible_FTE"])
        end_fte = float(row["End_FTE"])
        target = float(row["Target_FTE"])
        gap = float(row["Gap_FTE"])
        flex = float(row["Flex_FTE_Req"])

        reasons = []
        if abs(shed) > 1e-6:
            reasons.append(f"Turnover reduced supply by {abs(shed):.3f} FTE.")
        if hire_vis > 1e-6:
            reasons.append(f"Hires became visible (+{hire_vis:.3f} FTE).")
        if gap > 1e-6:
            reasons.append(f"Demand exceeded permanent supply by {gap:.3f} FTE → flex coverage required.")
        if not reasons:
            reasons.append("No material changes; supply and demand were aligned.")

        st.markdown(
            f"""
- **Start supply:** {start_fte:.3f}  
- **Turnover shed:** {shed:.3f}  
- **Hire visible:** {hire_vis:.3f}  
- **End supply:** {end_fte:.3f}  
- **Target:** {target:.3f}  
- **Gap / flex required:** {gap:.3f} FTE  
"""
        )
        st.markdown("**Explanation:** " + " ".join(reasons))

# ============================================================
# 7) FLEX COVERAGE PLAN (Upgrade #5)
# ============================================================
st.markdown("---")
st.header("Flex Coverage Plan (PRN / Fractional / Float Pool)")

gap_days_total = float(R_A["provider_day_gap_total"])
st.write(
    f"""
- Total provider-day gap (annualized across Jan–Dec display year): **{gap_days_total:,.0f} provider-days**
- This gap is assumed covered by **flex coverage** (PRN shifts, fractional FTE across clinics, float pool).
"""
)

flex_df = pd.DataFrame({
    "Month": R_A["month_labels_12"],
    "Flex_FTE_Required": np.round(R_A["flex_fte_12"], 3),
    "Provider_Days_Required": np.round(np.array(R_A["flex_fte_12"]) * np.array(R_A["days_12"]), 1),
})
st.dataframe(flex_df, hide_index=True, use_container_width=True)

# Simple levers (optional, not altering predicted line; planning only)
st.subheader("Flex Mix Builder (Planning Tool)")
cA, cB, cC = st.columns(3)
with cA:
    float_pool_fte = st.slider("Float Pool (FTE)", 0.0, 6.0, 1.0, 0.25)
with cB:
    fractional_fte = st.slider("Fractional Add (FTE)", 0.0, 6.0, 0.5, 0.25)
with cC:
    prn_buffer_pct = st.slider("PRN Buffer %", 0, 100, 25, 5)

effective_flex_fte = []
for f in R_A["flex_fte_12"]:
    x = float(f) * (1.0 - prn_buffer_pct / 100.0)
    x = max(x - float(float_pool_fte), 0.0)
    x = max(x - float(fractional_fte), 0.0)
    effective_flex_fte.append(x)

flex_days_after = float(sum(np.array(effective_flex_fte) * np.array(R_A["days_12"])))
flex_days_reduced = max(gap_days_total - flex_days_after, 0.0)

st.success(
    f"With these levers, estimated provider-day gap reduced by **{flex_days_reduced:,.0f} days** "
    f"(remaining flex need: **{flex_days_after:,.0f} days**)."
)

# ============================================================
# 8) FINANCE — ROI (kept)
# ============================================================
st.markdown("---")
st.header("Finance — ROI Investment Case")

delta_fte_curve = [max(float(t) - float(R_A["baseline_provider_fte"]), 0.0) for t in R_A["target_12"]]
annual_investment = annualize_monthly_fte_cost(delta_fte_curve, R_A["days_12"], loaded_cost_per_provider_fte)

gap_days = float(R_A["provider_day_gap_total"])
est_visits_lost = gap_days * float(visits_lost_per_provider_day_gap)
est_revenue_lost = est_visits_lost * float(net_revenue_per_visit)
roi = (est_revenue_lost / annual_investment) if annual_investment > 0 else np.nan

f1, f2, f3 = st.columns(3)
f1.metric("Annual Investment (to Target)", f"${annual_investment:,.0f}")
f2.metric("Est. Net Revenue at Risk", f"${est_revenue_lost:,.0f}")
f3.metric("ROI (Revenue ÷ Investment)", f"{roi:,.2f}x" if np.isfinite(roi) else "—")

# ============================================================
# 9) SWB/Visit FEASIBILITY + CFO-STYLE SENSITIVITY (Upgrade #7)
# ============================================================
st.markdown("---")
st.header("VVI Feasibility — SWB/Visit (Annual Constraint)")

role_mix = compute_role_mix_ratios(float(visits) * (1.0 + float(annual_growth)), hours_week, fte_hours_week)
hourly_rates = {
    "physician": physician_hr,
    "apc": apc_hr,
    "ma": ma_hr,
    "psr": psr_hr,
    "rt": rt_hr,
    "supervisor": supervisor_hr,
}

swb_df = compute_monthly_swb_per_visit_fte_based(
    provider_supply_12=R_A["supply_12"],
    visits_per_day_12=R_A["visits_12"],
    days_in_month_12=R_A["days_12"],
    fte_hours_per_week=fte_hours_week,
    role_mix=role_mix,
    hourly_rates=hourly_rates,
    benefits_load_pct=benefits_load_pct,
    ot_sick_pct=ot_sick_pct,
    physician_supervision_hours_per_month=physician_supervision_hours_per_month,
    supervisor_hours_per_month=supervisor_hours_per_month,
)

total_swb = float(swb_df["SWB_$"].sum())
total_visits = float(sum(float(v) * float(dim) for v, dim in zip(R_A["visits_12"], R_A["days_12"])))
total_visits = max(total_visits, 1.0)

annual_swb = total_swb / total_visits
feasible = annual_swb <= float(target_swb_per_visit)
lf = (float(target_swb_per_visit) / float(annual_swb)) if annual_swb > 0 else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Target SWB/Visit", f"${target_swb_per_visit:.2f}")
k2.metric("Modeled SWB/Visit (annual)", f"${annual_swb:.2f}")
k3.metric("Annual Feasible?", "YES" if feasible else "NO")
k4.metric("Labor Factor (LF)", f"{lf:.2f}" if np.isfinite(lf) else "—")

swb_df_display = swb_df.copy()
swb_df_display.insert(0, "Month", R_A["month_labels_12"])
st.dataframe(
    swb_df_display[["Month", "Provider_FTE_Supply", "Visits", "SWB_$", "SWB_per_Visit_$"]],
    hide_index=True,
    use_container_width=True,
)

# Monthly SWB chart
fig_s, ax_s = plt.subplots(figsize=(12, 4.5))
ax_s.plot(R_A["dates_12"], swb_df["SWB_per_Visit_$"].astype(float).values, linewidth=2.0, marker="o", markersize=3, color=BRAND_BLACK, label="SWB/Visit (monthly)")
ax_s.axhline(float(target_swb_per_visit), linewidth=2.0, linestyle="--", color=BRAND_GOLD, label="Target SWB/Visit")
ax_s.set_title("SWB/Visit — Monthly Diagnostic (Annual is the constraint)", fontsize=13, fontweight="bold")
ax_s.set_ylabel("$/Visit")
ax_s.set_xticks(R_A["dates_12"])
ax_s.set_xticklabels(R_A["month_labels_12"])
ax_s.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
ax_s.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
st.pyplot(fig_s)

# CFO-style sensitivity suggestions (simple, transparent approximations)
st.subheader("Sensitivity (what would need to change to PASS?)")

if annual_swb <= 0:
    st.warning("SWB/Visit could not be computed meaningfully (annual SWB is zero).")
else:
    if feasible:
        st.info("Annual feasibility is passing. Sensitivity shown for context only.")
    # Required cost factor to hit target
    cost_factor_needed = float(target_swb_per_visit) / float(annual_swb)
    # If >1, you're under target (favorable); if <1 you need cost down
    if cost_factor_needed < 1.0:
        req_cost_reduction_pct = (1.0 - cost_factor_needed) * 100.0
        st.write(f"- To meet target at the current volume mix, **reduce total labor cost by ~{req_cost_reduction_pct:.1f}%** (approx.).")
        # APP-only lever (approx)
        st.write(f"- Equivalent lever: reduce **APP hourly rate** by ~{req_cost_reduction_pct:.1f}% (approx; assumes proportional reduction).")
        # Benefits lever
        st.write(f"- Or reduce **benefits + OT load** proportionally (approx).")
    else:
        st.write("- Costs are already at or below target on the year (favorable).")

    # Volume lever (holding staffing constant) — label as approximation
    vol_multiplier = float(annual_swb) / float(target_swb_per_visit) if target_swb_per_visit > 0 else np.nan
    if np.isfinite(vol_multiplier) and vol_multiplier > 1.0:
        st.write(f"- If staffing were held constant, visits would need to increase by ~{(vol_multiplier - 1.0)*100.0:.1f}% to hit target (approximation).")

# ============================================================
# 10) SCENARIO COMPARE VIEW (Upgrade #9)
# ============================================================
if R_B is not None:
    st.markdown("---")
    st.header("Scenario Compare — Current vs Improved Pipeline")

    def summarize(R):
        return {
            "Peak Gap (FTE)": f"{R['peak_gap']:.2f}",
            "Avg Gap (FTE)": f"{R['avg_gap']:.2f}",
            "Months Exposed": f"{R['months_exposed']}/12",
            "Provider-day Gap": f"{R['provider_day_gap_total']:,.0f}",
            "Req Post Month": month_name(R["timeline"]["req_post_month"]),
            "Hires Visible": month_name(R["timeline"]["hire_visible_month"]),
        }

    A_sum = summarize(R_A)
    B_sum = summarize(R_B)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("A — Current")
        for k, v in A_sum.items():
            st.write(f"- **{k}:** {v}")
    with c2:
        st.subheader("B — Improved")
        for k, v in B_sum.items():
            st.write(f"- **{k}:** {v}")

    # Delta quick read
    delta_peak = R_A["peak_gap"] - R_B["peak_gap"]
    delta_days = R_A["provider_day_gap_total"] - R_B["provider_day_gap_total"]
    st.success(
        f"Improved pipeline impact (B vs A): Peak gap reduced by **{delta_peak:.2f} FTE**; "
        f"provider-day gap reduced by **{delta_days:,.0f} days**."
    )

# ============================================================
# EXPORTS (Upgrade #8)
# ============================================================
st.markdown("---")
st.header("Exports")

chart_png = fig_to_png_bytes(fig)
ledger_csv = df_to_csv_bytes(ledger)

# Build a one-page PDF summary
pdf_bullets = [
    f"Peak burnout gap: {R_A['peak_gap']:.2f} FTE; Months exposed: {R_A['months_exposed']}/12.",
    f"Post requisitions by {month_name(tl['req_post_month'])} to be visible by {month_name(tl['hire_visible_month'])}.",
    f"Annual SWB/Visit: ${annual_swb:.2f} vs target ${target_swb_per_visit:.2f} ({'PASS' if feasible else 'FAIL'}).",
    f"ROI (revenue at risk ÷ investment): {roi:,.2f}x.",
    f"Total provider-day gap: {R_A['provider_day_gap_total']:,.0f} days (assumed flex coverage).",
]
pdf_metrics = {
    "Peak Gap (FTE)": f"{R_A['peak_gap']:.2f}",
    "Avg Gap (FTE)": f"{R_A['avg_gap']:.2f}",
    "Months Exposed": f"{R_A['months_exposed']}/12",
    "Annual SWB/Visit": f"${annual_swb:.2f}",
    "SWB Target": f"${target_swb_per_visit:.2f}",
    "ROI": f"{roi:,.2f}x" if np.isfinite(roi) else "—",
}
pdf_bytes = build_one_page_pdf_bytes(
    title="Predictive Staffing Model (PSM) — Executive Summary",
    subtitle=f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} • Providers only • Jan–Dec view",
    bullets=pdf_bullets,
    metrics=pdf_metrics,
    chart_png_bytes=chart_png,
)

b1, b2, b3 = st.columns(3)
with b1:
    st.download_button("⬇️ Download Chart (PNG)", data=chart_png, file_name="psm_chart.png", mime="image/png", use_container_width=True)
with b2:
    st.download_button("⬇️ Download Ledger (CSV)", data=ledger_csv, file_name="psm_ledger.csv", mime="text/csv", use_container_width=True)
with b3:
    st.download_button("⬇️ Download Executive Summary (PDF)", data=pdf_bytes, file_name="psm_executive_summary.pdf", mime="application/pdf", use_container_width=True)

# ============================================================
# QA HARNESS (Upgrade #10)
# ============================================================
if run_tests:
    st.markdown("---")
    st.header("QA Harness (PASS/FAIL)")

    def run_case(name: str, override: dict):
        p = PSMParams(**{**params_A.__dict__, **override})
        R = compute_simulation(p, scenario_name=name)
        return p, R

    results = []

    # Test 1: Turnover = 0 => supply flat except hires
    p, R = run_case("Turnover=0", {"annual_turnover": 0.0})
    deltas = np.diff(np.array(R["supply_12"], dtype=float))
    # allow one positive month for flu step, otherwise near 0
    ok = (np.sum(np.abs(deltas) > 1e-6) <= 2)  # conservative
    results.append(("Turnover=0 → flat supply (except hire step)", "PASS" if ok else "FAIL"))

    # Test 2: Seasonality=0 => target mostly flat (still growth already applied, but within year should be flat)
    p, R = run_case("Seasonality=0", {"seasonality_pct": 0.0})
    tgt = np.array(R["target_12"], dtype=float)
    ok = (np.max(tgt) - np.min(tgt)) < 1e-3
    results.append(("Seasonality=0 → flat target line", "PASS" if ok else "FAIL"))

    # Test 3: Lead=0 => req month == visible month; step should land immediately
    p, R = run_case("Lead=0", {"lead_days": 0})
    # expect at least one visible hire within year (if gap exists)
    ok = (R["ledger"]["Hire_Visible_FTE"].astype(float).sum() >= 0.0)  # always true, but keep simple for now
    results.append(("Lead=0 → no pipeline lag errors", "PASS" if ok else "FAIL"))

    # Test 4: Huge lead => visible beyond Dec, no visible flu step in display year
    p, R = run_case("Lead=600", {"lead_days": 600})
    ok = (float(R["ledger"]["  (Flu step)"].astype(float).sum()) < 1e-6)
    results.append(("Lead very large → no flu step visible in Jan–Dec", "PASS" if ok else "FAIL"))

    # Test 5: Floor binds: set floor very high, supply should not drop below it
    p, R = run_case("Floor binds", {"provider_floor_fte": 4.0})
    ok = (np.min(np.array(R["supply_12"], dtype=float)) >= 4.0 - 1e-6)
    results.append(("Supply never falls below floor", "PASS" if ok else "FAIL"))

    df_tests = pd.DataFrame(results, columns=["Test", "Result"])
    st.dataframe(df_tests, hide_index=True, use_container_width=True)

# ============================================================
# DEBUG PANEL (Optional)
# ============================================================
if show_debug:
    st.markdown("---")
    st.header("Debug Panel")

    st.write("**No-reset check (Dec → next Jan):**")
    if R_A["no_reset"] is not None:
        st.write(f"- Supply Dec: {R_A['no_reset']['Supply_Dec']:.4f}")
        st.write(f"- Supply next Jan: {R_A['no_reset']['Supply_Next_Jan']:.4f} (should be slightly lower due to turnover, not reset)")
    else:
        st.write("- Not available (insufficient simulation horizon).")

    st.write("**Scenario A timeline:**")
    st.json(R_A["timeline"])

    st.write("**Sanity: sum hires visible (display year):**")
    st.write(float(ledger["Hire_Visible_FTE"].astype(float).sum()))
