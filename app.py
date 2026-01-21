# app.py — Predictive Staffing Model (PSM) — Client-Grade (Providers Only)
# Capacity-aware demand logic + coverage realism + ramp + flu uplift window + finance alignment
# Jan–Dec display • Continuous simulation (no year reset) • Audit-ready outputs

import math
import io
from dataclasses import dataclass
from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages

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
      .note { background: #f3f7ff; border: 1px solid #cfe0ff; border-radius: 10px; padding: 12px 14px; }
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

DISPLAY_START = 12  # Jan of display year (Year 1)
DISPLAY_END = 23    # Dec of display year (Year 1)

# ============================================================
# HELPERS
# ============================================================
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

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def build_one_page_pdf_bytes_matplotlib(
    title: str,
    subtitle: str,
    bullets: list[str],
    metrics: dict[str, str],
    chart_fig,
) -> bytes:
    """Creates a 1-page PDF using Matplotlib only."""
    pdf_buf = io.BytesIO()
    page_fig = plt.figure(figsize=(8.5, 11))  # Letter portrait
    page_fig.patch.set_facecolor("white")

    page_fig.text(0.06, 0.965, title, fontsize=16, fontweight="bold", va="top")
    page_fig.text(0.06, 0.942, subtitle, fontsize=10.5, color="#444444", va="top")

    y = 0.90
    page_fig.text(0.06, y, "Key metrics", fontsize=11, fontweight="bold", va="top")
    y -= 0.02
    for k, v in metrics.items():
        page_fig.text(0.075, y, f"• {k}: {v}", fontsize=10, va="top")
        y -= 0.018

    y -= 0.01
    page_fig.text(0.06, y, "Executive summary", fontsize=11, fontweight="bold", va="top")
    y -= 0.02
    for b in bullets[:10]:
        page_fig.text(0.075, y, f"• {b}", fontsize=10, va="top")
        y -= 0.018

    chart_png = fig_to_png_bytes(chart_fig)
    img = plt.imread(io.BytesIO(chart_png))
    ax_img = page_fig.add_axes([0.06, 0.08, 0.88, 0.38])
    ax_img.imshow(img)
    ax_img.axis("off")

    with PdfPages(pdf_buf) as pdf:
        pdf.savefig(page_fig, bbox_inches="tight")

    plt.close(page_fig)
    pdf_buf.seek(0)
    return pdf_buf.read()

def provider_day_gap(target_curve, supply_curve, days_in_month):
    return float(sum(max(float(t) - float(s), 0.0) * float(dim) for t, s, dim in zip(target_curve, supply_curve, days_in_month)))

def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    return float(sum(float(df) * float(loaded_cost_per_provider_fte) * (float(dim) / 365.0) for df, dim in zip(delta_fte_curve, days_in_month)))

def monthly_hours_from_fte(fte: float, fte_hours_per_week: float, days_in_month: int) -> float:
    return float(fte) * float(fte_hours_per_week) * (float(days_in_month) / 7.0)

def loaded_hourly_rate(base_hourly: float, benefits_load_pct: float, ot_sick_pct: float, bonus_pct: float) -> float:
    # bonus modeled as % of base
    return float(base_hourly) * (1.0 + float(bonus_pct)) * (1.0 + float(benefits_load_pct)) * (1.0 + float(ot_sick_pct))

def compute_role_mix_ratios(visits_per_day: float, hours_week: float, fte_hours_week: float):
    """
    Stable ratios: lock to the baseline volume level rather than month-to-month noise.
    """
    return model.get_role_mix_ratios(float(visits_per_day))

def compute_monthly_swb_per_visit_fte_based(
    provider_supply_12,
    visits_per_day_12,
    days_in_month_12,
    fte_hours_per_week,
    role_mix,
    hourly_rates,
    benefits_load_pct,
    ot_sick_pct,
    bonus_pct,
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

        apc_rate = loaded_hourly_rate(hourly_rates["apc"], benefits_load_pct, ot_sick_pct, bonus_pct)
        psr_rate = loaded_hourly_rate(hourly_rates["psr"], benefits_load_pct, ot_sick_pct, bonus_pct)
        ma_rate = loaded_hourly_rate(hourly_rates["ma"], benefits_load_pct, ot_sick_pct, bonus_pct)
        rt_rate = loaded_hourly_rate(hourly_rates["rt"], benefits_load_pct, ot_sick_pct, bonus_pct)

        phys_rate = loaded_hourly_rate(hourly_rates["physician"], benefits_load_pct, ot_sick_pct, bonus_pct)
        sup_rate = loaded_hourly_rate(hourly_rates["supervisor"], benefits_load_pct, ot_sick_pct, bonus_pct)

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

# ----------------------------
# Capacity-aware provider target logic
# ----------------------------
def compute_provider_target_fte(
    visits_per_day: float,
    hours_week: float,
    fte_hours_week: float,
    productivity_pct: float,
    days_open_per_week: float,
    capacity_mode: str,
    max_pts_per_provider_day: float,
    pts_per_provider_hour: float,
    min_concurrent_providers: float,
    pct_hours_two_providers: float,
) -> float:
    """
    Target Provider FTE = Coverage need (open hours & concurrency, adjusted for productivity)
                       × Demand multiplier (only when demand > capacity)
    """
    prod = max(float(productivity_pct), 0.50)
    days_open = max(float(days_open_per_week), 1.0)
    hours_per_day = float(hours_week) / days_open

    # concurrency:
    # avg_concurrent = max(min_concurrent, 1 + pct_hours_two_providers)
    avg_concurrent = max(float(min_concurrent_providers), 1.0 + float(pct_hours_two_providers))

    # coverage (1 provider seat for open hours) adjusted for productivity + concurrency
    coverage_fte = (float(hours_week) / (max(float(fte_hours_week), 1.0) * prod)) * avg_concurrent

    # capacity threshold
    if capacity_mode == "Patients per hour":
        cap_day = max(float(pts_per_provider_hour), 0.5) * max(hours_per_day, 1.0)
    else:
        cap_day = max(float(max_pts_per_provider_day), 1.0)

    demand_multiplier = max(1.0, float(visits_per_day) / cap_day)
    return float(coverage_fte) * float(demand_multiplier)

def rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        j0 = max(0, i - window + 1)
        out.append(float(np.mean(values[j0:i+1])))
    return out

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

def apply_flu_uplift(visits_curve: list[float], months: list[int], flu_months: set[int], flu_uplift_pct: float) -> list[float]:
    out = []
    for v, m in zip(visits_curve, months):
        if int(m) in flu_months:
            out.append(float(v) * (1.0 + float(flu_uplift_pct)))
        else:
            out.append(float(v))
    return out

def find_month_index_in_range(months: list[int], target_month: int, start_i: int, end_i: int):
    for i in range(start_i, end_i + 1):
        if int(months[i]) == int(target_month):
            return i
    return None

# ============================================================
# PARAMS + CORE SIMULATION
# ============================================================
@dataclass
class PSMParams:
    # Demand
    visits: float
    hours_week: float
    fte_hours_week: float
    days_open_per_week: float

    # Capacity & productivity
    capacity_mode: str
    max_patients_per_provider_day: float
    patients_per_provider_hour: float
    productivity_pct: float
    peak_factor: float
    demand_smoothing_months: int

    # Coverage concurrency realism
    provider_floor_fte: float
    min_concurrent_providers: float
    pct_hours_two_providers: float

    # Seasonality + flu
    seasonality_pct: float
    flu_uplift_pct: float
    flu_months: set[int]

    # Workforce
    annual_turnover: float
    annual_growth: float

    lead_days: int
    ramp_months: int
    ramp_productivity: float
    fill_probability: float

    starting_supply_fte: float

    # Hiring strategy rules
    ready_month: int = 11
    flu_anchor_month: int = 12
    hire_step_cap_fte: float = 1.25
    allow_floor_maintenance_pipeline: bool = True
    freeze_except_flu_and_floor: bool = True

def compute_simulation(params: PSMParams, scenario_name: str = "Current"):
    """
    Continuous simulation:
    - 37 months: 12 warm-up + 12 display + 12 forward + 1 no-reset check
    - Flu-cycle step hire applied in BOTH years so Jan display inherits prior Nov
    - Ramp reduces effective supply for first N months after visible
    - Capacity-aware target: coverage + demand multiplier (only above capacity)
    """
    today = datetime.today()
    year0 = today.year

    N = 37
    dates = pd.date_range(start=datetime(year0, 1, 1), periods=N, freq="MS")
    months = [int(d.month) for d in dates]
    days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

    lead_months = lead_days_to_months(int(params.lead_days))
    monthly_turnover = float(params.annual_turnover) / 12.0

    base_year0 = float(params.visits)
    base_year1 = float(params.visits) * (1.0 + float(params.annual_growth))

    visits_curve_base = compute_visits_curve(months, base_year0, base_year1, float(params.seasonality_pct))
    visits_curve_flu = apply_flu_uplift(visits_curve_base, months, set(params.flu_months), float(params.flu_uplift_pct))

    # peak-to-average factor
    effective_visits_curve = [float(v) * float(params.peak_factor) for v in visits_curve_flu]

    # raw target (capacity-aware)
    raw_target = []
    for v in effective_visits_curve:
        raw_target.append(
            compute_provider_target_fte(
                visits_per_day=float(v),
                hours_week=params.hours_week,
                fte_hours_week=params.fte_hours_week,
                productivity_pct=params.productivity_pct,
                days_open_per_week=params.days_open_per_week,
                capacity_mode=params.capacity_mode,
                max_pts_per_provider_day=params.max_patients_per_provider_day,
                pts_per_provider_hour=params.patients_per_provider_hour,
                min_concurrent_providers=params.min_concurrent_providers,
                pct_hours_two_providers=params.pct_hours_two_providers,
            )
        )

    window = max(int(params.demand_smoothing_months), 1)
    smoothed_target = rolling_mean(raw_target, window)
    target_curve = [max(float(t), float(params.provider_floor_fte)) for t in smoothed_target]

    baseline_provider_fte = max(
        compute_provider_target_fte(
            visits_per_day=float(base_year1) * float(params.peak_factor),
            hours_week=params.hours_week,
            fte_hours_week=params.fte_hours_week,
            productivity_pct=params.productivity_pct,
            days_open_per_week=params.days_open_per_week,
            capacity_mode=params.capacity_mode,
            max_pts_per_provider_day=params.max_patients_per_provider_day,
            pts_per_provider_hour=params.patients_per_provider_hour,
            min_concurrent_providers=params.min_concurrent_providers,
            pct_hours_two_providers=params.pct_hours_two_providers,
        ),
        float(params.provider_floor_fte),
    )

    req_post_month = wrap_month(params.ready_month - lead_months)
    hire_visible_month = wrap_month(req_post_month + lead_months)

    # Anchor month target for flu sizing (default December in display year)
    anchor_month = int(params.flu_anchor_month)
    anchor_idx_display = find_month_index_in_range(months, anchor_month, DISPLAY_START, DISPLAY_END)
    if anchor_idx_display is None:
        anchor_idx_display = DISPLAY_START + 11
    flu_target_fte = max(float(target_curve[anchor_idx_display]), float(params.provider_floor_fte))

    hires_visible = [0.0] * N
    hires_reason = [""] * N
    planned_floor_hires_visible = [0.0] * N
    planned_floor_reason = [""] * N

    def can_post_req(month_num: int, is_floor_maintenance: bool) -> bool:
        if not params.freeze_except_flu_and_floor:
            return True
        if is_floor_maintenance and params.allow_floor_maintenance_pipeline:
            return True
        return int(month_num) == int(req_post_month)

    def apply_fill(fte: float) -> float:
        return float(fte) * max(min(float(params.fill_probability), 1.0), 0.0)

    # YEAR 0 flu step (carry into display)
    req_idx_y0 = find_month_index_in_range(months, req_post_month, 0, 11)
    if req_idx_y0 is not None:
        vis_idx_y0 = req_idx_y0 + lead_months
        if 0 <= vis_idx_y0 <= 11:
            loss_factor = (1.0 - monthly_turnover) ** float(lead_months) if lead_months > 0 else 1.0
            expected_at_ready = float(params.starting_supply_fte) * float(loss_factor)
            fte_needed = max(float(flu_target_fte) - float(expected_at_ready), 0.0)
            step = min(fte_needed, float(params.hire_step_cap_fte)) if params.hire_step_cap_fte > 0 else fte_needed
            step = apply_fill(step)
            if step > 1e-6:
                hires_visible[vis_idx_y0] += float(step)
                hires_reason[vis_idx_y0] = f"Flu step (Y0 carryover) — filled @ {params.fill_probability*100:.0f}%"

    # YEAR 1 flu step
    req_idx_y1 = find_month_index_in_range(months, req_post_month, DISPLAY_START, DISPLAY_END)
    if req_idx_y1 is None:
        req_idx_y1 = find_month_index_in_range(months, req_post_month, 0, N - 1)
    vis_idx_y1 = (req_idx_y1 + lead_months) if req_idx_y1 is not None else None

    # Cohort simulation: paid supply vs effective supply (ramp)
    cohorts: list[dict] = [{"fte": max(float(params.starting_supply_fte), float(params.provider_floor_fte)), "age": 9999}]

    def ramp_factor(age_months: int) -> float:
        rm = max(int(params.ramp_months), 0)
        if rm <= 0:
            return 1.0
        if age_months < rm:
            return max(min(float(params.ramp_productivity), 1.0), 0.1)
        return 1.0

    supply_paid = [0.0] * N
    supply_effective = [0.0] * N

    for t in range(0, N):
        # floor maintenance planning (lead-time aware)
        if params.allow_floor_maintenance_pipeline and lead_months > 0 and t > 0:
            v = t + lead_months
            if v < N:
                now_paid = float(supply_paid[t - 1]) if t - 1 >= 0 else float(params.starting_supply_fte)
                if now_paid <= float(params.provider_floor_fte) + 0.15:
                    proj = float(now_paid) * ((1.0 - monthly_turnover) ** float(lead_months))
                    proj += float(hires_visible[v]) + float(planned_floor_hires_visible[v])
                    if proj < float(params.provider_floor_fte) - 1e-6:
                        need = float(params.provider_floor_fte) - proj
                        if can_post_req(months[t], is_floor_maintenance=True):
                            planned_floor_hires_visible[v] += need
                            planned_floor_reason[v] = "Floor replacement pipeline"

        # attrition
        for c in cohorts:
            c["fte"] = max(c["fte"] * (1.0 - monthly_turnover), 0.0)

        # apply visible hires (existing scheduled)
        add_fte = float(hires_visible[t]) + float(planned_floor_hires_visible[t])
        if add_fte > 1e-8:
            cohorts.append({"fte": add_fte, "age": 0})

        # age cohorts
        for c in cohorts:
            c["age"] = int(c["age"]) + 1

        paid = max(sum(float(c["fte"]) for c in cohorts), float(params.provider_floor_fte))
        supply_paid[t] = float(paid)

        eff = 0.0
        for c in cohorts:
            eff += float(c["fte"]) * ramp_factor(int(c["age"]))
        eff = max(eff, float(params.provider_floor_fte))
        supply_effective[t] = float(eff)

    # size YEAR1 flu step based on projected paid supply at visible month
    if vis_idx_y1 is not None and 0 <= int(vis_idx_y1) < N:
        idx = int(vis_idx_y1)
        expected_at_ready = float(supply_paid[idx])
        fte_needed = max(float(flu_target_fte) - float(expected_at_ready), 0.0)
        step = min(fte_needed, float(params.hire_step_cap_fte)) if params.hire_step_cap_fte > 0 else fte_needed
        step = apply_fill(step)
        if step > 1e-6:
            hires_visible[idx] += float(step)
            hires_reason[idx] = f"Flu step (Y1) — filled @ {params.fill_probability*100:.0f}%"

            # Re-run from scratch for determinism
            cohorts = [{"fte": max(float(params.starting_supply_fte), float(params.provider_floor_fte)), "age": 9999}]
            for t in range(0, N):
                for c in cohorts:
                    c["fte"] = max(c["fte"] * (1.0 - monthly_turnover), 0.0)
                add_fte = float(hires_visible[t]) + float(planned_floor_hires_visible[t])
                if add_fte > 1e-8:
                    cohorts.append({"fte": add_fte, "age": 0})
                for c in cohorts:
                    c["age"] = int(c["age"]) + 1
                paid = max(sum(float(c["fte"]) for c in cohorts), float(params.provider_floor_fte))
                supply_paid[t] = float(paid)
                eff = 0.0
                for c in cohorts:
                    eff += float(c["fte"]) * ramp_factor(int(c["age"]))
                eff = max(eff, float(params.provider_floor_fte))
                supply_effective[t] = float(eff)

    # display slices
    idx12 = list(range(DISPLAY_START, DISPLAY_END + 1))
    dates_12 = [dates[i] for i in idx12]
    month_labels_12 = [d.strftime("%b") for d in dates_12]
    days_12 = [days_in_month[i] for i in idx12]

    visits_12 = [float(visits_curve_flu[i]) for i in idx12]
    visits_eff_12 = [float(effective_visits_curve[i]) for i in idx12]
    target_12 = [float(target_curve[i]) for i in idx12]
    supply_paid_12 = [float(supply_paid[i]) for i in idx12]
    supply_eff_12 = [float(supply_effective[i]) for i in idx12]

    hires_flu_12 = [float(hires_visible[i]) for i in idx12]
    hires_floor_12 = [float(planned_floor_hires_visible[i]) for i in idx12]
    hires_12 = [float(hires_visible[i] + planned_floor_hires_visible[i]) for i in idx12]
    reason_12 = [
        hires_reason[i] if hires_reason[i] else (planned_floor_reason[i] if planned_floor_reason[i] else "")
        for i in idx12
    ]

    # gap uses effective supply
    gap_12 = [max(t - s, 0.0) for t, s in zip(target_12, supply_eff_12)]
    months_exposed = int(sum(1 for g in gap_12 if g > 1e-6))
    peak_gap = float(max(gap_12)) if gap_12 else 0.0
    avg_gap = float(np.mean(gap_12)) if gap_12 else 0.0

    flex_fte_12 = gap_12[:]
    provider_day_gap_total = provider_day_gap(target_12, supply_eff_12, days_12)

    start_paid_12 = [supply_paid_12[0]] + supply_paid_12[:-1]
    turnover_shed_12 = [-(start_paid_12[i] * monthly_turnover) for i in range(12)]
    end_paid_12 = supply_paid_12
    end_eff_12 = supply_eff_12

    ledger = pd.DataFrame({
        "Month": month_labels_12,
        "Visits/Day (avg)": np.round(visits_12, 1),
        "Visits/Day (peak adj)": np.round(visits_eff_12, 1),
        "Start_FTE (Paid)": np.round(start_paid_12, 3),
        "Turnover_Shed_FTE": np.round(turnover_shed_12, 3),
        "Hire_Visible_FTE": np.round(hires_12, 3),
        "  (Flu step)": np.round(hires_flu_12, 3),
        "  (Floor maint)": np.round(hires_floor_12, 3),
        "Hire_Reason": reason_12,
        "End_FTE (Paid)": np.round(end_paid_12, 3),
        "End_FTE (Effective)": np.round(end_eff_12, 3),
        "Target_FTE": np.round(target_12, 3),
        "Gap_FTE (Target - Effective)": np.round(gap_12, 3),
        "Flex_FTE_Req": np.round(flex_fte_12, 3),
    })

    dec_idx = DISPLAY_END
    next_jan_idx = DISPLAY_END + 1
    no_reset = None
    if next_jan_idx < N:
        no_reset = {
            "Paid_Dec": float(supply_paid[dec_idx]),
            "Paid_Next_Jan": float(supply_paid[next_jan_idx]),
            "Effective_Dec": float(supply_effective[dec_idx]),
            "Effective_Next_Jan": float(supply_effective[next_jan_idx]),
        }

    timeline = {
        "lead_months": lead_months,
        "monthly_turnover": monthly_turnover,
        "req_post_month": req_post_month,
        "hire_visible_month": hire_visible_month,
        "ready_month": params.ready_month,
        "flu_anchor_month": params.flu_anchor_month,
        "flu_target_fte": float(flu_target_fte),
        "scenario": scenario_name,
    }

    return dict(
        dates_12=dates_12,
        month_labels_12=month_labels_12,
        days_12=days_12,
        visits_12=visits_12,
        visits_eff_12=visits_eff_12,
        target_12=target_12,
        supply_paid_12=supply_paid_12,
        supply_eff_12=supply_eff_12,
        gap_12=gap_12,
        flex_fte_12=flex_fte_12,
        months_exposed=months_exposed,
        peak_gap=peak_gap,
        avg_gap=avg_gap,
        provider_day_gap_total=provider_day_gap_total,
        baseline_provider_fte=baseline_provider_fte,
        ledger=ledger,
        timeline=timeline,
        no_reset=no_reset,
        dates_full=list(dates),
    )

# ============================================================
# SIDEBAR — INPUTS + OPTIONS (All with help tooltips)
# ============================================================
with st.sidebar:
    st.header("Inputs")

    visits = st.number_input(
        "Avg Visits/Day",
        min_value=1.0, value=36.0, step=1.0,
        help="Average daily visits across the year (before seasonality/flu)."
    )
    hours_week = st.number_input(
        "Hours of Operation / Week",
        min_value=1.0, value=84.0, step=1.0,
        help="Total clinic open hours per week. Drives minimum coverage need."
    )
    days_open_per_week = st.number_input(
        "Days Open / Week",
        min_value=1.0, max_value=7.0, value=7.0, step=1.0,
        help="Used to convert weekly hours into hours/day for capacity math when using patients-per-hour mode."
    )
    fte_hours_week = st.number_input(
        "FTE Hours / Week",
        min_value=1.0, value=36.0, step=1.0,
        help="Paid hours per FTE per week (e.g., 36). Used for coverage and cost."
    )

    st.subheader("Capacity + Productivity")
    capacity_mode = st.radio(
        "Capacity mode",
        options=["Patients per day", "Patients per hour"],
        index=0,
        help="Day-based is simple; hour-based scales with hours/day."
    )
    max_patients_per_provider_day = st.number_input(
        "Max Patients / Provider / Day",
        min_value=10.0, value=36.0, step=1.0,
        help="Demand multiplier only increases when peak-adjusted visits exceed this value."
    )
    patients_per_provider_hour = st.number_input(
        "Patients / Provider / Hour",
        min_value=0.5, value=3.0, step=0.1,
        help="If using hour mode, capacity/day = pts/hr × hours/day."
    )
    productivity_pct = st.slider(
        "Clinical productivity %",
        min_value=50, max_value=100, value=88, step=1,
        help="Clinical time as % of paid time (charting/inbox reduces this). Lower increases required coverage."
    ) / 100.0
    peak_factor = st.slider(
        "Peak-to-average factor",
        min_value=1.00, max_value=1.50, value=1.20, step=0.01,
        help="Effective visits = avg × peak factor (captures surge days within a month)."
    )
    demand_smoothing_months = st.slider(
        "Demand smoothing (months)",
        min_value=1, max_value=6, value=3, step=1,
        help="Trailing rolling mean applied to target FTE to reduce jitter."
    )

    st.subheader("Coverage realism (concurrency)")
    provider_floor_fte = st.number_input(
        "Provider Minimum Floor (FTE)",
        min_value=0.25, value=1.0, step=0.25,
        help="Hard floor for target and predicted supply."
    )
    min_concurrent_providers = st.number_input(
        "Minimum concurrent providers",
        min_value=1.0, value=1.0, step=1.0,
        help="Minimum providers that must be present during open hours."
    )
    pct_hours_two_providers = st.slider(
        "% of hours needing a 2nd provider",
        min_value=0, max_value=80, value=0, step=5,
        help="If some hours require 2 providers (peak blocks), this increases coverage FTE even if visits/day is below capacity."
    ) / 100.0

    st.subheader("Seasonality + Flu window")
    seasonality_pct = st.number_input(
        "Seasonality % Lift/Drop",
        min_value=0.0, value=20.0, step=5.0,
        help="Winter uplift (Dec–Feb) and Summer drop (Jun–Aug)."
    ) / 100.0
    flu_uplift_pct = st.number_input(
        "Flu uplift % (selected months)",
        min_value=0.0, value=0.0, step=5.0,
        help="Additional uplift applied only to selected flu months (separate from winter seasonality)."
    ) / 100.0
    flu_months = st.multiselect(
        "Flu months",
        options=[("Jan",1),("Feb",2),("Mar",3),("Apr",4),("May",5),("Jun",6),("Jul",7),("Aug",8),("Sep",9),("Oct",10),("Nov",11),("Dec",12)],
        default=[("Oct",10),("Nov",11),("Dec",12),("Jan",1),("Feb",2)],
        help="Months to apply flu uplift."
    )
    flu_months_set = {m for _, m in flu_months} if flu_months else set()

    st.subheader("Workforce + Pipeline")
    annual_turnover = st.number_input(
        "Annual Turnover %",
        min_value=0.0, value=16.0, step=1.0,
        help="Annual attrition; converted to monthly in simulation."
    ) / 100.0
    annual_growth = st.number_input(
        "Annual Visit Growth % (applied to display year)",
        min_value=0.0, value=10.0, step=1.0,
        help="Growth applied to the display year baseline volume."
    ) / 100.0
    lead_days = st.number_input(
        "Days to Independent (Req→Independent)",
        min_value=0, value=210, step=10,
        help="Req-to-independent lead time."
    )
    ramp_months = st.slider(
        "Ramp months after 'visible'",
        min_value=0, max_value=6, value=1, step=1,
        help="Effective supply is reduced for this many months after a hire becomes visible/independent."
    )
    ramp_productivity = st.slider(
        "Ramp productivity %",
        min_value=30, max_value=100, value=75, step=5,
        help="Effective contribution during ramp months."
    ) / 100.0
    fill_probability = st.slider(
        "Fill probability %",
        min_value=0, max_value=100, value=85, step=5,
        help="Not every requisition yields an independent provider. Applied to flu-step hires."
    ) / 100.0
    hire_step_cap_fte = st.number_input(
        "Hire Step Cap (FTE)",
        min_value=0.0, value=1.25, step=0.25,
        help="Caps visible flu-step hires in a single month."
    )

    st.divider()
    allow_floor_maintenance_pipeline = st.checkbox(
        "Maintain floor via replacement pipeline (lead-time aware)",
        value=True,
        help="Schedules replacement hires to prevent supply dropping below the floor due to attrition."
    )
    freeze_except_flu_and_floor = st.checkbox(
        "Freeze hiring except flu req month (and floor maintenance)",
        value=True,
        help="Only posts reqs in the flu planning month unless needed to maintain floor."
    )

    st.divider()
    use_calculated_baseline = st.checkbox(
        "Use calculated baseline as starting supply",
        value=True,
        help="If checked, starting supply = model baseline for the display year."
    )

    # Finance
    st.divider()
    st.header("Finance Inputs")
    net_revenue_per_visit = st.number_input(
        "Net Revenue per Visit (NRPV)",
        min_value=0.0, value=140.0, step=5.0,
        help="Used to estimate revenue at risk from access gaps."
    )
    target_swb_per_visit = st.number_input(
        "Target SWB/Visit",
        min_value=0.0, value=85.0, step=1.0,
        help="Annual SWB/Visit affordability threshold."
    )
    visits_lost_per_provider_day_gap = st.number_input(
        "Visits Lost per 1.0 Provider-Day Gap",
        min_value=0.0, value=18.0, step=1.0,
        help="Conversion from provider-day gap to visits lost."
    )

    st.subheader("Provider comp model (fully loaded)")
    benefits_load_pct = st.number_input(
        "Benefits Load %",
        min_value=0.0, value=30.0, step=1.0,
        help="Employer benefits load as % of base pay."
    ) / 100.0
    bonus_pct = st.number_input(
        "Bonus % of base",
        min_value=0.0, value=10.0, step=1.0,
        help="Bonus as % of base pay (default 10%)."
    ) / 100.0
    ot_sick_pct = st.number_input(
        "OT + Sick/PTO %",
        min_value=0.0, value=4.0, step=0.5,
        help="Premium/coverage factor for OT, sick, PTO, etc."
    ) / 100.0

    physician_hr = st.number_input("Physician (optional) $/hr", min_value=0.0, value=135.79, step=1.0,
                                   help="Optional supervision physician rate for SWB calculation.")
    apc_hr = st.number_input("APP $/hr", min_value=0.0, value=62.0, step=1.0,
                             help="APP base hourly rate (used for loaded cost + SWB).")
    ma_hr = st.number_input("MA $/hr", min_value=0.0, value=24.14, step=0.5, help="MA base hourly rate (SWB).")
    psr_hr = st.number_input("PSR $/hr", min_value=0.0, value=21.23, step=0.5, help="PSR base hourly rate (SWB).")
    rt_hr = st.number_input("RT $/hr", min_value=0.0, value=31.36, step=0.5, help="RT base hourly rate (SWB).")
    supervisor_hr = st.number_input("Supervisor (optional) $/hr", min_value=0.0, value=28.25, step=0.5,
                                    help="Optional supervisor hourly rate (SWB).")

    physician_supervision_hours_per_month = st.number_input(
        "Physician supervision hours/month", min_value=0.0, value=0.0, step=1.0,
        help="Monthly physician supervision hours (optional)."
    )
    supervisor_hours_per_month = st.number_input(
        "Supervisor hours/month", min_value=0.0, value=0.0, step=1.0,
        help="Monthly supervisor hours (optional)."
    )

    cme_licensure_annual = st.number_input(
        "CME / licensure (annual per provider)",
        min_value=0.0, value=3000.0, step=500.0,
        help="Annual per-provider overhead added to loaded cost."
    )

    st.divider()
    st.header("Display + Tools")
    show_visits_overlay = st.checkbox("Show Visits/Day overlay", value=True, help="Overlay avg & peak-adjusted visits/day.")
    show_heatmap = st.checkbox("Show gap heatmap", value=True, help="Shows month-by-month gap severity.")
    show_debug = st.checkbox("Show debug panel", value=False, help="Shows carryover checks and internal timeline.")

    st.subheader("Scenario Compare")
    enable_compare = st.checkbox("Compare scenarios", value=True, help="Compare current vs improved pipeline.")
    improved_lead_days = st.number_input("Improved pipeline days (scenario B)", min_value=0, value=150, step=10,
                                        help="Scenario B lead time assumption.")

    st.subheader("QA Harness")
    run_tests = st.checkbox("Run test mode (PASS/FAIL)", value=False, help="Runs integrity checks.")

    st.divider()
    run = st.button("▶️ Run PSM", use_container_width=True)

if not run:
    st.info("Set inputs and click **Run PSM**.")
    st.stop()

# ============================================================
# STARTING SUPPLY + LOADED COST
# ============================================================
baseline_for_start = max(
    compute_provider_target_fte(
        visits_per_day=float(visits) * (1.0 + float(annual_growth)) * float(peak_factor),
        hours_week=float(hours_week),
        fte_hours_week=float(fte_hours_week),
        productivity_pct=float(productivity_pct),
        days_open_per_week=float(days_open_per_week),
        capacity_mode=str(capacity_mode),
        max_pts_per_provider_day=float(max_patients_per_provider_day),
        pts_per_provider_hour=float(patients_per_provider_hour),
        min_concurrent_providers=float(min_concurrent_providers),
        pct_hours_two_providers=float(pct_hours_two_providers),
    ),
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
        help="Use when current staffing differs from baseline."
    )
    starting_supply_fte = max(float(starting_supply_fte), float(provider_floor_fte))

hours_per_year = float(fte_hours_week) * 52.0
apc_loaded_hr = loaded_hourly_rate(float(apc_hr), float(benefits_load_pct), float(ot_sick_pct), float(bonus_pct))
loaded_cost_per_provider_fte = apc_loaded_hr * hours_per_year + float(cme_licensure_annual)

# ============================================================
# RUN SIMS
# ============================================================
params_A = PSMParams(
    visits=float(visits),
    hours_week=float(hours_week),
    fte_hours_week=float(fte_hours_week),
    days_open_per_week=float(days_open_per_week),
    capacity_mode=str(capacity_mode),
    max_patients_per_provider_day=float(max_patients_per_provider_day),
    patients_per_provider_hour=float(patients_per_provider_hour),
    productivity_pct=float(productivity_pct),
    peak_factor=float(peak_factor),
    demand_smoothing_months=int(demand_smoothing_months),
    provider_floor_fte=float(provider_floor_fte),
    min_concurrent_providers=float(min_concurrent_providers),
    pct_hours_two_providers=float(pct_hours_two_providers),
    seasonality_pct=float(seasonality_pct),
    flu_uplift_pct=float(flu_uplift_pct),
    flu_months=set(flu_months_set),
    annual_turnover=float(annual_turnover),
    annual_growth=float(annual_growth),
    lead_days=int(lead_days),
    ramp_months=int(ramp_months),
    ramp_productivity=float(ramp_productivity),
    fill_probability=float(fill_probability),
    starting_supply_fte=float(starting_supply_fte),
    hire_step_cap_fte=float(hire_step_cap_fte),
    allow_floor_maintenance_pipeline=bool(allow_floor_maintenance_pipeline),
    freeze_except_flu_and_floor=bool(freeze_except_flu_and_floor),
)

R_A = compute_simulation(params_A, scenario_name="A (Current)")

R_B = None
if enable_compare:
    params_B = PSMParams(**{**params_A.__dict__, "lead_days": int(improved_lead_days)})
    R_B = compute_simulation(params_B, scenario_name="B (Improved pipeline)")

# ============================================================
# CONTRACT PANEL
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
    <li><b>Target:</b> coverage (open hours × concurrency, adjusted for productivity) × demand multiplier (only above capacity), smoothed over {params_A.demand_smoothing_months} months, with floor <b>{provider_floor_fte:.2f}</b>.</li>
    <li><b>Capacity:</b> {capacity_mode}; peak factor {peak_factor:.2f}× applied to volume.</li>
    <li><b>Supply:</b> Starting paid FTE − turnover + hires after lead time; effective supply reduced during ramp ({params_A.ramp_months} mo @ {params_A.ramp_productivity*100:.0f}%).</li>
    <li><b>Pipeline:</b> {lead_days} days ≈ <b>{lead_months_A}</b> months; post req by <b>{month_name(req_m_A)}</b>, visible <b>{month_name(vis_m_A)}</b>.</li>
    <li><b>Finance:</b> Loaded provider cost computed from APP $/hr + bonus + benefits + OT/PTO + CME/licensure.</li>
  </ul>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HERO CHART
# ============================================================
st.markdown("---")
st.header("Jan–Dec Staffing Outlook (Providers Only)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Peak Burnout Gap (FTE)", f"{R_A['peak_gap']:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{R_A['avg_gap']:.2f}")
m3.metric("Months Exposed", f"{R_A['months_exposed']}/12")
m4.metric("Loaded Cost / Provider FTE", f"${loaded_cost_per_provider_fte:,.0f}")

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

dates_12 = R_A["dates_12"]
labels_12 = R_A["month_labels_12"]

ax1.plot(dates_12, R_A["target_12"], linewidth=2.2, color=BRAND_GOLD, marker="o", markersize=4, label="Target Provider FTE")
ax1.plot(dates_12, R_A["supply_eff_12"], linewidth=2.2, color=BRAND_BLACK, marker="o", markersize=4, label="Predicted Provider FTE (Effective)")
ax1.plot(dates_12, R_A["supply_paid_12"], linewidth=1.6, color=MID_GRAY, linestyle="--", label="Predicted Provider FTE (Paid)")

ax1.fill_between(
    dates_12,
    np.array(R_A["supply_eff_12"], dtype=float),
    np.array(R_A["target_12"], dtype=float),
    where=np.array(R_A["target_12"], dtype=float) > np.array(R_A["supply_eff_12"], dtype=float),
    color=BRAND_GOLD,
    alpha=0.12,
    label="Burnout Risk (Gap)",
)

if R_B is not None:
    ax1.plot(dates_12, R_B["supply_eff_12"], linewidth=2.0, color=GRAY, linestyle=":", label=f"Predicted Effective — B ({improved_lead_days}d pipeline)")

ax1.set_ylabel("Provider FTE", fontsize=12, fontweight="bold")
ax1.set_xticks(dates_12)
ax1.set_xticklabels(labels_12, fontsize=11)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

ax2 = None
if show_visits_overlay:
    ax2 = ax1.twinx()
    ax2.plot(dates_12, R_A["visits_12"], linestyle="-.", linewidth=1.6, color=GRAY, label="Visits/Day (avg, incl flu)")
    ax2.plot(dates_12, R_A["visits_eff_12"], linestyle=":", linewidth=1.6, color=GRAY, alpha=0.8, label="Visits/Day (peak adj)")
    ax2.set_ylabel("Visits / Day", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
if ax2 is not None:
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
else:
    ax1.legend(lines1, labels1, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))

ax1.set_title("Target vs Predicted Provider Staffing (Effective vs Paid)", fontsize=14, fontweight="bold")
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# LEDGER + DEBUG CHECK
# ============================================================
st.markdown("---")
st.header("Monthly Staffing Ledger (Audit View)")
st.dataframe(R_A["ledger"], hide_index=True, use_container_width=True)

if show_debug:
    st.markdown("---")
    st.header("Debug Panel")
    st.write("**No-reset check (Dec → next Jan):**")
    if R_A["no_reset"] is not None:
        st.write(R_A["no_reset"])
    else:
        st.write("No no-reset data available.")
