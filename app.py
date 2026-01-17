# app.py — PSM v1 (Providers Only)
# Changes in this version:
# 1) Continuous simulation (warm-up year + display year), but hiring is anchored to DISPLAY YEAR
# 2) Flu hiring shows as a SINGLE STEP in the hire-visible month (e.g., Nov)
# 3) Optional input: Current Provider FTE (Starting Supply) (defaults to calculated baseline)
# 4) Debug expander to verify the model is behaving as expected

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

from psm.staffing_model import StaffingModel

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Predictive Staffing Model (PSM) — v1", layout="centered")
st.title("Predictive Staffing Model (PSM) — v1")
st.caption("Jan–Dec display • Continuous simulation (no January reset) • Providers only")

model = StaffingModel()

WINTER = {12, 1, 2}
SPRING = {3, 4, 5}
SUMMER = {6, 7, 8}
FALL   = {9, 10, 11}

BRAND_BLACK = "#000000"
BRAND_GOLD = "#7a6200"
GRAY = "#B0B0B0"
LIGHT_GRAY = "#EAEAEA"

# ============================================================
# HELPERS
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(float(x), hi))

def lead_days_to_months(days: int, avg_days_per_month: float = 30.4) -> int:
    return max(0, int(math.ceil(float(days) / float(avg_days_per_month))))

def wrap_month(m: int) -> int:
    m = int(m)
    while m <= 0:
        m += 12
    while m > 12:
        m -= 12
    return m

def provider_fte_needed(vpd: float, hours_week: float, fte_hours_week: float) -> float:
    res = model.calculate_fte_needed(
        visits_per_day=float(vpd),
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
        ma_fte  = prov_fte * float(role_mix["ma_per_provider"])
        rt_fte  = prov_fte * float(role_mix["xrt_per_provider"])

        apc_hours = monthly_hours_from_fte(prov_fte, fte_hours_per_week, dim)
        psr_hours = monthly_hours_from_fte(psr_fte, fte_hours_per_week, dim)
        ma_hours  = monthly_hours_from_fte(ma_fte,  fte_hours_per_week, dim)
        rt_hours  = monthly_hours_from_fte(rt_fte,  fte_hours_per_week, dim)

        apc_rate = loaded_hourly_rate(hourly_rates["apc"], benefits_load_pct, ot_sick_pct)
        psr_rate = loaded_hourly_rate(hourly_rates["psr"], benefits_load_pct, ot_sick_pct)
        ma_rate  = loaded_hourly_rate(hourly_rates["ma"], benefits_load_pct, ot_sick_pct)
        rt_rate  = loaded_hourly_rate(hourly_rates["rt"], benefits_load_pct, ot_sick_pct)

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

        rows.append({
            "Provider_FTE_Supply": prov_fte,
            "Visits": month_visits,
            "SWB_$": total_swb,
            "SWB_per_Visit_$": swb_per_visit,
        })
    return pd.DataFrame(rows)

def provider_day_gap(target_curve, supply_curve, days_in_month):
    return float(sum(max(float(t) - float(s), 0.0) * float(dim) for t, s, dim in zip(target_curve, supply_curve, days_in_month)))

def annualize_monthly_fte_cost(delta_fte_curve, days_in_month, loaded_cost_per_provider_fte):
    return float(sum(float(df) * float(loaded_cost_per_provider_fte) * (float(dim) / 365.0) for df, dim in zip(delta_fte_curve, days_in_month)))

# ============================================================
# SIDEBAR INPUTS (v1)
# ============================================================
with st.sidebar:
    st.header("Inputs")

    visits = st.number_input("Avg Visits/Day", min_value=1.0, value=36.0, step=1.0)
    hours_week = st.number_input("Hours of Operation / Week", min_value=1.0, value=84.0, step=1.0)
    fte_hours_week = st.number_input("FTE Hours / Week", min_value=1.0, value=36.0, step=1.0)

    seasonality_pct = st.number_input("Seasonality % Lift/Drop", min_value=0.0, value=20.0, step=5.0) / 100.0
    annual_turnover = st.number_input("Annual Turnover %", min_value=0.0, value=16.0, step=1.0) / 100.0
    annual_growth = st.number_input("Annual Visit Growth %", min_value=0.0, value=10.0, step=1.0) / 100.0

    days_to_independent = st.number_input("Days to Independent (Req→Independent)", min_value=0, value=210, step=10)

    st.divider()
    provider_floor_fte = st.number_input("Provider Minimum Floor (FTE)", min_value=0.25, value=1.0, step=0.25)

    st.divider()
    st.subheader("Starting Supply (Optional)")
    use_calculated_baseline = st.checkbox("Use calculated baseline as starting supply", value=True)
    # We can't calculate baseline until we compute below; we will reconcile after calc.
    current_provider_fte_ui = st.number_input(
        "Current Provider FTE (Starting Supply)",
        min_value=0.0,
        value=0.0,  # placeholder; overwritten in-code when using baseline
        step=0.25,
        help="If you don't know, leave baseline enabled. This affects Predicted Supply only.",
    )

    st.divider()
    # Finance
    net_revenue_per_visit = st.number_input("Net Revenue per Visit (NRPV)", min_value=0.0, value=140.0, step=5.0)
    target_swb_per_visit = st.number_input("Target SWB/Visit", min_value=0.0, value=85.0, step=1.0)
    visits_lost_per_provider_day_gap = st.number_input("Visits Lost per 1.0 Provider-Day Gap", min_value=0.0, value=18.0, step=1.0)
    loaded_cost_per_provider_fte = st.number_input("Loaded Cost per Provider FTE (annual)", min_value=0.0, value=260000.0, step=5000.0)

    st.divider()
    # Compensation (kept simple; can expand later)
    st.subheader("Comp (Hourly) + Loads")
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
    show_debug = st.checkbox("Show debug panel", value=False)
    run = st.button("▶️ Run PSM v1", use_container_width=True)

if not run:
    st.info("Set inputs and click **Run PSM v1**.")
    st.stop()

# ============================================================
# CONTINUOUS SIMULATION
# We simulate 25 months (warm-up year + display year + 1 extra month)
# so we can validate 'no January reset' behavior post-Dec.
# ============================================================
today = datetime.today()
year0 = today.year  # only month-of-year matters
N = 25
dates = pd.date_range(start=datetime(year0, 1, 1), periods=N, freq="MS")
months = [int(d.month) for d in dates]
days_in_month = [pd.Period(d, "M").days_in_month for d in dates]

# Baseline visits with growth (v1 simple annual uplift)
baseline_adjusted = float(visits) * (1.0 + float(annual_growth))

# 1) Visits curve (month-of-year seasonality)
visits_curve = []
for m in months:
    if m in WINTER:
        v = baseline_adjusted * (1.0 + float(seasonality_pct))
    elif m in SUMMER:
        v = baseline_adjusted * (1.0 - float(seasonality_pct))
    else:
        v = baseline_adjusted
    visits_curve.append(float(v))

# 2) Target provider FTE curve, floor enforced
target_curve = []
for v in visits_curve:
    t = provider_fte_needed(v, hours_week, fte_hours_week)
    target_curve.append(max(float(t), float(provider_floor_fte)))

# 3) Baseline provider FTE = Spring/Fall baseline staffing level (baseline_adjusted)
baseline_provider_fte = max(
    provider_fte_needed(baseline_adjusted, hours_week, fte_hours_week),
    float(provider_floor_fte),
)

# Reconcile starting supply input
if use_calculated_baseline:
    starting_supply_fte = float(baseline_provider_fte)
else:
    starting_supply_fte = max(float(current_provider_fte_ui), float(provider_floor_fte))

# 4) Flu planning anchor
lead_months = lead_days_to_months(int(days_to_independent))
READY_MONTH = 11  # Nov
req_post_month = wrap_month(READY_MONTH - lead_months)
hire_visible_month = wrap_month(req_post_month + lead_months)

# Determine December target (month-of-year consistent)
dec_visits = baseline_adjusted * (1.0 + float(seasonality_pct))  # winter uplift
dec_target_fte = max(provider_fte_needed(dec_visits, hours_week, fte_hours_week), float(provider_floor_fte))

# Turnover
monthly_turnover = float(annual_turnover) / 12.0

# Include simple turnover projection into required flu add (between req posting and readiness)
loss_factor = (1.0 - monthly_turnover) ** float(lead_months) if lead_months > 0 else 1.0
expected_start_at_ready = float(starting_supply_fte) * float(loss_factor)
fte_to_add_for_flu = max(float(dec_target_fte) - float(expected_start_at_ready), 0.0)

# 5) Hiring: SINGLE STEP in the display year's hire-visible month
hires_visible = [0.0] * N

# We define display year indices as 12..23 (Jan–Dec of the 2nd year)
DISPLAY_START = 12
DISPLAY_END = 23

# Anchor req-post month occurrence to DISPLAY YEAR
req_post_idx = None
for i in range(DISPLAY_START, DISPLAY_END + 1):
    if months[i] == req_post_month:
        req_post_idx = i
        break
if req_post_idx is None:
    raise ValueError("Could not locate req_post_month in display year indices 12..23.")

visible_start_idx = req_post_idx + lead_months  # hire-visible index for the display year cycle

# Step hire only if it lands within the display year window
if DISPLAY_START <= visible_start_idx <= DISPLAY_END:
    hires_visible[visible_start_idx] = float(fte_to_add_for_flu)
else:
    # If lead time pushes visibility outside displayed year, nothing will step up in Jan–Dec view
    pass

# 6) Simulate predicted supply (continuous, no reset)
supply = [0.0] * N
supply[0] = float(starting_supply_fte)

for i in range(1, N):
    prev = float(supply[i - 1])
    after_attrition = prev * (1.0 - monthly_turnover)
    after_hiring = after_attrition + float(hires_visible[i])
    supply[i] = max(float(after_hiring), float(provider_floor_fte))

# ============================================================
# DISPLAY YEAR VIEW (Jan–Dec of 2nd year)
# ============================================================
idx_12 = list(range(DISPLAY_START, DISPLAY_END + 1))
dates_12 = [dates[i] for i in idx_12]
labels_12 = [d.strftime("%b") for d in dates_12]
days_12 = [days_in_month[i] for i in idx_12]

visits_12 = [visits_curve[i] for i in idx_12]
target_12 = [target_curve[i] for i in idx_12]
supply_12 = [supply[i] for i in idx_12]
hires_12 = [hires_visible[i] for i in idx_12]

gap_12 = [max(t - s, 0.0) for t, s in zip(target_12, supply_12)]
months_exposed = sum(1 for g in gap_12 if g > 1e-6)
peak_gap = max(gap_12) if gap_12 else 0.0
avg_gap = float(np.mean(gap_12)) if gap_12 else 0.0

# Visible hire month label (if within display)
hire_step_label = None
if DISPLAY_START <= visible_start_idx <= DISPLAY_END:
    hire_step_label = dates[visible_start_idx]

# ============================================================
# MAIN OUTPUT: GRAPH
# ============================================================
st.markdown("---")
st.header("Jan–Dec Staffing Outlook (Providers Only)")

m1, m2, m3 = st.columns(3)
m1.metric("Peak Burnout Gap (FTE)", f"{peak_gap:.2f}")
m2.metric("Avg Burnout Gap (FTE)", f"{avg_gap:.2f}")
m3.metric("Months Exposed", f"{months_exposed}/12")

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

ax1.plot(dates_12, target_12, linewidth=2.0, color=BRAND_GOLD, marker="o", markersize=4, label="Target Provider FTE")
ax1.plot(dates_12, supply_12, linewidth=2.0, color=BRAND_BLACK, marker="o", markersize=4, label="Predicted Provider FTE (Pipeline-Constrained)")

ax1.fill_between(
    dates_12,
    np.array(supply_12, dtype=float),
    np.array(target_12, dtype=float),
    where=np.array(target_12, dtype=float) > np.array(supply_12, dtype=float),
    alpha=0.12,
    color=BRAND_GOLD,
    label="Burnout Risk (Gap)",
)

# Mark the step hire month
if hire_step_label is not None and float(fte_to_add_for_flu) > 1e-6:
    ax1.axvline(hire_step_label, color=BRAND_BLACK, linewidth=1.0, linestyle="--", alpha=0.6)

ax1.set_ylabel("Provider FTE")
ax1.set_xticks(dates_12)
ax1.set_xticklabels(labels_12)
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRAY)

ax2 = ax1.twinx()
ax2.plot(dates_12, visits_12, linestyle="-.", linewidth=1.6, color=GRAY, label="Visits/Day (Forecast)")
ax2.set_ylabel("Visits / Day")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
)

ax1.set_title("Target vs Predicted Provider Staffing (Nov Step Hire, No January Reset)", fontsize=14, fontweight="bold")
plt.tight_layout()
st.pyplot(fig)

monthly_turnover_pct = monthly_turnover * 100.0
st.info(
    f"**Flu planning anchor:** staff needed for **December** should be independent by **Nov 1**.\n\n"
    f"- Lead time: **{int(days_to_independent)} days ≈ {lead_months} months**\n"
    f"- **Req post month (display year):** {datetime(2000, req_post_month, 1).strftime('%b')}\n"
    f"- **Hires visible month (display year):** {datetime(2000, hire_visible_month, 1).strftime('%b')} "
    f"{'(step hire shown)' if hire_step_label is not None and fte_to_add_for_flu > 1e-6 else '(not visible in this Jan–Dec window)'}\n"
    f"- Turnover: **{annual_turnover*100:.1f}% annual ≈ {monthly_turnover_pct:.2f}% monthly**\n"
    f"- Starting supply: **{starting_supply_fte:.2f} FTE** "
    f"({'calculated baseline' if use_calculated_baseline else 'user-entered'})"
)

# ============================================================
# FINANCE — ROI
# ============================================================
st.markdown("---")
st.header("Finance — ROI Investment Case")

delta_fte_curve = [max(t - baseline_provider_fte, 0.0) for t in target_12]
annual_investment = annualize_monthly_fte_cost(delta_fte_curve, days_12, loaded_cost_per_provider_fte)

gap_days = provider_day_gap(target_12, supply_12, days_12)
est_visits_lost = gap_days * float(visits_lost_per_provider_day_gap)
est_revenue_lost = est_visits_lost * float(net_revenue_per_visit)

roi = (est_revenue_lost / annual_investment) if annual_investment > 0 else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("Annual Investment (to Target)", f"${annual_investment:,.0f}")
c2.metric("Est. Net Revenue at Risk", f"${est_revenue_lost:,.0f}")
c3.metric("ROI (Revenue ÷ Investment)", f"{roi:,.2f}x" if np.isfinite(roi) else "—")

# ============================================================
# VVI FEASIBILITY — SWB/Visit (Annual Constraint)
# ============================================================
st.markdown("---")
st.header("VVI Feasibility — SWB/Visit (Annual Constraint)")

role_mix = compute_role_mix_ratios(baseline_adjusted, hours_week, fte_hours_week)
hourly_rates = {
    "physician": physician_hr,
    "apc": apc_hr,
    "ma": ma_hr,
    "psr": psr_hr,
    "rt": rt_hr,
    "supervisor": supervisor_hr,
}

swb_df = compute_monthly_swb_per_visit_fte_based(
    provider_supply_12=supply_12,
    visits_per_day_12=visits_12,
    days_in_month_12=days_12,
    fte_hours_per_week=fte_hours_week,
    role_mix=role_mix,
    hourly_rates=hourly_rates,
    benefits_load_pct=benefits_load_pct,
    ot_sick_pct=ot_sick_pct,
    physician_supervision_hours_per_month=physician_supervision_hours_per_month,
    supervisor_hours_per_month=supervisor_hours_per_month,
)

total_swb = float(swb_df["SWB_$"].sum())
total_visits = float(sum(float(v) * float(dim) for v, dim in zip(visits_12, days_12)))
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
swb_df_display.insert(0, "Month", labels_12)
st.dataframe(
    swb_df_display[["Month", "Provider_FTE_Supply", "Visits", "SWB_$", "SWB_per_Visit_$"]],
    hide_index=True,
    use_container_width=True,
)

fig2, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(dates_12, swb_df["SWB_per_Visit_$"].astype(float).values, linewidth=2.0, marker="o", markersize=3, color=BRAND_BLACK, label="SWB/Visit (monthly)")
ax.axhline(float(target_swb_per_visit), linewidth=2.0, linestyle="--", color=BRAND_GOLD, label="Target SWB/Visit")
ax.set_title("SWB/Visit — Monthly Diagnostic (Annual is the constraint)", fontsize=13, fontweight="bold")
ax.set_ylabel("$/Visit")
ax.set_xticks(dates_12)
ax.set_xticklabels(labels_12)
ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
st.pyplot(fig2)

# ============================================================
# EXECUTIVE SUMMARY (v1)
# ============================================================
st.markdown("---")
st.header("Executive Summary (v1)")

st.write(
    f"""
**What this model does**
- Forecasts visits/day with quarter-based seasonality (Winter up, Summer down).
- Converts demand into a **Target Provider FTE** curve (with a floor).
- Simulates **Predicted Provider FTE** under:
  - continuous monthly turnover
  - a single flu-planning requisition posting window (posting frozen otherwise)
  - lead-time delay before hires become visible
- Shows the **burnout risk area** where predicted supply is below target.

**Key planning dates**
- Lead time: **{int(days_to_independent)} days ≈ {lead_months} months**
- Post requisitions by: **{datetime(2000, req_post_month, 1).strftime('%b')}** (display year)
- Hires become visible: **{datetime(2000, hire_visible_month, 1).strftime('%b')}** (display year)
- Predicted staffing is continuous; **no January reset**.

**Financial reasonableness**
- Annual SWB/Visit: **${annual_swb:.2f}** vs Target **${target_swb_per_visit:.2f}** → **{"PASS" if feasible else "FAIL"}**
- ROI (Revenue at risk ÷ Investment): **{roi:,.2f}x**
"""
)

st.success("✅ v1 is wired for pressure testing: the predicted line must respond to turnover monthly and step up in the hire-visible month when applicable.")

# ============================================================
# DEBUG PANEL
# ============================================================
if show_debug:
    with st.expander("Debug — model sanity checks", expanded=True):
        st.write("**Key indices (continuous sim):**")
        st.write(f"- DISPLAY_START index: {DISPLAY_START} (Jan of display year)")
        st.write(f"- DISPLAY_END index: {DISPLAY_END} (Dec of display year)")
        st.write(f"- req_post_idx: {req_post_idx} (month={months[req_post_idx]})")
        st.write(f"- visible_start_idx: {visible_start_idx} (month={(months[visible_start_idx] if 0 <= visible_start_idx < N else 'out of range')})")

        st.write("**Step hire within display window?**")
        st.write(f"- fte_to_add_for_flu: {fte_to_add_for_flu:.3f}")
        st.write(f"- hire applied in display year: {'YES' if any(h > 1e-6 for h in hires_12) else 'NO'}")

        # Month-to-month deltas in display year
        deltas = [supply_12[i] - supply_12[i-1] for i in range(1, 12)]
        df_dbg = pd.DataFrame({
            "Month": labels_12,
            "Visits/Day": np.round(visits_12, 1),
            "Target_FTE": np.round(target_12, 3),
            "Supply_FTE": np.round(supply_12, 3),
            "Hire_Visible_FTE": np.round(hires_12, 3),
            "Gap_FTE": np.round(gap_12, 3),
        })
        st.dataframe(df_dbg, hide_index=True, use_container_width=True)

        st.write("**Supply deltas (display year)** (positive should occur at the hire step; otherwise mostly negative unless floor binds):")
        st.write([round(x, 4) for x in deltas])

        # No-reset check: compare Dec display year vs next Jan (requires N>=25)
        if N >= 25:
            dec_idx = DISPLAY_END
            next_jan_idx = DISPLAY_END + 1  # Jan after displayed Dec
            if next_jan_idx < N:
                st.write("**No-reset check (Dec → next Jan):**")
                st.write(f"- Supply Dec: {supply[dec_idx]:.4f}")
                st.write(f"- Supply next Jan: {supply[next_jan_idx]:.4f} (should be slightly lower due to turnover, not reset)")
