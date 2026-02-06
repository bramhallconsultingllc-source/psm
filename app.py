if annual is None or len(annual) == 0:
    st.error("Annual summary is empty — unable to render scorecard.")
else:
    swb_y1 = float(annual.loc[0, "SWB_per_Visit"])
    util_y1 = float(annual.loc[0, "Avg_Utilization"])
    min_y1 = float(annual.loc[0, "Min_Perm_Paid_FTE"])
    max_y1 = float(annual.loc[0, "Max_Perm_Paid_FTE"])

    fte_range = max_y1 - min_y1
    fte_avg = (max_y1 + min_y1) / 2.0
    fte_volatility = fte_range / max(fte_avg, 1e-9)

    peak_pre = float(R.get("peak_load_pre", R.get("peak_load_post", 0.0)))
    peak_post = float(R.get("peak_load_post", peak_pre))
    red_months = int(R.get("months_red", 0))
    flex_share = float(R.get("flex_share", 0.0))

    def swb_color():
        diff = abs(swb_y1 - params.target_swb_per_visit)
        if diff <= params.swb_tolerance:
            return "#27ae60", "On Target"
        if diff <= params.swb_tolerance * 2:
            return "#f39c12", "Close"
        return "#e74c3c", "Off Target"

    def util_color():
        diff = abs(util_y1 * 100 - ui["target_utilization"])
        if diff <= 2:
            return "#27ae60", "On Target"
        if diff <= 5:
            return "#f39c12", "Close"
        return "#e74c3c", "Off Target"

    def peak_color():
        if peak_pre <= params.budgeted_pppd:
            return "#27ae60", "Green"
        if peak_pre <= params.yellow_max_pppd:
            return "#f1c40f", "Yellow"
        if peak_pre <= params.red_start_pppd:
            return "#f39c12", "Orange"
        return "#e74c3c", "Red"

    def fte_color():
        if fte_volatility <= 0.10:
            return "#27ae60", "Stable"
        if fte_volatility <= 0.20:
            return "#f1c40f", "Moderate"
        return "#e74c3c", "Volatile"

    swb_c, swb_s = swb_color()
    util_c, util_s = util_color()
    peak_c, peak_s = peak_color()
    fte_c, fte_s = fte_color()

    hero_html = f"""
<div class="hero">
  <div class="hero-title">Policy Performance Scorecard</div>
  <div class="grid">
    <div class="card" style="border-left-color:{GOLD}; background:white;">
      <div class="kpi-label">Staffing Policy</div>
      <div class="kpi-value">{ui["target_utilization"]}% Target</div>
      <div class="kpi-detail">
        Coverage: {policy.base_coverage_pct*100:.0f}% base / {policy.winter_coverage_pct*100:.0f}% winter<br/>
        Posture: {POSTURE_LABEL[int(ui["risk_posture"])]}
      </div>
    </div>

    <div class="card" style="border-left-color:{swb_c};">
      <div class="kpi-label">SWB per Visit (Y1)</div>
      <div class="kpi-value" style="color:{swb_c};">${swb_y1:.2f}</div>
      <div class="kpi-detail">
        Target ${params.target_swb_per_visit:.0f} ± ${params.swb_tolerance:.0f}
        <b style="color:{swb_c};">({swb_s})</b>
      </div>
    </div>

    <div class="card" style="border-left-color:{util_c};">
      <div class="kpi-label">Utilization (Y1)</div>
      <div class="kpi-value" style="color:{util_c};">{util_y1*100:.0f}%</div>
      <div class="kpi-detail">
        Target {ui["target_utilization"]}%
        <b style="color:{util_c};">({util_s})</b>
      </div>
    </div>

    <div class="card" style="border-left-color:{peak_c};">
      <div class="kpi-label">Peak Load (PPPD)</div>
      <div class="kpi-value" style="color:{peak_c};">{peak_pre:.1f}</div>
      <div class="kpi-detail">
        Pre-flex <b style="color:{peak_c};">({peak_s})</b><br/>
        Post-flex: {peak_post:.1f}
      </div>
    </div>

    <div class="card" style="border-left-color:{fte_c};">
      <div class="kpi-label">FTE Range (Y1)</div>
      <div class="kpi-value" style="color:{fte_c};">{min_y1:.2f}–{max_y1:.2f}</div>
      <div class="kpi-detail">
        Volatility {fte_volatility*100:.0f}% <b style="color:{fte_c};">({fte_s})</b>
      </div>
    </div>

    <div class="card" style="border-left-color:#666;">
      <div class="kpi-label">Flex Share</div>
      <div class="kpi-value">{flex_share*100:.1f}%</div>
      <div class="kpi-detail">Share of total provider-days (perm + flex)</div>
    </div>

    <div class="card" style="border-left-color:{'#e74c3c' if red_months>0 else '#27ae60'};">
      <div class="kpi-label">Red Months</div>
      <div class="kpi-value">{red_months}</div>
      <div class="kpi-detail">Months above red PPPD threshold</div>
    </div>

    <div class="card" style="border-left-color:{GOLD_MUTED};">
      <div class="kpi-label">Model Version</div>
      <div class="kpi-value" style="font-size:1.05rem;">{MODEL_VERSION}</div>
      <div class="kpi-detail">36-month horizon</div>
    </div>
  </div>
</div>
""".strip()

    # 🔥 hard-stop Markdown from seeing indentation as a code block
    hero_html = "\n".join(line.lstrip() for line in hero_html.splitlines()).strip()

    st.markdown(hero_html, unsafe_allow_html=True)
