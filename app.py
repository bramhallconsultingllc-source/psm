# ENHANCEMENTS TO ADD TO YOUR EXISTING FILE
# Add these sections where indicated in comments

# ============================================================
# ENHANCEMENT 1: Current State Input
# Add this in the Core Settings section after annual_turnover
# ============================================================

# Current state (add after line with annual_turnover)
st.markdown("#### Current Staffing")  # ✅ CORRECT - markdown header syntax
use_current_state = st.checkbox(
    "Start from actual current FTE (not policy target)", 
    value=False,
    help="Check this to model from your actual current staffing level instead of policy target"
)
if use_current_state:
    current_fte = st.number_input(
        "Current FTE on Staff", 
        0.1, 
        value=3.5, 
        step=0.1,
        help="Your actual current provider FTE (paid headcount)"
    )
else:
    current_fte = None

# ============================================================
# ENHANCEMENT 2: Update simulate_policy to use current_fte
# Modify the initialization section (around line 623)
# ============================================================

# REPLACE THIS LINE:
# initial_target = target_fte_for_month(0)
# cohorts = [{"fte": initial_target, "age": 9999}]

# WITH THIS:
def simulate_policy(params: ModelParams, policy: Policy, current_fte_override=None) -> Dict[str, object]:
    # ... existing code ...
    
    # Initialize cohorts (MODIFIED)
    if current_fte_override is not None:
        initial_staff = float(current_fte_override)
    else:
        initial_staff = target_fte_for_month(0)
    
    cohorts = [{"fte": initial_staff, "age": 9999}]
    
    # ... rest of existing code ...

# ============================================================
# ENHANCEMENT 3: Pass current_fte to simulation
# Update the call sites (around line 1245)
# ============================================================

# REPLACE:
# R = cached_simulate(params_dict, policy_dict)

# WITH:
R = cached_simulate(params_dict, policy_dict, current_fte)

# Update cached_simulate signature:
@st.cache_data(show_spinner=False)
def cached_simulate(
    params_dict: Dict[str, Any], 
    policy_dict: Dict[str, float],
    current_fte_override=None
) -> Dict[str, object]:
    params_dict = dict(params_dict)
    params_dict["flu_months"] = set(params_dict.get("flu_months", []))
    params_dict["freeze_months"] = set(params_dict.get("freeze_months", []))
    params = ModelParams(**params_dict)
    policy = Policy(**policy_dict)
    return simulate_policy(params, policy, current_fte_override)

# ============================================================
# ENHANCEMENT 4: Hiring Action Plan Export
# Add this in the Smart Hiring Insights section (after line 1342)
# ============================================================

# Add after the upcoming_hires visualization
if len(upcoming_hires) > 0:
    # Export hiring plan
    st.markdown("### 📥 Export Hiring Plan")
    
    # Create clean hiring plan dataframe
    hiring_export = upcoming_hires[["Month", "Hires Visible (FTE)", "Hire Reason"]].copy()
    hiring_export.columns = ["Month", "FTE to Hire", "Hiring Rationale"]
    hiring_export["FTE to Hire"] = hiring_export["FTE to Hire"].apply(lambda x: f"{x:.2f}")
    
    # Add summary row
    total_row = pd.DataFrame([{
        "Month": "TOTAL (12 months)",
        "FTE to Hire": f"{total_hires_12mo:.2f}",
        "Hiring Rationale": f"Total hiring volume: {(total_hires_12mo/max(avg_fte,1e-9))*100:.0f}% of avg staff"
    }])
    hiring_export = pd.concat([hiring_export, total_row], ignore_index=True)
    
    # Convert to CSV
    hiring_csv = hiring_export.to_csv(index=False).encode("utf-8")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.download_button(
            "⬇️ Download Hiring Action Plan (CSV)",
            hiring_csv,
            "hiring_action_plan.csv",
            "text/csv",
            use_container_width=True,
            help="Download detailed 12-month hiring plan with rationale"
        )
    with c2:
        st.download_button(
            "⬇️ Download Full Ledger (CSV)",
            ledger.to_csv(index=False).encode("utf-8"),
            "staffing_ledger_full.csv",
            "text/csv",
            use_container_width=True
        )

# ============================================================
# ENHANCEMENT 5: Scenario Comparison Tool
# Add this as a new section before the charts (after line 1344)
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("## 🔀 Scenario Comparison")

st.markdown("""
Compare multiple scenarios side-by-side to understand how changing key assumptions 
affects your staffing needs and costs.
""")

with st.expander("🎯 **Run Scenario Analysis**", expanded=False):
    st.markdown("### Define Scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Base Case (Current)**")
        st.info(f"Utilization: {ui['target_utilization']}%  \nGrowth: {params.annual_growth*100:.0f}%  \nTurnover: {params.annual_turnover*100:.0f}%")
    
    with col2:
        st.markdown("**🔺 Optimistic Scenario**")
        opt_growth = st.number_input("Growth %", 0.0, value=15.0, step=1.0, key="opt_growth") / 100.0
        opt_turnover = st.number_input("Turnover %", 0.0, value=12.0, step=1.0, key="opt_turn") / 100.0
        opt_util = st.slider("Utilization %", 80, 98, 94, 2, key="opt_util")
    
    with col3:
        st.markdown("**🔻 Conservative Scenario**")
        cons_growth = st.number_input("Growth %", 0.0, value=5.0, step=1.0, key="cons_growth") / 100.0
        cons_turnover = st.number_input("Turnover %", 0.0, value=20.0, step=1.0, key="cons_turn") / 100.0
        cons_util = st.slider("Utilization %", 80, 98, 88, 2, key="cons_util")
    
    if st.button("🔄 Run Scenario Comparison", use_container_width=True):
        with st.spinner("Running scenarios..."):
            # Base case (already have R)
            base_metrics = pack_exec_metrics(R)
            
            # Optimistic scenario
            opt_params = ModelParams(**{**params.__dict__, 
                                        "annual_growth": opt_growth,
                                        "annual_turnover": opt_turnover})
            opt_cov = 1.0 / (opt_util/100.0)
            opt_policy = Policy(base_coverage_pct=opt_cov, 
                               winter_coverage_pct=opt_cov * 1.03)
            R_opt = simulate_policy(opt_params, opt_policy, current_fte)
            opt_metrics = pack_exec_metrics(R_opt)
            
            # Conservative scenario
            cons_params = ModelParams(**{**params.__dict__,
                                         "annual_growth": cons_growth,
                                         "annual_turnover": cons_turnover})
            cons_cov = 1.0 / (cons_util/100.0)
            cons_policy = Policy(base_coverage_pct=cons_cov,
                                winter_coverage_pct=cons_cov * 1.03)
            R_cons = simulate_policy(cons_params, cons_policy, current_fte)
            cons_metrics = pack_exec_metrics(R_cons)
            
            # Create comparison table
            df_scenarios = pd.DataFrame([
                {"Scenario": "Base Case", **base_metrics},
                {"Scenario": "Optimistic", **opt_metrics},
                {"Scenario": "Conservative", **cons_metrics}
            ])
            
            st.markdown("### 📊 Scenario Results")
            st.dataframe(
                df_scenarios.style.format({
                    "SWB/Visit (Y1)": lambda x: f"${x:.2f}",
                    "EBITDA Proxy (Y1)": lambda x: f"${x:,.0f}",
                    "EBITDA Proxy (3yr total)": lambda x: f"${x:,.0f}",
                    "Flex Share": lambda x: f"{x*100:.1f}%",
                    "Peak Load (PPPD)": lambda x: f"{x:.1f}"
                }),
                hide_index=True,
                use_container_width=True
            )
            
            # Key insights
            st.markdown("### 💡 Key Insights")
            
            ebitda_range = max([base_metrics["EBITDA Proxy (3yr total)"], 
                               opt_metrics["EBITDA Proxy (3yr total)"],
                               cons_metrics["EBITDA Proxy (3yr total)"]]) - \
                          min([base_metrics["EBITDA Proxy (3yr total)"],
                               opt_metrics["EBITDA Proxy (3yr total)"],
                               cons_metrics["EBITDA Proxy (3yr total)"]])
            
            st.info(f"""
            **Financial Range:** 3-year EBITDA varies by **${ebitda_range:,.0f}** across scenarios.
            
            - **Optimistic** (high growth, low turnover): ${opt_metrics['EBITDA Proxy (3yr total)']:,.0f}
            - **Base Case**: ${base_metrics['EBITDA Proxy (3yr total)']:,.0f}
            - **Conservative** (low growth, high turnover): ${cons_metrics['EBITDA Proxy (3yr total)']:,.0f}
            
            This helps you understand your financial exposure to growth and retention assumptions.
            """)
            
            # Export scenarios
            scenario_csv = df_scenarios.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Scenario Comparison (CSV)",
                scenario_csv,
                "scenario_comparison.csv",
                "text/csv",
                use_container_width=True
            )

# ============================================================
# ENHANCEMENT 6: Sensitivity Analysis (Bonus)
# Add this as a collapsible section
# ============================================================

with st.expander("📈 **Sensitivity Analysis**", expanded=False):
    st.markdown("""
    See how sensitive your results are to key assumptions. 
    Helps identify which variables matter most for planning.
    """)
    
    if st.button("🔬 Run Sensitivity Analysis", key="sensitivity"):
        with st.spinner("Analyzing sensitivity..."):
            base_swb = R["annual_swb_per_visit"]
            base_ebitda = R["ebitda_proxy_annual"]
            
            # Test turnover sensitivity
            turnover_tests = [0.12, 0.16, 0.20, 0.24]  # 12%, 16%, 20%, 24%
            turnover_results = []
            
            for turn in turnover_tests:
                test_params = ModelParams(**{**params.__dict__, "annual_turnover": turn})
                test_result = simulate_policy(test_params, policy, current_fte)
                turnover_results.append({
                    "Turnover %": f"{turn*100:.0f}%",
                    "SWB/Visit": test_result["annual_swb_per_visit"],
                    "EBITDA": test_result["ebitda_proxy_annual"],
                    "Δ SWB": test_result["annual_swb_per_visit"] - base_swb,
                    "Δ EBITDA": test_result["ebitda_proxy_annual"] - base_ebitda
                })
            
            st.markdown("**Turnover Sensitivity**")
            df_turn = pd.DataFrame(turnover_results)
            st.dataframe(
                df_turn.style.format({
                    "SWB/Visit": "${:.2f}",
                    "EBITDA": "${:,.0f}",
                    "Δ SWB": "${:+.2f}",
                    "Δ EBITDA": "${:+,.0f}"
                }),
                hide_index=True,
                use_container_width=True
            )
            
            st.caption("💡 **Insight:** Every 4% increase in turnover costs you approximately " + 
                      f"${abs(turnover_results[-1]['Δ EBITDA'] - turnover_results[0]['Δ EBITDA'])/3:,.0f} " +
                      "in 3-year EBITDA")

# ============================================================
# WHERE TO ADD IN YOUR FILE
# ============================================================
"""
1. Current State Input: Add after line ~280 (after annual_turnover input)
2. Update simulate_policy: Modify function signature around line ~520
3. Update cached_simulate: Modify around line ~512
4. Hiring Export: Add around line ~1342 (in Smart Hiring Insights section)
5. Scenario Comparison: Add around line ~1344 (before charts section)
6. Sensitivity: Add in expandable section near scenarios

Remember to:
- Pass current_fte through the ui dict from build_sidebar
- Update all calls to cached_simulate to include current_fte parameter
- Test each enhancement independently before combining
"""
