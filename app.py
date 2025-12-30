"""
RetainOps: Customer Retention Policy & Budget Optimization Simulator
Main Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.data_utils import load_and_validate_data, clean_data, prepare_features
from src.eda import (
    compute_basic_stats, segment_customers, analyze_churn_drivers,
    create_churn_distribution_plot, create_segment_churn_plot, create_feature_correlation_plot
)
from src.ml_model import train_churn_model, predict_churn_risk, create_probability_distribution_plot, create_feature_importance_plot
from src.revenue import (
    calculate_revenue_at_risk, calculate_customer_value_risk_matrix,
    create_value_risk_scatter_plot, get_segment_summary
)
from src.policy_simulator import allocate_interventions, simulate_scenario, create_allocation_summary_plot
from src.evaluation import compare_scenarios, calculate_best_worst_case, create_scenario_comparison_plot, create_roi_comparison_plot, create_risk_analysis_plot
from src.llm_helper import explain_churn_drivers, explain_retention_policy, analyze_what_if_scenario, explain_risk_uncertainty

# Page configuration
st.set_page_config(
    page_title="RetainOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Demo Mode Indicator (No External APIs Required)
DEMO_MODE = True  # Explainable ML Mode - all insights are rule-based and deterministic

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'churn_column' not in st.session_state:
    st.session_state.churn_column = None
if 'categorical_features' not in st.session_state:
    st.session_state.categorical_features = []
if 'numeric_features' not in st.session_state:
    st.session_state.numeric_features = []
if 'churn_probabilities' not in st.session_state:
    st.session_state.churn_probabilities = None
if 'customer_revenue' not in st.session_state:
    st.session_state.customer_revenue = None
if 'revenue_at_risk' not in st.session_state:
    st.session_state.revenue_at_risk = None
if 'scenarios_cache' not in st.session_state:
    st.session_state.scenarios_cache = {}
if 'scenarios_cache_key' not in st.session_state:
    st.session_state.scenarios_cache_key = None


def main():
    # Sidebar - Business Control Panel
    with st.sidebar:
        st.header("📋 Business Control Panel")
        
        # CSV Upload
        uploaded_file = st.file_uploader(
            "Upload Customer Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with customer data including churn information"
        )
        
        if uploaded_file is not None:
            if not st.session_state.data_loaded or st.session_state.get('uploaded_file_name') != uploaded_file.name:
                with st.spinner("Loading and validating data..."):
                    df, validation_errors = load_and_validate_data(uploaded_file)
                    
                    if validation_errors:
                        st.error("Validation errors:")
                        for error in validation_errors:
                            st.error(f"• {error}")
                        st.session_state.data_loaded = False
                    elif df is not None:
                        # Clean data
                        df_clean, churn_col, cleaning_info = clean_data(df)
                        st.session_state.df_processed = df_clean
                        st.session_state.churn_column = churn_col
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.session_state.data_loaded = True
                        st.session_state.model_trained = False
                        
                        # Prepare features
                        X, y, cat_features, num_features = prepare_features(df_clean, churn_col)
                        st.session_state.categorical_features = cat_features
                        st.session_state.numeric_features = num_features
                        
                        if cleaning_info:
                            st.success("Data loaded successfully")
                            with st.expander("Data Cleaning Details"):
                                for info in cleaning_info:
                                    st.text(info)
        
        if st.session_state.data_loaded:
            st.divider()
            
            # Retention Budget
            if 'retention_budget' not in st.session_state:
                st.session_state.retention_budget = 50000.0
            retention_budget = st.number_input(
                "Retention Budget ($)",
                min_value=0.0,
                value=st.session_state.retention_budget,
                step=1000.0,
                format="%.0f",
                key='sidebar_budget'
            )
            st.session_state.retention_budget = retention_budget
            
            # Cost per Intervention
            if 'intervention_cost' not in st.session_state:
                st.session_state.intervention_cost = 100.0
            intervention_cost = st.number_input(
                "Cost per Intervention ($)",
                min_value=1.0,
                value=st.session_state.intervention_cost,
                step=10.0,
                format="%.2f",
                key='sidebar_cost'
            )
            st.session_state.intervention_cost = intervention_cost
            
            # Risk Tolerance
            if 'risk_tolerance' not in st.session_state:
                st.session_state.risk_tolerance = 1.0
            risk_tolerance = st.slider(
                "Risk Tolerance",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.risk_tolerance,
                step=0.1,
                help="Higher values prioritize high-risk customers, lower values prioritize high-value customers",
                key='sidebar_risk'
            )
            st.session_state.risk_tolerance = risk_tolerance
            
            st.divider()
            
            # Run Simulation Button
            run_simulation = st.button(
                "🚀 Run Simulation",
                type="primary",
                use_container_width=True
            )
            
            if run_simulation:
                # Clear scenario cache when running new simulation
                st.session_state.scenarios_cache = {}
                st.session_state.scenarios_cache_key = None
                
                with st.spinner("Training model and running simulation..."):
                    # Train model if not already trained
                    if not st.session_state.model_trained:
                        X, y, _, _ = prepare_features(
                            st.session_state.df_processed,
                            st.session_state.churn_column
                        )
                        
                        # Validate target variable before training
                        unique_classes = y.unique()
                        class_counts = y.value_counts()
                        
                        if len(unique_classes) == 1:
                            st.warning(f"⚠️ Warning: Your dataset has only one class in the target variable (all values are {unique_classes[0]}). "
                                     "The model may not be meaningful. Please ensure your dataset contains both churned and non-churned customers.")
                        elif len(unique_classes) > 100:
                            st.warning(f"⚠️ Warning: Your churn column appears to have continuous values ({len(unique_classes)} unique values). "
                                     "This suggests the churn column may not be binary (0/1). The model will still train, but results may be unexpected.")
                        elif any(count < 2 for count in class_counts.tolist()):
                            st.warning(f"⚠️ Warning: One or more classes have very few samples. "
                                     f"Class distribution: {dict(class_counts.head(10))}. "
                                     "The model will use a non-stratified split, which may affect performance.")
                        
                        model, train_metrics, val_metrics, feature_importance = train_churn_model(
                            X, y,
                            st.session_state.categorical_features
                        )
                        
                        st.session_state.model = model
                        st.session_state.train_metrics = train_metrics
                        st.session_state.val_metrics = val_metrics
                        st.session_state.feature_importance = feature_importance
                        st.session_state.model_trained = True
                        
                        # Predict churn probabilities
                        churn_probs = predict_churn_risk(model, X)
                        st.session_state.churn_probabilities = churn_probs
                        
                        # Calculate revenue at risk
                        rev_at_risk, customer_rev = calculate_revenue_at_risk(
                            st.session_state.df_processed,
                            churn_probs
                        )
                        st.session_state.revenue_at_risk = rev_at_risk
                        st.session_state.customer_revenue = customer_rev
                    
                    st.session_state.simulation_run = True
                    st.rerun()
    
    # Main Content Area
    if not st.session_state.data_loaded:
        st.title("RetainOps")
        st.subheader("Customer Retention Policy & Budget Optimization Simulator")
        
        # Demo Mode Badge
        st.success("🔬 **Explainable ML Mode**: This application uses deterministic, rule-based explanations. "
                   "No external APIs or API keys required.")
        
        st.markdown("---")
        st.info("👈 Please upload a CSV file with customer data in the sidebar to begin.")
        st.markdown("""
        ### About RetainOps
        
        RetainOps is an executive-level customer retention strategy tool that helps leadership teams:
        
        - **Identify** customers at risk of churning
        - **Analyze** churn drivers and customer segments
        - **Simulate** retention policy strategies under budget constraints
        - **Compare** multiple scenarios to optimize ROI
        - **Get explainable insights** and recommendations (rule-based, no external APIs)
        
        Upload your customer data to get started with churn risk analysis and retention policy optimization.
        """)
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "👥 Customer Insights",
        "🎯 Churn Risk Model",
        "💰 Retention Policy Simulator",
        "📈 Scenario Comparison",
        "📊 Explainable Insights"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Executive Overview")
        
        if not st.session_state.model_trained:
            st.info("Run simulation from the sidebar to view model results and KPIs.")
        else:
            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_customers = len(st.session_state.df_processed)
                st.metric("Total Customers", f"{total_customers:,}")
            
            with col2:
                churn_rate = st.session_state.df_processed[st.session_state.churn_column].mean()
                st.metric("Overall Churn Rate", f"{churn_rate:.1%}")
            
            with col3:
                total_rev_at_risk = st.session_state.revenue_at_risk.sum()
                st.metric("Total Revenue at Risk", f"${total_rev_at_risk:,.0f}")
            
            with col4:
                val_accuracy = st.session_state.val_metrics.get('accuracy', 0)
                st.metric("Model Accuracy", f"{val_accuracy:.1%}")
            
            st.divider()
            
            # High-level summary
            st.subheader("Summary")
            
            stats = compute_basic_stats(
                st.session_state.df_processed,
                st.session_state.churn_column,
                st.session_state.numeric_features
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset Statistics**")
                st.write(f"- Total Customers: {stats['total_customers']:,}")
                st.write(f"- Churn Rate: {stats['churn_rate']:.1%}")
                st.write(f"- Churned: {stats['churn_count']:,}")
                st.write(f"- Retained: {stats['retention_count']:,}")
            
            with col2:
                st.markdown("**Model Performance**")
                st.write(f"- Validation Accuracy: {st.session_state.val_metrics['accuracy']:.1%}")
                st.write(f"- Precision: {st.session_state.val_metrics['precision']:.3f}")
                st.write(f"- Recall: {st.session_state.val_metrics['recall']:.3f}")
                st.write(f"- F1 Score: {st.session_state.val_metrics['f1']:.3f}")
                st.write(f"- ROC AUC: {st.session_state.val_metrics['roc_auc']:.3f}")
            
            # Churn distribution chart
            st.subheader("Churn Distribution")
            fig_churn = create_churn_distribution_plot(
                st.session_state.df_processed,
                st.session_state.churn_column
            )
            st.plotly_chart(fig_churn, use_container_width=True)
    
    # Tab 2: Customer Insights
    with tab2:
        st.header("Customer Insights & Segmentation")
        
        if not st.session_state.model_trained:
            st.info("Run simulation from the sidebar to view customer insights.")
        else:
            # Customer Segmentation
            st.subheader("Customer Segmentation")
            
            df_segmented, segment_stats = segment_customers(
                st.session_state.df_processed,
                st.session_state.numeric_features
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_seg_churn = create_segment_churn_plot(
                    df_segmented,
                    st.session_state.churn_column
                )
                st.plotly_chart(fig_seg_churn, use_container_width=True)
            
            with col2:
                st.markdown("**Segment Statistics**")
                for segment, stats in segment_stats.items():
                    st.write(f"**{segment}**")
                    st.write(f"- Count: {stats['count']:,} ({stats['percentage']:.1f}%)")
            
            st.divider()
            
            # Churn Drivers
            st.subheader("Churn Drivers")
            
            drivers = analyze_churn_drivers(
                st.session_state.df_processed,
                st.session_state.churn_column,
                st.session_state.numeric_features,
                st.session_state.categorical_features
            )
            
            if drivers['numeric_correlations']:
                fig_corr = create_feature_correlation_plot(
                    drivers['numeric_correlations'],
                    top_n=10
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            st.divider()
            
            # Value-Risk Analysis
            st.subheader("Customer Value-Risk Matrix")
            
            df_value_risk = calculate_customer_value_risk_matrix(
                st.session_state.df_processed,
                st.session_state.churn_probabilities,
                st.session_state.customer_revenue
            )
            
            fig_value_risk = create_value_risk_scatter_plot(df_value_risk)
            st.plotly_chart(fig_value_risk, use_container_width=True)
            
            st.markdown("**Value-Risk Segment Summary**")
            segment_summary = get_segment_summary(df_value_risk)
            st.dataframe(segment_summary, use_container_width=True)
    
    # Tab 3: Churn Risk Model
    with tab3:
        st.header("Churn Risk Model")
        
        if not st.session_state.model_trained:
            st.info("Run simulation from the sidebar to view model details.")
        else:
            # Model Performance Metrics
            st.subheader("Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{st.session_state.val_metrics['accuracy']:.1%}")
            with col2:
                st.metric("Precision", f"{st.session_state.val_metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{st.session_state.val_metrics['recall']:.3f}")
            with col4:
                st.metric("F1 Score", f"{st.session_state.val_metrics['f1']:.3f}")
            
            st.divider()
            
            # Probability Distribution
            st.subheader("Churn Probability Distribution")
            fig_prob_dist = create_probability_distribution_plot(
                st.session_state.churn_probabilities
            )
            st.plotly_chart(fig_prob_dist, use_container_width=True)
            
            st.divider()
            
            # Feature Importance
            st.subheader("Feature Importance")
            fig_feat_imp = create_feature_importance_plot(
                st.session_state.feature_importance,
                top_n=15
            )
            st.plotly_chart(fig_feat_imp, use_container_width=True)
            
            # Technical Details (Hidden in Expander)
            with st.expander("Technical Model Details"):
                st.markdown("**Model Configuration**")
                st.write("- Algorithm: CatBoost Classifier")
                st.write("- Tree Depth: 4 (shallow trees for stability)")
                st.write("- Learning Rate: 0.05 (conservative)")
                st.write("- L2 Regularization: 10 (strong regularization)")
                st.write("- Early Stopping: Enabled (20 rounds)")
                st.write("- Validation Strategy: Stratified 80/20 split")
                
                st.markdown("**Training Metrics**")
                train_metrics = st.session_state.train_metrics
                st.write(f"- Accuracy: {train_metrics['accuracy']:.3f}")
                st.write(f"- Precision: {train_metrics['precision']:.3f}")
                st.write(f"- Recall: {train_metrics['recall']:.3f}")
                st.write(f"- F1: {train_metrics['f1']:.3f}")
                
                st.markdown("**Feature Importance Table**")
                st.dataframe(
                    st.session_state.feature_importance,
                    use_container_width=True
                )
    
    # Tab 4: Retention Policy Simulator
    with tab4:
        st.header("Retention Policy Simulator")
        
        if not st.session_state.model_trained:
            st.info("Run simulation from the sidebar to view policy simulation results.")
        elif not st.session_state.get('simulation_run', False):
            st.info("Click 'Run Simulation' in the sidebar to allocate interventions.")
        else:
            # Get parameters from session state
            retention_budget = st.session_state.get('retention_budget', 50000.0)
            intervention_cost = st.session_state.get('intervention_cost', 100.0)
            risk_tolerance = st.session_state.get('risk_tolerance', 1.0)
            
            # Allocate interventions
            df_allocated = allocate_interventions(
                st.session_state.df_processed,
                st.session_state.churn_probabilities,
                st.session_state.customer_revenue,
                retention_budget,
                intervention_cost,
                risk_tolerance
            )
            
            # Display results
            n_interventions = df_allocated['intervention_allocated'].sum()
            total_cost = n_interventions * intervention_cost
            expected_revenue_saved = df_allocated['revenue_saved'].sum()
            net_benefit = expected_revenue_saved - total_cost
            roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Interventions Allocated", f"{n_interventions:,}")
            with col2:
                st.metric("Total Cost", f"${total_cost:,.0f}")
            with col3:
                st.metric("Expected Revenue Saved", f"${expected_revenue_saved:,.0f}")
            with col4:
                st.metric("Net Benefit", f"${net_benefit:,.0f}", f"{roi:.1f}% ROI")
            
            st.divider()
            
            # Allocation Visualization
            st.subheader("Intervention Allocation")
            fig_alloc = create_allocation_summary_plot(df_allocated)
            st.plotly_chart(fig_alloc, use_container_width=True)
            
            st.divider()
            
            # Allocation Details Table
            st.subheader("Allocation Details")
            display_cols = ['churn_probability', 'customer_revenue', 'revenue_at_risk_before', 
                           'revenue_at_risk_after', 'revenue_saved', 'intervention_allocated']
            
            # Add segment column if available
            if 'segment' in df_allocated.columns:
                display_cols.insert(0, 'segment')
            
            df_display = df_allocated[display_cols].copy()
            df_display = df_display.sort_values('revenue_saved', ascending=False)
            
            st.dataframe(df_display, use_container_width=True)
    
    # Tab 5: Scenario Comparison
    with tab5:
        st.header("Scenario Comparison")
        
        if not st.session_state.model_trained:
            st.info("Run simulation from the sidebar to view scenario comparisons.")
        elif not st.session_state.get('simulation_run', False):
            st.info("Click 'Run Simulation' in the sidebar to generate scenarios.")
        else:
            # Get parameters from session state
            retention_budget = st.session_state.get('retention_budget', 50000.0)
            intervention_cost = st.session_state.get('intervention_cost', 100.0)
            
            # Create cache key based on parameters (deterministic caching)
            cache_key = (
                len(st.session_state.df_processed),
                retention_budget,
                intervention_cost,
                tuple(st.session_state.churn_probabilities[:10]) if len(st.session_state.churn_probabilities) > 10 else tuple(st.session_state.churn_probabilities)
            )
            
            # Check if scenarios are cached
            if (st.session_state.scenarios_cache_key == cache_key and 
                'scenarios' in st.session_state.scenarios_cache):
                scenarios = st.session_state.scenarios_cache['scenarios']
            else:
                # Simulate multiple scenarios (only if not cached)
                scenarios = {}
                
                # No intervention
                scenarios['No Intervention'] = simulate_scenario(
                    st.session_state.df_processed,
                    st.session_state.churn_probabilities,
                    st.session_state.customer_revenue,
                    0,  # No budget
                    intervention_cost,
                    1.0
                )
                
                # Conservative (lower risk tolerance)
                scenarios['Conservative'] = simulate_scenario(
                    st.session_state.df_processed,
                    st.session_state.churn_probabilities,
                    st.session_state.customer_revenue,
                    retention_budget,
                    intervention_cost,
                    0.7  # Lower risk tolerance
                )
                
                # Balanced (default)
                scenarios['Balanced'] = simulate_scenario(
                    st.session_state.df_processed,
                    st.session_state.churn_probabilities,
                    st.session_state.customer_revenue,
                    retention_budget,
                    intervention_cost,
                    1.0  # Balanced
                )
                
                # Aggressive (higher risk tolerance)
                scenarios['Aggressive'] = simulate_scenario(
                    st.session_state.df_processed,
                    st.session_state.churn_probabilities,
                    st.session_state.customer_revenue,
                    retention_budget,
                    intervention_cost,
                    1.5  # Higher risk tolerance
                )
                
                # Cache scenarios
                st.session_state.scenarios_cache['scenarios'] = scenarios
                st.session_state.scenarios_cache_key = cache_key
            
            # Compare scenarios
            comparison_df = compare_scenarios(scenarios)
            
            # Display comparison metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Net Benefit Comparison")
                fig_net = create_scenario_comparison_plot(comparison_df)
                st.plotly_chart(fig_net, use_container_width=True)
            
            with col2:
                st.subheader("ROI Comparison")
                fig_roi = create_roi_comparison_plot(comparison_df)
                st.plotly_chart(fig_roi, use_container_width=True)
            
            st.divider()
            
            # Comparison Table
            st.subheader("Scenario Comparison Table")
            st.dataframe(comparison_df, use_container_width=True)
            
            st.divider()
            
            # Risk Analysis (Best/Worst Case)
            st.subheader("Risk Analysis: Best/Worst Case Outcomes")
            
            # Use balanced scenario for risk analysis
            balanced_allocation = scenarios['Balanced']['allocation_df']
            best_worst = calculate_best_worst_case(balanced_allocation)
            
            fig_risk = create_risk_analysis_plot(best_worst)
            st.plotly_chart(fig_risk, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Revenue Saved Distribution**")
                rev_stats = best_worst['revenue_saved']
                st.write(f"- Expected: ${rev_stats['mean']:,.0f}")
                st.write(f"- Best Case (95th %ile): ${rev_stats['best_case_95']:,.0f}")
                st.write(f"- Worst Case (5th %ile): ${rev_stats['worst_case_5']:,.0f}")
                st.write(f"- Std Dev: ${rev_stats['std']:,.0f}")
            
            with col2:
                st.markdown("**Customers Saved Distribution**")
                cust_stats = best_worst['customers_saved']
                st.write(f"- Expected: {cust_stats['mean']:.1f}")
                st.write(f"- Best Case (95th %ile): {cust_stats['best_case_95']:.1f}")
                st.write(f"- Worst Case (5th %ile): {cust_stats['worst_case_5']:.1f}")
                st.write(f"- Std Dev: {cust_stats['std']:.1f}")
    
    # Tab 6: Explainable Insights (formerly AI Strategy Assistant)
    with tab6:
        st.header("📊 Explainable Insights")
        
        # Display Demo Mode badge
        st.info("🔬 **Explainable ML Mode (No External APIs)**: All insights are generated using "
                "rule-based logic, feature importance analysis, and statistical heuristics. "
                "No API keys or external services required.")
        
        if not st.session_state.model_trained:
            st.info("Run simulation from the sidebar to access explainable insights.")
        else:
            # Churn Drivers Explanation
            st.subheader("Churn Drivers Analysis")
            
            feature_importance_dict = dict(zip(
                st.session_state.feature_importance['feature'],
                st.session_state.feature_importance['importance']
            ))
            
            with st.spinner("Analyzing feature importance..."):
                churn_drivers_explanation = explain_churn_drivers(
                    feature_importance_dict,
                    top_n=5
                )
            st.markdown(churn_drivers_explanation)
            
            st.divider()
            
            # Retention Policy Explanation
            if st.session_state.get('simulation_run', False):
                st.subheader("Retention Policy Strategy")
                
                # Get parameters from session state
                retention_budget = st.session_state.get('retention_budget', 50000.0)
                intervention_cost = st.session_state.get('intervention_cost', 100.0)
                
                # Strategy selector
                selected_strategy = st.selectbox(
                    "Select Strategy to Analyze:",
                    ["Conservative", "Balanced", "Aggressive"],
                    index=1,  # Default to Balanced
                    help="Choose which retention strategy to analyze and get insights for"
                )
                
                # Map strategy to risk tolerance
                strategy_params = {
                    "Conservative": {"risk_tolerance": 0.7, "name": "Conservative"},
                    "Balanced": {"risk_tolerance": 1.0, "name": "Balanced"},
                    "Aggressive": {"risk_tolerance": 1.5, "name": "Aggressive"}
                }
                
                selected_params = strategy_params[selected_strategy]
                
                # Run scenario for selected strategy
                selected_scenario = simulate_scenario(
                    st.session_state.df_processed,
                    st.session_state.churn_probabilities,
                    st.session_state.customer_revenue,
                    retention_budget,
                    intervention_cost,
                    selected_params["risk_tolerance"]
                )
                
                with st.spinner(f"Analyzing {selected_strategy.lower()} retention strategy..."):
                    policy_explanation = explain_retention_policy(
                        selected_scenario,
                        retention_budget,
                        intervention_cost,
                        selected_params["name"],
                        selected_params["risk_tolerance"]
                    )
                st.markdown(policy_explanation)
                
                st.divider()
                
                # Risk & Uncertainty Explanation
                st.subheader("Risk & Uncertainty Analysis")
                st.caption(f"Risk analysis for {selected_strategy} strategy")
                
                selected_allocation = selected_scenario['allocation_df']
                best_worst = calculate_best_worst_case(selected_allocation)
                
                with st.spinner("Analyzing risk..."):
                    risk_explanation = explain_risk_uncertainty(best_worst)
                st.markdown(risk_explanation)
                
                st.divider()
                
                # What-If Analysis
                st.subheader("What-If Scenario Analysis")
                st.caption("Enter a question about budget, costs, or strategy to compare scenarios.")
                
                user_question = st.text_input(
                    "Ask a what-if question:",
                    placeholder="e.g., What if we double the budget? What if intervention cost increases by 50%?"
                )
                
                if user_question and user_question.strip():
                    # Get scenario comparison for what-if (reuse cached scenarios if available)
                    cache_key_whatif = (
                        len(st.session_state.df_processed),
                        retention_budget,
                        intervention_cost,
                        tuple(st.session_state.churn_probabilities[:10]) if len(st.session_state.churn_probabilities) > 10 else tuple(st.session_state.churn_probabilities)
                    )
                    
                    if (st.session_state.scenarios_cache_key == cache_key_whatif and 
                        'scenarios' in st.session_state.scenarios_cache):
                        scenarios_whatif = st.session_state.scenarios_cache['scenarios']
                    else:
                        # Get scenario comparison for what-if
                        scenarios_whatif = {
                            'No Intervention': simulate_scenario(
                                st.session_state.df_processed,
                                st.session_state.churn_probabilities,
                                st.session_state.customer_revenue,
                                0,
                                intervention_cost,
                                1.0
                            ),
                            'Conservative': simulate_scenario(
                                st.session_state.df_processed,
                                st.session_state.churn_probabilities,
                                st.session_state.customer_revenue,
                                retention_budget,
                                intervention_cost,
                                0.7
                            ),
                            'Balanced': simulate_scenario(
                                st.session_state.df_processed,
                                st.session_state.churn_probabilities,
                                st.session_state.customer_revenue,
                                retention_budget,
                                intervention_cost,
                                1.0
                            ),
                            'Aggressive': simulate_scenario(
                                st.session_state.df_processed,
                                st.session_state.churn_probabilities,
                                st.session_state.customer_revenue,
                                retention_budget,
                                intervention_cost,
                                1.5
                            )
                        }
                    
                    with st.spinner("Analyzing scenarios..."):
                        whatif_analysis = analyze_what_if_scenario(
                            scenarios_whatif,
                            user_question
                        )
                    st.markdown(whatif_analysis)
            else:
                st.info("Run simulation from the sidebar to access policy and risk analysis.")


if __name__ == "__main__":
    main()

