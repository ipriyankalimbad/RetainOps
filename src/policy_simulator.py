"""
Budget-constrained retention policy simulator.
"""

import pandas as pd
import numpy as np


def calculate_intervention_score(churn_probability, customer_revenue, intervention_cost, risk_weight=1.0):
    """
    Calculate intervention priority score for a customer.
    
    Higher score = higher priority for intervention.
    
    Args:
        churn_probability: Customer's churn probability
        customer_revenue: Customer's annual revenue
        intervention_cost: Cost of intervention
        risk_weight: Weight for risk vs value (default 1.0 = balanced)
        
    Returns:
        float: Intervention score
    """
    # Revenue at risk
    revenue_at_risk = churn_probability * customer_revenue
    
    # Expected ROI if intervention succeeds (assume 50% effectiveness)
    intervention_effectiveness = 0.5
    expected_saved_revenue = revenue_at_risk * intervention_effectiveness
    expected_roi = (expected_saved_revenue - intervention_cost) / intervention_cost if intervention_cost > 0 else 0
    
    # Combined score: prioritize high risk, high value, positive ROI
    score = (churn_probability ** risk_weight) * customer_revenue * max(0, 1 + expected_roi)
    
    return score


def allocate_interventions(df, churn_probabilities, customer_revenue, total_budget, intervention_cost, risk_tolerance=1.0):
    """
    Allocate retention interventions based on budget constraints.
    
    Args:
        df: DataFrame with customer data
        churn_probabilities: Array of churn probabilities
        customer_revenue: Series of customer revenue values
        total_budget: Total budget available for interventions
        intervention_cost: Cost per intervention
        risk_tolerance: Risk weight (1.0 = balanced, >1.0 = prioritize risk, <1.0 = prioritize value)
        
    Returns:
        DataFrame: Customer data with intervention allocation decisions
    """
    result_df = df.copy()
    result_df['churn_probability'] = churn_probabilities
    result_df['customer_revenue'] = customer_revenue
    
    # Calculate intervention scores
    result_df['intervention_score'] = result_df.apply(
        lambda row: calculate_intervention_score(
            row['churn_probability'],
            row['customer_revenue'],
            intervention_cost,
            risk_tolerance
        ),
        axis=1
    )
    
    # Sort by intervention score (descending)
    result_df = result_df.sort_values('intervention_score', ascending=False)
    
    # Allocate interventions within budget
    remaining_budget = total_budget
    result_df['intervention_allocated'] = False
    
    for idx in result_df.index:
        if remaining_budget >= intervention_cost:
            result_df.loc[idx, 'intervention_allocated'] = True
            remaining_budget -= intervention_cost
        else:
            break
    
    # Calculate expected outcomes
    intervention_effectiveness = 0.5  # Assume 50% reduction in churn probability
    
    result_df['churn_probability_after'] = result_df.apply(
        lambda row: row['churn_probability'] * (1 - intervention_effectiveness) if row['intervention_allocated'] else row['churn_probability'],
        axis=1
    )
    
    result_df['revenue_at_risk_before'] = result_df['churn_probability'] * result_df['customer_revenue']
    result_df['revenue_at_risk_after'] = result_df['churn_probability_after'] * result_df['customer_revenue']
    result_df['revenue_saved'] = result_df['revenue_at_risk_before'] - result_df['revenue_at_risk_after']
    
    # Sort back to original order
    result_df = result_df.sort_index()
    
    return result_df


def simulate_scenario(df, churn_probabilities, customer_revenue, total_budget, intervention_cost, risk_tolerance, intervention_effectiveness=0.5):
    """
    Simulate a retention scenario with given parameters.
    
    Args:
        df: DataFrame with customer data
        churn_probabilities: Array of churn probabilities
        customer_revenue: Series of customer revenue values
        total_budget: Total budget available
        intervention_cost: Cost per intervention
        risk_tolerance: Risk weight parameter
        intervention_effectiveness: Expected reduction in churn probability
        
    Returns:
        dict: Scenario results with metrics
    """
    # Allocate interventions
    df_allocated = allocate_interventions(
        df, churn_probabilities, customer_revenue, 
        total_budget, intervention_cost, risk_tolerance
    )
    
    # Calculate metrics
    n_interventions = df_allocated['intervention_allocated'].sum()
    total_cost = n_interventions * intervention_cost
    
    # Expected outcomes
    expected_revenue_at_risk_before = df_allocated['revenue_at_risk_before'].sum()
    expected_revenue_at_risk_after = df_allocated['revenue_at_risk_after'].sum()
    expected_revenue_saved = expected_revenue_at_risk_before - expected_revenue_at_risk_after
    
    # ROI calculation
    net_benefit = expected_revenue_saved - total_cost
    roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0
    
    # Expected customer loss
    expected_churn_before = df_allocated['churn_probability'].sum()
    expected_churn_after = df_allocated['churn_probability_after'].sum()
    customers_saved = expected_churn_before - expected_churn_after
    
    return {
        'n_interventions': n_interventions,
        'total_cost': total_cost,
        'expected_revenue_at_risk_before': expected_revenue_at_risk_before,
        'expected_revenue_at_risk_after': expected_revenue_at_risk_after,
        'expected_revenue_saved': expected_revenue_saved,
        'net_benefit': net_benefit,
        'roi': roi,
        'expected_churn_before': expected_churn_before,
        'expected_churn_after': expected_churn_after,
        'customers_saved': customers_saved,
        'allocation_df': df_allocated
    }


def create_allocation_summary_plot(df_allocated):
    """
    Create visualization of intervention allocation.
    
    Args:
        df_allocated: DataFrame with intervention_allocated column
        
    Returns:
        plotly figure
    """
    import plotly.graph_objects as go
    
    allocated = df_allocated[df_allocated['intervention_allocated'] == True]
    not_allocated = df_allocated[df_allocated['intervention_allocated'] == False]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=not_allocated['churn_probability'],
        y=not_allocated['customer_revenue'],
        mode='markers',
        name='No Intervention',
        marker=dict(color='#CCCCCC', size=3, opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=allocated['churn_probability'],
        y=allocated['customer_revenue'],
        mode='markers',
        name='Intervention Allocated',
        marker=dict(color='#4A90E2', size=5, opacity=0.7)
    ))
    
    fig.update_layout(
        title="Intervention Allocation",
        xaxis_title="Churn Probability",
        yaxis_title="Customer Revenue",
        font=dict(size=12),
        height=450,
        hovermode='closest'
    )
    
    return fig


