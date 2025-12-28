"""
Scenario comparison and evaluation metrics.
"""

import pandas as pd
import numpy as np


def compare_scenarios(scenarios_dict):
    """
    Compare multiple retention scenarios.
    
    Args:
        scenarios_dict: Dictionary of scenario_name -> scenario_results
        
    Returns:
        DataFrame: Comparison table
    """
    comparison_data = []
    
    for scenario_name, results in scenarios_dict.items():
        comparison_data.append({
            'Scenario': scenario_name,
            'Interventions': results['n_interventions'],
            'Total Cost': results['total_cost'],
            'Expected Revenue Saved': results['expected_revenue_saved'],
            'Net Benefit': results['net_benefit'],
            'ROI (%)': results['roi'],
            'Expected Churn': results['expected_churn_after'],
            'Customers Saved': results['customers_saved']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df


def calculate_best_worst_case(df_allocated, n_simulations=1000):
    """
    Calculate best-case and worst-case outcomes using Monte Carlo simulation.
    
    Args:
        df_allocated: DataFrame with intervention allocation
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        dict: Best/worst case statistics
    """
    np.random.seed(42)
    
    revenues_saved = []
    customers_saved_list = []
    
    for _ in range(n_simulations):
        # Simulate actual churn (random based on probability)
        df_sim = df_allocated.copy()
        df_sim['actual_churn'] = np.random.binomial(1, df_sim['churn_probability_after'])
        
        # Calculate actual revenue saved (only for customers who would have churned)
        df_sim['actual_revenue_saved'] = df_sim.apply(
            lambda row: row['customer_revenue'] if (row['churn_probability'] > row['churn_probability_after'] and row['actual_churn'] == 0) else 0,
            axis=1
        )
        
        total_revenue_saved = df_sim['actual_revenue_saved'].sum()
        total_customers_saved = len(df_sim[(df_sim['churn_probability'] > df_sim['churn_probability_after']) & (df_sim['actual_churn'] == 0)])
        
        revenues_saved.append(total_revenue_saved)
        customers_saved_list.append(total_customers_saved)
    
    revenues_saved = np.array(revenues_saved)
    customers_saved_list = np.array(customers_saved_list)
    
    return {
        'revenue_saved': {
            'mean': np.mean(revenues_saved),
            'std': np.std(revenues_saved),
            'best_case_95': np.percentile(revenues_saved, 95),
            'worst_case_5': np.percentile(revenues_saved, 5),
            'best_case_max': np.max(revenues_saved),
            'worst_case_min': np.min(revenues_saved)
        },
        'customers_saved': {
            'mean': np.mean(customers_saved_list),
            'std': np.std(customers_saved_list),
            'best_case_95': np.percentile(customers_saved_list, 95),
            'worst_case_5': np.percentile(customers_saved_list, 5),
            'best_case_max': np.max(customers_saved_list),
            'worst_case_min': np.min(customers_saved_list)
        }
    }


def create_scenario_comparison_plot(comparison_df):
    """
    Create bar chart comparing scenarios.
    
    Args:
        comparison_df: DataFrame with scenario comparison data
        
    Returns:
        plotly figure
    """
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Bar(
            name='Net Benefit',
            x=comparison_df['Scenario'],
            y=comparison_df['Net Benefit'],
            marker_color='#4A90E2'
        )
    ])
    
    fig.update_layout(
        title="Scenario Comparison: Net Benefit",
        xaxis_title="Scenario",
        yaxis_title="Net Benefit ($)",
        font=dict(size=12),
        height=350
    )
    
    return fig


def create_roi_comparison_plot(comparison_df):
    """
    Create bar chart comparing ROI across scenarios.
    
    Args:
        comparison_df: DataFrame with scenario comparison data
        
    Returns:
        plotly figure
    """
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Bar(
            name='ROI',
            x=comparison_df['Scenario'],
            y=comparison_df['ROI (%)'],
            marker_color='#4A90E2'
        )
    ])
    
    fig.update_layout(
        title="Scenario Comparison: ROI",
        xaxis_title="Scenario",
        yaxis_title="ROI (%)",
        font=dict(size=12),
        height=350
    )
    
    return fig


def create_risk_analysis_plot(best_worst_case):
    """
    Create visualization of risk analysis (best/worst case outcomes).
    
    Args:
        best_worst_case: Dictionary with best/worst case statistics
        
    Returns:
        plotly figure
    """
    import plotly.graph_objects as go
    
    revenue_stats = best_worst_case['revenue_saved']
    
    fig = go.Figure()
    
    # Mean
    fig.add_trace(go.Bar(
        name='Expected',
        x=['Revenue Saved'],
        y=[revenue_stats['mean']],
        marker_color='#4A90E2',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[revenue_stats['best_case_95'] - revenue_stats['mean']],
            arrayminus=[revenue_stats['mean'] - revenue_stats['worst_case_5']]
        )
    ))
    
    fig.update_layout(
        title="Risk Analysis: Revenue Saved (Best/Worst Case)",
        yaxis_title="Revenue Saved ($)",
        font=dict(size=12),
        height=350,
        showlegend=False
    )
    
    return fig


