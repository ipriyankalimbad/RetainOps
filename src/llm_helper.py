"""
Rule-based explainable insights generator (Demo Mode - No External APIs).
Replaces LLM-based explanations with deterministic, interpretable explanations
derived from ML feature importance, statistics, and business heuristics.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def explain_churn_drivers(feature_importance: Dict[str, float], top_n: int = 5) -> str:
    """
    Generate rule-based explanation of top churn drivers using feature importance.
    
    Args:
        feature_importance: Dictionary of feature names to importance scores
        top_n: Number of top features to explain
        
    Returns:
        str: Explanation text
    """
    if not feature_importance:
        return "No feature importance data available."
    
    # Get top N features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    if not top_features:
        return "No features available for analysis."
    
    # Generate explanation based on feature names and importance scores
    explanation_parts = [
        "## Top Churn Drivers Analysis\n\n",
        "Based on the machine learning model's feature importance analysis, "
        "the following factors are most predictive of customer churn:\n\n"
    ]
    
    # Common business interpretations for feature names
    feature_interpretations = {
        'contract_type': 'contract type and commitment level',
        'tenure_months': 'customer tenure and relationship length',
        'monthly_charges': 'pricing and monthly cost',
        'total_charges': 'total customer value',
        'payment_method': 'payment preference and convenience',
        'internet_service': 'service type and quality',
        'age': 'customer demographics',
        'revenue': 'customer revenue contribution'
    }
    
    for idx, (feature, importance) in enumerate(top_features, 1):
        feature_display = feature_interpretations.get(feature.lower(), feature.replace('_', ' ').title())
        importance_pct = (importance / sum([f[1] for f in top_features])) * 100
        
        explanation_parts.append(
            f"**{idx}. {feature_display.title()}** (Importance: {importance:.4f}, "
            f"Relative Weight: {importance_pct:.1f}%)\n"
        )
        
        # Add specific interpretation based on feature type
        if 'contract' in feature.lower() or 'tenure' in feature.lower():
            explanation_parts.append(
                "   - Customer commitment level strongly predicts retention. "
                "Shorter contracts or newer customers show higher churn risk.\n"
            )
        elif 'charge' in feature.lower() or 'revenue' in feature.lower() or 'price' in feature.lower():
            explanation_parts.append(
                "   - Pricing sensitivity is a key churn driver. "
                "Both high-cost and low-value customers may be at risk.\n"
            )
        elif 'payment' in feature.lower():
            explanation_parts.append(
                "   - Payment method preferences correlate with churn behavior. "
                "Different payment types indicate different customer segments.\n"
            )
        else:
            explanation_parts.append(
                "   - This feature significantly contributes to churn prediction accuracy.\n"
            )
    
    explanation_parts.append(
        "\n### Business Implications\n\n"
        "Focus retention efforts on customers with characteristics matching the highest-importance features. "
        "These drivers represent the most actionable levers for reducing churn. "
        "Consider targeting interventions (e.g., discounts, outreach, service improvements) based on "
        "these predictive factors to maximize retention ROI."
    )
    
    return "".join(explanation_parts)


def explain_retention_policy(scenario_results: Dict[str, Any], budget: float, intervention_cost: float, strategy_name: str = "Balanced", risk_tolerance: float = 1.0) -> str:
    """
    Generate rule-based explanation of retention policy strategy.
    
    Args:
        scenario_results: Dictionary with scenario metrics
        budget: Total budget used
        intervention_cost: Cost per intervention
        strategy_name: Name of the strategy (Conservative, Balanced, Aggressive)
        risk_tolerance: Risk tolerance value used (0.7, 1.0, 1.5)
        
    Returns:
        str: Explanation text
    """
    n_interventions = scenario_results.get('n_interventions', 0)
    expected_revenue_saved = scenario_results.get('expected_revenue_saved', 0)
    roi = scenario_results.get('roi', 0)
    net_benefit = scenario_results.get('net_benefit', 0)
    total_cost = scenario_results.get('total_cost', n_interventions * intervention_cost)
    
    explanation_parts = [
        f"## {strategy_name} Strategy Analysis\n\n",
        "### Strategy Overview\n\n"
    ]
    
    # Strategy-specific explanation
    if strategy_name == "Conservative":
        explanation_parts.append(
            "**Conservative Strategy** (Risk Tolerance: 0.7): This strategy prioritizes **high-value customers** "
            "over high-risk customers. It uses a lower risk weight, meaning customer revenue has more influence "
            "than churn probability in the scoring. This approach is best when you want to protect your most "
            "valuable customers, even if they have lower churn risk.\n\n"
        )
    elif strategy_name == "Aggressive":
        explanation_parts.append(
            "**Aggressive Strategy** (Risk Tolerance: 1.5): This strategy prioritizes **high-risk customers** "
            "regardless of their value. It uses a higher risk weight, meaning churn probability has more influence "
            "than customer revenue in the scoring. This approach is best when you want to prevent as many churns "
            "as possible, focusing on customers most likely to leave.\n\n"
        )
    else:  # Balanced
        explanation_parts.append(
            "**Balanced Strategy** (Risk Tolerance: 1.0): This strategy balances both **churn risk and customer value** "
            "equally. It uses equal weighting for both factors in the scoring. This approach is best when you want "
            "to optimize for overall ROI, considering both the probability of churn and the financial impact.\n\n"
        )
    
    explanation_parts.append("### Allocation Summary\n\n")
    
    # Budget utilization analysis
    budget_utilization = (total_cost / budget * 100) if budget > 0 else 0
    explanation_parts.append(
        f"- **Budget Utilized**: ${total_cost:,.0f} of ${budget:,.0f} available ("
        f"{budget_utilization:.1f}% utilization)\n"
    )
    explanation_parts.append(f"- **Interventions Allocated**: {n_interventions:,} customers\n")
    explanation_parts.append(f"- **Cost per Intervention**: ${intervention_cost:,.2f}\n\n")
    
    # ROI analysis
    explanation_parts.append("### Financial Performance\n\n")
    explanation_parts.append(f"- **Expected Revenue Saved**: ${expected_revenue_saved:,.0f}\n")
    explanation_parts.append(f"- **Net Benefit**: ${net_benefit:,.0f}\n")
    explanation_parts.append(f"- **Return on Investment (ROI)**: {roi:.1f}%\n\n")
    
    # Strategy-specific targeting analysis
    explanation_parts.append("### Targeting Analysis\n\n")
    
    if strategy_name == "Conservative":
        explanation_parts.append(
            "**Targeting Focus**: High-value customers with moderate to high churn risk. "
            "This strategy protects revenue by focusing on customers who contribute the most financially, "
            "even if their immediate churn risk is not the highest.\n\n"
        )
    elif strategy_name == "Aggressive":
        explanation_parts.append(
            "**Targeting Focus**: High-risk customers regardless of value. "
            "This strategy prevents churn by targeting customers most likely to leave, "
            "prioritizing risk reduction over revenue protection.\n\n"
        )
    else:  # Balanced
        explanation_parts.append(
            "**Targeting Focus**: Customers with balanced risk-value profiles. "
            "This strategy optimizes ROI by targeting customers where both churn probability "
            "and revenue impact are considered equally.\n\n"
        )
    
    # Strategic interpretation
    explanation_parts.append("### Financial Performance Assessment\n\n")
    
    if roi > 200:
        explanation_parts.append(
            "**Excellent ROI**: This strategy demonstrates very strong financial performance. "
            "The expected revenue saved significantly exceeds intervention costs, indicating "
            "high-value targets were prioritized effectively.\n\n"
        )
    elif roi > 100:
        explanation_parts.append(
            "**Strong ROI**: This strategy shows solid financial returns. The allocation "
            "effectively balances intervention costs with expected revenue protection.\n\n"
        )
    elif roi > 50:
        explanation_parts.append(
            "**Positive ROI**: This strategy generates positive returns, indicating effective "
            "prioritization of high-value, high-risk customers.\n\n"
        )
    elif roi > 0:
        explanation_parts.append(
            "**Modest ROI**: While positive, this strategy shows relatively modest returns. "
            "Consider optimizing targeting criteria or intervention effectiveness.\n\n"
        )
    else:
        explanation_parts.append(
            "**Negative ROI**: This strategy does not generate positive returns. "
            "Review targeting criteria, intervention costs, or consider alternative approaches.\n\n"
        )
    
    # Efficiency analysis
    if n_interventions > 0:
        revenue_per_intervention = expected_revenue_saved / n_interventions
        explanation_parts.append(
            f"**Efficiency Metrics**: Average revenue saved per intervention: "
            f"${revenue_per_intervention:,.0f}. "
        )
        
        if revenue_per_intervention > intervention_cost * 2:
            explanation_parts.append("Interventions are highly cost-effective.\n\n")
        elif revenue_per_intervention > intervention_cost:
            explanation_parts.append("Interventions generate positive returns.\n\n")
        else:
            explanation_parts.append(
                "Consider reviewing intervention targeting to improve efficiency.\n\n"
            )
    
    # Strategy-specific recommendations
    explanation_parts.append("### Recommendations\n\n")
    
    if strategy_name == "Conservative":
        explanation_parts.append(
            "- **Strategy Fit**: This conservative approach is ideal when customer lifetime value "
            "is critical and you want to protect your highest-revenue customers.\n"
        )
        if roi < 50:
            explanation_parts.append(
                "- **Consider Alternative**: Low ROI may indicate that high-value customers have low "
                "churn risk. Consider switching to Balanced or Aggressive strategy.\n"
            )
    elif strategy_name == "Aggressive":
        explanation_parts.append(
            "- **Strategy Fit**: This aggressive approach is ideal when preventing churn volume "
            "is the priority, even if some interventions target lower-value customers.\n"
        )
        if roi < 50:
            explanation_parts.append(
                "- **Consider Alternative**: Low ROI may indicate that high-risk customers have low "
                "value. Consider switching to Conservative or Balanced strategy.\n"
            )
    else:  # Balanced
        explanation_parts.append(
            "- **Strategy Fit**: This balanced approach optimizes for overall ROI by considering "
            "both risk and value equally.\n"
        )
    
    if budget_utilization < 80:
        explanation_parts.append(
            "- **Budget Opportunity**: Not all budget was utilized ({budget_utilization:.1f}%). "
            "Consider increasing intervention scope or adjusting targeting criteria to capture "
            "more high-value targets.\n"
        )
    
    if roi > 100 and n_interventions < budget / intervention_cost * 0.8:
        explanation_parts.append(
            "- **Scale Opportunity**: Strong ROI ({roi:.1f}%) suggests potential to scale interventions "
            "within remaining budget capacity. Consider expanding the intervention pool.\n"
        )
    
    explanation_parts.append(
        "- **Monitor Performance**: Track actual outcomes against these expected metrics "
        "to validate model predictions and refine strategy.\n"
    )
    explanation_parts.append(
        "- **Compare Strategies**: Review the Scenario Comparison tab to see how this strategy "
        "performs relative to Conservative and Aggressive approaches.\n"
    )
    explanation_parts.append(
        "- **Continuous Improvement**: Use these insights to optimize intervention targeting, "
        "improve customer segmentation, and enhance retention program effectiveness."
    )
    
    return "".join(explanation_parts)


def analyze_what_if_scenario(scenario_comparison: Dict[str, Dict[str, Any]], question: str) -> str:
    """
    Analyze what-if scenarios using rule-based logic and scenario comparison.
    
    Args:
        scenario_comparison: Dictionary of scenario_name -> scenario_results
        question: User's what-if question (analyzed for keywords)
        
    Returns:
        str: Analysis and recommendations
    """
    if not scenario_comparison:
        return "No scenario data available for analysis."
    
    question_lower = question.lower()
    
    explanation_parts = [
        "## What-If Scenario Analysis\n\n",
        f"**Your Question**: {question}\n\n",
        "### Scenario Comparison\n\n"
    ]
    
    # Extract key metrics from scenarios
    scenarios_data = []
    for name, results in scenario_comparison.items():
        scenarios_data.append({
            'name': name,
            'roi': results.get('roi', 0),
            'net_benefit': results.get('net_benefit', 0),
            'interventions': results.get('n_interventions', 0),
            'cost': results.get('total_cost', 0),
            'revenue_saved': results.get('expected_revenue_saved', 0)
        })
    
    # Sort by ROI for recommendation
    scenarios_data.sort(key=lambda x: x['roi'], reverse=True)
    
    # Display scenario comparison
    for scenario in scenarios_data:
        explanation_parts.append(
            f"**{scenario['name']} Strategy**:\n"
            f"- ROI: {scenario['roi']:.1f}%\n"
            f"- Net Benefit: ${scenario['net_benefit']:,.0f}\n"
            f"- Interventions: {scenario['interventions']:,}\n"
            f"- Revenue Saved: ${scenario['revenue_saved']:,.0f}\n\n"
        )
    
    # Identify best performing scenario
    best_scenario = scenarios_data[0] if scenarios_data else None
    
    explanation_parts.append("### Analysis\n\n")
    
    # Keyword-based analysis
    if 'budget' in question_lower or 'spend' in question_lower or 'cost' in question_lower:
        explanation_parts.append(
            "**Budget Impact Analysis**: Higher budgets typically enable more interventions, "
            "but ROI may decrease as lower-priority targets are included. The optimal budget "
            "balances total revenue protection with cost efficiency.\n\n"
        )
        
        if best_scenario and best_scenario['name'] != 'No Intervention':
            explanation_parts.append(
                f"The {best_scenario['name']} strategy shows the best ROI ({best_scenario['roi']:.1f}%). "
                f"This suggests the current budget allocation is effectively targeting high-value customers.\n\n"
            )
    
    if 'double' in question_lower or 'increase' in question_lower or 'more' in question_lower:
        explanation_parts.append(
            "**Scaling Analysis**: Increasing budget or interventions generally increases total "
            "revenue saved, but may reduce ROI as marginal returns decrease. Monitor the "
            "trade-off between scale and efficiency.\n\n"
        )
    
    if 'risk' in question_lower or 'tolerance' in question_lower or 'conservative' in question_lower or 'aggressive' in question_lower or 'strategy' in question_lower:
        explanation_parts.append(
            "**Risk Strategy Comparison**: Different risk tolerance levels prioritize different "
            "customer segments:\n\n"
            "- **Conservative (0.7)**: Prioritizes high-value customers over high-risk customers. "
            "Best when protecting revenue from your most valuable customers is the priority.\n"
            "- **Balanced (1.0)**: Equally weights churn risk and customer value. "
            "Best for optimizing overall ROI.\n"
            "- **Aggressive (1.5)**: Prioritizes high-risk customers regardless of value. "
            "Best when preventing as many churns as possible is the priority.\n\n"
        )
        
        conservative = next((s for s in scenarios_data if 'Conservative' in s['name']), None)
        balanced = next((s for s in scenarios_data if 'Balanced' in s['name']), None)
        aggressive = next((s for s in scenarios_data if 'Aggressive' in s['name']), None)
        
        if conservative and balanced and aggressive:
            # Compare strategies
            strategies_compared = [
                ("Conservative", conservative),
                ("Balanced", balanced),
                ("Aggressive", aggressive)
            ]
            strategies_compared.sort(key=lambda x: x[1]['roi'], reverse=True)
            
            explanation_parts.append("**Strategy Performance Ranking**:\n\n")
            for rank, (name, data) in enumerate(strategies_compared, 1):
                explanation_parts.append(
                    f"{rank}. **{name}**: ROI {data['roi']:.1f}%, "
                    f"Net Benefit ${data['net_benefit']:,.0f}, "
                    f"{data['interventions']:,} interventions\n"
                )
            
            explanation_parts.append("\n")
            
            best_strategy = strategies_compared[0]
            if best_strategy[0] == "Conservative":
                explanation_parts.append(
                    "The **Conservative** strategy performs best, indicating that prioritizing "
                    "high-value customers generates the highest ROI in this dataset. This suggests "
                    "that customer value is a stronger predictor of retention ROI than churn risk alone.\n\n"
                )
            elif best_strategy[0] == "Aggressive":
                explanation_parts.append(
                    "The **Aggressive** strategy performs best, indicating that prioritizing "
                    "high-risk customers generates the highest ROI. This suggests that churn risk "
                    "is a stronger predictor of retention ROI than customer value alone.\n\n"
                )
            else:
                explanation_parts.append(
                    "The **Balanced** strategy performs best, indicating that equally weighting "
                    "churn risk and customer value optimizes ROI. This suggests both factors are "
                    "important for effective retention targeting.\n\n"
                )
    
    # General recommendation
    if best_scenario and best_scenario['name'] != 'No Intervention':
        explanation_parts.append(
            f"### Recommendation\n\n"
            f"Based on the scenario comparison, the **{best_scenario['name']}** strategy "
            f"demonstrates the best performance with {best_scenario['roi']:.1f}% ROI and "
            f"${best_scenario['net_benefit']:,.0f} net benefit. "
        )
        
        explanation_parts.append(
            "This strategy effectively balances intervention costs with expected revenue protection. "
            "Consider implementing this approach while monitoring actual outcomes to validate predictions."
        )
    else:
        explanation_parts.append(
            "### Recommendation\n\n"
            "Compare the scenario metrics above to identify the strategy that best aligns "
            "with your business objectives, risk tolerance, and resource constraints."
        )
    
    return "".join(explanation_parts)


def explain_risk_uncertainty(best_worst_case: Dict[str, Any]) -> str:
    """
    Explain risk and uncertainty using Monte Carlo simulation statistics.
    
    Args:
        best_worst_case: Dictionary with best/worst case statistics
        
    Returns:
        str: Explanation text
    """
    revenue_stats = best_worst_case.get('revenue_saved', {})
    customers_stats = best_worst_case.get('customers_saved', {})
    
    if not revenue_stats:
        return "No risk analysis data available."
    
    mean_rev = revenue_stats.get('mean', 0)
    std_rev = revenue_stats.get('std', 0)
    best_95 = revenue_stats.get('best_case_95', 0)
    worst_5 = revenue_stats.get('worst_case_5', 0)
    
    explanation_parts = [
        "## Risk & Uncertainty Analysis\n\n",
        "This analysis uses Monte Carlo simulation (1,000 iterations) to quantify uncertainty "
        "in retention strategy outcomes, accounting for the probabilistic nature of churn predictions.\n\n",
        "### Revenue Saved Distribution\n\n",
        f"- **Expected Value (Mean)**: ${mean_rev:,.0f}\n",
        f"- **Best Case (95th Percentile)**: ${best_95:,.0f}\n",
        f"- **Worst Case (5th Percentile)**: ${worst_5:,.0f}\n",
        f"- **Standard Deviation**: ${std_rev:,.0f}\n",
        f"- **Coefficient of Variation**: {(std_rev / mean_rev * 100) if mean_rev > 0 else 0:.1f}%\n\n"
    ]
    
    # Risk assessment
    explanation_parts.append("### Risk Assessment\n\n")
    
    cv = (std_rev / mean_rev * 100) if mean_rev > 0 else 0
    
    if cv < 20:
        explanation_parts.append(
            "**Low Uncertainty**: The coefficient of variation is below 20%, indicating "
            "relatively predictable outcomes. You can expect results close to the expected value "
            "with high confidence.\n\n"
        )
    elif cv < 40:
        explanation_parts.append(
            "**Moderate Uncertainty**: The coefficient of variation suggests moderate variability "
            "in outcomes. While the expected value is a good estimate, actual results may vary "
            "within a reasonable range.\n\n"
        )
    else:
        explanation_parts.append(
            "**High Uncertainty**: Significant variability in potential outcomes. The expected value "
            "represents an average, but actual results could vary substantially. Consider this "
            "uncertainty in decision-making and planning.\n\n"
        )
    
    # Range analysis
    range_pct = ((best_95 - worst_5) / mean_rev * 100) if mean_rev > 0 else 0
    
    explanation_parts.append(
        f"**Outcome Range**: The 90% confidence interval spans from ${worst_5:,.0f} to "
        f"${best_95:,.0f}, representing {range_pct:.1f}% of the expected value. "
    )
    
    if range_pct < 50:
        explanation_parts.append("This narrow range indicates reliable predictions.\n\n")
    elif range_pct < 100:
        explanation_parts.append("This moderate range suggests some variability should be expected.\n\n")
    else:
        explanation_parts.append(
            "This wide range indicates significant uncertainty—plan for various outcome scenarios.\n\n"
        )
    
    # Worst case analysis
    explanation_parts.append("### Worst-Case Scenario Planning\n\n")
    explanation_parts.append(
        f"The 5th percentile outcome (${worst_5:,.0f}) represents a conservative estimate "
        "of revenue saved. Even in this pessimistic scenario, the strategy may still generate "
        "positive returns, depending on intervention costs.\n\n"
    )
    
    # Best case analysis
    explanation_parts.append("### Best-Case Scenario Potential\n\n")
    explanation_parts.append(
        f"The 95th percentile outcome (${best_95:,.0f}) represents an optimistic estimate. "
        "While unlikely, this outcome demonstrates the strategy's upside potential under "
        "favorable conditions.\n\n"
    )
    
    # Decision-making guidance
    explanation_parts.append("### Decision-Making Guidance\n\n")
    explanation_parts.append(
        "- **Use Expected Value** for planning and budget allocation\n"
        "- **Consider Worst Case** when assessing risk tolerance and setting expectations\n"
        "- **Understand Variability** when communicating uncertainty to stakeholders\n"
        "- **Monitor Actuals** to validate predictions and improve model accuracy over time\n\n"
    )
    
    explanation_parts.append(
        "The probabilistic nature of churn means actual outcomes will vary. This analysis "
        "helps quantify that uncertainty and supports informed decision-making under conditions "
        "of risk."
    )
    
    return "".join(explanation_parts)
