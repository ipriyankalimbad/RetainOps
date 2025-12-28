"""
LLM integration for strategy explanations and what-if analysis.
"""

import os
import openai
from typing import Dict, Any, Optional


def get_llm_client():
    """
    Initialize OpenAI client from environment variable.
    
    Returns:
        OpenAI client or None if API key not set
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    
    try:
        client = openai.OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


def explain_churn_drivers(feature_importance: Dict[str, float], top_n: int = 5) -> str:
    """
    Generate explanation of top churn drivers using LLM.
    
    Args:
        feature_importance: Dictionary of feature names to importance scores
        top_n: Number of top features to explain
        
    Returns:
        str: Explanation text
    """
    client = get_llm_client()
    if not client:
        return "LLM explanations require OPENAI_API_KEY environment variable to be set."
    
    # Get top N features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    feature_list = "\n".join([f"- {name}: {score:.4f}" for name, score in top_features])
    
    prompt = f"""As a data science consultant, explain the top churn drivers based on feature importance analysis.
Focus on business implications and actionable insights, not technical ML details.

Top features by importance:
{feature_list}

Provide a concise explanation (2-3 paragraphs) of what these features tell us about customer churn drivers.
Be specific and business-focused."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def explain_retention_policy(scenario_results: Dict[str, Any], budget: float, intervention_cost: float) -> str:
    """
    Explain the retention policy strategy and rationale.
    
    Args:
        scenario_results: Dictionary with scenario metrics
        budget: Total budget used
        intervention_cost: Cost per intervention
        
    Returns:
        str: Explanation text
    """
    client = get_llm_client()
    if not client:
        return "LLM explanations require OPENAI_API_KEY environment variable to be set."
    
    n_interventions = scenario_results.get('n_interventions', 0)
    expected_revenue_saved = scenario_results.get('expected_revenue_saved', 0)
    roi = scenario_results.get('roi', 0)
    net_benefit = scenario_results.get('net_benefit', 0)
    
    prompt = f"""As a retention strategy consultant, explain the retention policy allocation strategy.

Key metrics:
- Total budget: ${budget:,.0f}
- Cost per intervention: ${intervention_cost:,.0f}
- Number of interventions allocated: {n_interventions}
- Expected revenue saved: ${expected_revenue_saved:,.0f}
- Net benefit: ${net_benefit:,.0f}
- ROI: {roi:.1f}%

Explain the strategic rationale behind this allocation approach. What does this tell us about the retention strategy?
Focus on business logic and decision-making framework, not technical implementation."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def analyze_what_if_scenario(scenario_comparison: Dict[str, Dict[str, Any]], question: str) -> str:
    """
    Analyze what-if scenarios based on comparison data.
    
    Args:
        scenario_comparison: Dictionary of scenario_name -> scenario_results
        question: User's what-if question
        
    Returns:
        str: Analysis and recommendations
    """
    client = get_llm_client()
    if not client:
        return "LLM explanations require OPENAI_API_KEY environment variable to be set."
    
    # Format scenario data
    scenario_summary = []
    for name, results in scenario_comparison.items():
        scenario_summary.append(
            f"{name}: ROI={results.get('roi', 0):.1f}%, "
            f"Net Benefit=${results.get('net_benefit', 0):,.0f}, "
            f"Interventions={results.get('n_interventions', 0)}"
        )
    
    scenario_text = "\n".join(scenario_summary)
    
    prompt = f"""As a retention strategy consultant, answer this what-if question based on the scenario comparison data.

Scenarios analyzed:
{scenario_text}

Question: {question}

Provide a data-driven analysis and recommendation. Be specific about trade-offs and uncertainties.
Keep response concise (3-4 paragraphs)."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"


def explain_risk_uncertainty(best_worst_case: Dict[str, Any]) -> str:
    """
    Explain risk and uncertainty in the retention strategy.
    
    Args:
        best_worst_case: Dictionary with best/worst case statistics
        
    Returns:
        str: Explanation text
    """
    client = get_llm_client()
    if not client:
        return "LLM explanations require OPENAI_API_KEY environment variable to be set."
    
    revenue_stats = best_worst_case.get('revenue_saved', {})
    
    prompt = f"""As a risk analyst, explain the uncertainty and risk in this retention strategy based on Monte Carlo simulation results.

Revenue saved statistics:
- Expected (mean): ${revenue_stats.get('mean', 0):,.0f}
- Best case (95th percentile): ${revenue_stats.get('best_case_95', 0):,.0f}
- Worst case (5th percentile): ${revenue_stats.get('worst_case_5', 0):,.0f}
- Standard deviation: ${revenue_stats.get('std', 0):,.0f}

Explain what this uncertainty means for decision-making. What should executives consider when evaluating this strategy?
Focus on practical risk management implications."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


