# RetainOps: Customer Retention Policy & Budget Optimization Simulator

RetainOps is a production-style executive dashboard for customer retention strategy and budget optimization. It combines machine learning-powered churn prediction with budget-constrained policy simulation to help leadership teams make data-driven retention decisions.

## Features

- **Data-Driven Churn Prediction**: CatBoost-based probabilistic churn risk modeling
- **Customer Segmentation**: Automatic customer segmentation and value-risk analysis
- **Budget Optimization**: Allocate retention interventions under budget constraints
- **Scenario Comparison**: Compare no-intervention, conservative, balanced, and aggressive strategies
- **AI-Powered Insights**: LLM-generated explanations of churn drivers, retention policies, and risk analysis
- **Executive Dashboard**: Clean, professional interface optimized for leadership decision-making

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set OpenAI API key for AI Strategy Assistant features:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

1. Start the Streamlit application:
```bash
python -m streamlit run app.py
```

**Windows Note**: If `streamlit` command is not recognized, use `python -m streamlit run app.py` instead.

2. Upload a CSV file with customer data containing:
   - Customer features (numeric and categorical)
   - A churn column (binary: 0/1, True/False, or auto-detected)
   - Optional: revenue column for more accurate revenue-at-risk calculations

3. Configure parameters in the sidebar:
   - Retention Budget
   - Cost per Intervention
   - Risk Tolerance (prioritize risk vs. value)

4. Click "Run Simulation" to:
   - Train the churn prediction model
   - Calculate churn probabilities
   - Allocate interventions
   - Generate scenario comparisons

5. Explore results across six tabs:
   - **Overview**: Executive KPIs and high-level summary
   - **Customer Insights**: Segmentation, churn drivers, value-risk analysis
   - **Churn Risk Model**: Model performance and feature importance
   - **Retention Policy Simulator**: Budget allocation and intervention decisions
   - **Scenario Comparison**: Compare multiple strategies with ROI and risk metrics
   - **AI Strategy Assistant**: LLM-generated insights and what-if analysis

## Data Format

Your CSV file should contain:
- **Required**: Customer features (any columns)
- **Required**: Churn column (binary target: 0/1, True/False, or column name containing "churn")
- **Optional**: Revenue column (for accurate revenue-at-risk calculations)

The system automatically:
- Detects the churn column
- Handles missing values
- Identifies categorical vs. numeric features
- Validates data quality

## Architecture

```
RetainOps/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── EXPLANATION.md           # Comprehensive system documentation
└── src/
    ├── data_utils.py        # Data loading, validation, cleaning
    ├── eda.py               # Exploratory data analysis, segmentation
    ├── ml_model.py          # CatBoost churn risk modeling
    ├── revenue.py           # Revenue at risk calculations
    ├── policy_simulator.py  # Budget-constrained allocation logic
    ├── evaluation.py        # Scenario comparison and metrics
    └── llm_helper.py        # OpenAI API integration for insights
```

## Model Details

- **Algorithm**: CatBoost Classifier (exclusively)
- **Training Strategy**: Conservative tuning with shallow trees (depth=4), strong regularization (L2=10), early stopping
- **Target Performance**: 70-80% accuracy with emphasis on probability calibration and business-aligned decisions
- **Validation**: Stratified 80/20 train/validation split

## License

This project is provided as-is for demonstration and educational purposes.

## Support

For detailed explanations of the system architecture, design decisions, and implementation details, see `EXPLANATION.md`.


